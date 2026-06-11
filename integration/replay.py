#!/usr/bin/env python3
"""
Virtual sensor sources -- Green Wave++  (run.py --virtual)

Replays a video file and a WAV file as if they were the live camera and
microphone.  Both sources are paced against ONE shared media clock, so an
event at t=7.0s in the video and t=7.0s in the audio reaches the fusion
engine at the same wall-clock moment (within one hop).

    VirtualClock      wall-clock <-> media-time mapping (speed multiplier)
    VideoFileSource   cv2.VideoCapture -> (frame, t_media) at native FPS
    AudioFileSource   WAV -> fixed-size chunks (default 100 ms), all channels
    AudioDetectorAdapter   chunks -> {p_siren, bearing_deg, ...}   (CRNN [+GCC-PHAT])
    VisionDetectorAdapter  frames -> [fusion detection dicts]      (YOLOv11)
    VirtualSensorRig  owns the above; fills the pipeline's two queues

Mono WAVs cannot carry bearing information, so the rig uses the configured
camera lane's heading as the bearing (a fixed roadside unit knows which
approach it listens to).  Multichannel WAVs (>= 2ch) get real GCC-PHAT.

A monocular, uncalibrated camera cannot measure metric speed or distance;
config 'virtual.default_speed_kmh' / 'assume_approaching' fill those gaps
honestly (documented stand-ins, not measurements).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

class VirtualClock:
    """Maps wall time to media time with a speed multiplier."""

    def __init__(self, speed: float = 1.0):
        self.speed = max(0.01, float(speed))
        self._t0: Optional[float] = None

    def start(self) -> None:
        self._t0 = time.time()

    @property
    def started(self) -> bool:
        return self._t0 is not None

    def media_now(self) -> float:
        if self._t0 is None:
            return 0.0
        return (time.time() - self._t0) * self.speed

    def sleep_until(self, t_media: float) -> None:
        """Block until the clock reaches t_media (no-op if already past)."""
        if self._t0 is None:
            return
        wall_target = self._t0 + t_media / self.speed
        delay = wall_target - time.time()
        if delay > 0:
            time.sleep(delay)


# ---------------------------------------------------------------------------
# File sources
# ---------------------------------------------------------------------------

class VideoFileSource:
    """Iterate (bgr_frame, t_media) over a video file via cv2.VideoCapture."""

    def __init__(self, path: str, loop: bool = False):
        import cv2
        self._cv2 = cv2
        self.path = str(path)
        self.loop = loop
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._idx = 0
        self._loops = 0

    @property
    def duration_sec(self) -> float:
        return self.n_frames / self.fps if self.n_frames > 0 else 0.0

    def __iter__(self) -> Iterator[tuple]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if not self.loop:
                    break
                self._loops += 1
                self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            t_media = (self._idx + self._loops * max(self.n_frames, 1)) / self.fps
            self._idx += 1
            yield frame, t_media

    def release(self) -> None:
        self.cap.release()


class AudioFileSource:
    """
    Iterate (channels, t_media) over a WAV in fixed-size chunks.

    channels: list of 1-D float32 arrays (one per channel), resampled to sr.
    """

    def __init__(self, path: str, sr: int = 16000, chunk_sec: float = 0.1,
                 loop: bool = False):
        import librosa
        self.path = str(path)
        self.sr = int(sr)
        self.chunk = int(round(chunk_sec * sr))
        self.chunk_sec = chunk_sec
        self.loop = loop

        y, _ = librosa.load(self.path, sr=self.sr, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        self.data = y.astype(np.float32)          # (n_ch, n_samples)
        self.n_channels = self.data.shape[0]

    @property
    def duration_sec(self) -> float:
        return self.data.shape[1] / self.sr

    def __iter__(self) -> Iterator[tuple]:
        n = self.data.shape[1]
        start, loops = 0, 0
        while True:
            end = start + self.chunk
            if end > n:
                if not self.loop:
                    break
                start, loops = 0, loops + 1
                continue
            t_media = (start + loops * n) / self.sr
            yield [self.data[c, start:end] for c in range(self.n_channels)], t_media
            start = end


# ---------------------------------------------------------------------------
# Detector adapters (model-level objects are injectable for tests)
# ---------------------------------------------------------------------------

class AudioDetectorAdapter:
    """
    Chunks -> the dict shape the pipeline's audio queue expects:
        {p_siren, detected, bearing_deg, bearing_confidence}

    Multichannel input -> StreamingDetector (CRNN + GCC-PHAT bearing).
    Mono input        -> SirenDetector (CRNN) + fixed fallback bearing.
    """

    def __init__(self, fallback_bearing_deg: float = 0.0,
                 detector=None, model_path: Optional[str] = None,
                 config_path: Optional[str] = None):
        self.fallback_bearing = float(fallback_bearing_deg)
        self._det = detector            # injected stub in tests
        self._doppler = None
        self._t_audio = 0.0
        self._sr = 16000
        if self._det is None:
            cfg = config_path or str(ROOT / "common" / "config.yaml")
            ckpt = model_path or str(ROOT / "checkpoints" / "audio_best.pt")
            if not Path(ckpt).exists():
                raise FileNotFoundError(
                    f"Audio checkpoint missing: {ckpt}\n  Train first: python audio/train.py")
            sys.path.insert(0, str(ROOT / "audio"))
            from stream_detector import StreamingDetector, DopplerTracker
            from infer import SirenDetector
            self._multi_det = StreamingDetector(model_path=ckpt, config_path=cfg)
            self._mono_det = SirenDetector(model_path=ckpt, config_path=cfg)
            # mono replay bypasses StreamingDetector, so it gets its own tracker
            import yaml
            with open(cfg) as f:
                full_cfg = yaml.safe_load(f)
            self._sr = int(full_cfg["audio"]["sample_rate"])
            self._doppler = DopplerTracker.from_config(full_cfg)

    def process(self, channels: List[np.ndarray]) -> dict:
        if self._det is not None:                     # test stub
            return self._det(channels)

        if len(channels) >= 2:
            return self._multi_det.process_multichannel(channels)

        r = self._mono_det.process_chunk(channels[0])
        self._t_audio += len(channels[0]) / self._sr
        detected = bool(r["detected"])

        doppler = {"doppler_factor": 1.0, "pitch_slope_hz_s": 0.0}
        if self._doppler is not None:
            doppler = (self._doppler.update(channels[0], self._t_audio)
                       if detected else self._doppler.verdict())

        return {
            "p_siren": float(r["p_siren"]),
            "detected": detected,
            "bearing_deg": self.fallback_bearing if detected else 0.0,
            "bearing_confidence": 0.8 * float(r["p_siren"]) if detected else 0.0,
            **doppler,
        }


class VisionDetectorAdapter:
    """
    Frames -> the list of fusion detection dicts:
        {lane_id, confidence, approaching, distance_m, speed_kmh, speed_mps}

    A fixed camera watches exactly one approach, so every accepted detection
    maps to config 'virtual.camera_lane'.  Tracker speed/approach estimates
    are used when available; configured stand-ins otherwise (see module doc).
    """

    def __init__(self, camera_lane: str, assume_approaching: bool = True,
                 default_speed_kmh: float = 50.0,
                 detector=None, weights: Optional[str] = None,
                 config_path: Optional[str] = None):
        self.camera_lane = camera_lane
        self.assume_approaching = assume_approaching
        self.default_speed_kmh = float(default_speed_kmh)
        self._det = detector            # injected stub in tests
        if self._det is None:
            cfg = config_path or str(ROOT / "common" / "config.yaml")
            w = weights or str(ROOT / "vision" / "weights" / "yolov11s-ambulance.pt")
            if not Path(w).exists():
                raise FileNotFoundError(
                    f"Vision weights missing: {w}\n  Train first: python vision/train.py")
            from vision.infer import AmbulanceDetector
            self._yolo = AmbulanceDetector(model_path=w, config_path=cfg,
                                           device="cuda" if _cuda_ok() else "cpu")

    def process(self, frame: np.ndarray, t_media: float) -> List[dict]:
        if self._det is not None:                     # test stub
            raw = self._det(frame, t_media)
        else:
            raw = self._yolo.detect_with_lanes(frame, timestamp=t_media)["detections"]

        out = []
        for d in raw:
            approaching = d.get("approaching")
            if approaching is None:
                approaching = self.assume_approaching
            speed_kmh = d.get("speed_kmh") or self.default_speed_kmh
            out.append({
                "lane_id":     self.camera_lane,
                "confidence":  float(d.get("confidence", 0.0)),
                "approaching": bool(approaching),
                "distance_m":  d.get("distance_m"),       # None: monocular, uncalibrated
                "speed_kmh":   float(speed_kmh),
                "speed_mps":   float(speed_kmh) / 3.6,
            })
        return out


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Rig: sources + adapters -> pipeline queues
# ---------------------------------------------------------------------------

class VirtualSensorRig:
    """
    Drives the pipeline's audio/vision queues from files, time-synchronized.

    put_audio / put_vision are callables (item) -> None supplied by the
    pipeline (its drop-on-overflow queue writers).
    """

    def __init__(
        self,
        audio_source: Optional[AudioFileSource],
        video_source: Optional[VideoFileSource],
        audio_adapter: Optional[AudioDetectorAdapter],
        vision_adapter: Optional[VisionDetectorAdapter],
        speed: float = 1.0,
        on_finished: Optional[Callable[[str], None]] = None,
    ):
        if audio_source is None and video_source is None:
            raise ValueError("virtual mode needs at least one of --wav / --video")
        self.audio_source = audio_source
        self.video_source = video_source
        self.audio_adapter = audio_adapter
        self.vision_adapter = vision_adapter
        self.clock = VirtualClock(speed)
        self.on_finished = on_finished or (lambda name: print(f"[virtual] {name} finished"))
        self._threads: List[threading.Thread] = []
        self.running = False

    def start(self, put_audio: Callable, put_vision: Callable) -> None:
        self.running = True
        self.clock.start()
        if self.audio_source is not None:
            t = threading.Thread(target=self._audio_loop, args=(put_audio,),
                                 daemon=True, name="virtual-audio")
            t.start(); self._threads.append(t)
        if self.video_source is not None:
            t = threading.Thread(target=self._video_loop, args=(put_vision,),
                                 daemon=True, name="virtual-vision")
            t.start(); self._threads.append(t)

    def stop(self) -> None:
        self.running = False

    def join(self, timeout: Optional[float] = None) -> None:
        for t in self._threads:
            t.join(timeout)

    @property
    def finished(self) -> bool:
        return bool(self._threads) and all(not t.is_alive() for t in self._threads)

    # ------------------------------------------------------------------

    def _audio_loop(self, put_audio: Callable) -> None:
        for channels, t_media in self.audio_source:
            if not self.running:
                return
            self.clock.sleep_until(t_media)
            result = self.audio_adapter.process(channels)
            result["t_media"] = round(t_media, 3)
            put_audio((result, time.time()))
        self.on_finished("audio")

    def _video_loop(self, put_vision: Callable) -> None:
        for frame, t_media in self.video_source:
            if not self.running:
                return
            self.clock.sleep_until(t_media)
            dets = self.vision_adapter.process(frame, t_media)
            put_vision((dets, time.time()))
        self.video_source.release()
        self.on_finished("video")


# ---------------------------------------------------------------------------
# Factory from config
# ---------------------------------------------------------------------------

def build_rig_from_config(
    config: dict,
    video: Optional[str] = None,
    wav: Optional[str] = None,
    lane_headings: Optional[dict] = None,
) -> VirtualSensorRig:
    """
    Build a VirtualSensorRig from config['virtual'] (CLI args override).

    lane_headings: {lane_name: heading_deg} from the pipeline, used to derive
    the mono-audio fallback bearing from the camera lane.
    """
    vc = dict(config.get("virtual", {}))
    video = video or vc.get("video") or None
    wav = wav or vc.get("wav") or None
    camera_lane = vc.get("camera_lane", "approach_north")
    speed = float(vc.get("speed", 1.0))
    loop = bool(vc.get("loop", False))
    sr = int(config["audio"]["sample_rate"])
    hop_sec = float(config["audio"]["hop_sec"])

    bearing = vc.get("audio_bearing_deg")
    if bearing is None:
        bearing = (lane_headings or {}).get(camera_lane, 0.0)

    audio_source = audio_adapter = None
    if wav:
        wav_path = Path(wav) if Path(wav).is_absolute() else ROOT / wav
        audio_source = AudioFileSource(str(wav_path), sr=sr, chunk_sec=hop_sec, loop=loop)
        audio_adapter = AudioDetectorAdapter(fallback_bearing_deg=float(bearing))

    video_source = vision_adapter = None
    if video:
        vid_path = Path(video) if Path(video).is_absolute() else ROOT / video
        video_source = VideoFileSource(str(vid_path), loop=loop)
        vision_adapter = VisionDetectorAdapter(
            camera_lane=camera_lane,
            assume_approaching=bool(vc.get("assume_approaching", True)),
            default_speed_kmh=float(vc.get("default_speed_kmh", 50.0)),
        )

    rig = VirtualSensorRig(audio_source, video_source, audio_adapter,
                           vision_adapter, speed=speed)
    a = f"{audio_source.duration_sec:.1f}s/{audio_source.n_channels}ch" if audio_source else "none"
    v = f"{video_source.duration_sec:.1f}s@{video_source.fps:.0f}fps" if video_source else "none"
    print(f"[virtual] audio={a}  video={v}  lane={camera_lane}  "
          f"bearing={bearing:.0f}deg  speed=x{speed}")
    return rig
