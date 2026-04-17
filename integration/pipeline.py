#!/usr/bin/env python3
"""
End-to-end pipeline -- Green Wave++

Wires:
  audio thread   ->  SirenDetector + BearingEstimator
  vision thread  ->  AmbulanceDetector (YOLOv11)
                 ↓
            event queues  (non-blocking, drop-on-overflow)
                 ↓
          async fusion loop  ->  TemporalFusionEngine
                             ->  SumoController
                             ->  WebSocket broadcast

Modes:
  demo=True    Fully synthetic data -- no hardware, no trained models needed.
               Use this for demos, development, and CI.
  demo=False   Expects real mic + camera; calls into stream_detector.py
               and vision/infer.py.  Placeholder threads for now -- swap
               in the real capture code when hardware is available.

Threading model:
  Two daemon threads fill per-modality queues.  The async fusion loop
  drains both queues on each 100ms tick, runs the fusion engine, fires
  SUMO commands, and broadcasts the payload to all dashboard clients.
  Queue depth is bounded; frames dropped rather than buffered indefinitely.
"""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Coroutine, List, Optional

import numpy as np
import yaml

# Anchor imports to the greenwave/ root regardless of CWD
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.fuser import FusionCommand, Lane, TemporalFusionEngine
from fusion.route_predictor import RoutePredictor
from fusion.sumo_controller import SumoController
from integration.logger import E2ELogger

_QUEUE_MAX = 60   # max frames buffered per modality before we start dropping


# ---------------------------------------------------------------------------
# Synthetic sources (demo mode)
# ---------------------------------------------------------------------------

class _DemoAudio:
    """
    Simulates a siren approaching from the north over ~15 seconds then leaving.
    Outputs the same dict structure as StreamingDetector.process_multichannel().
    """

    def __init__(self, hop_sec: float = 0.1):
        self._hop  = hop_sec
        self._t    = 0.0

    def next(self) -> dict:
        self._t += self._hop
        # Confidence ramps up 0 -> 0.92 over first 10s, then fades
        if self._t < 10.0:
            conf    = min(0.92, 0.15 + 0.077 * self._t)
            bearing = max(0.0, 25.0 - 2.5 * self._t)   # drifts toward north
        else:
            conf    = max(0.0, 0.92 - 0.12 * (self._t - 10.0))
            bearing = 2.0

        time.sleep(self._hop)
        return {
            "p_siren":           round(conf, 3),
            "detected":          conf > 0.50,
            "bearing_deg":       round(bearing, 1),
            "bearing_confidence": round(conf * 0.85, 3),
        }


class _DemoVision:
    """
    Simulates an ambulance detection on approach_north starting at t=2s.
    Distance decreases, speed is constant at 60 km/h.
    """

    _FPS_DELAY = 1.0 / 25.0

    def __init__(self):
        self._t = 0.0

    def next(self) -> list:
        self._t += self._FPS_DELAY
        time.sleep(self._FPS_DELAY)

        if self._t < 2.0:
            return []

        dist = max(15.0, 200.0 - 22.0 * (self._t - 2.0))
        conf = min(0.94, 0.45 + 0.038 * (self._t - 2.0))

        return [{
            "lane_id":    "approach_north",
            "confidence": round(conf, 3),
            "approaching": True,
            "distance_m": round(dist, 1),
            "speed_kmh":  60.0,
            "speed_mps":  60.0 / 3.6,
        }]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class EndToEndPipeline:
    """
    Main controller.  Call start() to launch source threads, then await
    run_fusion_loop() inside an asyncio event loop (done automatically
    by run.py).
    """

    def __init__(self, config: dict, demo: bool = False):
        self.config = config
        self.demo   = demo

        self._q_audio  = queue.Queue(maxsize=_QUEUE_MAX)
        self._q_vision = queue.Queue(maxsize=_QUEUE_MAX)
        self._running  = False

        # Core components
        lanes = [
            Lane("approach_north", heading_deg=0.0,   corridor_tls=["J_N1", "J_N2", "J_N3"]),
            Lane("approach_south", heading_deg=180.0, corridor_tls=["J_S1", "J_S2", "J_S3"]),
            Lane("approach_east",  heading_deg=90.0,  corridor_tls=["J_E1", "J_E2"]),
            Lane("approach_west",  heading_deg=270.0, corridor_tls=["J_W1", "J_W2"]),
        ]
        self.fusion    = TemporalFusionEngine(lanes, config)
        self.predictor = RoutePredictor(config=config)
        self.sumo      = SumoController(config, mock=True)
        self.logger    = E2ELogger()

        self._broadcast: Optional[Callable[..., Coroutine]] = None

    def set_broadcast(self, fn: Callable[..., Coroutine]) -> None:
        """Inject the WebSocket broadcast coroutine from server.py."""
        self._broadcast = fn

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        if self.demo:
            self._audio_src  = _DemoAudio()
            self._vision_src = _DemoVision()
            threading.Thread(target=self._audio_thread, daemon=True, name="audio").start()
            threading.Thread(target=self._vision_thread, daemon=True, name="vision").start()
        else:
            threading.Thread(target=self._live_audio,  daemon=True, name="audio").start()
            threading.Thread(target=self._live_vision, daemon=True, name="vision").start()

    def stop(self) -> None:
        self._running = False
        self.sumo.stop()
        self.logger.save()

    # ------------------------------------------------------------------
    # Source threads
    # ------------------------------------------------------------------

    def _audio_thread(self) -> None:
        while self._running:
            result = self._audio_src.next()
            _q_put(self._q_audio, (result, time.time()))

    def _vision_thread(self) -> None:
        while self._running:
            dets = self._vision_src.next()
            _q_put(self._q_vision, (dets, time.time()))

    def _live_audio(self) -> None:
        """
        Placeholder for live microphone capture.
        Replace with:
            from audio.stream_detector import StreamingDetector
            det = StreamingDetector(...)
            while self._running:
                chunk = capture_mic_chunk()    # e.g. sounddevice.read()
                result = det.process_multichannel([ch0, ch1, ch2])
                _q_put(self._q_audio, (result, time.time()))
        """
        print("[WARN]  Live audio: mic capture not wired -- use --demo")
        while self._running:
            time.sleep(1.0)

    def _live_vision(self) -> None:
        """
        Placeholder for live camera capture.
        Replace with:
            from vision.infer import AmbulanceDetector
            det = AmbulanceDetector(model_path=..., config_path=...)
            cap = cv2.VideoCapture(0)
            while self._running:
                ok, frame = cap.read()
                result = det.detect_with_lanes(frame, timestamp=time.time())
                _q_put(self._q_vision, (result['detections'], time.time()))
        """
        print("[WARN]  Live vision: camera capture not wired -- use --demo")
        while self._running:
            time.sleep(1.0)

    # ------------------------------------------------------------------
    # Fusion loop (async -- runs inside FastAPI's event loop)
    # ------------------------------------------------------------------

    async def run_fusion_loop(self) -> None:
        """
        10 Hz async loop.  Drains both queues, runs the fusion engine,
        issues SUMO preemptions, logs, and broadcasts to the dashboard.
        """
        print("[LOOP] Fusion loop running  (10 Hz)")
        while self._running:
            ts = time.time()

            # Drain audio -- take only the latest result; discard stale ones
            audio_conf, audio_bearing = 0.0, None
            while not self._q_audio.empty():
                a, _ = self._q_audio.get_nowait()
                if a.get("detected"):
                    audio_conf    = float(a.get("p_siren", 0.0))
                    audio_bearing = float(a.get("bearing_deg", 0.0))

            # Drain vision -- accumulate all pending detections
            vision_dets: List[dict] = []
            while not self._q_vision.empty():
                dets, _ = self._q_vision.get_nowait()
                vision_dets.extend(dets)

            # Fusion tick
            commands: List[FusionCommand] = self.fusion.update(
                audio_conf, audio_bearing, vision_dets, ts
            )

            # Handle new preemptions
            for cmd in commands:
                print(
                    f"[!!] PREEMPT  lane={cmd.target_lane}"
                    f"  belief={cmd.belief:.3f}"
                    f"  ETAs={cmd.eta_seconds}"
                )
                self.sumo.trigger_preemption(cmd.target_lane, cmd.corridor_tls, cmd.eta_seconds)
                self.logger.log_preempt(cmd)

            # Frame log
            self.logger.log_frame(ts, audio_conf, len(vision_dets))

            # Dashboard broadcast
            if self._broadcast is not None:
                await self._broadcast(self._build_payload(
                    ts, audio_conf, audio_bearing, vision_dets, commands
                ))

            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------

    def _build_payload(
        self,
        ts:           float,
        audio_conf:   float,
        audio_bearing: Optional[float],
        vision_dets:  List[dict],
        commands:     List[FusionCommand],
    ) -> dict:
        return {
            "timestamp":      round(ts, 3),
            "audio_conf":     round(audio_conf, 3),
            "audio_bearing":  round(audio_bearing, 1) if audio_bearing is not None else 0.0,
            "audio_detected": audio_conf > 0.50,
            "fusion_beliefs": self.fusion.get_beliefs(),
            "fusion_phases":  self.fusion.get_phases(),
            "tls_states":     self.sumo.get_tls_states(),
            "vision_count":   len(vision_dets),
            "active_preemptions": [c.target_lane for c in commands],
            "preempt_fired":  len(commands) > 0,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _q_put(q: queue.Queue, item) -> None:
    """Non-blocking enqueue; evicts the oldest entry if the queue is full."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass   # genuinely unlucky race -- just drop it


def load_config(path: Optional[Path] = None) -> dict:
    cfg_path = path or ROOT / "common" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Standalone entry (no FastAPI -- just prints fusion output)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run fusion pipeline standalone")
    ap.add_argument("--demo", action="store_true", default=True)
    ap.add_argument("--duration", type=float, default=20.0, help="Seconds to run")
    args = ap.parse_args()

    cfg      = load_config()
    pipeline = EndToEndPipeline(cfg, demo=args.demo)
    pipeline.start()

    async def _run():
        deadline = time.time() + args.duration
        task = asyncio.create_task(pipeline.run_fusion_loop())
        while time.time() < deadline:
            await asyncio.sleep(1.0)
        task.cancel()

    asyncio.run(_run())
    pipeline.stop()
