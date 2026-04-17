import cv2
import numpy as np
import librosa
import time

class SynchronizedReplayer:
    def __init__(self, video_path, audio_path, speed_multiplier=1.0):
        self.video_path = video_path
        self.audio_path = audio_path
        self.speed = speed_multiplier

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0

        self.audio, self.sr = librosa.load(audio_path, sr=16000)
        self.start_time = None

    def get_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        timestamp = self.current_frame / self.fps
        self.current_frame += 1
        return frame, timestamp

    def get_audio_window(self, timestamp, window_sec=1.0):
        center = int(timestamp * self.sr)
        half = int(window_sec * self.sr / 2)
        start = max(0, center - half)
        end = start + int(window_sec * self.sr)
        return self.audio[start:end]

    def sleep(self):
        time.sleep(1.0 / self.fps)

    def release(self):
        self.cap.release()
