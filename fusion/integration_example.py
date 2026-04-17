#!/usr/bin/env python3
"""
Example showing how to integrate fusion with audio + vision pipelines.
This is a stub for demonstration purposes only.
"""

import time
import numpy as np
from fuser import TemporalFusionEngine, Lane


# 1️⃣ Define intersection lanes
lanes = [
    Lane("approach_north", 0.0, ["J1", "J2"]),
    Lane("approach_south", 180.0, ["J3", "J4"]),
    Lane("approach_east", 90.0, ["J5"]),
    Lane("approach_west", 270.0, ["J6"])
]


# 2️⃣ Initialize fusion engine
fusion = TemporalFusionEngine(
    lanes=lanes,
    decay_rate=0.92,
    fuse_gain=0.3,
    belief_threshold=0.75
)


# 3️⃣ Simulated main loop
print("=== GreenWave++ Fusion Integration Example ===")
print("Press Ctrl+C to stop.\n")

try:
    t = 0.0
    while t < 3.0:  # run for a few simulated seconds
        # Fake audio inference output
        audio_conf = 0.8
        audio_bearing = 5.0 if t < 1.5 else 185.0  # changes from north to south

        # Fake vision detections
        if t < 1.5:
            vision_dets = [{
                "lane_id": "approach_north",
                "confidence": 0.7,
                "approaching": True,
                "distance_m": 120.0,
                "speed_mps": 15.0
            }]
        else:
            vision_dets = [{
                "lane_id": "approach_south",
                "confidence": 0.8,
                "approaching": True,
                "distance_m": 100.0,
                "speed_mps": 14.0
            }]

        # Fusion update
        cmd = fusion.update(audio_conf, audio_bearing, vision_dets, timestamp=t)

        beliefs = fusion.get_beliefs()
        print(f"t={t:.1f}s | beliefs={beliefs}")

        if cmd:
            print(f"[!!] Preempt Triggered for {cmd.target_lane}")
            print(f"  -> Corridor: {cmd.corridor_tls}")
            print(f"  -> ETA: {cmd.eta_seconds[0]:.1f}s")

        time.sleep(0.2)
        t += 0.2

except KeyboardInterrupt:
    print("\nStopped manually.")
