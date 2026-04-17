import asyncio
import time
import threading
import random
from ui.backend.server import broadcast

def run_demo():
    print("[>>] Starting Green Wave++ Demo...")

    async def stream_loop():
        while True:
            payload = {
                "timestamp": time.time(),
                "audio_conf": round(random.uniform(0.2, 1.0), 2),
                "audio_bearing": random.randint(0, 360),
                "vision_detections": [
                    {"id": 1, "conf": round(random.uniform(0.5, 1.0), 2)}
                ],
                "fusion_beliefs": {
                    "approach_north": random.random(),
                    "approach_south": random.random(),
                    "approach_east": random.random(),
                    "approach_west": random.random()
                },
                "tls_states": {
                    "J1": "green",
                    "J2": "yellow",
                    "J3": "red"
                }
            }

            await broadcast(payload)
            await asyncio.sleep(0.2)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream_loop())


if __name__ == "__main__":
    print("[>>] GreenWave Demo Running")
    run_demo()
