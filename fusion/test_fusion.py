import sys
sys.path.append('..')

from fusion.fuser import TemporalFusionEngine, Lane
import numpy as np
import time

def test_fusion_basic():
    """Test basic fusion logic"""
    
    # Define 4 approach lanes (N, E, S, W)
    lanes = [
        Lane("north", 0.0, ["TLS1", "TLS2", "TLS3"]),
        Lane("east", 90.0, ["TLS4", "TLS5"]),
        Lane("south", 180.0, ["TLS6", "TLS7"]),
        Lane("west", 270.0, ["TLS8", "TLS9"])
    ]
    
    engine = TemporalFusionEngine(lanes)
    
    print("=== Test 1: Audio-only detection from North ===")
    t = 0.0
    for i in range(15):
        cmd = engine.update(
            audio_conf=0.85,
            audio_bearing=5.0,  # Near north
            vision_detections=[],
            timestamp=t
        )
        beliefs = engine.get_beliefs()
        print(f"t={t:.1f}s | North belief: {beliefs['north']:.3f} | Preempt: {cmd is not None}")
        
        if cmd:
            print(f"  -> Triggered! Lane: {cmd.target_lane}, Corridor: {cmd.corridor_tls}")
        
        t += 0.1
        time.sleep(0.05)
    
    print("\n=== Test 2: Vision + Audio fusion ===")
    engine = TemporalFusionEngine(lanes)
    
    t = 0.0
    for i in range(12):
        vision = [{
            'lane_id': 'east',
            'confidence': 0.75,
            'approaching': True,
            'distance_m': 150.0,
            'speed_mps': 12.0
        }] if i > 3 else []
        
        cmd = engine.update(
            audio_conf=0.6,
            audio_bearing=85.0,  # Near east
            vision_detections=vision,
            timestamp=t
        )
        beliefs = engine.get_beliefs()
        print(f"t={t:.1f}s | East belief: {beliefs['east']:.3f} | Armed: {engine.states['east'].armed} | Preempt: {cmd is not None}")
        
        if cmd:
            print(f"  -> ETAs: {[f'{eta:.1f}s' for eta in cmd.eta_seconds]}")
        
        t += 0.1
        time.sleep(0.05)
    
    print("\n=== Test 3: Decay without input ===")
    engine = TemporalFusionEngine(lanes)
    
    # Boost north lane
    engine.update(0.9, 0.0, [], 0.0)
    
    t = 0.1
    for i in range(20):
        engine.update(0.0, None, [], t)
        belief = engine.get_beliefs()['north']
        print(f"t={t:.1f}s | North belief: {belief:.3f}")
        t += 0.1
        time.sleep(0.03)
    
    print("\n[DONE] All fusion tests completed!")

if __name__ == "__main__":
    test_fusion_basic()
