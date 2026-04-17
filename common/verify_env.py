#!/usr/bin/env python3
"""
Environment verification script for Green Wave++
"""
import sys
import subprocess

def check_import(module_name, package_name=None):
    """Try importing a module and report status"""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"[OK] {package_name} imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] {package_name} import failed: {e}")
        return False

def check_torch_cuda():
    """Check PyTorch CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] PyTorch CUDA available")
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - Device count: {torch.cuda.device_count()}")
            print(f"  - Device name: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("[WARN] PyTorch installed but CUDA not available")
            return False
    except Exception as e:
        print(f"[FAIL] PyTorch CUDA check failed: {e}")
        return False

def check_yolo():
    """Check YOLOv11 availability"""
    try:
        from ultralytics import YOLO
        print("[OK] Ultralytics YOLOv11 available")
        # Try loading a model
        model = YOLO('yolov8n.pt')  # Will download if not present
        print(f"  - Successfully loaded YOLOv8n as test")
        return True
    except Exception as e:
        print(f"[FAIL] YOLOv11 check failed: {e}")
        return False

def check_sumo():
    """Check SUMO installation"""
    try:
        import traci
        print("[OK] TraCI (SUMO) imported successfully")
        
        # Check SUMO_HOME environment variable
        import os
        sumo_home = os.environ.get('SUMO_HOME')
        if sumo_home:
            print(f"  - SUMO_HOME: {sumo_home}")
        else:
            print("  [WARN] SUMO_HOME not set (may need to set for TraCI)")
        
        return True
    except Exception as e:
        print(f"[FAIL] SUMO/TraCI check failed: {e}")
        return False

def check_librosa():
    """Check librosa and audio processing"""
    try:
        import librosa
        import soundfile
        print(f"[OK] Librosa {librosa.__version__} available")
        print(f"[OK] SoundFile available")
        return True
    except Exception as e:
        print(f"[FAIL] Audio libraries check failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Green Wave++ Environment Verification")
    print("=" * 60)
    
    checks = []
    
    print("\n[1/6] Core Dependencies")
    checks.append(check_import("numpy"))
    checks.append(check_import("scipy"))
    checks.append(check_import("yaml", "pyyaml"))
    
    print("\n[2/6] PyTorch & CUDA")
    checks.append(check_import("torch"))
    checks.append(check_torch_cuda())
    
    print("\n[3/6] YOLOv11")
    checks.append(check_yolo())
    
    print("\n[4/6] Audio Processing")
    checks.append(check_librosa())
    
    print("\n[5/6] Computer Vision")
    checks.append(check_import("cv2", "opencv-python"))
    
    print("\n[6/6] SUMO Traffic Simulation")
    checks.append(check_sumo())
    
    print("\n" + "=" * 60)
    passed = sum(checks)
    total = len(checks)
    print(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("[OK] Environment ready for Green Wave++ development!")
        return 0
    else:
        print("[WARN] Some checks failed. Install missing dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main())