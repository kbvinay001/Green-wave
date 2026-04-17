#!/usr/bin/env python3
"""
Helper script for downloading ambulance images
Provides instructions and utilities
"""
import requests
from pathlib import Path
import json


def download_instructions():
    """Print data collection instructions"""
    print("=" * 60)
    print("Ambulance Dataset Collection Guide")
    print("=" * 60)
    
    print("\n📥 Recommended Data Sources:\n")
    
    print("1. Roboflow Universe (Pre-labeled)")
    print("   - https://universe.roboflow.com/")
    print("   - Search: 'ambulance', 'emergency vehicle'")
    print("   - Download in YOLOv8 format")
    print("   - Many datasets with 100-1000+ images")
    
    print("\n2. Google Open Images V7")
    print("   - https://storage.googleapis.com/openimages/web/index.html")
    print("   - Search class: '/m/012n7d' (ambulance)")
    print("   - Use OIDv4 toolkit to download")
    
    print("\n3. Kaggle Datasets")
    print("   - https://www.kaggle.com/datasets")
    print("   - Search: 'ambulance', 'emergency vehicle', 'traffic CCTV'")
    print("   - Download and convert to YOLO format")
    
    print("\n4. YouTube Frame Extraction")
    print("   - Search: 'dash cam ambulance', 'ambulance POV'")
    print("   - Use: youtube-dl + ffmpeg to extract frames")
    print("   - Label with LabelImg or Roboflow")
    
    print("\n5. Custom Collection")
    print("   - Smartphone camera (with permission)")
    print("   - Dash cam footage")
    print("   - Public traffic cameras")
    
    print("\n" + "=" * 60)
    print("📝 Labeling Tools:")
    print("=" * 60)
    
    print("\n- LabelImg (Desktop): https://github.com/heartexlabs/labelImg")
    print("- Roboflow (Web): https://roboflow.com/ (free tier)")
    print("- CVAT (Web/Self-hosted): https://cvat.org/")
    print("- Labelbox (Web): https://labelbox.com/")
    
    print("\n" + "=" * 60)
    print("🎯 Quick Start Option:")
    print("=" * 60)
    
    print("\n1. Visit: https://universe.roboflow.com/")
    print("2. Search: 'ambulance detection'")
    print("3. Choose dataset with 500+ images")
    print("4. Download in 'YOLOv8' format")
    print("5. Extract to vision/data/")
    print("6. Run: python prepare_data.py --roboflow <extracted_dir>")


def sample_roboflow_format():
    """Show expected Roboflow directory structure"""
    print("\n" + "=" * 60)
    print("Expected Roboflow Directory Structure:")
    print("=" * 60)
    
    structure = """
roboflow_download/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt
│       └── img2.txt
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml  # Contains class names
"""
    print(structure)


def create_sample_data_yaml():
    """Create sample data.yaml for YOLO training"""
    data_yaml = {
        'path': '../data',
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'ambulance',
            1: 'lightbar'
        },
        'nc': 2
    }
    
    output_path = Path('data/dataset.yaml')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"path: {data_yaml['path']}\n")
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        f.write(f"test: {data_yaml['test']}\n\n")
        f.write(f"nc: {data_yaml['nc']}\n")
        f.write("names:\n")
        for k, v in data_yaml['names'].items():
            f.write(f"  {k}: {v}\n")
    
    print(f"\n[OK] Created template: {output_path}")


def verify_yolo_format(label_file: str):
    """Verify YOLO label file format"""
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[FAIL] Line {i+1}: Expected 5 values, got {len(parts)}")
                return False
            
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                print(f"[FAIL] Line {i+1}: Coordinates not normalized [0,1]")
                return False
            
            if class_id not in [0, 1]:
                print(f"[FAIL] Line {i+1}: Invalid class_id {class_id} (expected 0 or 1)")
                return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error reading {label_file}: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-label', help='Verify YOLO label file format')
    parser.add_argument('--create-yaml', action='store_true', 
                       help='Create sample dataset.yaml')
    args = parser.parse_args()
    
    if args.check_label:
        if verify_yolo_format(args.check_label):
            print(f"[OK] {args.check_label} is valid YOLO format")
    elif args.create_yaml:
        create_sample_data_yaml()
    else:
        download_instructions()
        sample_roboflow_format()
        
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("=" * 60)
        print("\n1. Download dataset from one of the sources above")
        print("2. Place in vision/data/raw/ or use Roboflow structure")
        print("3. Run: python prepare_data.py")
        print("4. Verify with: python prepare_data.py --verify")


if __name__ == "__main__":
    main()
