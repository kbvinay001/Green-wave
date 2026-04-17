#!/usr/bin/env python3
"""
Prepare vision dataset for YOLOv11 training
- Convert formats if needed
- Apply train/val/test split
- Verify annotations
"""
import shutil
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import numpy as np
import argparse


def convert_roboflow_to_standard(roboflow_dir: str, output_dir: str = "data"):
    """
    Convert Roboflow download structure to our standard format
    """
    print("=" * 60)
    print("Converting Roboflow Dataset")
    print("=" * 60)
    
    roboflow_dir = Path(roboflow_dir)
    output_dir = Path(output_dir)
    
    # Check for data.yaml
    data_yaml_path = roboflow_dir / "data.yaml"
    if data_yaml_path.exists():
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        print(f"[OK] Found data.yaml with {data_config.get('nc', 0)} classes")
    
    # Copy splits
    for split in ['train', 'valid', 'test']:
        roboflow_split = roboflow_dir / split
        
        if not roboflow_split.exists():
            if split == 'valid':
                # Try 'val' instead
                roboflow_split = roboflow_dir / 'val'
        
        if roboflow_split.exists():
            output_split = output_dir / ('val' if split == 'valid' else split)
            
            # Copy images
            src_images = roboflow_split / "images"
            dst_images = output_split / "images"
            dst_images.mkdir(parents=True, exist_ok=True)
            
            if src_images.exists():
                for img_file in src_images.glob("*.*"):
                    shutil.copy2(img_file, dst_images / img_file.name)
            
            # Copy labels
            src_labels = roboflow_split / "labels"
            dst_labels = output_split / "labels"
            dst_labels.mkdir(parents=True, exist_ok=True)
            
            if src_labels.exists():
                for lbl_file in src_labels.glob("*.txt"):
                    shutil.copy2(lbl_file, dst_labels / lbl_file.name)
            
            n_images = len(list(dst_images.glob("*.*")))
            n_labels = len(list(dst_labels.glob("*.txt")))
            print(f"[OK] {split}: {n_images} images, {n_labels} labels")
    
    # Create dataset.yaml
    create_dataset_yaml(output_dir)
    
    print(f"\n[OK] Dataset converted to {output_dir}")


def create_dataset_yaml(data_dir: Path):
    """Create dataset.yaml for YOLO training"""
    data_yaml = {
        'path': str(data_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'ambulance',
            1: 'lightbar'
        },
        'nc': 2
    }
    
    yaml_path = data_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"[OK] Created {yaml_path}")


def verify_dataset(data_dir: str = "data"):
    """
    Verify dataset integrity
    - Check image/label pairs
    - Validate YOLO format
    - Check class distribution
    - Detect issues
    """
    print("=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    
    data_dir = Path(data_dir)
    
    issues = []
    stats = {
        'train': {'images': 0, 'labels': 0, 'boxes': 0},
        'val': {'images': 0, 'labels': 0, 'boxes': 0},
        'test': {'images': 0, 'labels': 0, 'boxes': 0}
    }
    
    class_counts = {0: 0, 1: 0}  # ambulance, lightbar
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        
        if not split_dir.exists():
            print(f"[WARN] Split '{split}' not found")
            continue
        
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            issues.append(f"{split}: Missing images or labels directory")
            continue
        
        image_files = list(images_dir.glob("*.*"))
        stats[split]['images'] = len(image_files)
        
        for img_file in tqdm(image_files, desc=f"Checking {split}"):
            # Check corresponding label
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            if not label_file.exists():
                issues.append(f"{split}/{img_file.name}: No label file")
                continue
            
            stats[split]['labels'] += 1
            
            # Verify image readability
            img = cv2.imread(str(img_file))
            if img is None:
                issues.append(f"{split}/{img_file.name}: Cannot read image")
                continue
            
            # Verify label format
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        issues.append(f"{split}/{label_file.name}: Invalid format")
                        continue
                    
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    
                    # Check normalization
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        issues.append(f"{split}/{label_file.name}: Coordinates not normalized")
                    
                    # Check class ID
                    if class_id not in [0, 1]:
                        issues.append(f"{split}/{label_file.name}: Invalid class {class_id}")
                    else:
                        class_counts[class_id] += 1
                        stats[split]['boxes'] += 1
                    
            except Exception as e:
                issues.append(f"{split}/{label_file.name}: Parse error - {e}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    for split, counts in stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Images: {counts['images']}")
        print(f"  Labels: {counts['labels']}")
        print(f"  Bounding boxes: {counts['boxes']}")
    
    print(f"\nClass Distribution:")
    print(f"  Ambulance: {class_counts[0]}")
    print(f"  Lightbar: {class_counts[1]}")
    
    # Print issues
    if issues:
        print("\n" + "=" * 60)
        print(f"[WARN] Found {len(issues)} Issues")
        print("=" * 60)
        for issue in issues[:20]:  # Show first 20
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("\n[OK] No issues found!")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    
    total_images = sum(s['images'] for s in stats.values())
    
    if total_images < 100:
        print("[WARN] Very small dataset (<100 images)")
        print("  - Recommend: 500+ images for production")
        print("  - Can proceed for testing, but expect lower accuracy")
    elif total_images < 500:
        print("[WARN] Small dataset (100-500 images)")
        print("  - Recommend: Add more diverse images")
        print("  - Focus on: different weather, lighting, angles")
    else:
        print("[OK] Good dataset size (500+ images)")
    
    if stats['val']['images'] < stats['train']['images'] * 0.1:
        print("[WARN] Validation set too small")
        print("  - Recommend: 15-20% of training set")
    
    if class_counts[0] == 0:
        print("[FAIL] No ambulance annotations found!")
    
    return len(issues) == 0


def visualize_samples(data_dir: str = "data", n_samples: int = 9):
    """
    Visualize random samples with annotations
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    print("=" * 60)
    print("Visualizing Dataset Samples")
    print("=" * 60)
    
    data_dir = Path(data_dir)
    train_images = list((data_dir / "train" / "images").glob("*.*"))
    
    if len(train_images) == 0:
        print("[WARN] No training images found")
        return
    
    # Random samples
    samples = np.random.choice(train_images, min(n_samples, len(train_images)), replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    class_names = {0: 'ambulance', 1: 'lightbar'}
    colors = {0: 'red', 1: 'yellow'}
    
    for idx, img_file in enumerate(samples):
        # Read image
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Read labels
        label_file = data_dir / "train" / "labels" / f"{img_file.stem}.txt"
        
        ax = axes[idx]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(img_file.name, fontsize=8)
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        class_id = int(class_id)
                        
                        # Convert to pixel coordinates
                        x_center *= w
                        y_center *= h
                        width *= w
                        height *= h
                        
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        
                        # Draw box
                        rect = patches.Rectangle(
                            (x1, y1), width, height,
                            linewidth=2,
                            edgecolor=colors.get(class_id, 'white'),
                            facecolor='none'
                        )
                        ax.add_patch(rect)
                        
                        # Add label
                        ax.text(
                            x1, y1 - 5,
                            class_names.get(class_id, 'unknown'),
                            color=colors.get(class_id, 'white'),
                            fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
                        )
    
    plt.tight_layout()
    output_path = Path("../outputs/dataset_samples.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare vision dataset")
    parser.add_argument('--roboflow', help='Convert from Roboflow directory')
    parser.add_argument('--verify', action='store_true', help='Verify dataset integrity')
    parser.add_argument('--visualize', action='store_true', help='Visualize samples')
    parser.add_argument('--data-dir', default='data', help='Dataset directory')
    args = parser.parse_args()
    
    if args.roboflow:
        convert_roboflow_to_standard(args.roboflow, args.data_dir)
    
    if args.verify:
        verify_dataset(args.data_dir)
    
    if args.visualize:
        visualize_samples(args.data_dir)
    
    if not any([args.roboflow, args.verify, args.visualize]):
        print("No action specified. Use --help for options")
        print("\nQuick start:")
        print("  1. Download dataset from Roboflow")
        print("  2. python prepare_data.py --roboflow <download_dir>")
        print("  3. python prepare_data.py --verify")
        print("  4. python prepare_data.py --visualize")


if __name__ == "__main__":
    main()