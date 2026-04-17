#!/usr/bin/env python3
"""
Advanced augmentation for emergency vehicle detection
Focus: motion blur, fog, rain, glare, night conditions
"""
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation():
    """
    Heavy augmentation pipeline for training
    Simulates challenging real-world conditions
    """
    return A.Compose([
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.7
        ),
        
        # Motion blur (moving vehicles)
        A.MotionBlur(blur_limit=(5, 25), p=0.3),
        
        # Weather conditions
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, p=1.0),  # Fog/haze
            A.RandomRain(
                slant_lower=-10, slant_upper=10,
                drop_length=20, drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                p=1.0
            ),
            A.RandomSnow(
                snow_point_lower=0.1, snow_point_upper=0.3,
                brightness_coeff=2.5,
                p=1.0
            ),
        ], p=0.3),
        
        # Lighting conditions
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3),
                contrast_limit=(-0.3, 0.3),
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(60, 140), p=1.0),  # Night/day
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=3,
                num_flare_circles_upper=6,
                src_radius=200,
                p=1.0
            ),  # Sun glare
        ], p=0.4),
        
        # Color/quality
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
            A.ChannelShuffle(p=1.0),
        ], p=0.3),
        
        # Blur and noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.2),
        
        # Compression artifacts
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
        
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3  # Keep boxes with >30% visibility
    ))


def get_validation_augmentation():
    """Minimal augmentation for validation"""
    return A.Compose([
        # Only resize/normalize, no data augmentation
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))


def test_augmentation():
    """Test augmentation pipeline with sample image"""
    print("=" * 60)
    print("Augmentation Pipeline Test")
    print("=" * 60)
    
    # Check if albumentations is installed
    try:
        import albumentations
        print(f"[OK] Albumentations version: {albumentations.__version__}")
    except ImportError:
        print("[FAIL] Albumentations not installed")
        print("  Install: pip install albumentations")
        return
    
    # Create synthetic test image
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Add a rectangle to simulate vehicle
    cv2.rectangle(img, (200, 200), (440, 440), (255, 0, 0), -1)
    
    # Test bounding box
    bboxes = [[0.5, 0.5, 0.375, 0.375]]  # center_x, center_y, width, height
    class_labels = [0]
    
    # Apply augmentation
    transform = get_training_augmentation()
    
    print("\nApplying augmentations...")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Apply multiple times to show variety
    for i in range(1, 6):
        augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        
        # Draw boxes
        aug_img_viz = aug_img.copy()
        if len(aug_bboxes) > 0:
            h, w = aug_img.shape[:2]
            for bbox in aug_bboxes:
                x_c, y_c, bw, bh = bbox
                x1 = int((x_c - bw/2) * w)
                y1 = int((y_c - bh/2) * h)
                x2 = int((x_c + bw/2) * w)
                y2 = int((y_c + bh/2) * h)
                cv2.rectangle(aug_img_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        axes[i].imshow(cv2.cvtColor(aug_img_viz, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    output_path = Path("../outputs/augmentation_test.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved to {output_path}")


if __name__ == "__main__":
    test_augmentation()