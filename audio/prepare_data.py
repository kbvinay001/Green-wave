#!/usr/bin/env python3
"""
Prepare audio dataset: validate, augment, split
"""
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

from preprocess import AudioPreprocessor, AudioAugmenter, verify_audio_file


def prepare_dataset(
    siren_dir: str = "data/raw/sirens",
    noise_dir: str = "data/raw/noise",
    output_dir: str = "data/processed",
    augment_dir: str = "data/augmented",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
):
    """
    Full data preparation pipeline
    """
    print("=" * 60)
    print("Green Wave++ Audio Data Preparation")
    print("=" * 60)
    
    siren_dir = Path(siren_dir)
    noise_dir = Path(noise_dir)
    output_dir = Path(output_dir)
    augment_dir = Path(augment_dir)
    
    # Step 1: Validate raw files
    print("\n[1/4] Validating raw audio files...")
    siren_files = list(siren_dir.glob("*.wav"))
    noise_files = list(noise_dir.glob("*.wav"))
    
    print(f"Found {len(siren_files)} siren files")
    print(f"Found {len(noise_files)} noise files")
    
    if len(siren_files) < 10 or len(noise_files) < 10:
        print("[WARN] Warning: Very few samples. Need at least 100+ of each for good results.")
        print("  Download from Freesound.org or UrbanSound8K dataset")
        return
    
    # Validate
    valid_sirens = [f for f in tqdm(siren_files) if verify_audio_file(str(f))]
    valid_noise = [f for f in tqdm(noise_files) if verify_audio_file(str(f))]
    
    print(f"[OK] Valid sirens: {len(valid_sirens)}/{len(siren_files)}")
    print(f"[OK] Valid noise: {len(valid_noise)}/{len(noise_files)}")
    
    # Step 2: Generate augmented (mixed) data
    print("\n[2/4] Generating SNR-augmented dataset...")
    augmenter = AudioAugmenter()
    
    try:
        augmenter.create_mixed_dataset(
            siren_dir=siren_dir,
            noise_dir=noise_dir,
            output_dir=augment_dir / "positive",
            snr_levels=[-5, 0, 5, 10, 15],
            n_samples_per_snr=min(50, len(valid_sirens) // 2),  # Limit if few samples
            duration_sec=2.0
        )
    except Exception as e:
        print(f"[WARN] Augmentation failed: {e}")
        print("  Continuing with raw data only...")
    
    # Step 3: Combine datasets
    print("\n[3/4] Combining datasets...")
    
    # Positive: raw sirens + augmented
    positive_files = valid_sirens.copy()
    augmented_positives = list((augment_dir / "positive").glob("*.wav"))
    positive_files.extend(augmented_positives)
    
    # Negative: raw noise
    negative_files = valid_noise
    
    print(f"Total positive samples: {len(positive_files)}")
    print(f"Total negative samples: {len(negative_files)}")
    
    # Create labels
    positive_labels = [(f, 1) for f in positive_files]
    negative_labels = [(f, 0) for f in negative_files]
    
    all_data = positive_labels + negative_labels
    files, labels = zip(*all_data)
    
    # Step 4: Train/val/test split
    print("\n[4/4] Splitting into train/val/test...")
    
    # First split: train+val vs test
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        files, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels,
        test_size=val_size_adjusted,
        stratify=train_val_labels,
        random_state=random_state
    )
    
    # Copy files to split directories
    splits = {
        'train': (train_files, train_labels),
        'val': (val_files, val_labels),
        'test': (test_files, test_labels)
    }
    
    for split_name, (split_files, split_labels) in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create class subdirs
        (split_dir / "positive").mkdir(exist_ok=True)
        (split_dir / "negative").mkdir(exist_ok=True)
        
        for file, label in zip(split_files, split_labels):
            class_name = "positive" if label == 1 else "negative"
            dest = split_dir / class_name / file.name
            shutil.copy2(file, dest)
        
        n_pos = sum(split_labels)
        n_neg = len(split_labels) - n_pos
        print(f"  {split_name}: {len(split_files)} samples (pos={n_pos}, neg={n_neg})")
    
    # Save split info
    split_info = output_dir / "split_info.txt"
    with open(split_info, 'w') as f:
        f.write(f"Train: {len(train_files)} samples\n")
        f.write(f"Val: {len(val_files)} samples\n")
        f.write(f"Test: {len(test_files)} samples\n")
        f.write(f"\nPositive class: {sum(labels)} samples\n")
        f.write(f"Negative class: {len(labels) - sum(labels)} samples\n")
    
    print(f"\n[OK] Dataset prepared in {output_dir}")
    print(f"[OK] Split info saved to {split_info}")
    print("\nNext: Train audio CRNN model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--siren_dir", default="data/raw/sirens")
    parser.add_argument("--noise_dir", default="data/raw/noise")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()
    
    prepare_dataset(
        siren_dir=args.siren_dir,
        noise_dir=args.noise_dir,
        output_dir=args.output_dir
    )