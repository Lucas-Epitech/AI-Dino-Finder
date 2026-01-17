#!/usr/bin/env python3
"""
Split dataset into train/val/test sets.

This script takes images from data/raw/ and copies them into
data/train/, data/val/, and data/test/ with a 70/15/15 split.

Usage:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --train-ratio 0.7 --val-ratio 0.15
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def get_images_by_class(raw_dir: Path) -> dict:
    """
    Scan raw directory and get all images organized by class.

    Args:
        raw_dir: Path to data/raw/ directory

    Returns:
        Dictionary mapping class names to lists of image paths
    """
    images_by_class = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    for class_dir in raw_dir.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue

        class_name = class_dir.name
        images = [
            img for img in class_dir.iterdir()
            if img.suffix.lower() in valid_extensions
        ]

        if images:
            images_by_class[class_name] = images
            print(f"  Found {len(images)} images for class '{class_name}'")

    return images_by_class


def split_images(
    images: List[Path],
    train_ratio: float,
    val_ratio: float
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split a list of images into train/val/test sets.

    Args:
        images: List of image paths
        train_ratio: Proportion for training (e.g., 0.7)
        val_ratio: Proportion for validation (e.g., 0.15)

    Returns:
        Tuple of (train_images, val_images, test_images)
    """
    # Shuffle images randomly
    images_shuffled = images.copy()
    random.shuffle(images_shuffled)

    total = len(images_shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_images = images_shuffled[:train_count]
    val_images = images_shuffled[train_count:train_count + val_count]
    test_images = images_shuffled[train_count + val_count:]

    return train_images, val_images, test_images


def copy_images(images: List[Path], dest_dir: Path, class_name: str):
    """
    Copy images to destination directory, preserving class structure.

    Args:
        images: List of image paths to copy
        dest_dir: Destination directory (e.g., data/train/)
        class_name: Name of the class (e.g., 'trex')
    """
    class_dest = dest_dir / class_name
    class_dest.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        dest_path = class_dest / img_path.name
        shutil.copy2(img_path, dest_path)


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Path to raw images directory (default: data/raw)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for splits (default: data)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion for training set (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion for validation set (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Paths
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    # Validate ratios
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0 or test_ratio > 1:
        print("Error: train_ratio + val_ratio must be <= 1.0")
        return

    print("🦖 AI-Dino-Finder Dataset Splitter")
    print("=" * 50)
    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={test_ratio:.2f}")
    print(f"Random seed: {args.seed}")
    print("=" * 50)

    # Check if raw directory exists
    if not raw_dir.exists():
        print(f"Error: Raw directory '{raw_dir}' not found.")
        print("Please run the data download script first or add images manually.")
        return

    # Get images by class
    print("\n📂 Scanning raw directory...")
    images_by_class = get_images_by_class(raw_dir)

    if not images_by_class:
        print("Error: No images found in raw directory.")
        return

    print(f"\n✅ Found {len(images_by_class)} classes:")
    for class_name, images in images_by_class.items():
        print(f"  - {class_name}: {len(images)} images")

    # Clean existing split directories
    print("\n🧹 Cleaning existing split directories...")
    for split_dir in [train_dir, val_dir, test_dir]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    # Split and copy images for each class
    print("\n📊 Splitting and copying images...\n")

    total_train = 0
    total_val = 0
    total_test = 0

    for class_name, images in images_by_class.items():
        print(f"Processing '{class_name}':")

        # Split images
        train_images, val_images, test_images = split_images(
            images, args.train_ratio, args.val_ratio
        )

        # Copy to respective directories
        copy_images(train_images, train_dir, class_name)
        copy_images(val_images, val_dir, class_name)
        copy_images(test_images, test_dir, class_name)

        print(f"  Train: {len(train_images)} images → {train_dir / class_name}")
        print(f"  Val:   {len(val_images)} images → {val_dir / class_name}")
        print(f"  Test:  {len(test_images)} images → {test_dir / class_name}")
        print()

        total_train += len(train_images)
        total_val += len(val_images)
        total_test += len(test_images)

    # Summary
    total_images = total_train + total_val + total_test
    print("=" * 50)
    print("✅ Dataset Split Complete!")
    print("=" * 50)
    print(f"Total images: {total_images}")
    print(f"  Training:   {total_train} ({total_train/total_images*100:.1f}%)")
    print(f"  Validation: {total_val} ({total_val/total_images*100:.1f}%)")
    print(f"  Test:       {total_test} ({total_test/total_images*100:.1f}%)")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Review the split images in data/train/, data/val/, data/test/")
    print("2. Run: python scripts/train.py (when implemented)")
    print("=" * 50)


if __name__ == "__main__":
    main()
