#!/usr/bin/env python3
"""
Image Optimization Script for Photography Portfolio

This script:
1. Generates multiple sizes for responsive images (srcset)
2. Converts images to WebP format with JPEG fallbacks

Requirements:
    pip install pillow

Usage:
    python optimize_images.py
"""

import os
import re
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install pillow")
    exit(1)

# Configuration
INPUT_DIR = Path("../Portfolio Selections")  # full-resolution masters, not deployed
OUTPUT_DIR = Path("images")  # derived variants, deployed
SIZES = [400, 800, 1200, 2000]  # Width breakpoints for srcset
WEBP_QUALITY = 85
JPEG_QUALITY = 85

def get_image_files():
    """Get all JPEG images in the input directory."""
    extensions = {'.jpg', '.jpeg', '.png'}
    return [f for f in INPUT_DIR.iterdir()
            if f.suffix.lower() in extensions
            and not any(f.stem.endswith(f"-{s}") for s in SIZES)]

def optimize_image(input_path: Path):
    """Generate responsive sizes and WebP version of an image."""
    print(f"Processing: {input_path.name}")

    try:
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (for PNG with transparency)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            original_width, original_height = img.size
            stem = input_path.stem

            # Generate sized versions
            for width in SIZES:
                if width >= original_width:
                    continue  # Skip if larger than original

                # Calculate new height maintaining aspect ratio
                ratio = width / original_width
                height = int(original_height * ratio)

                resized = img.resize((width, height), Image.Resampling.LANCZOS)

                # Save JPEG version
                jpeg_path = OUTPUT_DIR / f"{stem}-{width}.jpg"
                resized.save(jpeg_path, 'JPEG', quality=JPEG_QUALITY, optimize=True)
                print(f"  Created: {jpeg_path.name}")

                # Save WebP version
                webp_path = OUTPUT_DIR / f"{stem}-{width}.webp"
                resized.save(webp_path, 'WEBP', quality=WEBP_QUALITY)
                print(f"  Created: {webp_path.name}")

    except Exception as e:
        print(f"  Error processing {input_path.name}: {e}")

def main():
    if not INPUT_DIR.exists():
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        print("Make sure you're running this from the Portfolio Website directory.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    image_files = get_image_files()

    if not image_files:
        print("No images found to process.")
        return

    print(f"Found {len(image_files)} images to process.\n")

    for img_path in image_files:
        optimize_image(img_path)

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nProcessed {len(image_files)} images.")
    print("Generated responsive sizes: " + ", ".join(f"{s}px" for s in SIZES))

if __name__ == "__main__":
    main()
