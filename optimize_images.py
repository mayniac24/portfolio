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
INPUT_DIR = Path("../Portfolio Corrected")  # colour-corrected masters; these are what ship
# ../Portfolio Selections holds the uncorrected originals. With the RAWs gone they are
# the only masters left, so nothing writes to that directory.
OUTPUT_DIR = Path("images")  # derived variants, deployed
SIZES = [400, 800, 1200, 2000]  # Width breakpoints for srcset
WEBP_QUALITY = 85

CREATOR = "Mayniac Creations"
SITE = "https://mayniac-portfolio.netlify.app/"


def _known_year(stem: str) -> str | None:
    """Capture year recorded in gallery.json, when the filename cannot be trusted.

    The 20260202- prefix on the maternity set came from a Lightroom export
    timestamp, five months after the shoot. Deriving a copyright year from the
    filename would assert 2026 for photographs taken in August 2025.
    """
    try:
        import json
        data = json.loads((Path(__file__).parent / "data" / "gallery.json")
                          .read_text(encoding="utf-8"))
        for p in data["photos"]:
            if p["id"] == stem and p.get("captured"):
                return str(p["captured"])[:4]
    except Exception:
        pass
    return None


def rights_metadata(stem: str, source: Path):
    """Build a MINIMAL copyright block for the deployed variants.

    Deliberately not a copy of the source EXIF: the originals carry GPS, and
    publishing the coordinates of client shoots is a privacy problem. Only
    authorship and rights go out.

    Visible watermarks stop casual reposting and nothing else -- generative fill
    removes one in seconds. Embedded rights survive reposting, cost nothing
    visually, and give provenance if a claim is ever needed.
    """
    year = _known_year(stem) or (stem[:4] if stem[:4].isdigit() else "")
    try:
        ex = Image.open(source).getexif()
        raw = str(ex.get_ifd(0x8769).get(36867, ""))     # DateTimeOriginal
        if raw[:4].isdigit() and not _known_year(stem):
            year = raw[:4]
    except Exception:
        pass
    notice = f"(c) {year} {CREATOR}. All rights reserved." if year else              f"(c) {CREATOR}. All rights reserved."

    exif = Image.Exif()
    exif[0x013B] = CREATOR      # Artist
    exif[0x8298] = notice       # Copyright
    exif[0x010E] = f"{notice} {SITE}"   # ImageDescription

    xmp = (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:xmpRights="http://ns.adobe.com/xap/1.0/rights/">'
        f'<dc:creator><rdf:Seq><rdf:li>{CREATOR}</rdf:li></rdf:Seq></dc:creator>'
        f'<dc:rights><rdf:Alt><rdf:li xml:lang="x-default">{notice}</rdf:li>'
        '</rdf:Alt></dc:rights>'
        '<xmpRights:Marked>True</xmpRights:Marked>'
        f'<xmpRights:WebStatement>{SITE}</xmpRights:WebStatement>'
        '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
    )
    return exif.tobytes(), xmp
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

            exif_bytes, xmp_str = rights_metadata(stem, input_path)

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
                resized.save(jpeg_path, 'JPEG', quality=JPEG_QUALITY,
                             optimize=True, exif=exif_bytes)
                print(f"  Created: {jpeg_path.name}")

                # Save WebP version
                webp_path = OUTPUT_DIR / f"{stem}-{width}.webp"
                resized.save(webp_path, 'WEBP', quality=WEBP_QUALITY,
                             exif=exif_bytes, xmp=xmp_str)
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
