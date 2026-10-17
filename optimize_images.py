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

import argparse
import os
import re
import sys
from pathlib import Path

try:
    from PIL import Image, ImageOps
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


#: What `share_metadata` may name (settings key in _studio_settings.json). The published
#: copies (images/) carry exactly these and nothing else; the masters are never touched.
SHARE_CHOICES = ("copyright", "artist", "camera", "lens", "exposure", "icc_profile")
EXIF_IFD, IFD_BASE = 0x8769, 0
_TAGS = {                       # choice -> [(ifd, tag)] copied FRESH from the source
    "camera": [(IFD_BASE, 0x010F), (IFD_BASE, 0x0110)],                 # Make, Model
    "lens": [(EXIF_IFD, 0xA433), (EXIF_IFD, 0xA434)],                   # LensMake, LensModel
    "exposure": [(EXIF_IFD, 0x829A), (EXIF_IFD, 0x829D),                # ExposureTime, FNumber
                 (EXIF_IFD, 0x8827), (EXIF_IFD, 0x920A)],               # ISO, FocalLength
}


def share_allowlist(settings=None):
    """The set of metadata kinds that may be published.

    `share_metadata` in _studio_settings.json (same loader as the picker). Absent file or
    absent key: only the ICC profile (colour must survive; nothing identifying does).
    A present list is taken exactly, an empty list meaning nothing at all. Unknown names are
    ignored with a warning.
    """
    if settings is None:
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent / "tools" / "picker"))
            import studio_link
            settings = studio_link.load_settings()
        except Exception as exc:  # unreadable settings fail closed: strip everything
            print(f"  settings unreadable ({exc}); publishing ICC profile only")
            settings = {}
    listed = settings.get("share_metadata")
    if listed is None:
        return {"icc_profile"}
    unknown = [x for x in listed if x not in SHARE_CHOICES]
    if unknown:
        print(f"  share_metadata: ignoring unknown {unknown}")
    return {x for x in listed if x in SHARE_CHOICES}


def published_metadata(stem: str, source_exif, source_icc, allowed):
    """-> (exif bytes or None, icc bytes or None) for a published variant.

    Every tag is written FRESH into an empty Exif block; the source EXIF is only read.
    So GPS, dates and times, serial numbers, software and editing history, XMP, IPTC,
    MakerNotes, thumbnails and comments cannot come through: nothing copies them. What
    goes out is exactly the kinds in `allowed` (see SHARE_CHOICES).

    Visible watermarks stop casual reposting and nothing else -- generative fill
    removes one in seconds. Embedded rights survive reposting, cost nothing
    visually, and give provenance if a claim is ever needed.
    """
    out = Image.Exif()
    wrote = False
    if "artist" in allowed:
        out[0x013B] = CREATOR                    # Artist
        wrote = True
    if "copyright" in allowed:
        year = _known_year(stem) or (stem[:4] if stem[:4].isdigit() else "")
        try:
            raw = str(source_exif.get_ifd(EXIF_IFD).get(36867, ""))   # DateTimeOriginal: year only
            if raw[:4].isdigit() and not _known_year(stem):
                year = raw[:4]
        except Exception:
            pass
        out[0x8298] = (f"(c) {year} {CREATOR}. All rights reserved." if year
                       else f"(c) {CREATOR}. All rights reserved.")
        wrote = True
    for choice in ("camera", "lens", "exposure"):
        if choice not in allowed:
            continue
        for ifd, tag in _TAGS[choice]:
            try:
                value = (source_exif if ifd == IFD_BASE else source_exif.get_ifd(ifd)).get(tag)
            except Exception:
                value = None
            if value in (None, "", b""):
                continue
            (out if ifd == IFD_BASE else out.get_ifd(ifd))[tag] = value
            wrote = True
    return (out.tobytes() if wrote else None), (source_icc if "icc_profile" in allowed else None)


JPEG_QUALITY = 85


def face_guard(input_path: Path):
    """None when this source may be published, else the reason it may not.

    Evan's rule: our kids' faces are never posted. Defence in depth behind the gate in
    publish_external: a master with a detected face gets no variants unless a `faces` event
    exists for its id (the filename stem): `cleared` by a clearer, or `masked` for this very
    file (its sha1 is the recorded one). Fails CLOSED: a detector that
    cannot run, an unreadable image or a missing faces module all refuse."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "tools" / "picker"))
        import faces
        if faces.has_clearance(Path(input_path).stem, image=input_path):
            return None
        found = faces.check_image(input_path, faces.link.load_settings())["faces"]
    except Exception as exc:  # noqa: BLE001
        return f"the face check could not run ({exc})"
    if found:
        return (f"{len(found)} face(s) detected and no faces event for {Path(input_path).stem}: "
                "mask it with tools/picker/faces.py mask, or have a face_clearer clear it")
    return None


def get_image_files():
    """Get all JPEG images in the input directory."""
    extensions = {'.jpg', '.jpeg', '.png'}
    return [f for f in INPUT_DIR.iterdir()
            if f.suffix.lower() in extensions
            and not any(f.stem.endswith(f"-{s}") for s in SIZES)]

def optimize_image(input_path: Path, allowed=None):
    """Generate responsive sizes and WebP version of an image.

    `allowed` is the set of metadata kinds to publish (default: share_allowlist())."""
    if allowed is None:
        allowed = share_allowlist()
    print(f"Processing: {input_path.name}")

    refused = face_guard(input_path)
    if refused:
        print(f"  REFUSED (face guard): {input_path.name}: {refused}")
        return False

    try:
        with Image.open(input_path) as img:
            source_exif = img.getexif()
            source_icc = img.info.get("icc_profile")
            # Bake in the EXIF Orientation before any resize. Variants ship
            # with no Orientation tag (see published_metadata), so a source shot
            # in portrait but stored wide-with-rotate-tag would otherwise
            # ship wide and untagged -- sideways with nothing to correct it.
            img = ImageOps.exif_transpose(img)

            # Convert to RGB if necessary (for PNG with transparency)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            original_width, original_height = img.size
            stem = input_path.stem

            exif_bytes, icc = published_metadata(stem, source_exif, source_icc, allowed)
            meta = {}
            if exif_bytes:
                meta["exif"] = exif_bytes
            if icc:
                meta["icc_profile"] = icc

            # Generate sized versions
            for width in SIZES:
                if width >= original_width:
                    continue  # Skip if larger than original

                # Calculate new height maintaining aspect ratio
                ratio = width / original_width
                height = int(original_height * ratio)

                resized = img.resize((width, height), Image.Resampling.LANCZOS)
                # resize() carries img.info over (comment, xmp, jfif...) and Pillow writes
                # some of it on save. Only `meta` below may reach the file.
                resized.info = {}

                # Save JPEG version
                jpeg_path = OUTPUT_DIR / f"{stem}-{width}.jpg"
                resized.save(jpeg_path, 'JPEG', quality=JPEG_QUALITY,
                             optimize=True, **meta)
                print(f"  Created: {jpeg_path.name}")

                # Save WebP version
                webp_path = OUTPUT_DIR / f"{stem}-{width}.webp"
                resized.save(webp_path, 'WEBP', quality=WEBP_QUALITY, **meta)
                print(f"  Created: {webp_path.name}")

    except Exception as e:
        print(f"  Error processing {input_path.name}: {e}")
    return True

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", metavar="STEM",
                    help="process only these master stems (publish_external passes what it staged)")
    args = ap.parse_args(argv)
    if not INPUT_DIR.exists():
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        print("Make sure you're running this from the Portfolio Website directory.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    image_files = get_image_files()
    if args.only:
        image_files = [f for f in image_files if f.stem in set(args.only)]

    if not image_files:
        print("No images found to process.")
        return

    print(f"Found {len(image_files)} images to process.\n")

    refused = [f.name for f in image_files if optimize_image(f) is False]

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nProcessed {len(image_files)} images.")
    print("Generated responsive sizes: " + ", ".join(f"{s}px" for s in SIZES))
    if refused:
        print(f"\nREFUSED by the face guard ({len(refused)}): " + ", ".join(refused))
        return 3
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
