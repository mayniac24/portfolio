#!/usr/bin/env python3
"""Index the photo library for the picker.

Walks the library once, keeps real-camera frames, and records what is useful for
BROWSING -- date, outing, camera, and the sun's elevation at capture. It
deliberately does not score or rank: two attempts at automated quality ranking
failed (see docs/DECISIONS.md), and this tool exists because the judgement has
to be the owner's.

    python tools/picker/index_library.py
    python tools/picker/index_library.py --root "M:/Photos & Videos/Hiking"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_ROOT = REPO.parent
INDEX = HERE / "picker_index.json"
EXIF_IFD = 0x8769
LAT, LON = 40.35, -105.55          # Front Range / RMNP

SKIP_PARTS = {
    "Portfolio Website", "Portfolio Corrected", "Portfolio Selections",
    "Screenshots", "CacheClip", "Resolve Project Backups",
    "Plex Favorites Photos", "_Watermarked for Sharing", "Snapchat Video",
    "_QUARANTINE dedup 2026-08-25", "$RECYCLE.BIN",
}


def solar_elevation(local: datetime) -> float:
    """NOAA low-precision solar position. Camera clock is US Mountain time."""
    offset = -6 if 3 <= local.month <= 11 else -7
    utc = local.replace(tzinfo=timezone(timedelta(hours=offset))).astimezone(timezone.utc)
    n = utc.timestamp() / 86400.0 + 2440587.5 - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)
    dec = math.asin(math.sin(eps) * math.sin(lam))
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    lst = math.radians(((18.697374558 + 24.06570982441908 * n) % 24) * 15 + LON)
    lat = math.radians(LAT)
    sin_el = (math.sin(lat) * math.sin(dec) +
              math.cos(lat) * math.cos(dec) * math.cos(lst - ra))
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))


def light_band(elev: float) -> str:
    """Descriptive only. Overcast defeats this entirely -- a low sun behind
    cloud still reads as 'golden', which is exactly why it must not be a score."""
    if elev < -10:
        return "night"
    if elev < -6:
        return "blue hour"
    if elev <= 10:
        return "golden"
    if elev <= 25:
        return "mid"
    return "harsh"


# Camera-dump folder names carry no meaning. The useful label is an ancestor
# like "07.18.21 - Trail Ridge Road & Echo Lake", so walk up past these.
GENERIC = {"dcim", "camera", "originals", "original", "edits", "photos", "pics",
           "images", "raw", "jpg", "jpeg", "export", "exports", "ogs", "misc"}


def outing_of(rel: Path) -> str:
    """First ancestor folder that actually names an occasion."""
    for part in reversed(rel.parts[:-1]):
        low = part.lower()
        if low in GENERIC or part.isdigit():
            continue
        if low.endswith("'s camera") or low.endswith("s camera"):
            continue
        if re.fullmatch(r"\d{3}[a-z]?d\d{3,4}", low):      # 100D3200, 101D3200
            continue
        if re.fullmatch(r"(pt|part)\s*\d+", low):           # Pt 1, Part 2
            continue
        return part
    return rel.parts[-2] if len(rel.parts) > 1 else "(root)"


def exif_of(p: Path) -> dict | None:
    try:
        ex = Image.open(p).getexif()
        if not ex:
            return None
        return {**{TAGS.get(k, k): v for k, v in ex.items()},
                **{TAGS.get(k, k): v for k, v in ex.get_ifd(EXIF_IFD).items()}}
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--all-cameras", action="store_true",
                    help="include phone photos (default: real cameras only)")
    args = ap.parse_args()

    root = Path(args.root)
    photos, scanned = [], 0

    for p in root.rglob("*"):
        if p.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        if any(part in SKIP_PARTS for part in p.parts) or \
           any(part.startswith("_Portfolio Cull") for part in p.parts):
            continue
        scanned += 1
        if scanned % 2500 == 0:
            print(f"  ...{scanned} scanned, {len(photos)} kept", flush=True)

        ex = exif_of(p)
        if not ex:
            continue
        make = str(ex.get("Make", "")).upper()
        if not args.all_cameras and "NIKON" not in make:
            continue

        raw = str(ex.get("DateTimeOriginal") or ex.get("DateTime") or "")
        try:
            when = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            continue

        elev = solar_elevation(when)
        photos.append({
            "id": str(p.relative_to(root)).replace("\\", "/"),
            "outing": outing_of(p.relative_to(root)),
            "date": when.strftime("%Y-%m-%d"),
            "time": when.strftime("%H:%M"),
            "camera": str(ex.get("Model", "")).replace("NIKON ", ""),
            "iso": ex.get("ISOSpeedRatings"),
            "elev": round(elev, 1),
            "light": light_band(elev),
        })

    photos.sort(key=lambda r: (r["date"], r["time"]))
    INDEX.write_text(json.dumps({"root": str(root), "photos": photos}, indent=1),
                     encoding="utf-8", newline="\n")

    from collections import Counter
    print(f"\nscanned {scanned}, indexed {len(photos)} -> {INDEX.name}")
    print("  by light:", dict(Counter(p["light"] for p in photos)))
    print("  by camera:", dict(Counter(p["camera"] for p in photos)))
    print(f"  outings: {len({p['outing'] for p in photos})}")


if __name__ == "__main__":
    main()
