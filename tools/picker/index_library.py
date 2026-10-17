#!/usr/bin/env python3
"""Index the photo library for the picker.

Walks the library once, keeps real-camera frames, and records what is useful for
BROWSING -- date, outing, camera, and the sun's elevation at capture. It
deliberately does not score or rank: two attempts at automated quality ranking
failed (see docs/DECISIONS.md), and this tool exists because the judgement has
to be the owner's.

    python tools/picker/index_library.py                 # rebuild the whole index
    python tools/picker/index_library.py --root "M:/Photos & Videos/Hiking"
    python tools/picker/index_library.py --add <outing>  # merge ONE outing in

The camera filter is `camera_match` in _studio_settings.json (default "" = no filter): a
frame is kept when that text is in its EXIF Model or Make. The finish folder is never
indexed. `--camera-match` overrides it; `--all-cameras` keeps everything. `--add` never replaces the index: it drops that
outing's old entries and adds the fresh ones, ids staying relative to the index root
(photos_root). Index and root come from settings (see studio_link.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import studio_link as link  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_ROOT = REPO.parent
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


def finish_skip(index_root: Path, finish_dir: str):
    """-> predicate(rel path parts) true inside the finish folder. A relative finish_dir
    is skipped wherever its first part appears under the root; an absolute one by path."""
    fd = Path(finish_dir or "_Finish")
    if fd.is_absolute():
        base = fd.resolve()

        def inside(rel_parts):
            try:
                index_root.joinpath(*rel_parts).resolve().relative_to(base)
                return True
            except ValueError:
                return False
        return inside
    first = fd.parts[0].lower()
    return lambda rel_parts: any(part.lower() == first for part in rel_parts[:-1])


def scan(index_root: Path, scan_dir: Path, camera_match: str, all_cameras: bool,
         finish_dir: str = "_Finish") -> tuple[list, int]:
    """Index the JPEGs under scan_dir; ids are relative to index_root. -> (photos, scanned).

    The finish folder holds copies of starred frames and his exports; indexing it would
    list every pick twice, so it is skipped. An empty camera_match keeps every camera."""
    photos, scanned = [], 0
    match = (camera_match or "").upper()
    in_finish = finish_skip(index_root, finish_dir)
    for p in scan_dir.rglob("*"):
        if p.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        rel = p.relative_to(index_root)
        if (any(part in SKIP_PARTS for part in rel.parts)
                or any(part.startswith("_Portfolio Cull") for part in rel.parts)
                or in_finish(rel.parts)):
            continue
        scanned += 1
        if scanned % 2500 == 0:
            print(f"  ...{scanned} scanned, {len(photos)} kept", flush=True)

        ex = exif_of(p)
        if not ex:
            continue
        model = str(ex.get("Model", ""))
        if match and not all_cameras and match not in (model + " " + str(ex.get("Make", ""))).upper():
            continue

        raw = str(ex.get("DateTimeOriginal") or ex.get("DateTime") or "")
        try:
            when = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            continue

        elev = solar_elevation(when)
        photos.append({
            "id": str(rel).replace("\\", "/"),
            "outing": outing_of(rel),
            "date": when.strftime("%Y-%m-%d"),
            "time": when.strftime("%H:%M"),
            "camera": model.replace("NIKON ", ""),
            "iso": ex.get("ISOSpeedRatings"),
            "elev": round(elev, 1),
            "light": light_band(elev),
        })
    return photos, scanned


def main(argv=None) -> int:
    link.stdout_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="library root (default: settings photos_root)")
    ap.add_argument("--add", metavar="OUTING",
                    help="index ONE outing folder into the existing index (never replaces it)")
    ap.add_argument("--all-cameras", action="store_true",
                    help="include every camera, phones too (default: camera_match only)")
    ap.add_argument("--camera-match", help="override settings camera_match")
    ap.add_argument("--settings", help="_studio_settings.json (default: env / $VAULT_PATH)")
    args = ap.parse_args(argv)

    try:
        settings = link.load_settings(args.settings)
    except ValueError as exc:
        print("error: %s" % exc)
        return 2
    index_file = link.index_path(settings)
    camera_match = args.camera_match if args.camera_match is not None else (settings.get("camera_match") or "")
    finish_dir = settings.get("finish_dir") or link.DEFAULTS["finish_dir"]

    if args.add:
        if index_file.is_file():
            existing = json.loads(index_file.read_text(encoding="utf-8"))
            root = Path(args.root or existing["root"])
            old = existing["photos"]
        else:
            try:
                root = link.photos_root(settings, args.root)
            except ValueError as exc:
                print("error: %s" % exc)
                return 2
            old = []
        try:
            outing_dir = link.resolve_outing_dir(args.add, root)
            prefix = outing_dir.relative_to(root.resolve()).as_posix() + "/"
        except (FileNotFoundError, ValueError) as exc:
            print("error: outing must be a folder under %s (%s)" % (root, exc))
            return 2
        fresh, scanned = scan(root.resolve(), outing_dir, camera_match, args.all_cameras, finish_dir)
        kept = [p for p in old if not p["id"].startswith(prefix)]
        photos = kept + fresh
        print(f"--add {prefix}: scanned {scanned}, {len(fresh)} indexed "
              f"(replacing {len(old) - len(kept)} old entries)")
        root_text = existing["root"] if index_file.is_file() else str(root)
    else:
        root = Path(args.root or settings.get("photos_root") or DEFAULT_ROOT)
        photos, scanned = scan(root, root, camera_match, args.all_cameras, finish_dir)
        root_text = str(root)

    photos.sort(key=lambda r: (r["date"], r["time"], r["id"]))
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text(json.dumps({"root": root_text, "photos": photos}, indent=1),
                          encoding="utf-8", newline="\n")

    from collections import Counter
    print(f"\nindexed {len(photos)} -> {index_file.name}  (camera_match "
          f"{'ALL' if args.all_cameras or not camera_match else repr(camera_match)})")
    print("  by light:", dict(Counter(p["light"] for p in photos)))
    print("  by camera:", dict(Counter(p["camera"] for p in photos)))
    print(f"  outings: {len({p['outing'] for p in photos})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
