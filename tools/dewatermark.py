#!/usr/bin/env python3
"""Rebuild watermark-free masters without discarding the owner's grade.

The published masters carry a baked-in wordmark. Unwatermarked originals exist
elsewhere in the library, but for most frames they are also *less graded* --
re-sourcing them directly would strip the watermark and the owner's colour work
together.

So instead of inpainting (lossy, and generative fill is overkill here), fit the
tone mapping that turns the clean original into the graded export, using only
pixels OUTSIDE the watermark region, then apply that mapping to the clean file.
The result carries the grade and never had a watermark in it.

Per-channel monotonic LUTs via histogram matching. That is enough because the
difference between the two files is a global grade, not local retouching -- and
the residual check below verifies exactly that assumption per photo.

    python tools/dewatermark.py --dry-run
    python tools/dewatermark.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

REPO = Path(__file__).resolve().parents[1]
LIB = REPO.parent
SELECTIONS = LIB / "Portfolio Selections"
OUT = LIB / "Portfolio Unwatermarked"

# The wordmark sits in the bottom-right. Exclude generously: a few percent of
# contaminated pixels would bias the fit toward the mark's own brightness.
WM_X0, WM_Y0 = 0.40, 0.82


def mask_outside(shape) -> np.ndarray:
    h, w = shape[:2]
    m = np.ones((h, w), dtype=bool)
    m[int(h * WM_Y0):, int(w * WM_X0):] = False
    return m


def fit_lut(clean: np.ndarray, graded: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel histogram match, computed only where the mask is True."""
    luts = np.empty((3, 256), dtype=np.uint8)
    for c in range(3):
        src = clean[:, :, c][mask].astype(np.uint8)
        dst = graded[:, :, c][mask].astype(np.uint8)
        s_hist = np.bincount(src, minlength=256).astype(np.float64)
        d_hist = np.bincount(dst, minlength=256).astype(np.float64)
        s_cdf = np.cumsum(s_hist) / max(s_hist.sum(), 1)
        d_cdf = np.cumsum(d_hist) / max(d_hist.sum(), 1)
        luts[c] = np.interp(s_cdf, d_cdf, np.arange(256)).round().clip(0, 255)
    return luts


def apply_lut(img: np.ndarray, luts: np.ndarray) -> np.ndarray:
    out = np.empty_like(img)
    for c in range(3):
        out[:, :, c] = luts[c][img[:, :, c]]
    return out


def load(p: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


def process(stem: str, source: Path, dry: bool) -> dict:
    graded_im = load(SELECTIONS / f"{stem}.jpg")
    clean_im = load(source)
    if clean_im.size != graded_im.size:
        clean_im = clean_im.resize(graded_im.size, Image.LANCZOS)

    clean = np.asarray(clean_im, dtype=np.uint8)
    graded = np.asarray(graded_im, dtype=np.uint8)
    mask = mask_outside(clean.shape)

    before = float(np.abs(clean[mask].astype(float) - graded[mask].astype(float)).mean())
    luts = fit_lut(clean, graded, mask)
    matched = apply_lut(clean, luts)
    after = float(np.abs(matched[mask].astype(float) - graded[mask].astype(float)).mean())

    if not dry:
        OUT.mkdir(parents=True, exist_ok=True)
        src_img = Image.open(source)
        out = Image.fromarray(matched, "RGB")
        kw = {"quality": 96, "subsampling": 0}
        # Prefer the clean original's EXIF, but fall back to the graded export.
        # For the maternity set the library original had been stripped while the
        # Lightroom export kept it, so without this fallback capture date, camera
        # and lens are lost for good -- the RAWs no longer exist.
        graded_img = Image.open(SELECTIONS / f"{stem}.jpg")
        if (e := src_img.info.get("exif") or graded_img.info.get("exif")):
            kw["exif"] = e
        if (i := src_img.info.get("icc_profile") or graded_img.info.get("icc_profile")):
            kw["icc_profile"] = i
        out.save(OUT / f"{stem}.jpg", "JPEG", **kw)

    return {"stem": stem, "before": round(before, 2), "after": round(after, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(REPO / "data" / "watermark_sources.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    srcs = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    print(f"{'photo':<24} {'before':>7} {'after':>7}   grade recovered")
    print("-" * 62)
    rows = []
    for stem, v in sorted(srcs.items()):
        r = process(stem, LIB / v["source"], args.dry_run)
        gain = (1 - r["after"] / r["before"]) * 100 if r["before"] > 0.01 else 0
        rows.append(r)
        print(f"{r['stem']:<24} {r['before']:7.2f} {r['after']:7.2f}   {gain:5.1f}%")

    worst = max(rows, key=lambda r: r["after"])
    print(f"\nresidual outside the watermark: "
          f"mean {np.mean([r['after'] for r in rows]):.2f}, worst {worst['after']:.2f} "
          f"({worst['stem']})")
    print("A low residual means the two files differed only by a global grade,"
          "\nwhich is the assumption this whole approach rests on.")
    if not args.dry_run:
        print(f"\nwrote {len(rows)} files -> {OUT}")


if __name__ == "__main__":
    main()
