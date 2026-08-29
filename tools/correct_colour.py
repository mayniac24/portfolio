#!/usr/bin/env python3
"""Non-destructive corrections for already-exported JPEGs.

Targets the three things measurably wrong across the portfolio set:
  1. colour casts (yellow-green and teal skies)
  2. no black point (nothing near 0 -> flat, hazy)
  3. oversaturation on a few frames

Deliberately conservative. Gains are clamped so a genuine golden hour stays
golden -- the goal is to remove a cast, not to neutralise the light.

Vignetting is NOT corrected here: it is baked into the pixels and inverting it
amplifies corner noise. That one has to be fixed at the source.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# --- tunables -------------------------------------------------------------
WB_STRENGTH = 0.70      # 0 = off, 1 = fully neutralise the shadows
WB_GAIN_MIN = 0.88      # clamp so warm light survives correction
WB_GAIN_MAX = 1.14
SHADOW_PCT = 22.0       # shadows are skylit and should be near-neutral
SHADOW_FLOOR = 8        # ignore near-black pixels: that is noise, not colour
CAST_DEADBAND = 15.0     # below this the shadows are already neutral -- leave WB alone
BLACK_PCT = 0.05        # percentile mapped to true black
BLACK_MAX_SHIFT = 14    # never lift/crush more than this many levels
SAT_TARGET = 118        # only pulled down when above SAT_TRIGGER
SAT_TRIGGER = 128


def white_balance(rgb: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Neutralise the SHADOWS, not the highlights.

    Outdoor shadow is lit by skylight and should sit close to neutral, so a warm
    or teal shadow is a cast. Highlights carry the real colour of the sun -- a
    golden hour sky is legitimately orange, and balancing against it destroys
    exactly the light worth keeping.
    """
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    mask = (lum <= np.percentile(lum, SHADOW_PCT)) & (lum > SHADOW_FLOOR)
    if mask.sum() < 500:
        return rgb, (1.0, 1.0, 1.0)

    means = rgb[mask].mean(axis=0)
    if abs(means[0] - means[2]) < CAST_DEADBAND:
        return rgb, (1.0, 1.0, 1.0)  # already neutral; touching it can only hurt

    gains = np.clip((means.mean() / means) ** WB_STRENGTH, WB_GAIN_MIN, WB_GAIN_MAX)
    return np.clip(rgb * gains, 0, 255), tuple(round(g, 3) for g in gains)


def black_point(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Set a real black point. This is what removes the hazy, lifted look."""
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    lo = float(np.percentile(lum, BLACK_PCT))
    shift = min(lo, BLACK_MAX_SHIFT)
    if shift <= 0.5:
        return rgb, 0.0
    return np.clip((rgb - shift) * (255.0 / (255.0 - shift)), 0, 255), round(shift, 1)


def desaturate(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Pull back only the frames that measure oversaturated. Most are untouched."""
    hsv = np.asarray(Image.fromarray(rgb.astype(np.uint8), "RGB").convert("HSV"),
                     dtype=np.float64)
    sat = hsv[:, :, 1].mean()
    if sat <= SAT_TRIGGER:
        return rgb, 1.0
    factor = max(SAT_TARGET / sat, 0.85)
    grey = (rgb @ np.array([0.2126, 0.7152, 0.0722]))[:, :, None]
    return np.clip(grey + (rgb - grey) * factor, 0, 255), round(factor, 3)


def process(src: Path, dst: Path, review_dir: Path | None = None) -> str:
    original = Image.open(src)
    # Carry EXIF across. Pillow drops it silently on save, and an earlier
    # version of this script therefore stripped camera, lens and capture date
    # from every corrected master. With the RAWs gone those masters are the
    # only originals left, so losing their metadata is not recoverable.
    exif = original.info.get("exif")
    icc = original.info.get("icc_profile")
    rgb = np.asarray(original.convert("RGB"), dtype=np.float64)
    rgb, gains = white_balance(rgb)
    rgb, shift = black_point(rgb)
    rgb, satf = desaturate(rgb)

    out = Image.fromarray(rgb.astype(np.uint8), "RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_kw = {"quality": 95, "subsampling": 0}
    if exif:
        save_kw["exif"] = exif
    if icc:
        save_kw["icc_profile"] = icc
    out.save(dst, "JPEG", **save_kw)

    if review_dir:
        review_dir.mkdir(parents=True, exist_ok=True)
        before = Image.open(src).convert("RGB")
        w = 900
        b = before.resize((w, round(w * before.height / before.width)), Image.LANCZOS)
        a = out.resize((w, round(w * out.height / out.width)), Image.LANCZOS)
        pair = Image.new("RGB", (w * 2 + 12, b.height), (20, 20, 20))
        pair.paste(b, (0, 0))
        pair.paste(a, (w + 12, 0))
        pair.save(review_dir / f"{src.stem}-compare.jpg", "JPEG", quality=88)

    return f"{src.stem:<24} wb={gains}  black=-{shift}  sat x{satf}"


if __name__ == "__main__":
    src_dir = Path(sys.argv[1])
    dst_dir = Path(sys.argv[2])
    review = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    names = sys.argv[4:]

    files = ([src_dir / n for n in names] if names
             else sorted(f for f in src_dir.glob("*.jpg")))
    for f in files:
        print(process(f, dst_dir / f.name, review), flush=True)
