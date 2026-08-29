#!/usr/bin/env python3
"""One-off rewrite of the gallery markup in index.html.

Caps every srcset at 2000w, repoints the src fallback away from the original,
adds intrinsic width/height (fixes layout shift), and adds explicit lightbox
targets so the lightbox stops reading the src attribute.

Superseded by build.py in phase 3, which must reproduce this exact markup.
"""

import re
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "index.html"
IMAGES = REPO / "images"
SIZES = [400, 800, 1200, 2000]

SOURCE_RE = re.compile(
    r'<source type="image/webp" srcset="images/(?P<stem>[^"]+)-400\.webp [^"]*" '
    r'sizes="(?P<sizes>[^"]*)">'
)
IMG_RE = re.compile(
    r'<img src="images/(?P<stem>[^"]+)\.jpg" srcset="[^"]*" sizes="(?P<sizes>[^"]*)" '
    r'alt="(?P<alt>[^"]*)" loading="lazy">'
)


def available(stem: str, ext: str) -> list[int]:
    return [s for s in SIZES if (IMAGES / f"{stem}-{s}.{ext}").exists()]


def srcset(stem: str, ext: str) -> str:
    return ", ".join(f"images/{stem}-{s}.{ext} {s}w" for s in available(stem, ext))


def rebuild_source(m: re.Match) -> str:
    stem = m.group("stem")
    return (
        f'<source type="image/webp" srcset="{srcset(stem, "webp")}" '
        f'sizes="{m.group("sizes")}">'
    )


def rebuild_img(m: re.Match) -> str:
    stem, sizes_attr, alt = m.group("stem"), m.group("sizes"), m.group("alt")
    jpgs = available(stem, "jpg")
    webps = available(stem, "webp")
    if not jpgs or not webps:
        raise SystemExit(f"no variants on disk for {stem}; run optimize_images.py first")

    largest_jpg = max(jpgs)
    largest_webp = max(webps)
    fallback = max([s for s in jpgs if s <= 1200], default=largest_jpg)

    with Image.open(IMAGES / f"{stem}-{largest_jpg}.jpg") as im:
        width, height = im.size

    return (
        f'<img src="images/{stem}-{fallback}.jpg" srcset="{srcset(stem, "jpg")}" '
        f'sizes="{sizes_attr}" width="{width}" height="{height}" '
        f'data-lightbox-src="images/{stem}-{largest_webp}.webp" '
        f'data-lightbox-fallback="images/{stem}-{largest_jpg}.jpg" '
        f'alt="{alt}" loading="lazy">'
    )


def main() -> None:
    html = INDEX.read_text(encoding="utf-8")
    html, n_source = SOURCE_RE.subn(rebuild_source, html)
    html, n_img = IMG_RE.subn(rebuild_img, html)
    INDEX.write_text(html, encoding="utf-8", newline="\n")
    print(f"rewrote {n_source} <source> and {n_img} <img> tags")
    if n_source != n_img:
        raise SystemExit(f"mismatch: {n_source} sources vs {n_img} images")


if __name__ == "__main__":
    main()
