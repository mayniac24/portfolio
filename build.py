#!/usr/bin/env python3
"""Render the gallery and filter bar into index.html from data/gallery.json.

gallery.json is the source of truth. This regenerates only the two marked
regions of index.html, leaving the CSS, JS and page structure untouched -- a
narrower and safer contract than templating the whole page, and the one an
admin UI can write against.

Image dimensions and available variants are read from disk rather than stored
in JSON, so they can never drift from the actual files.

    python build.py            regenerate index.html
    python build.py --check    exit 1 if index.html is out of date (for CI)
"""

import json
import re
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent
INDEX = REPO / "index.html"
DATA = REPO / "data" / "gallery.json"
IMAGES = REPO / "images"
SIZES = [400, 800, 1200, 2000]
# Wide tiles span two grid columns, tall tiles span one -- so a tall image
# displays at half the width and must advertise half the sizes, or the browser
# picks a variant twice as large as it needs. Getting this wrong is a quieter
# rerun of F2.
SIZES_WIDE = "(max-width: 480px) 100vw, (max-width: 768px) 50vw, 25vw"
SIZES_TALL = "(max-width: 480px) 50vw, (max-width: 768px) 25vw, 12.5vw"

GALLERY_RE = re.compile(
    r"(?P<open><!-- GALLERY:START -->\n).*?(?P<close>[ \t]*<!-- GALLERY:END -->)",
    re.DOTALL)
FILTERS_RE = re.compile(
    r"(?P<open><!-- FILTERS:START -->\n).*?(?P<close>[ \t]*<!-- FILTERS:END -->)",
    re.DOTALL)


def variants(stem: str, ext: str) -> list[int]:
    return [s for s in SIZES if (IMAGES / f"{stem}-{s}.{ext}").exists()]


def render_photo(p: dict, indent: str = "        ") -> str:
    stem = p["id"]
    jpgs, webps = variants(stem, "jpg"), variants(stem, "webp")
    if not jpgs or not webps:
        raise SystemExit(f"no variants on disk for {stem}; run optimize_images.py")

    largest_jpg, largest_webp = max(jpgs), max(webps)
    fallback = max([s for s in jpgs if s <= 1200], default=largest_jpg)
    with Image.open(IMAGES / f"{stem}-{largest_jpg}.jpg") as im:
        w, h = im.size

    srcset = lambda ext, sizes: ", ".join(
        f"images/{stem}-{s}.{ext} {s}w" for s in sizes)
    attrs = (f' data-attributes="{" ".join(p["attributes"])}"'
             if p.get("attributes") else "")
    orient = "is-wide" if w > h else "is-tall"
    sizes_attr = SIZES_WIDE if orient == "is-wide" else SIZES_TALL
    alt = p["alt"].replace('"', "&quot;")

    return (
        f'{indent}<div class="gallery-item {orient}" data-category="{p["category"]}"{attrs}>\n'
        f'{indent}    <picture>\n'
        f'{indent}        <source type="image/webp" srcset="{srcset("webp", webps)}" '
        f'sizes="{sizes_attr}">\n'
        f'{indent}        <img src="images/{stem}-{fallback}.jpg" '
        f'srcset="{srcset("jpg", jpgs)}" sizes="{sizes_attr}" '
        f'width="{w}" height="{h}" '
        f'data-lightbox-src="images/{stem}-{largest_webp}.webp" '
        f'data-lightbox-fallback="images/{stem}-{largest_jpg}.jpg" '
        f'alt="{alt}" loading="lazy">\n'
        f'{indent}    </picture>\n'
        f'{indent}</div>'
    )


def render_filters(cats: list[dict], photos: list[dict], indent: str = "        ") -> str:
    used = {p["category"] for p in photos}
    lines = [f'{indent}<button class="filter-btn active" data-filter="all">All</button>']
    for c in cats:
        # A button matching nothing is dead weight; one matching everything is a
        # lie. Both were true of the old hand-written bar.
        if c["id"] in used and len(used) > 1:
            lines.append(f'{indent}<button class="filter-btn" '
                         f'data-filter="{c["id"]}">{c["label"]}</button>')
    return "\n".join(lines)


def build() -> str:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    photos, cats = data["photos"], data["categories"]

    known = {c["id"] for c in cats}
    bad = [p["id"] for p in photos if p["category"] not in known]
    if bad:
        raise SystemExit(f"photos with unknown category: {bad}")

    html = INDEX.read_text(encoding="utf-8")
    for regex, body in (
        (GALLERY_RE, "\n".join(render_photo(p) for p in photos)),
        (FILTERS_RE, render_filters(cats, photos)),
    ):
        if not regex.search(html):
            raise SystemExit("marker comments missing from index.html")
        html = regex.sub(lambda m: m.group("open") + body + "\n" + m.group("close"),
                         html, count=1)
    return html


def main() -> None:
    rendered = build()
    if "--check" in sys.argv:
        if rendered != INDEX.read_text(encoding="utf-8"):
            print("index.html is out of date; run python build.py")
            sys.exit(1)
        print("index.html is up to date")
        return
    INDEX.write_text(rendered, encoding="utf-8", newline="\n")
    n = len(json.loads(DATA.read_text(encoding="utf-8"))["photos"])
    print(f"rendered {n} photos into index.html")


if __name__ == "__main__":
    main()
