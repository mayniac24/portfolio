#!/usr/bin/env python3
"""One-off migration: hand-edited gallery markup -> data/gallery.json.

Everything up to now has been regex surgery on index.html, which is fragile and
impossible for a CMS to edit safely. This lifts the gallery into structured data
so build.py can regenerate the markup and an admin UI can edit the data.

Run once. After this, gallery.json is the source of truth and this script is
history.
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "index.html"
OUT = REPO / "data" / "gallery.json"

ITEM_RE = re.compile(
    r'<div class="gallery-item (?P<orient>is-wide|is-tall)" '
    r'data-category="(?P<cat>[^"]+)"(?P<attrs>[^>]*)>.*?'
    r'data-lightbox-src="images/(?P<id>[\w.-]+)-2000\.webp".*?'
    r'alt="(?P<alt>[^"]*)"',
    re.DOTALL,
)
ATTR_RE = re.compile(r'data-attributes="([^"]+)"')

# New landscape work, vetted by eye rather than by any scoring metric.
# Both automated ranking attempts surfaced snapshots; these were chosen by
# looking at contact sheets.
NEW_LANDSCAPES = [
    ("20220712-DSC_0358",
     "Chimney Rock sandstone spires lit by low evening sun against a clear sky."),
    ("20240326-DSC_0482",
     "Palm trees on a green sea cliff above the surf at Pololu Valley, Hawaii."),
    ("20241113-DSC_0059",
     "Longs Peak silhouetted against a burning orange sunset sky."),
    ("20250818-DSC_0296",
     "Notchtop Mountain rising behind conifers in golden evening light."),
    ("20240326-DSC_0473",
     "Pololu Valley coastline with palms leaning out over the Pacific."),
    ("20241113-DSC_0084",
     "Orange and teal sunset banding above a dark Front Range ridgeline."),
    ("20210718-DSC_5526",
     "Layered mountain ridgelines fading into dusk from Trail Ridge Road."),
    ("20250601-DSC_0087",
     "Snow-covered peak catching warm light through a screen of pines."),
]

CATEGORIES = [
    {"id": "landscape", "label": "Landscapes"},
    {"id": "wedding", "label": "Weddings"},
    {"id": "maternity", "label": "Maternity"},
]


def main() -> None:
    html = INDEX.read_text(encoding="utf-8")
    photos = []

    for m in ITEM_RE.finditer(html):
        attrs = ATTR_RE.search(m.group("attrs"))
        photos.append({
            "id": m.group("id"),
            "category": m.group("cat"),
            "attributes": attrs.group(1).split() if attrs else [],
            "alt": m.group("alt"),
        })
    print(f"extracted {len(photos)} existing photos")

    existing = {p["id"] for p in photos}
    added = [{"id": i, "category": "landscape", "attributes": [], "alt": a}
             for i, a in NEW_LANDSCAPES if i not in existing]
    print(f"adding {len(added)} landscape frames")

    # Landscapes lead: that is the stated identity of the site.
    ordered = added + photos

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps({"categories": CATEGORIES, "photos": ordered}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(f"wrote {len(ordered)} photos -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
