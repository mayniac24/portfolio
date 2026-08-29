"""Guards for the build pipeline.

data/gallery.json is the source of truth; index.html is generated. These tests
keep that true, so a hand-edit to the markup cannot silently survive the next
build, and an admin UI writing JSON can trust what comes out.
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "index.html"
DATA = REPO / "data" / "gallery.json"

import sys
sys.path.insert(0, str(REPO))
import build  # noqa: E402


def _gallery():
    return json.loads(DATA.read_text(encoding="utf-8"))


def test_index_is_up_to_date_with_gallery_json():
    """The committed HTML must equal what build.py produces."""
    assert build.build() == INDEX.read_text(encoding="utf-8"), (
        "index.html is out of date; run python build.py"
    )


def test_build_is_idempotent():
    once = build.build()
    INDEX.write_text(once, encoding="utf-8", newline="\n")
    assert build.build() == once, "build.py is not idempotent"


def test_every_photo_has_alt_text():
    bad = [p["id"] for p in _gallery()["photos"] if not p.get("alt", "").strip()]
    assert bad == [], f"photos missing alt text: {bad}"


def test_photo_ids_are_unique():
    ids = [p["id"] for p in _gallery()["photos"]]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate photo ids: {dupes}"


def test_every_photo_has_a_known_category():
    data = _gallery()
    known = {c["id"] for c in data["categories"]}
    bad = [(p["id"], p["category"]) for p in data["photos"]
           if p["category"] not in known]
    assert bad == [], f"photos with unknown category: {bad}"


@pytest.mark.parametrize("orient,expected", [
    ("is-wide", build.SIZES_WIDE),
    ("is-tall", build.SIZES_TALL),
])
def test_sizes_attribute_matches_grid_span(orient, expected):
    """Tall tiles span one column, wide tiles two.

    A tall image advertising the wide sizes makes the browser fetch a variant
    twice as large as it renders -- a quieter rerun of finding F2. build.py had
    exactly this bug until the byte-comparison against the hand-edited markup
    caught it.
    """
    html = INDEX.read_text(encoding="utf-8")
    items = re.findall(
        r'<div class="gallery-item ' + orient + r'".*?</div>', html, re.DOTALL)
    assert items, f"no {orient} items found"
    for it in items:
        for s in re.findall(r'sizes="([^"]+)"', it):
            assert s == expected, f"{orient} item has wrong sizes: {s}"


def test_landscapes_lead_the_gallery():
    """The site's stated identity is landscape-first; the order encodes it."""
    cats = [p["category"] for p in _gallery()["photos"]]
    assert cats[0] == "landscape", f"first photo is {cats[0]}, not a landscape"
