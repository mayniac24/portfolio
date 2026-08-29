"""Guards for copyright metadata on the deployed images.

The visible watermark was removed on 2026-08-29. Embedded rights are what
replaced it: invisible, no cost to the image, and they survive reposting in a
way a corner wordmark does not (generative fill erases one in seconds).

That trade only holds if the metadata is actually present on every file, so
these tests exist to keep it there.
"""

import json
import re
from pathlib import Path

import pytest
from PIL import Image
from PIL.ExifTags import TAGS

REPO = Path(__file__).resolve().parents[1]
IMAGES = REPO / "images"
GALLERY = REPO / "data" / "gallery.json"

COPYRIGHT_TAG = 0x8298
ARTIST_TAG = 0x013B


def _deployed():
    return sorted(p for p in IMAGES.iterdir()
                  if p.suffix.lower() in {".jpg", ".webp"})


def _exif(p: Path) -> dict:
    ex = Image.open(p).getexif()
    return {TAGS.get(k, k): v for k, v in ex.items()} if ex else {}


def test_every_deployed_image_has_copyright_and_creator():
    missing = []
    for p in _deployed():
        d = _exif(p)
        if not d.get("Copyright") or not d.get("Artist"):
            missing.append(p.name)
    assert missing == [], f"{len(missing)} deployed images lack rights metadata: {missing[:5]}"


def test_copyright_year_matches_the_capture_year():
    """A wrong year on a registered work is a real problem.

    The maternity set carries a 20260202- filename prefix taken from a Lightroom
    export timestamp, five months after the shoot. gallery.json records the true
    capture date; the notice must follow that, not the filename.
    """
    captured = {p["id"]: str(p["captured"])[:4]
                for p in json.loads(GALLERY.read_text(encoding="utf-8"))["photos"]
                if p.get("captured")}
    assert captured, "no capture dates recorded"

    bad = []
    for stem, year in captured.items():
        for p in IMAGES.glob(f"{stem}-*.jpg"):
            notice = _exif(p).get("Copyright", "")
            m = re.search(r"\(c\) (\d{4})", notice)
            if not m or m.group(1) != year:
                bad.append((p.name, notice, year))
    assert bad == [], f"copyright year disagrees with capture year: {bad[:5]}"


def test_no_gps_is_published():
    """Rights metadata goes out; client shoot locations do not.

    optimize_images.py writes a minimal fresh block rather than copying source
    EXIF, precisely so GPS never reaches the web.
    """
    leaked = []
    for p in _deployed():
        ex = Image.open(p).getexif()
        if ex and ex.get_ifd(0x8825):
            leaked.append(p.name)
    assert leaked == [], f"GPS data published on: {leaked[:5]}"


@pytest.mark.parametrize("field", ["Copyright", "Artist"])
def test_rights_fields_name_the_owner(field):
    sample = next(iter(_deployed()))
    assert "Mayniac Creations" in str(_exif(sample).get(field, "")), \
        f"{field} does not name the owner"
