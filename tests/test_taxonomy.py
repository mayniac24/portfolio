"""Guards for the category taxonomy (spec finding F3).

The original bug was not a typo -- it was that "landscape" named both a CSS
grid-span class and a session type, so the categoriser applied both senses to
every image. "Portraits" then matched all 23 images and "Landscapes" matched 22,
which meant the filter bar looked functional and did nothing.

These tests encode the two axes staying separate, and that every button on the
bar actually selects a real, proper subset of the gallery.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "index.html"

# Subject categories. "landscape" is legitimate here now: orientation moved to
# is-wide/is-tall, so the word no longer names two different things. That
# collision -- not the word itself -- was finding F3.
CATEGORIES = {"landscape", "wedding", "engagement", "maternity",
              "family", "portrait", "event"}
ORIENTATIONS = {"is-wide", "is-tall"}

ITEM_RE = re.compile(r'<div class="gallery-item([^"]*)" data-category="([^"]*)"')
BUTTON_RE = re.compile(r'<button class="filter-btn[^"]*" data-filter="([^"]+)"')


def _html():
    return INDEX.read_text(encoding="utf-8")


def _items():
    return [(cls.split(), cat.split()) for cls, cat in ITEM_RE.findall(_html())]


def test_every_item_has_exactly_one_category():
    bad = [c for _, c in _items() if len(c) != 1]
    assert bad == [], f"items with zero or multiple category values: {bad}"


def test_categories_come_from_the_agreed_vocabulary():
    used = {c[0] for _, c in _items() if c}
    assert used <= CATEGORIES, f"unknown category values: {used - CATEGORIES}"


def test_no_orientation_class_is_used_as_a_category():
    # F3 was one word naming both axes. The axes are now disjoint by
    # construction, and this keeps them that way.
    leaked = [c for _, c in _items() if ORIENTATIONS & set(c)]
    assert leaked == [], f"orientation classes leaking into data-category: {leaked}"


def test_every_item_has_exactly_one_orientation_class():
    bad = [cls for cls, _ in _items() if len(ORIENTATIONS & set(cls)) != 1]
    assert bad == [], f"items with zero or multiple orientation classes: {bad}"


def test_every_filter_button_selects_a_real_subset():
    """A button matching nothing is dead. One matching everything is a lie."""
    items = _items()
    total = len(items)
    for f in BUTTON_RE.findall(_html()):
        if f == "all":
            continue
        n = sum(1 for _, cats in items if f in cats)
        assert n > 0, f"filter '{f}' matches no images"
        assert n < total, f"filter '{f}' matches all {total} images, so it filters nothing"
