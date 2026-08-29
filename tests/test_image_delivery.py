"""Regression guards for image delivery (spec findings F1, F2).

These encode the contract that no full-resolution original is ever
referenced by the deployed page.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "index.html"
MAX_WIDTH = 2000

SRCSET_RE = re.compile(r'srcset="([^"]+)"')
CANDIDATE_RE = re.compile(r"(\S+)\s+(\d+)w")
IMG_SRC_RE = re.compile(r'<img[^>]*\ssrc="(images/[^"]+)"')
LIGHTBOX_RE = re.compile(r'data-lightbox-(?:src|fallback)="(images/[^"]+)"')
# Must list the known tiers explicitly. A loose `-(\d+)\.jpg$` would treat
# originals like `20260202-DSC_0026-1.jpg` as variants and silently pass.
VARIANT_RE = re.compile(r"-(?:400|800|1200|2000)\.(?:jpg|webp)$")
ASSET_RE = re.compile(r"images/[\w.-]+\.(?:jpg|webp)")


def _html():
    return INDEX.read_text(encoding="utf-8")


def _srcset_candidates():
    for srcset in SRCSET_RE.findall(_html()):
        for url, width in CANDIDATE_RE.findall(srcset):
            yield url, int(width)


def test_no_srcset_candidate_exceeds_max_width():
    oversized = [(u, w) for u, w in _srcset_candidates() if w > MAX_WIDTH]
    assert oversized == [], f"srcset candidates above {MAX_WIDTH}w: {oversized}"


def test_img_src_never_points_at_an_original():
    bad = [s for s in IMG_SRC_RE.findall(_html()) if not VARIANT_RE.search(s)]
    assert bad == [], f"<img src> pointing at full-resolution originals: {bad}"


def test_lightbox_targets_are_variants():
    targets = LIGHTBOX_RE.findall(_html())
    assert targets, "no data-lightbox-src attributes found in index.html"
    bad = [t for t in targets if not VARIANT_RE.search(t)]
    assert bad == [], f"lightbox targets pointing at originals: {bad}"


def test_lightbox_js_does_not_read_img_src():
    assert "lightboxImg.src = img.src" not in _html(), (
        "lightbox is still reading the src attribute (F1)"
    )


def test_every_referenced_asset_exists_on_disk():
    refs = sorted(set(ASSET_RE.findall(_html())))
    missing = [r for r in refs if not (REPO / r).exists()]
    assert missing == [], f"referenced but missing from disk: {missing}"
