"""Regression guard: EXIF-rotated masters must ship tall, not sideways.

optimize_images.py resized before applying the source's EXIF Orientation tag,
so a master stored wide with an Orientation tag (portrait shot, camera-rotated)
shipped as wide, untagged variants -- displayed sideways on the live site.
Fixed by ImageOps.exif_transpose() right after opening, before any resize.
"""

import importlib.util
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def _load_optimize_images():
    spec = importlib.util.spec_from_file_location("optimize_images", REPO / "optimize_images.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["optimize_images"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_exif_rotated_master_yields_tall_untagged_variants(tmp_path, monkeypatch):
    optimize_images = _load_optimize_images()

    # Build a wide (600x400) master tagged Orientation=6 (rotate 90 CW to display),
    # matching how the three real masters (20240326-DSC_0482 etc.) are stored.
    src_dir = tmp_path / "corrected"
    src_dir.mkdir()
    out_dir = tmp_path / "images"

    img = Image.new("RGB", (600, 400), color=(10, 20, 30))
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation
    src_path = src_dir / "20990101-TEST_0001.jpg"
    img.save(src_path, "JPEG", exif=exif.tobytes())

    monkeypatch.setattr(optimize_images, "INPUT_DIR", src_dir)
    monkeypatch.setattr(optimize_images, "OUTPUT_DIR", out_dir)
    # Post-transpose the master is 400 wide x 600 tall, so the requested
    # width must be smaller than the TRANSPOSED width, not the stored one.
    monkeypatch.setattr(optimize_images, "SIZES", [300])
    out_dir.mkdir()

    optimize_images.optimize_image(src_path)

    jpeg_path = out_dir / "20990101-TEST_0001-300.jpg"
    assert jpeg_path.exists()

    with Image.open(jpeg_path) as out:
        width, height = out.size
        assert height > width, f"expected a tall variant, got {width}x{height}"
        assert out.getexif().get(0x0112) is None, (
            "variant must carry no Orientation tag -- exif_transpose bakes "
            "the rotation into pixels; a leftover tag would rotate it twice"
        )
