"""Published copies carry only the metadata `share_metadata` lists (_studio_settings.json).

The generator (optimize_images.py) writes a FRESH block from an allowlist and never carries
the source EXIF over. Local copies - the masters, the originals, the finish folders - are
never modified. Everything runs on tmp_path; the real images/ folder is not regenerated.
"""
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageCms

REPO = Path(__file__).resolve().parents[1]
ALL = ["copyright", "artist", "camera", "lens", "exposure", "icc_profile"]
SECRET_SERIAL = "SN-SECRET-12345"
SECRET_COMMENT = b"secret comment"
ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _gen():
    spec = importlib.util.spec_from_file_location("optimize_images_share", REPO / "optimize_images.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _splice(jpeg: bytes, marker: int, payload: bytes) -> bytes:
    """Insert an APPn segment right after SOI."""
    return jpeg[:2] + b"\xff" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload + jpeg[2:]


def make_source(path: Path) -> Path:
    """A master with every kind of metadata the policy has to handle."""
    ex = Image.Exif()
    ex[0x010F] = "ACME"                       # Make
    ex[0x0110] = "Cam One"                    # Model
    ex[0x0131] = "Editing Suite 9"            # Software
    ex[0x0132] = "2025:08:01 10:00:00"        # DateTime
    ex[0x013B] = "Someone Else"               # Artist (the source's own)
    ex[0x8298] = "(c) source copyright"       # Copyright
    ex[0x010E] = "a description with a place name"
    sub = ex.get_ifd(0x8769)
    sub[0x9003] = "2025:08:01 09:59:59"       # DateTimeOriginal
    sub[0x9004] = "2025:08:01 09:59:59"
    sub[0x829A] = 1 / 250                     # ExposureTime
    sub[0x829D] = 8.0                         # FNumber
    sub[0x8827] = 100                         # ISO
    sub[0x920A] = 35.0                        # FocalLength
    sub[0xA431] = SECRET_SERIAL               # BodySerialNumber
    sub[0xA435] = SECRET_SERIAL               # LensSerialNumber
    sub[0xA433] = "ACME Optics"               # LensMake
    sub[0xA434] = "ACME 18-55"                # LensModel
    sub[0x927C] = b"MAKERNOTE-SECRET"         # MakerNote
    gps = ex.get_ifd(0x8825)
    gps[1], gps[2], gps[3], gps[4] = "N", (40.0, 21.0, 0.0), "W", (105.0, 33.0, 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (800, 600))
    im.putdata([(x % 256, y % 256, (x + y) % 256) for y in range(600) for x in range(800)])
    im.save(path, "JPEG", exif=ex.tobytes(), comment=SECRET_COMMENT, icc_profile=ICC)
    raw = path.read_bytes()
    xmp = (b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta><dc:creator>SECRET-XMP</dc:creator></x:xmpmeta>")
    iptc = b"Photoshop 3.0\x00" + b"8BIM\x04\x04\x00\x00\x00\x00\x00\x0aSECRET-IPTC"
    path.write_bytes(_splice(_splice(raw, 0xE1, xmp), 0xED, iptc))
    return path


def _tags(path: Path):
    with Image.open(path) as im:
        ex = im.getexif()
        base = {k for k in ex.keys()} - {0x8769, 0x8825}
        sub = set(ex.get_ifd(0x8769).keys())
        gps = set(ex.get_ifd(0x8825).keys())
        return base, sub, gps, dict(im.info), ex


def _run(tmp_path, monkeypatch, allowed):
    mod = _gen()
    src_dir, out = tmp_path / "corrected", tmp_path / "images"
    src = make_source(src_dir / "20250801-DSC_0001.jpg")
    out.mkdir()
    before = hashlib.sha1(src.read_bytes()).hexdigest()
    monkeypatch.setattr(mod, "INPUT_DIR", src_dir)
    monkeypatch.setattr(mod, "OUTPUT_DIR", out)
    monkeypatch.setattr(mod, "SIZES", [200, 400])
    mod.optimize_image(src, allowed)
    files = sorted(out.iterdir())
    assert [f.name for f in files] == ["20250801-DSC_0001-200.jpg", "20250801-DSC_0001-200.webp",
                                       "20250801-DSC_0001-400.jpg", "20250801-DSC_0001-400.webp"]
    assert hashlib.sha1(src.read_bytes()).hexdigest() == before          # the master is untouched
    return files


def _no_secrets(files):
    for f in files:
        raw = f.read_bytes()
        for needle in (SECRET_SERIAL.encode(), SECRET_COMMENT, b"SECRET-XMP", b"SECRET-IPTC", b"MAKERNOTE-SECRET",
                       b"Editing Suite", b"2025:08:01", b"a description with", b"ns.adobe.com/xap",
                       b"Photoshop 3.0", b"Someone Else", b"source copyright"):
            assert needle not in raw, (f.name, needle)


def test_each_published_size_carries_exactly_the_allowed_tags(tmp_path, monkeypatch):
    mod = _gen()
    files = _run(tmp_path, monkeypatch, mod.share_allowlist({"share_metadata": ALL}))
    for f in files:
        base, sub, gps, info, ex = _tags(f)
        assert base == {0x013B, 0x8298, 0x010F, 0x0110}, f.name          # Artist Copyright Make Model
        assert sub == {0xA433, 0xA434, 0x829A, 0x829D, 0x8827, 0x920A}, f.name
        assert gps == set(), f.name
        assert ex[0x013B] == "Mayniac Creations" and ex[0x8298].startswith("(c) 2025 Mayniac Creations")
        assert ex[0x010F] == "ACME" and ex.get_ifd(0x8769)[0xA434] == "ACME 18-55"
        assert info.get("icc_profile") == ICC, f.name
        assert not {"comment", "xmp", "photoshop"} & set(info), (f.name, set(info))
    _no_secrets(files)


def test_a_subset_publishes_only_that_subset(tmp_path, monkeypatch):
    mod = _gen()
    files = _run(tmp_path, monkeypatch, mod.share_allowlist({"share_metadata": ["camera", "copyright"]}))
    for f in files:
        base, sub, gps, info, ex = _tags(f)
        assert base == {0x010F, 0x0110, 0x8298} and sub == set() and gps == set()
        assert "icc_profile" not in info
    _no_secrets(files)


def test_with_the_setting_absent_only_the_icc_profile_goes_out(tmp_path, monkeypatch):
    mod = _gen()
    assert mod.share_allowlist({}) == {"icc_profile"}
    files = _run(tmp_path, monkeypatch, mod.share_allowlist({}))
    for f in files:
        base, sub, gps, info, ex = _tags(f)
        assert base == set() and sub == set() and gps == set(), f.name
        assert info.get("icc_profile") == ICC
        assert not {"comment", "xmp", "photoshop", "exif"} & set(info)
    _no_secrets(files)


def test_an_empty_list_publishes_nothing_at_all(tmp_path, monkeypatch):
    mod = _gen()
    files = _run(tmp_path, monkeypatch, mod.share_allowlist({"share_metadata": []}))
    for f in files:
        base, sub, gps, info, ex = _tags(f)
        assert base == sub == gps == set() and "icc_profile" not in info


def test_the_allowlist_is_read_from_the_studio_settings_file(tmp_path, monkeypatch):
    mod = _gen()
    monkeypatch.delenv("VAULT_PATH", raising=False)
    settings = tmp_path / "_studio_settings.json"
    settings.write_text(json.dumps({"share_metadata": ["copyright", "bogus", "artist"]}), encoding="utf-8")
    monkeypatch.setenv("MAYNIAC_STUDIO_SETTINGS", str(settings))
    assert mod.share_allowlist() == {"copyright", "artist"}                  # unknown names ignored
    monkeypatch.setenv("MAYNIAC_STUDIO_SETTINGS", str(tmp_path / "nope.json"))
    assert mod.share_allowlist() == {"icc_profile"}                          # no file: the safe default
    settings.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("MAYNIAC_STUDIO_SETTINGS", str(settings))
    assert mod.share_allowlist() == {"icc_profile"}                          # unreadable: fail closed
