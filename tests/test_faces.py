"""The face guard (tools/picker/faces.py): Evan's rule, 2026-10-17: our kids' faces are never
published, and unknown means refuse. Everything runs on tmp fixtures with injectable detectors;
one smoke test uses the real detector and skips when this machine has none. No person in a
fixture has a real name.
"""
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageStat

REPO = Path(__file__).resolve().parents[1]
PICKER = REPO / "tools" / "picker"
sys.path.insert(0, str(PICKER))
sys.path.insert(0, str(REPO / "tools"))
import faces  # noqa: E402

SECRET = "SN-SECRET-98765"
FACE = (0.40, 0.30, 0.10, 0.15)           # a "face" as fractions of the image: x, y, w, h


# ---------------------------------------------------------------- fakes and fixtures

class NoiseDetector:
    """Finds a face in each given box (fractions of the image) while that region is still
    high-frequency noise. A mask (flat ellipse + emoji) therefore really removes it, and a mask
    that misses the box really leaves it: the detector reads the pixels, not the file name."""
    name = "noise"

    def __init__(self, rel_boxes=(FACE,)):
        self.rel = list(rel_boxes)

    def detect(self, image):
        W, H = image.size
        out = []
        for x, y, w, h in self.rel:
            box = (int(x * W), int(y * H), int((x + w) * W), int((y + h) * H))
            reg = image.crop(box).convert("L")
            busy = ImageStat.Stat(ImageChops.difference(reg, ImageChops.offset(reg, 1, 0))).mean[0]
            if busy > 40:
                out.append([box[0], box[1], box[2] - box[0], box[3] - box[1], 0.9])
        return out


class NoFace:
    name = "none"

    def detect(self, image):
        return []


class Boom:
    name = "boom"

    def detect(self, image):
        raise RuntimeError("detector exploded")


def _sha(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def noise_image(path, size=(600, 400), seed=1, exif=False):
    rng = random.Random(seed)
    im = Image.frombytes("RGB", size, bytes(rng.randrange(256) for _ in range(size[0] * size[1] * 3)))
    kw = {}
    if exif:
        ex = Image.Exif()
        ex[0x010F], ex[0x0110], ex[0x0131] = "ACME", "Cam One", "Editing Suite 9"
        sub = ex.get_ifd(0x8769)
        sub[0xA431] = SECRET
        sub[0x9003] = "2025:08:01 09:59:59"
        gps = ex.get_ifd(0x8825)
        gps[1], gps[2], gps[3], gps[4] = "N", (40.0, 21.0, 0.0), "W", (105.0, 33.0, 0.0)
        kw["exif"] = ex.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=95, **kw)
    return path


def smooth_image(path, size=(300, 200)):
    im = Image.linear_gradient("L").resize(size).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=95)
    return path


def note(path, **fm):
    lines = ["---"]
    for k, v in fm.items():
        lines.append("%s: %s" % (k, json.dumps(v)))
    lines += ["---", "", "body"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def raw_note(path, frontmatter):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\n" + frontmatter + "\n---\n\nbody\n", encoding="utf-8")
    return path


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A vault with settings, People notes, an outing projection, and an empty events dir."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "_vault_paths.json").write_text(json.dumps({"paths": {"people": "People"}}), encoding="utf-8")
    people = vault / "People"
    note(people / "Kid One.md", type="person", relationship="family", tags=["family", "child"])
    note(people / "Adult Two.md", type="person", relationship="family", tags=["family"])
    note(people / "Pal Three.md", type="person", relationship="friend", tags=["friend"])
    note(people / "Owner Zero.md", type="person", relationship="self", tags=["person"])
    young = date.today().replace(day=1, year=date.today().year - 9).isoformat()
    note(people / "Minor Four.md", type="person", relationship="friend", tags=["friend"], date_of_birth=young)
    (vault / "_studio_outings.json").write_text(json.dumps({"outings": [
        {"id": "out-kids", "folder": "Hiking/kids", "hike_logs": [{"companions": ["[[People/Kid One|Kid One]]", "Pal"]}]},
        {"id": "out-adults", "folder": "Hiking/adults", "hike_logs": [{"companions": "Adult Two, Pal Three"}]},
        {"id": "out-minor", "folder": "Hiking/minor", "hike_logs": [{"companions": ["Minor Four"]}]},
        {"id": "out-stranger", "folder": "Hiking/stranger", "hike_logs": [{"companions": ["Adult Two", "Somebody Else"]}]},
        {"id": "out-solo", "folder": "Hiking/solo", "hike_logs": [{"companions": ["solo"]}]},
        {"id": "out-blank", "folder": "Hiking/blank", "hike_logs": [{"companions": []}]},
        {"id": "out-nolog", "folder": "Client/session", "hike_logs": []},
    ]}), encoding="utf-8")
    settings_file = vault / "_studio_settings.json"
    settings_file.write_text(json.dumps({
        "face_clearers": ["Evan"], "child_relationships": ["child"], "face_emoji": "\U0001F642",
        "share_metadata": ["camera"], "photos_root": str(tmp_path / "photos"),
    }), encoding="utf-8")
    events = tmp_path / "_events"
    monkeypatch.setattr(faces, "default_detectors", lambda settings=None: [NoiseDetector()])
    return {"tmp": tmp_path, "vault": vault, "settings_file": settings_file, "events": events,
            "settings": faces.link.load_settings(str(settings_file)), "root": tmp_path / "photos"}


@pytest.fixture
def no_face(monkeypatch):
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [NoFace()])


@pytest.fixture
def desk(monkeypatch):
    """Evan at an interactive terminal."""
    monkeypatch.setattr(faces, "_interactive", lambda: True)


def rows(w):
    f = w["events"] / "studio-picker.jsonl"
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []


def cli(w, *args):
    return faces.main([*args, "--settings", str(w["settings_file"]), "--events-dir", str(w["events"])])


def make_masked(src, name="a_masked.jpg", boxes=None):
    """A truly masked copy, made the way `mask` makes it, without the bus."""
    img, _, _ = faces.load_pixels(src)
    W, H = img.size
    boxes = boxes or [[FACE[0] * W, FACE[1] * H, FACE[2] * W, FACE[3] * H, 1.0]]
    path = Path(src).with_name(name)
    faces.write_masked(faces.mask_pixels(img, boxes, "\U0001F642"), path, None, None)
    return path


def record_masked(w, frame, path, outing="o", **extra):
    return faces.record(w["settings"], "frame", frame, "masked", outing=outing, events_dir=w["events"],
                        image=str(path), sha1=_sha(path), **extra)


def clear_row(w, ident, who="evan", kind="frame", outing="o", **extra):
    return faces.record(w["settings"], kind, ident, "cleared", outing=outing, by=who,
                        events_dir=w["events"], **extra)


# ---------------------------------------------------------------- geometry and scanning

@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_unrotate_box_maps_a_detection_back(k):
    W, H = 60, 40
    base = Image.new("RGB", (W, H), (0, 0, 0))
    for x in range(10, 20):
        for y in range(5, 25):
            base.putpixel((x, y), (255, 255, 255))
    rot = base if k == 0 else base.rotate(90 * k, expand=True)
    px = [(x, y) for x in range(rot.width) for y in range(rot.height) if rot.getpixel((x, y)) == (255, 255, 255)]
    x0, y0 = min(p[0] for p in px), min(p[1] for p in px)
    x1, y1 = max(p[0] for p in px) + 1, max(p[1] for p in px) + 1
    got = faces.unrotate_box(k, [x0, y0, x1 - x0, y1 - y0, 1.0], W, H)
    assert [round(v) for v in got[:4]] == [10, 5, 10, 20]


def test_merge_boxes_unions_overlaps_and_keeps_separate_faces():
    merged = faces.merge_boxes([[10, 10, 40, 40, 0.5], [12, 14, 40, 40, 0.9], [200, 10, 30, 30, 1.0]])
    assert len(merged) == 2
    big = min(merged, key=lambda b: b[0])
    assert big[0] == 10 and big[1] == 10 and big[2] == 42 and big[4] == 0.9


@pytest.mark.parametrize("length", [1601, 2000, 2400, 3000, 4000, 6000, 7000])
def test_tiles_overlap_so_a_face_under_400_px_lands_whole_in_one(length):
    starts = faces.tile_origins(length)
    assert starts[0] == 0 and starts[-1] + faces.LONG_EDGE == length
    for x in range(0, length - 400 + 1, 5):
        assert any(s <= x and x + 400 <= s + faces.LONG_EDGE for s in starts), (length, x)
    assert faces.tile_origins(1600) == [0] and faces.tile_origins(900) == [0]


class TileOnly:
    """Sees a white block only in a full-resolution 1600 px tile; never in the downscaled whole."""
    name = "tile-only"
    seen = []

    def detect(self, image):
        TileOnly.seen.append(image.size)
        if image.size != (1600, 1600):
            return []
        box = image.convert("L").point(lambda v: 255 if v >= 250 else 0).getbbox()
        return [[box[0], box[1], box[2] - box[0], box[3] - box[1], 1.0]] if box else []


def test_a_big_image_is_scanned_in_tiles_and_boxes_map_back(tmp_path):
    TileOnly.seen = []
    img = Image.new("RGB", (3300, 2200), (90, 90, 90))
    img.paste((255, 255, 255), (1500, 900, 1620, 1020))
    p = tmp_path / "big.png"
    img.save(p)
    res = faces.check_image(p, {}, [TileOnly()])
    assert (1600, 1067) in TileOnly.seen and TileOnly.seen.count((1600, 1600)) >= 6
    (f,) = res["faces"]
    assert abs(f["x"] - 1500) <= 3 and abs(f["y"] - 900) <= 3 and abs(f["w"] - 120) <= 4 and abs(f["h"] - 120) <= 4


def test_a_small_image_gets_an_upscaled_scan_and_no_tiles(tmp_path):
    seen = []

    class Spy:
        name = "spy"

        def detect(self, image):
            seen.append(image.size)
            return []
    faces.detect_faces(Image.new("RGB", (400, 300)), [Spy()])
    assert sorted(seen) == [(400, 300), (1600, 1200)]
    seen.clear()
    faces.detect_faces(Image.new("RGB", (1500, 1000)), [Spy()])
    assert seen == [(1500, 1000)]


# ---------------------------------------------------------------- detection wiring

def test_check_reports_faces_in_the_documented_shape(world, tmp_path):
    img = noise_image(tmp_path / "a.jpg")
    res = faces.check_image(img, world["settings"])
    assert set(res) == {"faces", "detector", "version"} and res["version"] == 1
    assert res["detector"] == "noise"
    assert len(res["faces"]) == 1
    f = res["faces"][0]
    assert set(f) == {"x", "y", "w", "h", "score"}
    assert abs(f["x"] - 240) <= 3 and abs(f["w"] - 60) <= 4


def test_check_cli_prints_json_and_exits_zero(world, tmp_path, capsys):
    img = noise_image(tmp_path / "a.jpg")
    assert cli(world, "check", str(img), "--json") == 0
    assert json.loads(capsys.readouterr().out)["faces"]


def test_real_detector_smoke(tmp_path):
    """The real detector runs on a generated image and reports none. Skips off Windows."""
    try:
        dets = faces.default_detectors({})
    except faces.FaceError:
        pytest.skip("no face detector on this machine")
    img = Image.new("RGB", (500, 300), (90, 140, 200))
    p = tmp_path / "sky.jpg"
    img.save(p, "JPEG")
    res = faces.check_image(p, {}, dets)
    assert res["faces"] == [] and res["detector"]


def test_real_detector_finds_the_astronaut(tmp_path):
    skimage_data = pytest.importorskip("skimage.data")
    try:
        dets = faces.default_detectors({})
    except faces.FaceError:
        pytest.skip("no face detector on this machine")
    p = tmp_path / "astronaut.png"
    Image.fromarray(skimage_data.astronaut()).save(p)
    assert len(faces.check_image(p, {}, dets)["faces"]) == 1


# ---------------------------------------------------------------- masking

def _ellipse_pixels(img_size, box):
    px, py, pw, ph = faces.pad_box(box)
    cx, cy, a, b = px + pw / 2, py + ph / 2, pw / 2, ph / 2
    W, H = img_size
    return [(x, y) for y in range(max(0, int(py)), min(H, int(py + ph) + 1))
            for x in range(max(0, int(px)), min(W, int(px + pw) + 1))
            if ((x + 0.5 - cx) / a) ** 2 + ((y + 0.5 - cy) / b) ** 2 <= 0.95]


def test_mask_leaves_no_original_pixel_in_the_padded_ellipse(tmp_path):
    img = noise_image(tmp_path / "a.jpg")
    src, _, _ = faces.load_pixels(img)
    box = [240.0, 120.0, 60.0, 60.0, 1.0]
    out = faces.mask_pixels(src, [box], "\U0001F642")
    same = [p for p in _ellipse_pixels(src.size, box) if out.getpixel(p) == src.getpixel(p)]
    assert len(_ellipse_pixels(src.size, box)) > 3000 and same == []
    tight = [(x, y) for x in range(240, 300) for y in range(120, 180) if out.getpixel((x, y)) == src.getpixel((x, y))]
    assert tight == []
    assert out.getpixel((5, 5)) == src.getpixel((5, 5))            # far away: untouched


def test_mask_without_the_emoji_font_is_a_plain_ellipse(tmp_path, monkeypatch):
    monkeypatch.setattr(faces, "_emoji_font_path", lambda: None)
    src, _, _ = faces.load_pixels(noise_image(tmp_path / "a.jpg"))
    out = faces.mask_pixels(src, [[240.0, 120.0, 60.0, 60.0, 1.0]], "\U0001F642")
    assert len({out.getpixel((x, 150)) for x in range(255, 285)}) == 1


def test_mask_cli_writes_a_new_file_and_never_touches_the_source(world, tmp_path, capsys):
    img = noise_image(tmp_path / "shot.jpg", exif=True)
    before = _sha(img)
    assert cli(world, "mask", str(img), "--auto", "--box", "10,10,30,30") == 0       # dry run
    assert not (tmp_path / "shot_masked.jpg").exists() and rows(world) == []
    assert cli(world, "mask", str(img), "--auto", "--box", "10,10,30,30", "--apply") == 0
    out = tmp_path / "shot_masked.jpg"
    assert out.is_file() and _sha(img) == before
    m = Image.open(out)
    assert m.size == (600, 400) and m.format == "JPEG"
    a, b = Image.open(img).crop((240, 120, 300, 180)), m.crop((240, 120, 300, 180))
    diff = sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3
    assert diff > 40
    assert not list(tmp_path.glob("*.part"))


def test_mask_output_carries_only_share_metadata(world, tmp_path):
    img = noise_image(tmp_path / "shot.jpg", exif=True)
    out = tmp_path / "m.jpg"
    assert cli(world, "mask", str(img), "--box", "240,120,60,60", "--out", str(out), "--apply") == 0
    raw = out.read_bytes()
    assert SECRET.encode() not in raw and b"Editing Suite" not in raw and b"2025:08:01" not in raw
    ex = Image.open(out).getexif()
    assert set(ex.keys()) == {0x010F, 0x0110}                       # camera, as share_metadata allows
    assert not ex.get_ifd(0x8825) and not ex.get_ifd(0x8769)


def test_mask_with_no_share_metadata_has_no_exif_at_all(world, tmp_path):
    world["settings_file"].write_text(json.dumps({"share_metadata": [], "photos_root": str(world["root"])}), encoding="utf-8")
    img = noise_image(tmp_path / "shot.jpg", exif=True)
    out = tmp_path / "m.jpg"
    assert cli(world, "mask", str(img), "--box", "240,120,60,60", "--out", str(out), "--apply") == 0
    assert dict(Image.open(out).getexif()) == {} and b"Exif" not in out.read_bytes()[:200]


def test_mask_refuses_to_overwrite_its_source_or_to_mask_nothing(world, tmp_path):
    img = noise_image(tmp_path / "shot.jpg")
    assert cli(world, "mask", str(img), "--box", "1,1,5,5", "--out", str(img), "--apply") == 2
    assert cli(world, "mask", str(img), "--apply") == 2


def test_mask_records_a_faces_event_for_the_frame(world, tmp_path):
    img = noise_image(world["root"] / "Out A" / "DSC_0001.jpg")
    assert cli(world, "mask", str(img), "--auto", "--frame", "Out A/DSC_0001.jpg", "--outing", "out-1",
               "--by", "claude", "--via", "test", "--checked-all-faces", "--apply") == 0
    (r,) = rows(world)
    out = world["root"] / "Out A" / "DSC_0001_masked.jpg"
    assert r["kind"] == "faces" and r["state"] == "masked" and r["frame"] == "Out A/DSC_0001.jpg"
    assert r["outing"] == "out-1" and r["sha1"] == _sha(out)
    assert r["boxes"] == 1 and r["detected"] == 1 and r["auto"] is True and r["checked"] is True
    assert r["image"] == "Out A/DSC_0001_masked.jpg"


def test_a_box_that_misses_the_face_cannot_be_recorded_as_masked(world, tmp_path, capsys):
    """The output is re-checked: the face is still there, so nothing is kept or recorded."""
    img = noise_image(world["root"] / "Out A" / "DSC_0001.jpg")
    assert cli(world, "mask", str(img), "--box", "10,10,30,30", "--frame", "Out A/DSC_0001.jpg",
               "--outing", "out-1", "--apply") == 3
    assert "still detectable" in capsys.readouterr().out
    assert not (img.parent / "DSC_0001_masked.jpg").exists() and not list(img.parent.glob("*.part"))
    assert rows(world) == []


def test_fewer_boxes_than_detected_faces_is_refused(world, monkeypatch, tmp_path, capsys):
    two = [FACE, (0.10, 0.60, 0.10, 0.15)]
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [NoiseDetector(two)])
    img = noise_image(tmp_path / "shot.jpg")
    assert cli(world, "mask", str(img), "--box", "240,120,60,60", "--apply") == 3
    assert "2 face(s) detected but only 1 box" in capsys.readouterr().out
    assert not (tmp_path / "shot_masked.jpg").exists()
    assert cli(world, "mask", str(img), "--auto", "--apply") == 0                   # both boxes
    assert not NoiseDetector(two).detect(Image.open(tmp_path / "shot_masked.jpg").convert("RGB"))


def test_mask_without_a_detector_refuses_rather_than_trusting_the_boxes(world, monkeypatch, tmp_path):
    def none(s=None):
        raise faces.FaceError("no face detector is available")
    monkeypatch.setattr(faces, "default_detectors", none)
    img = noise_image(tmp_path / "shot.jpg")
    assert cli(world, "mask", str(img), "--box", "240,120,60,60", "--apply") == 3
    assert not (tmp_path / "shot_masked.jpg").exists()


# ---------------------------------------------------------------- clear and status

def test_clear_is_refused_for_anyone_not_listed(world, desk):
    assert cli(world, "clear", "Out A/f.jpg", "--outing", "out-1", "--by", "stranger", "--apply") == 3
    assert cli(world, "clear", "Out A/f.jpg", "--outing", "out-1", "--apply") == 3          # no --by
    assert rows(world) == []
    world["settings_file"].write_text(json.dumps({}), encoding="utf-8")                     # nobody listed
    assert cli(world, "clear", "Out A/f.jpg", "--outing", "out-1", "--by", "Evan", "--apply") == 3
    assert rows(world) == []


def test_clear_needs_a_person_at_the_terminal_or_the_desk_flag(world, capsys):
    args = ("clear", "Out A/f.jpg", "--outing", "out-1", "--by", "evan", "--apply")
    assert cli(world, *args) == 3                                  # a script: no tty, no flag
    assert "interactive terminal" in capsys.readouterr().out and rows(world) == []
    assert cli(world, *args, "--i-am-at-the-desk") == 0            # the Horizons desktop door
    assert [r["state"] for r in rows(world)] == ["cleared"]


def test_clear_at_an_interactive_terminal_records(world, desk):
    assert cli(world, "clear", "Out A/f.jpg", "--outing", "out-1", "--by", "evan", "--apply") == 0
    assert [r["state"] for r in rows(world)] == ["cleared"]


def test_clear_and_status_follow_the_last_event(world, capsys, desk):
    fid = "Out A/f.jpg"
    assert cli(world, "status", fid, "--outing", "out-1", "--json") == 0
    assert json.loads(capsys.readouterr().out)["state"] == "none"
    assert cli(world, "clear", fid, "--outing", "out-1", "--by", "evan") == 0             # dry run
    assert rows(world) == []
    assert cli(world, "clear", fid, "--outing", "out-1", "--by", "evan", "--apply") == 0
    capsys.readouterr()
    cli(world, "status", fid, "--outing", "out-1", "--json")
    st = json.loads(capsys.readouterr().out)
    assert st["state"] == "cleared" and st["by"] == "evan"
    faces.record(world["settings"], "frame", fid, "masked", outing="out-1", events_dir=world["events"],
                 image="x_masked.jpg", sha1="abc")
    cli(world, "status", fid, "--outing", "out-1", "--json")
    st = json.loads(capsys.readouterr().out)
    assert (st["state"], st["image"], st["sha1"]) == ("masked", "x_masked.jpg", "abc")
    assert faces.status_of(world["settings"], fid, "out-other", world["events"])["state"] == "none"


def test_a_cleared_row_counts_only_while_its_author_is_a_clearer(world, tmp_path):
    img = noise_image(tmp_path / "a.jpg")
    clear_row(world, "f1", who="stranger")
    assert faces.status_of(world["settings"], "f1", "o", world["events"])["state"] == "none"
    assert not faces.gate(img, frame="f1", outing="out-adults", settings=world["settings"],
                          events_dir=world["events"])["allowed"]
    clear_row(world, "f2", who="evan")
    assert faces.status_of(world["settings"], "f2", "o", world["events"])["state"] == "cleared"
    # taken off the list later, the clear stops counting
    s = dict(world["settings"], face_clearers=[])
    assert faces.status_of(s, "f2", "o", world["events"])["state"] == "none"
    # a stranger's clear does not undo a mask, it is ignored
    m = make_masked(img)
    record_masked(world, "f3", m)
    clear_row(world, "f3", who="stranger")
    assert faces.status_of(world["settings"], "f3", "o", world["events"])["state"] == "masked"


def test_a_copied_key_row_is_honoured_by_cleared_by_not_by_the_publisher(world):
    s, ev = world["settings"], world["events"]
    faces.record(s, "key", "k1", "cleared", by="publisher", cleared_by="evan", events_dir=ev)
    faces.record(s, "key", "k2", "cleared", by="evan", cleared_by="stranger", events_dir=ev)
    faces.record(s, "key", "k3", "cleared", by="evan", events_dir=ev)
    assert [faces.status_of(s, k, None, ev)["state"] for k in ("k1", "k2", "k3")] == ["cleared", "none", "cleared"]


# ---------------------------------------------------------------- the gate

def _gate(w, img, **kw):
    kw.setdefault("events_dir", w["events"])
    kw.setdefault("settings", w["settings"])
    return faces.gate(img, **kw)


def test_gate_refuses_a_detected_face_and_allows_a_clean_frame(world, tmp_path, monkeypatch):
    img = noise_image(tmp_path / "a.jpg")
    res = _gate(world, img, frame="f1", outing="out-adults")
    assert not res["allowed"] and res["faces"] == 1 and "face" in res["reason"]
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [NoFace()])
    assert _gate(world, img, frame="f1", outing="out-adults")["allowed"]


def test_gate_masked_requires_the_masked_file_itself(world, tmp_path):
    img = noise_image(tmp_path / "a.jpg")
    masked = make_masked(img)
    record_masked(world, "f1", masked, auto=True, checked=True)
    assert _gate(world, masked, frame="f1", outing="o")["allowed"]            # unknown companions: needs the acknowledged --auto mask
    res = _gate(world, img, frame="f1", outing="o")
    assert not res["allowed"] and "publish the masked file" in res["reason"]
    faces.record(world["settings"], "frame", "f2", "masked", outing="o", events_dir=world["events"])
    assert not _gate(world, masked, frame="f2", outing="o")["allowed"]        # no recorded sha


def test_gate_re_checks_the_masked_file_and_refuses_if_a_face_is_still_there(world, tmp_path):
    img = noise_image(tmp_path / "a.jpg")
    bad = make_masked(img, "bad_masked.jpg", boxes=[[10.0, 10.0, 30.0, 30.0, 1.0]])      # misses the face
    record_masked(world, "f1", bad, outing="out-adults")
    res = _gate(world, bad, frame="f1", outing="out-adults")
    assert not res["allowed"] and "still detectable" in res["reason"]
    short = make_masked(img, "short_masked.jpg")
    record_masked(world, "f2", short, outing="out-adults", boxes=1, detected=2)                                  # fewer boxes than faces
    assert not _gate(world, short, frame="f2", outing="out-adults")["allowed"]


def test_gate_masked_with_children_aboard_needs_auto_and_the_acknowledgement(world, tmp_path):
    img = noise_image(tmp_path / "a.jpg")
    m = make_masked(img)
    record_masked(world, "f1", m, outing="out-kids", auto=True, checked=False)
    res = _gate(world, m, frame="f1", outing="out-kids")
    assert not res["allowed"] and "--checked-all-faces" in res["reason"]
    record_masked(world, "f1", m, outing="out-kids", auto=False, checked=True)              # manual boxes only
    assert not _gate(world, m, frame="f1", outing="out-kids")["allowed"]
    record_masked(world, "f1", m, outing="out-kids", auto=True, checked=True)
    assert _gate(world, m, frame="f1", outing="out-kids")["allowed"]
    # unidentifiable companions need the same
    record_masked(world, "f2", m, outing="out-stranger", auto=True, checked=False)
    assert not _gate(world, m, frame="f2", outing="out-stranger")["allowed"]
    record_masked(world, "f3", m, outing="out-adults", auto=False, checked=False)           # adults only: plain mask is fine
    assert _gate(world, m, frame="f3", outing="out-adults")["allowed"]


def test_masked_on_an_outing_with_no_hike_log_needs_the_acknowledgement(world, tmp_path, capsys):
    """R1: unknown companions are as bad as children for a masked frame. `mask --auto` alone is
    refused by the gate; with --checked-all-faces it is allowed."""
    src = noise_image(world["root"] / "Client" / "DSC_0001.jpg")
    masked = src.with_name("DSC_0001_masked.jpg")
    args = ("mask", str(src), "--auto", "--frame", "Client/DSC_0001.jpg", "--outing", "out-nolog", "--apply")
    assert cli(world, *args) == 0
    res = _gate(world, masked, frame="Client/DSC_0001.jpg", outing="out-nolog")
    assert not res["allowed"] and "--checked-all-faces" in res["reason"] and res["companions"]["known"] is False
    assert cli(world, *args, "--checked-all-faces") == 0
    assert _gate(world, masked, frame="Client/DSC_0001.jpg", outing="out-nolog")["allowed"]
    # a hike log of adults needs no acknowledgement
    assert cli(world, "mask", str(src), "--auto", "--frame", "Client/DSC_0001.jpg", "--outing", "out-adults", "--apply") == 0
    assert _gate(world, masked, frame="Client/DSC_0001.jpg", outing="out-adults")["allowed"]


def test_gate_cleared_is_allowed_even_with_a_face_and_a_child_aboard(world, tmp_path):
    img = noise_image(tmp_path / "a.jpg")
    clear_row(world, "f1", outing="out-kids")
    assert _gate(world, img, frame="f1", outing="out-kids")["allowed"]


def test_gate_later_event_wins_over_clear(world, tmp_path):
    img = noise_image(tmp_path / "a.jpg")
    clear_row(world, "f1")
    faces.record(world["settings"], "frame", "f1", "masked", outing="o", events_dir=world["events"], sha1="zz")
    assert not _gate(world, img, frame="f1", outing="o")["allowed"]


def test_gate_refuses_a_child_companion_even_with_no_face_found(world, tmp_path, no_face):
    img = noise_image(tmp_path / "a.jpg")
    res = _gate(world, img, frame="f1", outing="out-kids")
    assert not res["allowed"] and res["companions"]["children"] == 1 and res["faces"] == 0
    assert "Kid" not in json.dumps(res) and "Pal" not in json.dumps(res)         # never a name
    assert _gate(world, img, frame="f1", outing="out-adults")["allowed"]
    assert _gate(world, img, frame="f1", outing="out-solo")["allowed"]             # an explicit "solo"


def test_unknown_companions_are_refused(world, tmp_path, no_face):
    img = noise_image(tmp_path / "a.jpg")
    for oid in ("out-unknown", "out-nolog", "out-blank", None):
        res = _gate(world, img, frame="f1", outing=oid)
        assert not res["allowed"] and "companions unknown" in res["reason"], oid
    # ... unless the frame is cleared or its mask is verified
    clear_row(world, "f1", outing="out-nolog")
    assert _gate(world, img, frame="f1", outing="out-nolog")["allowed"]
    m = make_masked(img)
    record_masked(world, "f2", m, outing="out-nolog", auto=True, checked=True)
    assert _gate(world, m, frame="f2", outing="out-nolog")["allowed"]


def test_an_unresolved_companion_is_refused(world, tmp_path, no_face):
    img = noise_image(tmp_path / "a.jpg")
    res = _gate(world, img, frame="f1", outing="out-stranger")
    assert not res["allowed"] and res["companions"]["unresolved"] == 1 and "Somebody" not in json.dumps(res)


def test_the_unconfigured_default_refuses_any_companion(world, tmp_path, no_face):
    """child_relationships defaults to []: nobody can be ruled out, so a frame with companions is refused."""
    img = noise_image(tmp_path / "a.jpg")
    s = faces.link.load_settings(str(world["settings_file"]))
    s["child_relationships"] = []
    res = faces.gate(img, frame="f1", outing="out-adults", settings=s, events_dir=world["events"])
    assert not res["allowed"] and "not configured" in res["reason"]
    assert faces.gate(img, frame="f1", outing="out-solo", settings=s, events_dir=world["events"])["allowed"]
    # the age rule still works without it
    assert faces.gate(img, frame="f1", outing="out-minor", settings=s, events_dir=world["events"])["companions"]["children"] == 1
    d = faces.link.load_settings(str(world["tmp"] / "none.json"))
    assert d["child_relationships"] == [] and d["face_clearers"] == []


def test_gate_companions_file_and_the_age_rule(world, tmp_path, no_face):
    img = noise_image(tmp_path / "a.jpg")
    kid_log = note(tmp_path / "log1.md", type="hike-log", companions=["Adult Two", "[[Kid One]]"])
    adult_log = note(tmp_path / "log2.md", type="hike-log", companions="Adult Two and Pal Three")
    owner_log = note(tmp_path / "log3.md", type="hike-log", companions=["Owner Zero"])
    blank_log = note(tmp_path / "log4.md", type="hike-log", companions=[])
    assert not _gate(world, img, frame="f1", outing="o", companions_file=str(kid_log))["allowed"]
    assert _gate(world, img, frame="f1", outing="o", companions_file=str(adult_log))["allowed"]
    assert _gate(world, img, frame="f1", outing="o", companions_file=str(owner_log))["allowed"]
    assert not _gate(world, img, frame="f1", outing="o", companions_file=str(blank_log))["allowed"]
    assert not _gate(world, img, frame="f1", outing="out-minor")["allowed"]


def test_gate_child_words_may_name_a_relationship_value(world, tmp_path, no_face):
    img = noise_image(tmp_path / "a.jpg")
    s = dict(world["settings"], child_relationships=["acquaintance-kid"])
    note(world["vault"] / "People" / "Kid Five.md", type="person", relationship="acquaintance-kid", tags=["person"])
    log = note(tmp_path / "log.md", type="hike-log", companions=["Kid Five"])
    assert not faces.gate(img, frame="f", outing="o", settings=s, events_dir=world["events"], companions_file=str(log))["allowed"]


@pytest.mark.parametrize("yaml_present", [True, False])
@pytest.mark.parametrize("front", [
    "type: person\nrelationship: family\ntags:\n- family\n- child",              # Obsidian block list, unindented
    "type: person\nrelationship: family\ntags:\n  - family\n  - child",
    "type: person\nrelationship: family\ntags:\n  - family\n  - #child",         # YAML calls this a comment
    "type: person\nrelationship: family\ntags:\n  - '#child'",
    "type: person\nrelationship: family\ntags: [family, '#child']",
    "type: person\nrelationship: family\ntags: #child",
    "type: person\nrelationship: family\ntags: [family, child]",
])
def test_child_tags_in_every_obsidian_spelling(world, front, yaml_present, monkeypatch):
    if not yaml_present:
        monkeypatch.setitem(sys.modules, "yaml", None)                           # `import yaml` now fails
    raw_note(world["vault"] / "People" / "Kid Six.md", front)
    people = faces.People(world["vault"] / "People")
    assert faces.is_child(people.find("Kid Six"), ["child"])
    assert faces.is_child(people.find("Kid Six"), ["#child"])                    # a # in the setting is ignored too
    assert not faces.is_child(people.find("Adult Two"), ["child"])


def test_gate_no_frame_mode(world, tmp_path, monkeypatch):
    img = noise_image(tmp_path / "art.jpg")
    res = _gate(world, img, key="art-001")
    assert not res["allowed"] and res["faces"] == 1
    clear_row(world, "art-001", kind="key", outing=None)
    assert _gate(world, img, key="art-001")["allowed"]
    assert not _gate(world, img, key="art-002")["allowed"]                 # another key is not cleared
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [NoFace()])
    assert _gate(world, img, key="art-002")["allowed"]


def test_gate_no_frame_masked_row_allows_only_the_matching_clean_file(world, tmp_path):
    img = noise_image(tmp_path / "art.jpg")
    m = make_masked(img)
    faces.record(world["settings"], "key", "art-1", "masked", events_dir=world["events"], sha1=_sha(m))
    assert _gate(world, m, key="art-1")["allowed"]
    assert not _gate(world, img, key="art-1")["allowed"]                      # the unmasked original
    other = make_masked(img, "other_masked.jpg", boxes=[[10.0, 10.0, 30.0, 30.0, 1.0]])
    assert not _gate(world, other, key="art-1")["allowed"]


def test_has_clearance_masked_only_for_the_recorded_file(world, tmp_path):
    img = noise_image(tmp_path / "a.jpg")
    m = make_masked(img)
    s, ev = world["settings"], world["events"]
    faces.record(s, "key", "k1", "masked", events_dir=ev, sha1=_sha(m))
    assert faces.has_clearance("k1", s, ev, image=m)
    assert not faces.has_clearance("k1", s, ev, image=img)
    assert not faces.has_clearance("k1", s, ev)                                  # no image to compare
    faces.record(s, "key", "k2", "cleared", by="evan", events_dir=ev)
    faces.record(s, "key", "k3", "cleared", by="stranger", events_dir=ev)
    assert faces.has_clearance("k2", s, ev) and not faces.has_clearance("k3", s, ev)
    assert not faces.has_clearance("never-seen", s, ev)


def test_gate_fails_closed(world, tmp_path, monkeypatch, no_face):
    img = noise_image(tmp_path / "a.jpg")
    assert _gate(world, img, frame="f1", outing="out-adults")["allowed"]            # baseline
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [Boom()])
    res = _gate(world, img, frame="f1", outing="out-adults")
    assert not res["allowed"] and "refused" in res["reason"]

    def none(s=None):
        raise faces.FaceError("no face detector is available")
    monkeypatch.setattr(faces, "default_detectors", none)
    assert not _gate(world, img, frame="f1", outing="out-adults")["allowed"]
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [NoFace()])
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert not _gate(world, bad, frame="f1", outing="out-adults")["allowed"]
    assert not _gate(world, tmp_path / "missing.jpg", frame="f1", outing="out-adults")["allowed"]
    gone = faces.link.load_settings(str(tmp_path / "nope.json"))
    assert not faces.gate(img, frame="f1", outing="o", settings=gone, events_dir=world["events"])["allowed"]
    assert not faces.gate(img, frame="f1", outing="o", settings=None)["allowed"]
    assert not _gate(world, img, frame="f1", outing="o", companions_file=str(tmp_path / "nolog.md"))["allowed"]
    (world["vault"] / "_vault_paths.json").write_text(json.dumps({"paths": {}}), encoding="utf-8")
    assert not _gate(world, img, frame="f1", outing="out-adults")["allowed"]


def test_gate_cli_exit_codes(world, tmp_path, capsys, no_face):
    img = noise_image(tmp_path / "a.jpg")
    assert cli(world, "gate", str(img), "--frame", "f1", "--outing", "out-adults", "--json") == 0
    assert json.loads(capsys.readouterr().out)["allowed"] is True
    assert cli(world, "gate", str(img), "--frame", "f1", "--outing", "out-kids", "--json") == 3
    assert json.loads(capsys.readouterr().out)["allowed"] is False
    assert cli(world, "gate", str(img), "--no-frame", "--key", "k") == 0
    assert cli(world, "gate", str(img)) == 3                                  # neither mode: refused
    assert cli(world, "gate", str(img), "--frame", "f1") == 2                 # a frame needs its outing
    assert faces.main(["gate", str(img), "--frame", "f", "--outing", "o", "--settings", str(tmp_path / "nope.json"),
                       "--events-dir", str(world["events"])]) == 3


# ---------------------------------------------------------------- publish_external

def _load_pe():
    spec = importlib.util.spec_from_file_location("publish_external_t", REPO / "tools" / "publish_external.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def pub(world, tmp_path, monkeypatch):
    """Two starred frames in one outing (with a hike log of adults); frame 1's export has a face
    (noise), frame 2's is smooth."""
    root = world["root"]
    outing = root / "Out A"
    ids, exports = [], tmp_path / "export"
    exports.mkdir()
    for n in (1, 2):
        f = outing / ("DSC_%04d.jpg" % n)
        noise_image(f, size=(300, 200), seed=n)
        ids.append("Out A/" + f.name)
    noise_image(exports / "DSC_0001.jpg", size=(300, 200), seed=1)
    smooth_image(exports / "DSC_0002.jpg")
    proj = json.loads((world["vault"] / "_studio_outings.json").read_text(encoding="utf-8"))
    proj["outings"].append({"id": "out-pub", "folder": "Out A", "hike_logs": [{"companions": ["Adult Two"]}]})
    (world["vault"] / "_studio_outings.json").write_text(json.dumps(proj), encoding="utf-8")
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [NoiseDetector([(100 / 300, 60 / 200, 40 / 300, 40 / 200)])])
    sel = {i: {"category": "landscape", "alt": "a view"} for i in ids}
    (tmp_path / "sel.json").write_text(json.dumps(sel), encoding="utf-8")
    (tmp_path / "idx.json").write_text(json.dumps({"root": str(root), "photos": [
        {"id": i, "date": "2025-08-01", "outing": "Out A"} for i in ids]}), encoding="utf-8")
    cfg = json.loads(world["settings_file"].read_text(encoding="utf-8"))
    cfg["picker_tools"] = {"selections": str(tmp_path / "sel.json"), "index_file": str(tmp_path / "idx.json")}
    world["settings_file"].write_text(json.dumps(cfg), encoding="utf-8")
    world["settings"] = faces.link.load_settings(str(world["settings_file"]))
    gallery = tmp_path / "repo" / "data" / "gallery.json"
    gallery.parent.mkdir(parents=True)
    gallery.write_text(json.dumps({"photos": []}), encoding="utf-8")
    pe = _load_pe()
    monkeypatch.setattr(pe, "GALLERY", gallery)
    monkeypatch.setattr(pe, "CORRECTED", tmp_path / "corrected")
    monkeypatch.setattr(pe, "REPO", tmp_path / "repo")
    calls = []
    monkeypatch.setattr(pe.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    return {"pe": pe, "ids": ids, "exports": exports, "gallery": gallery, "calls": calls, "world": world}


def _pe(p, *extra):
    w = p["world"]
    return p["pe"].main([str(p["exports"]), "--settings", str(w["settings_file"]),
                         "--events-dir", str(w["events"]), "--by", "test", *extra])


def test_publish_external_skips_a_refused_frame_and_exits_nonzero(pub, capsys):
    w = pub["world"]
    assert _pe(pub) == 3                                                    # dry run: still refused
    assert "REFUSED by the face guard: DSC_0001.jpg" in capsys.readouterr().out
    assert _pe(pub, "--apply") == 3
    gal = json.loads(pub["gallery"].read_text(encoding="utf-8"))["photos"]
    assert [p["id"] for p in gal] == ["20250801-DSC_0002"]
    assert sorted(x.name for x in (w["tmp"] / "corrected").iterdir()) == ["20250801-DSC_0002.jpg"]
    posts = [r for r in rows(w) if r["kind"] == "post"]
    assert [r["slug"] for r in posts] == ["20250801-DSC_0002"]
    assert pub["calls"][0][-1] == "20250801-DSC_0002" and "--only" in pub["calls"][0]


def test_publish_external_with_every_frame_refused_runs_nothing(pub, monkeypatch, capsys):
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [Boom()])
    assert _pe(pub, "--apply") == 3
    assert pub["calls"] == [] and not (pub["world"]["tmp"] / "corrected").exists()
    assert "nothing left to publish" in capsys.readouterr().out
    assert [r for r in rows(pub["world"]) if r["kind"] == "post"] == []


def test_publish_external_with_nothing_matched_skips_the_subprocesses(pub, capsys):
    for f in pub["exports"].iterdir():
        f.rename(f.with_name("OTHER_" + f.name))
    assert _pe(pub, "--apply") == 0
    assert pub["calls"] == [] and "nothing" in capsys.readouterr().out


def test_publish_external_publishes_a_cleared_frame_and_copies_who_cleared_it(pub):
    w = pub["world"]
    clear_row(w, "Out A/DSC_0001.jpg", outing="out-pub")
    assert _pe(pub, "--apply") == 0
    gal = json.loads(pub["gallery"].read_text(encoding="utf-8"))["photos"]
    assert sorted(p["id"] for p in gal) == ["20250801-DSC_0001", "20250801-DSC_0002"]
    (k,) = [r for r in rows(w) if r["kind"] == "faces" and r.get("key") == "20250801-DSC_0001"]
    assert k["state"] == "cleared" and k["cleared_by"] == "evan"
    assert faces.has_clearance("20250801-DSC_0001", w["settings"], w["events"])


def test_publish_external_masked_flow_end_to_end(pub):
    """Mask through the CLI exactly as Evan would, with the real outing id, then publish."""
    w = pub["world"]
    settings, root = w["settings"], w["root"]
    oid = faces.link.outing_id_for(settings, root, root / "Out A", w["events"])
    assert oid == "out-pub"
    export1 = pub["exports"] / "DSC_0001.jpg"
    assert faces.main(["mask", str(export1), "--auto", "--frame", "Out A/DSC_0001.jpg", "--outing", oid,
                       "--settings", str(w["settings_file"]), "--events-dir", str(w["events"]), "--apply"]) == 0
    masked = pub["exports"] / "DSC_0001_masked.jpg"
    assert masked.is_file()
    assert _pe(pub, "--apply") == 0
    pub_dir = w["tmp"] / "corrected"
    assert _sha(pub_dir / "20250801-DSC_0001.jpg") == _sha(masked)
    keyed = [r for r in rows(w) if r["kind"] == "faces" and r.get("key") == "20250801-DSC_0001"]
    assert keyed and keyed[-1]["state"] == "masked"
    assert faces.has_clearance("20250801-DSC_0001", settings, w["events"], image=pub_dir / "20250801-DSC_0001.jpg")
    assert not faces.has_clearance("20250801-DSC_0001", settings, w["events"], image=export1)
    assert not faces.has_clearance("20250801-DSC_0002", settings, w["events"])


# ---------------------------------------------------------------- optimize_images

def _load_oi(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("optimize_images_faceguard", REPO / "optimize_images.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "INPUT_DIR", tmp_path / "masters")
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir()
    return mod


def _env(world, monkeypatch):
    monkeypatch.setenv("MAYNIAC_STUDIO_SETTINGS", str(world["settings_file"]))
    monkeypatch.setenv("MAYNIAC_STUDIO_EVENTS", str(world["events"]))


def test_optimize_images_refuses_a_master_with_a_face(world, tmp_path, monkeypatch, capsys):
    oi = _load_oi(tmp_path, monkeypatch)
    master = noise_image(tmp_path / "masters" / "20250801-DSC_0001.jpg", size=(1000, 700))
    _env(world, monkeypatch)
    assert oi.optimize_image(master, allowed=set()) is False
    assert list((tmp_path / "out").iterdir()) == []
    assert "REFUSED (face guard)" in capsys.readouterr().out
    assert oi.main(["--only", "20250801-DSC_0001"]) == 3
    assert list((tmp_path / "out").iterdir()) == []


def test_optimize_images_proceeds_with_a_clearance_or_no_face(world, tmp_path, monkeypatch):
    oi = _load_oi(tmp_path, monkeypatch)
    master = noise_image(tmp_path / "masters" / "20250801-DSC_0001.jpg", size=(1000, 700))
    _env(world, monkeypatch)
    clear_row(world, "20250801-DSC_0001", kind="key", outing=None)
    assert oi.optimize_image(master, allowed=set()) is True
    assert len(list((tmp_path / "out").glob("*.jpg"))) >= 2
    other = noise_image(tmp_path / "masters" / "20250801-DSC_0002.jpg", size=(1000, 700))
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [NoFace()])
    assert oi.optimize_image(other, allowed=set()) is True


def test_optimize_images_a_masked_row_allows_only_the_masked_file(world, tmp_path, monkeypatch):
    oi = _load_oi(tmp_path, monkeypatch)
    original = noise_image(tmp_path / "src.jpg", size=(1000, 700))
    masked = make_masked(original, "m.jpg")
    _env(world, monkeypatch)
    faces.record(world["settings"], "key", "20250801-DSC_0001", "masked", events_dir=world["events"], sha1=_sha(masked))
    (tmp_path / "masters").mkdir()
    good = tmp_path / "masters" / "20250801-DSC_0001.jpg"
    good.write_bytes(masked.read_bytes())
    assert oi.optimize_image(good, allowed=set()) is True                            # the same bytes
    shutil_copy = tmp_path / "masters" / "20250801-DSC_0001.jpg"
    shutil_copy.write_bytes(original.read_bytes())                                   # the unmasked original under that id
    assert oi.optimize_image(shutil_copy, allowed=set()) is False


def test_optimize_images_ignores_a_clear_by_a_non_clearer(world, tmp_path, monkeypatch):
    oi = _load_oi(tmp_path, monkeypatch)
    master = noise_image(tmp_path / "masters" / "20250801-DSC_0001.jpg", size=(1000, 700))
    _env(world, monkeypatch)
    clear_row(world, "20250801-DSC_0001", who="stranger", kind="key", outing=None)
    assert oi.optimize_image(master, allowed=set()) is False


def test_optimize_images_fails_closed_when_the_detector_cannot_run(world, tmp_path, monkeypatch):
    oi = _load_oi(tmp_path, monkeypatch)
    master = noise_image(tmp_path / "masters" / "20250801-DSC_0001.jpg", size=(1000, 700))
    _env(world, monkeypatch)
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [Boom()])
    assert oi.optimize_image(master, allowed=set()) is False
    assert list((tmp_path / "out").iterdir()) == []
