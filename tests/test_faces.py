"""The face guard (tools/picker/faces.py): Evan's rule, 2026-10-17: our kids' faces are never
published. Everything runs on tmp fixtures with an injectable detector; one smoke test uses the
real detector and skips when this machine has none. No person in a fixture has a real name.
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


# ---------------------------------------------------------------- fakes and fixtures

class FakeDetector:
    """Finds the given boxes, given as fractions of the image so every scale agrees."""
    name = "fake"

    def __init__(self, rel_boxes=()):
        self.rel = list(rel_boxes)

    def detect(self, image):
        W, H = image.size
        return [[x * W, y * H, w * W, h * H, 0.9] for x, y, w, h in self.rel]


class Boom:
    name = "boom"

    def detect(self, image):
        raise RuntimeError("detector exploded")


FACE = (0.40, 0.30, 0.10, 0.15)


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


def note(path, **fm):
    lines = ["---"]
    for k, v in fm.items():
        lines.append("%s: %s" % (k, json.dumps(v)))
    lines += ["---", "", "body"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
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
    young = date.today().replace(year=date.today().year - 9).isoformat()
    note(people / "Minor Four.md", type="person", relationship="friend", tags=["friend"], date_of_birth=young)
    (vault / "_studio_outings.json").write_text(json.dumps({"outings": [
        {"id": "out-kids", "folder": "Hiking/kids", "hike_logs": [{"companions": ["[[People/Kid One|Kid One]]", "Pal"]}]},
        {"id": "out-adults", "folder": "Hiking/adults", "hike_logs": [{"companions": "Adult Two, Pal Three"}]},
        {"id": "out-minor", "folder": "Hiking/minor", "hike_logs": [{"companions": ["Minor Four"]}]},
    ]}), encoding="utf-8")
    settings_file = vault / "_studio_settings.json"
    settings_file.write_text(json.dumps({
        "face_clearers": ["Evan"], "child_relationships": ["child"], "face_emoji": "\U0001F642",
        "share_metadata": ["camera"], "photos_root": str(tmp_path / "photos"),
    }), encoding="utf-8")
    events = tmp_path / "_events"
    monkeypatch.setattr(faces, "default_detectors", lambda settings=None: [FakeDetector([FACE])])
    return {"tmp": tmp_path, "vault": vault, "settings_file": settings_file, "events": events,
            "settings": faces.link.load_settings(str(settings_file)), "root": tmp_path / "photos"}


def rows(w):
    f = w["events"] / "studio-picker.jsonl"
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []


def cli(w, *args):
    return faces.main([*args, "--settings", str(w["settings_file"]), "--events-dir", str(w["events"])])


# ---------------------------------------------------------------- geometry

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


# ---------------------------------------------------------------- detection wiring

def test_check_reports_faces_in_the_documented_shape(world, tmp_path):
    img = noise_image(tmp_path / "a.jpg")
    res = faces.check_image(img, world["settings"])
    assert set(res) == {"faces", "detector", "version"} and res["version"] == 1
    assert res["detector"] == "fake"
    assert len(res["faces"]) == 1                      # the two scales agree and merge into one
    f = res["faces"][0]
    assert set(f) == {"x", "y", "w", "h", "score"}
    assert abs(f["x"] - 240) <= 2 and abs(f["w"] - 60) <= 2


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
    # the tight face box is fully covered, padded corners are what is left of the box
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
    # the masked region is not the source's noise
    a, b = Image.open(img).crop((240, 120, 300, 180)), m.crop((240, 120, 300, 180))
    diff = sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3
    assert diff > 40


def test_mask_output_carries_only_share_metadata(world, tmp_path):
    img = noise_image(tmp_path / "shot.jpg", exif=True)
    out = tmp_path / "m.jpg"
    assert cli(world, "mask", str(img), "--box", "100,100,50,50", "--out", str(out), "--apply") == 0
    raw = out.read_bytes()
    assert SECRET.encode() not in raw and b"Editing Suite" not in raw and b"2025:08:01" not in raw
    ex = Image.open(out).getexif()
    assert set(ex.keys()) == {0x010F, 0x0110}                       # camera, as share_metadata allows
    assert not ex.get_ifd(0x8825) and not ex.get_ifd(0x8769)


def test_mask_with_no_share_metadata_has_no_exif_at_all(world, tmp_path):
    world["settings_file"].write_text(json.dumps({"share_metadata": [], "photos_root": str(world["root"])}), encoding="utf-8")
    img = noise_image(tmp_path / "shot.jpg", exif=True)
    out = tmp_path / "m.jpg"
    assert cli(world, "mask", str(img), "--box", "100,100,50,50", "--out", str(out), "--apply") == 0
    assert dict(Image.open(out).getexif()) == {} and b"Exif" not in out.read_bytes()[:200]


def test_mask_refuses_to_overwrite_its_source_or_to_mask_nothing(world, tmp_path):
    img = noise_image(tmp_path / "shot.jpg")
    assert cli(world, "mask", str(img), "--box", "1,1,5,5", "--out", str(img), "--apply") == 2
    assert cli(world, "mask", str(img), "--apply") == 2


def test_mask_records_a_faces_event_for_the_frame(world, tmp_path):
    img = noise_image(world["root"] / "Out A" / "DSC_0001.jpg")
    assert cli(world, "mask", str(img), "--auto", "--frame", "Out A/DSC_0001.jpg", "--outing", "out-1",
               "--by", "claude", "--via", "test", "--apply") == 0
    (r,) = rows(world)
    out = world["root"] / "Out A" / "DSC_0001_masked.jpg"
    assert r["kind"] == "faces" and r["state"] == "masked" and r["frame"] == "Out A/DSC_0001.jpg"
    assert r["outing"] == "out-1" and r["sha1"] == _sha(out) and r["boxes"] == 1
    assert r["image"] == "Out A/DSC_0001_masked.jpg"


# ---------------------------------------------------------------- clear and status

def test_clear_is_refused_for_anyone_not_listed(world, capsys):
    assert cli(world, "clear", "Out A/f.jpg", "--outing", "out-1", "--by", "stranger", "--apply") == 3
    assert cli(world, "clear", "Out A/f.jpg", "--outing", "out-1", "--apply") == 3          # no --by
    assert rows(world) == []
    world["settings_file"].write_text(json.dumps({}), encoding="utf-8")                     # nobody listed
    assert cli(world, "clear", "Out A/f.jpg", "--outing", "out-1", "--by", "Evan", "--apply") == 3
    assert rows(world) == []


def test_clear_by_a_clearer_records_and_status_follows_the_last_event(world, capsys):
    fid = "Out A/f.jpg"
    assert cli(world, "status", fid, "--outing", "out-1", "--json") == 0
    assert json.loads(capsys.readouterr().out)["state"] == "none"
    assert cli(world, "clear", fid, "--outing", "out-1", "--by", "evan") == 0             # dry run
    assert rows(world) == []
    assert cli(world, "clear", fid, "--outing", "out-1", "--by", "evan", "--apply") == 0
    capsys.readouterr()
    cli(world, "status", fid, "--outing", "out-1", "--json")
    assert json.loads(capsys.readouterr().out)["state"] == "cleared"
    faces.record(world["settings"], "frame", fid, "masked", outing="out-1", events_dir=world["events"],
                 image="x_masked.jpg", sha1="abc")
    cli(world, "status", fid, "--outing", "out-1", "--json")
    st = json.loads(capsys.readouterr().out)
    assert st == {"state": "masked", "image": "x_masked.jpg", "sha1": "abc"}
    # another outing's event for the same path does not count
    assert faces.status_of(world["settings"], fid, "out-other", world["events"])["state"] == "none"


# ---------------------------------------------------------------- the gate

def _gate(w, img, **kw):
    kw.setdefault("events_dir", w["events"])
    kw.setdefault("settings", w["settings"])
    return faces.gate(img, **kw)


def test_gate_refuses_a_detected_face_and_allows_a_clean_frame(world, tmp_path, monkeypatch):
    img = noise_image(tmp_path / "a.jpg")
    res = _gate(world, img, frame="f1", outing="out-adults")
    assert not res["allowed"] and res["faces"] == 1 and "face" in res["reason"]
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [FakeDetector([])])
    assert _gate(world, img, frame="f1", outing="out-adults")["allowed"]


def test_gate_masked_requires_the_masked_file_itself(world, tmp_path):
    img = noise_image(tmp_path / "a.jpg")
    masked = noise_image(tmp_path / "a_masked.jpg", seed=2)
    faces.record(world["settings"], "frame", "f1", "masked", outing="o", events_dir=world["events"],
                 image="a_masked.jpg", sha1=_sha(masked))
    assert _gate(world, masked, frame="f1", outing="o")["allowed"]
    res = _gate(world, img, frame="f1", outing="o")
    assert not res["allowed"] and "publish the masked file" in res["reason"]
    # a masked event with no recorded sha is never enough
    faces.record(world["settings"], "frame", "f2", "masked", outing="o", events_dir=world["events"])
    assert not _gate(world, masked, frame="f2", outing="o")["allowed"]


def test_gate_cleared_is_allowed_even_with_a_face_and_a_child_aboard(world, tmp_path):
    img = noise_image(tmp_path / "a.jpg")
    faces.record(world["settings"], "frame", "f1", "cleared", outing="out-kids", by="evan", events_dir=world["events"])
    assert _gate(world, img, frame="f1", outing="out-kids")["allowed"]


def test_gate_later_event_wins_over_clear(world, tmp_path):
    img = noise_image(tmp_path / "a.jpg")
    faces.record(world["settings"], "frame", "f1", "cleared", outing="o", events_dir=world["events"])
    faces.record(world["settings"], "frame", "f1", "masked", outing="o", events_dir=world["events"], sha1="zz")
    assert not _gate(world, img, frame="f1", outing="o")["allowed"]


@pytest.fixture
def no_face(monkeypatch):
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [FakeDetector([])])


def test_gate_refuses_a_child_companion_even_with_no_face_found(world, tmp_path, no_face):
    img = noise_image(tmp_path / "a.jpg")
    res = _gate(world, img, frame="f1", outing="out-kids")
    assert not res["allowed"] and res["companions"]["children"] == 1 and res["faces"] == 0
    assert "Kid" not in json.dumps(res) and "Pal" not in json.dumps(res)         # never a name
    assert _gate(world, img, frame="f1", outing="out-adults")["allowed"]
    assert _gate(world, img, frame="f1", outing="out-unknown")["allowed"]          # no projection, no companions


def test_gate_companions_file_and_the_age_rule(world, tmp_path, no_face):
    img = noise_image(tmp_path / "a.jpg")
    kid_log = note(tmp_path / "log1.md", type="hike-log", companions=["Adult Two", "[[Kid One]]"])
    adult_log = note(tmp_path / "log2.md", type="hike-log", companions="Adult Two and Pal Three")
    assert not _gate(world, img, frame="f1", outing="o", companions_file=str(kid_log))["allowed"]
    assert _gate(world, img, frame="f1", outing="o", companions_file=str(adult_log))["allowed"]
    # a friend noted with a recent date_of_birth is a minor even without a child tag
    assert not _gate(world, img, frame="f1", outing="out-minor")["allowed"]


def test_gate_child_words_may_name_a_relationship_value(world, tmp_path, no_face):
    img = noise_image(tmp_path / "a.jpg")
    s = dict(world["settings"], child_relationships=["acquaintance-kid"])
    note(world["vault"] / "People" / "Kid Five.md", type="person", relationship="acquaintance-kid", tags=["person"])
    log = note(tmp_path / "log.md", type="hike-log", companions=["Kid Five"])
    assert not faces.gate(img, frame="f", outing="o", settings=s, events_dir=world["events"], companions_file=str(log))["allowed"]


def test_gate_no_frame_mode(world, tmp_path, monkeypatch):
    img = noise_image(tmp_path / "art.jpg")
    res = _gate(world, img, key="art-001")
    assert not res["allowed"] and res["faces"] == 1
    faces.record(world["settings"], "key", "art-001", "cleared", events_dir=world["events"])
    assert _gate(world, img, key="art-001")["allowed"]
    assert not _gate(world, img, key="art-002")["allowed"]                 # another key is not cleared
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [FakeDetector([])])
    assert _gate(world, img, key="art-002")["allowed"]


def test_gate_fails_closed(world, tmp_path, monkeypatch, no_face):
    img = noise_image(tmp_path / "a.jpg")
    assert _gate(world, img, frame="f1", outing="out-adults")["allowed"]            # baseline
    # a detector that raises
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [Boom()])
    res = _gate(world, img, frame="f1", outing="out-adults")
    assert not res["allowed"] and "refused" in res["reason"]
    # no detector at all on this machine
    def none(s=None):
        raise faces.FaceError("no face detector is available")
    monkeypatch.setattr(faces, "default_detectors", none)
    assert not _gate(world, img, frame="f1", outing="out-adults")["allowed"]
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [FakeDetector([])])
    # unreadable image, missing image
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert not _gate(world, bad, frame="f1", outing="out-adults")["allowed"]
    assert not _gate(world, tmp_path / "missing.jpg", frame="f1", outing="out-adults")["allowed"]
    # a missing settings file
    gone = faces.link.load_settings(str(tmp_path / "nope.json"))
    assert not faces.gate(img, frame="f1", outing="o", settings=gone, events_dir=world["events"])["allowed"]
    assert not faces.gate(img, frame="f1", outing="o", settings=None)["allowed"]
    # an unreadable companions note, and People notes that cannot be found
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
    """Two starred frames in one outing; frame 1 has a face (red corner), frame 2 does not."""
    root = world["root"]
    outing = root / "Out A"
    ids, exports = [], tmp_path / "export"
    exports.mkdir()
    for n in (1, 2):
        f = outing / ("DSC_%04d.jpg" % n)
        im = noise_image(f, size=(300, 200), seed=n)
        ids.append("Out A/" + f.name)
        shot = Image.open(f).convert("RGB")
        shot.paste((255, 0, 0) if n == 1 else (0, 0, 255), (0, 0, 16, 16))
        shot.save(exports / ("DSC_%04d.jpg" % n), "JPEG", quality=95)

    class RedCorner:
        name = "red-corner"

        def detect(self, image):
            r, g, b = image.getpixel((image.width // 100, image.width // 100))[:3]
            k = image.width / 300
            return [[100 * k, 60 * k, 40 * k, 40 * k, 1.0]] if r > 200 and b < 80 else []

    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [RedCorner()])
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


def test_publish_external_masked_flow_end_to_end(pub):
    """Mask through the CLI exactly as Evan would, with the real outing id, then publish."""
    w = pub["world"]
    pe = pub["pe"]
    settings = w["settings"]
    root = w["root"]
    oid = faces.link.outing_id_for(settings, root, root / "Out A", w["events"])
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
    assert faces.has_clearance("20250801-DSC_0001", settings, w["events"])
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


def test_optimize_images_refuses_a_master_with_a_face(world, tmp_path, monkeypatch, capsys):
    oi = _load_oi(tmp_path, monkeypatch)
    master = noise_image(tmp_path / "masters" / "20250801-DSC_0001.jpg", size=(1000, 700))
    monkeypatch.setenv("MAYNIAC_STUDIO_SETTINGS", str(world["settings_file"]))
    monkeypatch.setenv("MAYNIAC_STUDIO_EVENTS", str(world["events"]))
    assert oi.optimize_image(master, allowed=set()) is False
    assert list((tmp_path / "out").iterdir()) == []
    assert "REFUSED (face guard)" in capsys.readouterr().out
    assert oi.main(["--only", "20250801-DSC_0001"]) == 3
    assert list((tmp_path / "out").iterdir()) == []


def test_optimize_images_proceeds_with_a_faces_event_or_no_face(world, tmp_path, monkeypatch):
    oi = _load_oi(tmp_path, monkeypatch)
    master = noise_image(tmp_path / "masters" / "20250801-DSC_0001.jpg", size=(1000, 700))
    monkeypatch.setenv("MAYNIAC_STUDIO_SETTINGS", str(world["settings_file"]))
    monkeypatch.setenv("MAYNIAC_STUDIO_EVENTS", str(world["events"]))
    faces.record(world["settings"], "key", "20250801-DSC_0001", "cleared", events_dir=world["events"])
    assert oi.optimize_image(master, allowed=set()) is True
    assert len(list((tmp_path / "out").glob("*.jpg"))) >= 2
    other = noise_image(tmp_path / "masters" / "20250801-DSC_0002.jpg", size=(1000, 700))
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [FakeDetector([])])
    assert oi.optimize_image(other, allowed=set()) is True


def test_optimize_images_fails_closed_when_the_detector_cannot_run(world, tmp_path, monkeypatch):
    oi = _load_oi(tmp_path, monkeypatch)
    master = noise_image(tmp_path / "masters" / "20250801-DSC_0001.jpg", size=(1000, 700))
    monkeypatch.setenv("MAYNIAC_STUDIO_SETTINGS", str(world["settings_file"]))
    monkeypatch.setenv("MAYNIAC_STUDIO_EVENTS", str(world["events"]))
    monkeypatch.setattr(faces, "default_detectors", lambda s=None: [Boom()])
    assert oi.optimize_image(master, allowed=set()) is False
    assert list((tmp_path / "out").iterdir()) == []
