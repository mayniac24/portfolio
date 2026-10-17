"""Stage picks and detect finished edits (DESIGN-2026-10-17 D4/D5, 2.3 Pick and Finish).

Everything runs on a tmp fixture: a tiny photos_root, a settings file, a tmp event
directory. Nothing here touches the real library, the real bus or the real index.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PICKER = REPO / "tools" / "picker"
SCRIPTS_EVENTS = Path(r"C:\Users\Evan-PC\Documents\Organize IT\Notes\The inner ravings of a mayniac"
                      r"\000 System\050 Architecture\052 System Utilities\Scripts\lib\studio_events.py")
OUTING = "Hiking/2026/10.13.26 - Test"
T0 = 1_700_000_000


def _load(name, path):
    sys.path.insert(0, str(PICKER))
    sys.path.insert(0, str(REPO / "tools"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stage_mod():
    return _load("stage_picks", PICKER / "stage_picks.py")


def finish_mod():
    return _load("finish_watch", PICKER / "finish_watch.py")


@pytest.fixture
def lib(tmp_path):
    """5 frames in one outing, 3 starred; settings + selections + events dir."""
    root = tmp_path / "photos"
    cam = root / OUTING / "Camera"
    cam.mkdir(parents=True)
    ids = []
    for n in range(1, 6):
        f = cam / ("DSC_%04d.JPG" % n)
        f.write_bytes(("frame %d " % n).encode() * 50)
        os.utime(f, (T0 + n * 60, T0 + n * 60))
        ids.append(f"{OUTING}/Camera/{f.name}")
    sel = {ids[0]: {"category": "landscape", "alt": "a ridge"},
           ids[2]: {"category": "landscape", "alt": "a lake"},
           ids[4]: {"category": "mountain", "alt": "a peak"}}
    sel_file = tmp_path / "selections.json"
    sel_file.write_text(json.dumps(sel), encoding="utf-8")
    settings = tmp_path / "_studio_settings.json"
    settings.write_text(json.dumps({
        "photos_root": str(root), "finish_dir": "_Finish", "camera_match": "D3200",
        "picker_tools": {"selections": str(sel_file), "index_file": str(tmp_path / "idx.json")},
    }), encoding="utf-8")
    events = tmp_path / "_events"
    return {"root": root, "outing": root / OUTING, "ids": ids, "settings": settings,
            "events": events, "tmp": tmp_path, "sel_file": sel_file,
            "finish": root / "_Finish" / "10.13.26 - Test"}


def _args(lib, *more):
    return [str(lib["outing"]), "--settings", str(lib["settings"]),
            "--events-dir", str(lib["events"]), *more]


def _rows(lib):
    f = lib["events"] / "studio-picker.jsonl"
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []


def _snapshot(root):
    return {str(p): (hashlib.sha1(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_stage_dry_run_writes_nothing(lib, capsys):
    assert stage_mod().main(_args(lib)) == 0
    assert not lib["finish"].exists()
    assert not lib["events"].exists()
    assert "dry run" in capsys.readouterr().out.lower()


def test_stage_copies_three_starred_of_five_and_writes_three_pick_events(lib, capsys):
    before = _snapshot(lib["outing"])
    assert stage_mod().main(_args(lib, "--apply")) == 0
    out = capsys.readouterr().out
    copied = sorted(p.name for p in lib["finish"].glob("*.JPG"))
    assert copied == ["DSC_0001.JPG", "DSC_0003.JPG", "DSC_0005.JPG"]
    # the folder to import is printed for Lightroom
    assert str(lib["finish"]) in out
    rows = _rows(lib)
    assert len(rows) == 3
    assert [r["frame"] for r in rows] == [lib["ids"][0], lib["ids"][2], lib["ids"][4]]
    for r in rows:
        assert r["v"] == 2 and r["source"] == "studio" and r["actor"] == "studio-picker"
        assert r["kind"] == "pick" and r["state"] == "starred" and r["via"] == "cli"
        assert r["outing"].startswith("out-") and len(r["outing"]) == 14
        assert r["day"] == r["ts"][:10]
    assert rows[2]["category"] == "mountain" and rows[0]["alt"] == "a ridge"
    assert len({r["outing"] for r in rows}) == 1
    # originals untouched: same bytes, same mtime, same file set (copied, never moved)
    assert _snapshot(lib["outing"]) == before


def test_stage_outing_id_is_hash_of_the_earliest_frame(lib):
    stage_mod().main(_args(lib, "--apply"))
    first = lib["root"] / lib["ids"][0]            # lowest mtime
    sha = hashlib.sha1(first.read_bytes()).hexdigest()
    want = "out-" + hashlib.sha1(("outing:" + sha).encode()).hexdigest()[:10]
    assert _rows(lib)[0]["outing"] == want


def test_stage_is_idempotent_and_never_overwrites(lib):
    m = stage_mod()
    m.main(_args(lib, "--apply"))
    staged = lib["finish"] / "DSC_0003.JPG"
    staged.write_bytes(b"his edit")                # an edit in place must survive a re-stage
    os.utime(staged, (T0 + 99999, T0 + 99999))
    assert m.main(_args(lib, "--apply")) == 0
    assert staged.read_bytes() == b"his edit"
    assert len(_rows(lib)) == 3                    # no duplicate pick events


def test_stage_refuses_two_starred_frames_with_one_file_name(lib):
    other = lib["outing"] / "Phone" / "DSC_0001.JPG"
    other.parent.mkdir()
    other.write_bytes(b"other")
    sel = json.loads(lib["sel_file"].read_text(encoding="utf-8"))
    sel[f"{OUTING}/Phone/DSC_0001.JPG"] = {"category": "landscape", "alt": "x"}
    lib["sel_file"].write_text(json.dumps(sel), encoding="utf-8")
    assert stage_mod().main(_args(lib, "--apply")) == 1
    assert not lib["finish"].exists() and not lib["events"].exists()


def test_stage_ignores_stars_in_other_outings(lib):
    sel = json.loads(lib["sel_file"].read_text(encoding="utf-8"))
    sel["Hiking/2026/other outing/Camera/DSC_0001.JPG"] = {"category": "landscape", "alt": "x"}
    lib["sel_file"].write_text(json.dumps(sel), encoding="utf-8")
    stage_mod().main(_args(lib, "--apply"))
    assert len(_rows(lib)) == 3


def _export(lib, stem, ext=".jpg", plus=3600):
    p = lib["finish"] / (stem + ext)
    p.write_bytes(b"edited " + stem.encode())
    t = T0 + 5 * 60 + plus
    os.utime(p, (t, t))
    return p


def test_finish_watch_one_same_stem_export_reports_finished_1_of_3(lib, capsys):
    stage_mod().main(_args(lib, "--apply"))
    capsys.readouterr()
    exp = _export(lib, "DSC_0003", ".tif")
    before = _snapshot(lib["outing"])
    assert finish_mod().main(_args(lib, "--apply")) == 0
    assert "finished 1 of 3" in capsys.readouterr().out
    fin = [r for r in _rows(lib) if r["kind"] == "finish"]
    assert len(fin) == 1
    r = fin[0]
    assert r["frame"] == lib["ids"][2] and r["v"] == 2 and r["actor"] == "studio-picker"
    assert r["export"] == f"_Finish/10.13.26 - Test/{exp.name}"
    assert r["bytes"] == exp.stat().st_size
    assert _snapshot(lib["outing"]) == before      # originals untouched


def test_finish_watch_is_idempotent(lib, capsys):
    stage_mod().main(_args(lib, "--apply"))
    _export(lib, "DSC_0003", ".tif")
    m = finish_mod()
    m.main(_args(lib, "--apply"))
    capsys.readouterr()
    m.main(_args(lib, "--apply"))
    assert "finished 1 of 3" in capsys.readouterr().out
    assert len([r for r in _rows(lib) if r["kind"] == "finish"]) == 1


def test_finish_watch_dry_run_reports_but_writes_nothing(lib, capsys):
    stage_mod().main(_args(lib, "--apply"))
    _export(lib, "DSC_0001", ".tif")
    capsys.readouterr()
    assert finish_mod().main(_args(lib)) == 0
    assert "finished 1 of 3" in capsys.readouterr().out
    assert [r for r in _rows(lib) if r["kind"] == "finish"] == []


def test_the_untouched_staged_copy_and_an_older_file_are_not_finished(lib, capsys):
    stage_mod().main(_args(lib, "--apply"))
    capsys.readouterr()
    old = lib["finish"] / "DSC_0005-old.tif"          # same stem prefix but not newer
    old.write_bytes(b"x")
    os.utime(old, (T0, T0))
    assert finish_mod().main(_args(lib, "--apply")) == 0
    assert "finished 0 of 3" in capsys.readouterr().out
    assert [r for r in _rows(lib) if r["kind"] == "finish"] == []


def test_a_lightroom_edit_suffix_matches_its_frame(lib, capsys):
    stage_mod().main(_args(lib, "--apply"))
    capsys.readouterr()
    _export(lib, "DSC_0001-Edit")
    finish_mod().main(_args(lib, "--apply"))
    assert "finished 1 of 3" in capsys.readouterr().out


def test_a_second_export_with_new_bytes_writes_a_new_finish_event(lib):
    stage_mod().main(_args(lib, "--apply"))
    p = _export(lib, "DSC_0003", ".tif")
    m = finish_mod()
    m.main(_args(lib, "--apply"))
    p.write_bytes(b"re-edited, longer " * 5)
    t = T0 + 5 * 60 + 7200
    os.utime(p, (t, t))
    m.main(_args(lib, "--apply"))
    assert len([r for r in _rows(lib) if r["kind"] == "finish"]) == 2


@pytest.mark.skipif(not SCRIPTS_EVENTS.is_file(), reason="Scripts studio_events.py not on this machine")
def test_the_documented_line_matches_the_scripts_writer(lib):
    """Both paths of the contract: settings.events_module present vs absent."""
    link = _load("studio_link", PICKER / "studio_link.py")
    settings = link.load_settings(lib["settings"])
    now = "2026-10-17T12:00:00-06:00"
    mine = link.append_event(settings, "pick", outing="out-aaaaaaaaaa", frame="a/b.JPG",
                             state="starred", category="x", alt="y", now=now,
                             events_dir=lib["tmp"] / "mine")
    settings["events_module"] = str(SCRIPTS_EVENTS)
    theirs = link.append_event(settings, "pick", outing="out-aaaaaaaaaa", frame="a/b.JPG",
                               state="starred", category="x", alt="y", now=now,
                               events_dir=lib["tmp"] / "theirs")
    assert mine == theirs
    assert list(mine) == list(theirs)               # same key order too


def test_events_dir_is_required_under_pytest(lib):
    link = _load("studio_link", PICKER / "studio_link.py")
    with pytest.raises(RuntimeError):
        link.append_event(link.load_settings(lib["settings"]), "pick", outing="x", frame="y", state="starred")


# ---------------------------------------------------------------- index_library

def _jpeg(path, model, when="2026:10:13 10:00:00", make="NIKON CORPORATION"):
    path.parent.mkdir(parents=True, exist_ok=True)
    ex = Image.Exif()
    ex[0x010F] = make
    ex[0x0110] = model
    ex.get_ifd(0x8769)[0x9003] = when
    Image.new("RGB", (8, 8), "gray").save(path, "JPEG", exif=ex)


def index_mod():
    return _load("index_library", PICKER / "index_library.py")


def test_index_filters_on_camera_match_not_on_nikon(lib):
    _jpeg(lib["outing"] / "Camera" / "d3200.JPG", "NIKON D3200")
    _jpeg(lib["outing"] / "Camera" / "d7500.JPG", "NIKON D7500")
    _jpeg(lib["outing"] / "Phone" / "p.JPG", "Pixel", make="Google")
    assert index_mod().main(["--settings", str(lib["settings"])]) == 0
    idx = json.loads((lib["tmp"] / "idx.json").read_text(encoding="utf-8"))
    assert [p["id"] for p in idx["photos"]] == [f"{OUTING}/Camera/d3200.JPG"]
    assert idx["root"] == str(lib["root"])


def test_index_add_merges_one_outing_into_the_existing_index(lib):
    other = lib["root"] / "Hiking/2026/09.01.26 - Earlier"
    _jpeg(other / "Camera" / "e1.JPG", "NIKON D3200", when="2026:09:01 09:00:00")
    m = index_mod()
    assert m.main(["--settings", str(lib["settings"])]) == 0
    _jpeg(lib["outing"] / "Camera" / "n1.JPG", "NIKON D3200")
    _jpeg(lib["outing"] / "Camera" / "n2.JPG", "NIKON D7500")
    for _ in range(2):                                # twice: --add is idempotent
        assert m.main(["--settings", str(lib["settings"]), "--add", str(lib["outing"])]) == 0
    idx = json.loads((lib["tmp"] / "idx.json").read_text(encoding="utf-8"))
    ids = [p["id"] for p in idx["photos"]]
    assert ids == ["Hiking/2026/09.01.26 - Earlier/Camera/e1.JPG", f"{OUTING}/Camera/n1.JPG"]
    assert idx["root"] == str(lib["root"])            # ids stay relative to photos_root


def test_index_add_without_an_existing_index_builds_one_with_root_from_settings(lib):
    _jpeg(lib["outing"] / "Camera" / "n1.JPG", "NIKON D3200")
    assert index_mod().main(["--settings", str(lib["settings"]), "--add", str(lib["outing"])]) == 0
    idx = json.loads((lib["tmp"] / "idx.json").read_text(encoding="utf-8"))
    assert [p["id"] for p in idx["photos"]] == [f"{OUTING}/Camera/n1.JPG"]


def test_index_default_camera_match_is_d3200_when_settings_absent(tmp_path, monkeypatch):
    link = _load("studio_link", PICKER / "studio_link.py")
    monkeypatch.delenv(link.SETTINGS_ENV, raising=False)
    monkeypatch.delenv("VAULT_PATH", raising=False)
    assert link.load_settings()["camera_match"] == "D3200"


# ---------------------------------------------------------------- the star route

def _app(lib, monkeypatch):
    pytest.importorskip("flask")
    root = lib["root"]
    photos = [{"id": i, "outing": "10.13.26 - Test", "date": "2026-10-13", "time": "10:00",
               "camera": "D3200", "iso": 100, "elev": 10.0, "light": "golden"} for i in lib["ids"]]
    (lib["tmp"] / "idx.json").write_text(json.dumps({"root": str(root), "photos": photos}), encoding="utf-8")
    lib["sel_file"].write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MAYNIAC_STUDIO_SETTINGS", str(lib["settings"]))
    monkeypatch.setenv("MAYNIAC_STUDIO_EVENTS", str(lib["events"]))
    return _load("picker_app_under_test", PICKER / "app.py")


def test_star_route_appends_a_pick_event_on_star_and_unstar_only(lib, monkeypatch):
    app = _app(lib, monkeypatch)
    c = app.app.test_client()
    pid = lib["ids"][1]
    assert c.post("/api/star/" + pid, json={}).status_code == 200
    c.post("/api/star/" + pid, json={"alt": "typing", "category": "mountain"})   # an edit: no event
    c.post("/api/star/" + pid, json={"alt": "typing more"})
    assert c.post("/api/star/" + pid, json={"starred": False}).status_code == 200
    rows = _rows(lib)
    assert [(r["kind"], r["state"], r["frame"]) for r in rows] == [
        ("pick", "starred", pid), ("pick", "unstarred", pid)]
    assert all(r["actor"] == "studio-picker" and r["via"] == "cli" and r["outing"].startswith("out-")
               for r in rows)
    assert rows[0]["outing"] == rows[1]["outing"]


def test_star_route_still_stars_when_the_bus_is_unwritable(lib, monkeypatch):
    app = _app(lib, monkeypatch)
    monkeypatch.setenv("MAYNIAC_STUDIO_EVENTS", str(lib["tmp"] / "photos" / lib["ids"][0]))  # a file, not a dir
    r = app.app.test_client().post("/api/star/" + lib["ids"][1], json={})
    assert r.status_code == 200 and r.get_json()["starred"] == 1


# ---------------------------------------------------------------- publish_external

def _publisher(lib, monkeypatch):
    m = _load("publish_external", REPO / "tools" / "publish_external.py")
    gallery = lib["tmp"] / "gallery.json"
    gallery.write_text(json.dumps({"photos": []}), encoding="utf-8")
    monkeypatch.setattr(m, "CORRECTED", lib["tmp"] / "corrected")
    monkeypatch.setattr(m, "GALLERY", gallery)
    calls = []

    class Done:
        returncode, stdout, stderr = 0, "", ""
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: calls.append(a) or Done())
    return m, gallery, calls


def _publish_fixture(lib):
    """Index the 5 frames and make an export dir holding DSC_0001 and DSC_0003."""
    photos = [{"id": i, "outing": "10.13.26 - Test", "date": "2026-10-13", "time": "10:00",
               "camera": "D3200", "iso": 100, "elev": 10.0, "light": "golden"} for i in lib["ids"]]
    (lib["tmp"] / "idx.json").write_text(json.dumps({"root": str(lib["root"]), "photos": photos}),
                                         encoding="utf-8")
    exp = lib["tmp"] / "exports"
    exp.mkdir()
    for stem in ("DSC_0001", "DSC_0003"):
        (exp / (stem + ".jpg")).write_bytes(b"jpeg " + stem.encode())
    return exp


def test_publish_external_writes_a_post_event_per_new_frame_and_scopes_to_an_outing(lib, monkeypatch, capsys):
    m, gallery, calls = _publisher(lib, monkeypatch)
    exp = _publish_fixture(lib)
    # a star in ANOTHER outing with the same stem would make the unscoped check ambiguous
    sel = json.loads(lib["sel_file"].read_text(encoding="utf-8"))
    sel["Hiking/2026/other outing/Camera/DSC_0001.JPG"] = {"category": "landscape", "alt": "x"}
    lib["sel_file"].write_text(json.dumps(sel), encoding="utf-8")
    idx = json.loads((lib["tmp"] / "idx.json").read_text(encoding="utf-8"))
    idx["photos"].append({**idx["photos"][0], "id": "Hiking/2026/other outing/Camera/DSC_0001.JPG",
                          "outing": "other outing"})
    (lib["tmp"] / "idx.json").write_text(json.dumps(idx), encoding="utf-8")

    base = [str(exp), "--settings", str(lib["settings"]), "--events-dir", str(lib["events"])]
    assert m.main(base + ["--apply"]) == 1                      # unscoped: ambiguous stems, as before
    assert "ambiguous" in capsys.readouterr().out
    assert m.main(base + ["--outing", str(lib["outing"]), "--apply"]) == 0
    posts = [r for r in _rows(lib) if r["kind"] == "post"]
    assert sorted(r["slug"] for r in posts) == ["20261013-DSC_0001", "20261013-DSC_0003"]
    for r in posts:
        assert r["channel"] == "portfolio" and r["actor"] == "studio-picker"
        assert r["frames"] in ([lib["ids"][0]], [lib["ids"][2]]) and r["when"]
        assert r["outing"].startswith("out-")
    assert len(json.loads(gallery.read_text(encoding="utf-8"))["photos"]) == 2
    # re-publishing the same frames is not a new post
    assert m.main(base + ["--outing", str(lib["outing"]), "--apply"]) == 0
    assert len([r for r in _rows(lib) if r["kind"] == "post"]) == 2
