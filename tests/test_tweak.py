"""Finish helper tools and recipes (DESIGN-2026-10-17 D16): tools/picker/tweak.py.

Everything runs on a tmp fixture: a tiny photos_root with one synthetic frame (bright sky,
dark land), a settings file, a recipes file and an events directory under tmp_path. Nothing
here touches the real library, the real bus or the real recipes.yaml.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PICKER = REPO / "tools" / "picker"
SCRIPTS_EVENTS = Path(r"C:\Users\Evan-PC\Documents\Organize IT\Notes\The inner ravings of a mayniac"
                      r"\000 System\050 Architecture\052 System Utilities\Scripts\lib\studio_events.py")
OUTING = "Hiking/2026/10.13.26 - Test"
FRAME = OUTING + "/Camera/DSC_0001.jpg"
T0 = 1_700_000_000


def _load(name, path):
    sys.path.insert(0, str(PICKER))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def tweak():
    return _load("tweak", PICKER / "tweak.py")


def stage():
    return _load("stage_picks", PICKER / "stage_picks.py")


def finish():
    return _load("finish_watch", PICKER / "finish_watch.py")


def synthetic(path, sky=0.92, land=0.15, size=(80, 100)):
    """Top 40 percent bright sky, a mid band, bottom 40 percent dark land; a real JPEG."""
    w, h = size
    a = np.zeros((h, w, 3), dtype=np.float32)
    a[: h * 2 // 5] = sky
    a[h * 2 // 5: h * 3 // 5] = (sky + land) / 2
    a[h * 3 // 5:] = land
    Image.fromarray(np.rint(a * 255).astype(np.uint8), "RGB").save(path, "JPEG", quality=95)


@pytest.fixture
def lib(tmp_path):
    root = tmp_path / "photos"
    cam = root / OUTING / "Camera"
    cam.mkdir(parents=True)
    master = cam / "DSC_0001.jpg"
    synthetic(master, sky=0.80, land=0.20)
    os.utime(master, (T0, T0))
    sel = tmp_path / "selections.json"
    sel.write_text(json.dumps({FRAME: {"category": "landscape", "alt": "a ridge"}}), encoding="utf-8")
    settings = tmp_path / "_studio_settings.json"
    settings.write_text(json.dumps({
        "photos_root": str(root), "finish_dir": "_Finish", "camera_match": "D3200",
        "picker_tools": {"selections": str(sel), "index_file": str(tmp_path / "idx.json")},
    }), encoding="utf-8")
    d = {"root": root, "outing": root / OUTING, "master": master, "settings": settings,
         "events": tmp_path / "_events", "recipes": tmp_path / "recipes.yaml", "tmp": tmp_path,
         "finish": root / "_Finish" / "10.13.26 - Test"}
    assert stage().main([str(d["outing"]), "--settings", str(settings),
                         "--events-dir", str(d["events"]), "--apply"]) == 0
    # his export: Lightroom overwrote the staged copy; sky a little blown, land a little dark
    exp = d["finish"] / "DSC_0001.jpg"
    synthetic(exp, sky=0.97, land=0.12)
    os.utime(exp, (T0 + 1_000_000, T0 + 1_000_000))
    d["export"] = exp
    (d["events"] / "studio-picker.jsonl").unlink()      # forget the stage's pick event
    return d


def run(lib, *args, capsys=None):
    argv = list(args) + ["--settings", str(lib["settings"]), "--events-dir", str(lib["events"]),
                         "--recipes", str(lib["recipes"])]
    code = tweak().main(argv)
    return code, (capsys.readouterr().out if capsys else "")


def rows(lib):
    f = lib["events"] / "studio-picker.jsonl"
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []


def snap(*dirs):
    out = {}
    for d in dirs:
        for p in Path(d).rglob("*"):
            if p.is_file():
                out[str(p)] = (hashlib.sha1(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
    return out


def sha(p):
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


def saved(lib):
    return yaml.safe_load(lib["recipes"].read_text(encoding="utf-8"))["recipes"]


def save_recipe(lib, capsys, name="blown-sky", symptom="sky is blown out and the land is dark",
                extra=("--params", "sky_pull=0.3", "highlights=0.2", "shadows=0.1")):
    code, out = run(lib, "recipe", "save", name, "--frame", FRAME, "--symptom", symptom,
                    "--step", "Graduated filter on the sky, exposure -0.7",
                    "--step", "Shadows +30", *extra, "--apply", capsys=capsys)
    assert code == 0, out
    return out


# ------------------------------------------------------------------ measure

def test_measure_reports_the_sky_land_gap_and_clipping(lib, capsys):
    code, out = run(lib, "measure", FRAME, "--json", capsys=capsys)
    m = json.loads(out)
    assert code == 0
    assert m["source"].endswith("DSC_0001.jpg") and "_Finish" in m["source"]     # his export, not the master
    assert m["sky"] == pytest.approx(0.97, abs=0.02)
    assert m["land"] == pytest.approx(0.12, abs=0.02)
    assert m["gap"] == pytest.approx(0.85, abs=0.03)
    assert m["highlight_clip"] == 0.0 and m["shadow_clip"] == 0.0


def test_measure_prints_the_gap_in_words(lib, capsys):
    code, out = run(lib, "measure", FRAME, capsys=capsys)
    assert code == 0
    assert "sky luminance" in out and "land luminance" in out and "gap" in out and "clipping" in out


def test_measure_counts_clipping():
    t = tweak()
    a = np.zeros((10, 10, 3), dtype=np.float32)
    a[:5] = 1.0                                   # half the frame blown, half crushed
    m = t.measure_array(a)
    assert m["highlight_clip"] == pytest.approx(0.5) and m["shadow_clip"] == pytest.approx(0.5)


# ------------------------------------------------------------------ apply

def test_apply_dry_run_writes_nothing(lib, capsys):
    before = snap(lib["tmp"])
    code, out = run(lib, "apply", FRAME, "--params", "sky_pull=0.3", capsys=capsys)
    assert code == 0 and "DRY RUN" in out
    assert snap(lib["tmp"]) == before


def test_apply_writes_only_a_candidate_and_leaves_every_source_byte_identical(lib, capsys):
    before = snap(lib["outing"])
    export_sha, export_mtime = sha(lib["export"]), lib["export"].stat().st_mtime_ns
    files_before = {p.name for p in lib["finish"].iterdir()}
    code, out = run(lib, "apply", FRAME, "--params", "sky_pull=0.3", "highlights=0.2", "shadows=0.1",
                    "--apply", capsys=capsys)
    assert code == 0, out
    cand = lib["finish"] / "DSC_0001_helper.jpg"
    assert cand.is_file()
    assert {p.name for p in lib["finish"].iterdir()} - files_before == {"DSC_0001_helper.jpg"}
    assert snap(lib["outing"]) == before                              # master and outing folder untouched
    assert sha(lib["export"]) == export_sha and lib["export"].stat().st_mtime_ns == export_mtime
    t = tweak()
    mine, theirs = t.measure_file(lib["export"]), t.measure_file(cand)
    assert theirs["gap"] < mine["gap"] - 0.05                         # the sky came down
    assert theirs["land"] > mine["land"]                              # the land came up
    assert not list(lib["tmp"].rglob("*.part"))
    assert rows(lib) == []                                            # apply writes no event


def test_apply_candidate_carries_its_recipe_and_params(lib, capsys):
    save_recipe(lib, capsys)
    code, out = run(lib, "apply", FRAME, "--recipe", "blown-sky", "--apply", capsys=capsys)
    assert code == 0, out
    prov = tweak().read_provenance(lib["finish"] / "DSC_0001_helper.jpg")
    assert prov["recipe"] == "blown-sky" and prov["params"]["sky_pull"] == 0.3
    assert prov["export_sha1"] == sha(lib["export"])


@pytest.mark.parametrize("bad", ["sky_pull=0.9", "highlights=-0.1", "sky_height=0.05", "gamma=1"])
def test_apply_refuses_params_outside_their_bounds(lib, capsys, bad):
    code, out = run(lib, "apply", FRAME, "--params", bad, "--apply", capsys=capsys)
    assert code == 2 and ("out of bounds" in out or "unknown" in out)
    assert not (lib["finish"] / "DSC_0001_helper.jpg").exists()


def test_apply_refuses_a_request_with_no_effect(lib, capsys):
    code, out = run(lib, "apply", FRAME, "--params", "sky_height=0.5", "--apply", capsys=capsys)
    assert code == 2 and "no effect" in out
    assert not (lib["finish"] / "DSC_0001_helper.jpg").exists()


def test_apply_at_the_bounds_never_inverts_tones():
    t = tweak()
    ramp = np.linspace(0, 1, 256, dtype=np.float32)[None, :, None].repeat(3, axis=2).repeat(4, axis=0)
    p = {k: hi for k, (lo, hi) in t.BOUNDS.items()}
    p["sky_pull"] = 0.0          # the sky term is spatial, not a tone curve
    out = t.apply_params(ramp, p)
    assert (np.diff(out[0, :, 1]) >= -1e-6).all()
    assert out.min() >= 0 and out.max() <= 1


def test_apply_needs_an_export_and_never_works_from_the_master(lib, capsys):
    lib["export"].unlink()
    code, out = run(lib, "apply", FRAME, "--params", "sky_pull=0.3", "--apply", capsys=capsys)
    assert code == 3 and "no export" in out
    assert not list(lib["finish"].glob("*_helper*"))


def test_an_export_outside_the_finish_folder_is_refused(lib, capsys):
    before = snap(lib["outing"])
    code, out = run(lib, "apply", FRAME, "--export", str(lib["master"]), "--params", "sky_pull=0.3",
                    "--apply", capsys=capsys)
    assert code == 3 and "finish folder" in out
    assert snap(lib["outing"]) == before


# ------------------------------------------------------------------ recipes

def test_recipe_save_dry_run_writes_nothing(lib, capsys):
    code, out = run(lib, "recipe", "save", "blown-sky", "--frame", FRAME, "--symptom", "sky blown out",
                    "--step", "Dehaze -10", capsys=capsys)
    assert code == 0 and "DRY RUN" in out
    assert not lib["recipes"].exists()


def test_recipe_save_stores_his_words_the_measured_trigger_steps_params_and_origin(lib, capsys):
    save_recipe(lib, capsys)
    assert lib["recipes"].read_text(encoding="utf-8").startswith("# Finish-helper recipes")
    r = saved(lib)[0]
    assert r["name"] == "blown-sky" and r["symptom"] == "sky is blown out and the land is dark"
    assert r["origin_frame"] == FRAME and r["created"]
    assert r["lightroom_steps"] == ["Graduated filter on the sky, exposure -0.7", "Shadows +30"]
    assert r["params"] == {"sky_pull": 0.3, "highlights": 0.2, "shadows": 0.1}
    lo, hi = r["trigger"]["gap"]
    assert lo < 0.85 < hi and r["trigger"]["sky"][0] < 0.97 < r["trigger"]["sky"][1]
    assert set(r["trigger"]) == {"sky", "land", "gap", "highlight_clip", "shadow_clip"}


def test_recipe_save_refuses_a_taken_name_and_bad_input(lib, capsys):
    save_recipe(lib, capsys)
    first = lib["recipes"].read_bytes()
    code, out = run(lib, "recipe", "save", "blown-sky", "--frame", FRAME, "--symptom", "again",
                    "--step", "x", "--apply", capsys=capsys)
    assert code == 3 and "exists" in out
    code, _ = run(lib, "recipe", "save", "n2", "--frame", FRAME, "--symptom", "ok words", "--apply", capsys=capsys)
    assert code == 2                                                  # no steps
    code, _ = run(lib, "recipe", "save", "n3", "--frame", FRAME, "--symptom", "ok words", "--step", "x",
                  "--params", "sky_pull=5", "--apply", capsys=capsys)
    assert code == 2                                                  # out-of-bounds param
    assert lib["recipes"].read_bytes() == first


def test_suggest_returns_the_saved_recipe_by_trigger_and_by_words(lib, capsys):
    save_recipe(lib, capsys)
    recipes_before, snap_before = lib["recipes"].read_bytes(), snap(lib["tmp"])
    code, out = run(lib, "suggest", FRAME, capsys=capsys)             # trigger only, no words
    assert code == 0 and "blown-sky" in out and "trigger matches" in out
    assert "Graduated filter on the sky" in out and "no decisions yet" in out
    code, out = run(lib, "suggest", FRAME, "--note", "the sky looks blown", capsys=capsys)
    assert "blown-sky" in out and "words: blown, sky" in out
    assert lib["recipes"].read_bytes() == recipes_before and snap(lib["tmp"]) == snap_before   # suggest writes nothing


def test_suggest_matches_by_words_when_the_trigger_does_not(lib, capsys):
    save_recipe(lib, capsys, extra=("--trigger", "gap=0.0:0.1"))
    code, out = run(lib, "suggest", FRAME, capsys=capsys)
    assert "no recipe matches" in out                                 # trigger misses, no words
    code, out = run(lib, "suggest", FRAME, "--note", "land is too dark", capsys=capsys)
    assert "blown-sky" in out and "words: dark, land" in out
    code, out = run(lib, "suggest", FRAME, "--note", "horizon is crooked", capsys=capsys)
    assert "no recipe matches" in out


def test_suggest_json_and_ranking(lib, capsys):
    save_recipe(lib, capsys, name="a-weak", symptom="grainy shadows", extra=("--trigger", "gap=0.0:0.1"))
    save_recipe(lib, capsys, name="b-strong")
    code, out = run(lib, "suggest", FRAME, "--note", "sky blown and grainy", "--json", capsys=capsys)
    assert [m["name"] for m in json.loads(out)["matches"]] == ["b-strong", "a-weak"]


def test_recipe_edit_and_retire_are_through_the_one_writer(lib, capsys):
    save_recipe(lib, capsys)
    code, _ = run(lib, "recipe", "edit", "blown-sky", "--step", "Highlights -40", "--apply", capsys=capsys)
    assert code == 0
    r = saved(lib)[0]
    assert r["lightroom_steps"] == ["Highlights -40"] and r["origin_frame"] == FRAME
    code, out = run(lib, "recipe", "retire", "blown-sky", "--apply", capsys=capsys)
    assert code == 0 and saved(lib)[0]["retired"]
    code, out = run(lib, "suggest", FRAME, "--note", "sky blown", capsys=capsys)
    assert "no recipe matches" in out
    code, out = run(lib, "recipe", "list", "--json", capsys=capsys)
    assert json.loads(out)[0]["name"] == "blown-sky"                  # kept for its history
    code, out = run(lib, "apply", FRAME, "--recipe", "blown-sky", "--apply", capsys=capsys)
    assert code == 3 and "retired" in out


def test_recipes_yaml_has_one_writer():
    """No other module in the repo writes recipes.yaml."""
    py = list((REPO / "tools").rglob("*.py")) + list((REPO / "scripts").rglob("*.py"))
    offenders = [p.name for p in py if p.name != "tweak.py"
                 and "recipes.yaml" in p.read_text(encoding="utf-8", errors="replace")]
    assert offenders == []


def test_the_seed_recipes_file_loads_empty():
    assert tweak().load_recipes(PICKER / "recipes.yaml") == []


# ------------------------------------------------------------------ accept and reject

def make_candidate(lib, capsys, recipe=True):
    if recipe:
        save_recipe(lib, capsys)
        args = ("--recipe", "blown-sky")
    else:
        args = ("--params", "sky_pull=0.3")
    code, out = run(lib, "apply", FRAME, *args, "--apply", capsys=capsys)
    assert code == 0, out
    return lib["finish"] / "DSC_0001_helper.jpg"


def test_accept_swaps_the_export_keeps_the_old_one_and_writes_the_event(lib, capsys):
    cand = make_candidate(lib, capsys)
    old_export, cand_bytes = lib["export"].read_bytes(), cand.read_bytes()
    master_snap = snap(lib["outing"])
    code, out = run(lib, "accept", FRAME, capsys=capsys)              # dry run first
    assert code == 0 and "DRY RUN" in out and cand.exists() and rows(lib) == []
    code, out = run(lib, "accept", FRAME, "--apply", capsys=capsys)
    assert code == 0, out
    pre = lib["finish"] / "DSC_0001_pre-helper.jpg"
    assert lib["export"].read_bytes() == cand_bytes
    assert pre.read_bytes() == old_export
    assert not cand.exists()
    assert snap(lib["outing"]) == master_snap
    (row,) = rows(lib)
    assert (row["v"], row["source"], row["actor"], row["kind"]) == (2, "studio", "studio-picker", "tweak")
    assert row["frame"] == FRAME and row["recipe"] == "blown-sky" and row["accepted"] is True
    assert row["outing"].startswith("out-") and row["day"] == row["ts"][:10] and row["via"] == "cli"
    assert row["export"].endswith("_Finish/10.13.26 - Test/DSC_0001.jpg")


def test_a_second_accept_never_deletes_the_first_pre_helper(lib, capsys):
    make_candidate(lib, capsys, recipe=False)
    first_export = lib["export"].read_bytes()
    run(lib, "accept", FRAME, "--apply", capsys=capsys)
    code, out = run(lib, "apply", FRAME, "--params", "shadows=0.2", "--apply", capsys=capsys)
    assert code == 0, out
    middle_export = lib["export"].read_bytes()
    code, out = run(lib, "accept", FRAME, "--apply", capsys=capsys)
    assert code == 0, out
    assert (lib["finish"] / "DSC_0001_pre-helper.jpg").read_bytes() == first_export
    assert (lib["finish"] / "DSC_0001_pre-helper-2.jpg").read_bytes() == middle_export
    assert [r["recipe"] for r in rows(lib)] == [None, None]


def test_reject_deletes_only_the_candidate_and_writes_the_event(lib, capsys):
    cand = make_candidate(lib, capsys)
    export_sha = sha(lib["export"])
    before = snap(lib["outing"])
    names = {p.name for p in lib["finish"].iterdir()}
    code, out = run(lib, "reject", FRAME, capsys=capsys)
    assert code == 0 and "DRY RUN" in out and cand.exists()
    code, out = run(lib, "reject", FRAME, "--apply", capsys=capsys)
    assert code == 0, out
    assert not cand.exists()
    assert sha(lib["export"]) == export_sha and snap(lib["outing"]) == before
    assert {p.name for p in lib["finish"].iterdir()} == names - {cand.name}
    (row,) = rows(lib)
    assert row["kind"] == "tweak" and row["accepted"] is False and row["recipe"] == "blown-sky"


def test_accept_refuses_a_candidate_made_from_a_different_export(lib, capsys):
    make_candidate(lib, capsys)
    synthetic(lib["export"], sky=0.9, land=0.3)                       # he re-exported meanwhile
    os.utime(lib["export"], (T0 + 2_000_000, T0 + 2_000_000))
    snap_before = snap(lib["finish"])
    code, out = run(lib, "accept", FRAME, "--apply", capsys=capsys)
    assert code == 3 and "changed" in out
    assert snap(lib["finish"]) == snap_before and rows(lib) == []


def test_accept_and_reject_need_a_candidate(lib, capsys):
    for verb in ("accept", "reject"):
        code, out = run(lib, verb, FRAME, "--apply", capsys=capsys)
        assert code == 3 and "no candidate" in out
    assert rows(lib) == []


def test_accept_rolls_back_when_the_event_cannot_be_written(lib, capsys):
    cand = make_candidate(lib, capsys)
    export_bytes, cand_bytes = lib["export"].read_bytes(), cand.read_bytes()
    shutil.rmtree(lib["events"])
    lib["events"].write_text("a file where the bus directory should be")
    code, out = run(lib, "accept", FRAME, "--apply", capsys=capsys)
    assert code == 2
    assert lib["export"].read_bytes() == export_bytes and cand.read_bytes() == cand_bytes
    assert not list(lib["finish"].glob("*pre-helper*"))


def test_a_recipe_earns_a_hit_rate_from_the_events(lib, capsys):
    make_candidate(lib, capsys)
    run(lib, "reject", FRAME, "--apply", capsys=capsys)
    run(lib, "apply", FRAME, "--recipe", "blown-sky", "--apply", capsys=capsys)
    run(lib, "accept", FRAME, "--apply", capsys=capsys)
    code, out = run(lib, "suggest", FRAME, "--note", "sky blown", capsys=capsys)
    assert "1/2 accepted (50%)" in out
    code, out = run(lib, "recipe", "list", "--json", capsys=capsys)
    assert json.loads(out)[0]["hits"] == 1 and json.loads(out)[0]["decided"] == 2


def test_recipe_save_after_an_accept_measures_the_pre_helper_state(lib, capsys):
    make_candidate(lib, capsys, recipe=False)
    pre_measure = tweak().measure_file(lib["export"])
    run(lib, "accept", FRAME, "--apply", capsys=capsys)
    assert tweak().measure_file(lib["export"])["gap"] < pre_measure["gap"] - 0.05
    save_recipe(lib, capsys, name="after")
    lo, hi = saved(lib)[0]["trigger"]["gap"]
    assert lo <= pre_measure["gap"] <= hi


def test_the_event_goes_through_events_module_when_one_is_set(lib, capsys):
    if not SCRIPTS_EVENTS.is_file():
        pytest.skip("Scripts lib/studio_events.py not on this machine")
    cfg = json.loads(lib["settings"].read_text(encoding="utf-8"))
    cfg["events_module"] = str(SCRIPTS_EVENTS)
    lib["settings"].write_text(json.dumps(cfg), encoding="utf-8")
    make_candidate(lib, capsys)
    run(lib, "accept", FRAME, "--apply", capsys=capsys)
    (row,) = rows(lib)
    assert row["kind"] == "tweak" and row["actor"] == "studio-picker" and row["accepted"] is True


# ------------------------------------------------------------------ finish_watch stays honest

def test_helper_files_are_never_counted_as_his_export(lib):
    f = finish()
    folder = lib["tmp"] / "f"
    folder.mkdir()
    src = lib["master"]
    for name in ("DSC_0001.jpg", "DSC_0001_helper.jpg", "DSC_0001_pre-helper.jpg", "DSC_0001_pre-helper-2.jpg",
                 "DSC_0001-Edit_helper.jpg"):
        synthetic(folder / name)
        os.utime(folder / name, (T0 + 9_999_999, T0 + 9_999_999))
    got = f.find_export(folder, "DSC_0001.jpg", src.stat().st_mtime_ns, None)
    assert got[0].name == "DSC_0001.jpg"
    (folder / "DSC_0001.jpg").unlink()
    assert f.find_export(folder, "DSC_0001.jpg", src.stat().st_mtime_ns, None) is None


# ------------------------------------------------------------------ review fixes

def test_a_frame_id_with_dotdot_or_absolute_is_refused_for_every_verb(lib, capsys):
    top = lib["root"] / "top.jpg"                   # what Hiking/../top.jpg would reach
    synthetic(top)
    os.utime(top, (T0 + 5_000_000, T0 + 5_000_000))
    make_candidate(lib, capsys)
    before = snap(lib["tmp"])
    for bad in ("Hiking/../top.jpg", "../top.jpg", str(lib["master"]), "C:/Windows/x.jpg", "/etc/x.jpg"):
        for verb, extra in (("accept", ["--apply"]), ("reject", ["--apply"]), ("measure", []),
                            ("suggest", []), ("apply", ["--params", "sky_pull=0.3", "--apply"])):
            code, out = run(lib, verb, bad, *extra, capsys=capsys)
            assert code == 3 and "refusing frame" in out, (verb, bad, out)
    assert snap(lib["tmp"]) == before
    assert rows(lib) == []


def test_the_dotdot_check_also_covers_an_explicit_export_flag(lib, capsys):
    code, out = run(lib, "apply", "Hiking/../top.jpg", "--export", str(lib["export"]),
                    "--params", "sky_pull=0.3", "--apply", capsys=capsys)
    assert code == 3 and "refusing frame" in out


def test_a_candidate_outside_the_finish_folder_is_refused(lib):
    t = tweak()
    f = t.Found("x", None, lib["finish"], lib["root"] / "elsewhere" / "a.jpg", None, lib["root"] / "_Finish")
    with pytest.raises(t.Refused):
        t.need_export(f)
    with pytest.raises(t.Refused):
        t.need_candidate(f, lib["root"] / "elsewhere" / "a.jpg")


def test_accept_resolves_the_bus_before_any_rename(lib, capsys, monkeypatch):
    cand = make_candidate(lib, capsys)
    (lib["tmp"] / "_vault_paths.json").write_text("{ this is not json", encoding="utf-8")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("MAYNIAC_STUDIO_EVENTS", raising=False)
    monkeypatch.setenv("VAULT_PATH", str(lib["tmp"]))
    before = snap(lib["finish"])
    code = tweak().main(["accept", FRAME, "--settings", str(lib["settings"]), "--recipes", str(lib["recipes"]),
                         "--apply"])
    assert code == 2
    assert "error" in capsys.readouterr().out
    assert snap(lib["finish"]) == before               # nothing renamed, candidate still there
    assert cand.exists() and not list(lib["finish"].glob("*pre-helper*"))


def test_accept_rolls_back_on_any_exception_not_just_oserror(lib, capsys, monkeypatch):
    cand = make_candidate(lib, capsys)
    before = snap(lib["finish"])
    t = tweak()

    def boom(*a, **k):
        raise ValueError("bus exploded")
    monkeypatch.setattr(t.link, "append_event", boom)
    code = t.main(["accept", FRAME, "--settings", str(lib["settings"]), "--events-dir", str(lib["events"]),
                   "--recipes", str(lib["recipes"]), "--apply"])
    assert code == 2 and "bus exploded" in capsys.readouterr().out
    assert snap(lib["finish"]) == before and cand.exists()


def test_reject_that_cannot_delete_writes_no_event(lib, capsys, monkeypatch):
    cand = make_candidate(lib, capsys)
    t = tweak()

    def locked(self, *a, **k):
        raise PermissionError("locked by Lightroom")
    monkeypatch.setattr(Path, "unlink", locked)
    code = t.main(["reject", FRAME, "--settings", str(lib["settings"]), "--events-dir", str(lib["events"]),
                   "--recipes", str(lib["recipes"]), "--apply"])
    assert code == 2 and "locked" in capsys.readouterr().out
    assert cand.exists() and rows(lib) == []


def test_reject_deletes_first_then_writes_the_event(lib, capsys, monkeypatch):
    cand = make_candidate(lib, capsys)
    t = tweak()
    seen = {}
    real = t.link.append_event

    def spy(*a, **k):
        seen["candidate_existed_at_event"] = cand.exists()
        return real(*a, **k)
    monkeypatch.setattr(t.link, "append_event", spy)
    assert t.main(["reject", FRAME, "--settings", str(lib["settings"]), "--events-dir", str(lib["events"]),
                   "--recipes", str(lib["recipes"]), "--apply"]) == 0
    assert seen == {"candidate_existed_at_event": False}


def test_apply_refuses_a_non_jpeg_export(lib, capsys):
    png = lib["finish"] / "DSC_0001.png"
    Image.open(lib["export"]).save(png)
    lib["export"].unlink()
    os.utime(png, (T0 + 3_000_000, T0 + 3_000_000))
    before = snap(lib["tmp"])
    code, out = run(lib, "apply", FRAME, "--params", "sky_pull=0.3", "--apply", capsys=capsys)
    assert code == 3 and "JPEG" in out and ".png" in out
    assert snap(lib["tmp"]) == before


def test_params_on_top_of_a_recipe_make_it_a_modified_candidate_and_spare_the_hit_rate(lib, capsys):
    save_recipe(lib, capsys)
    code, out = run(lib, "apply", FRAME, "--recipe", "blown-sky", "--params", "sky_pull=0.2", "--apply",
                    capsys=capsys)
    assert code == 0 and "modified from recipe blown-sky" in out
    prov = tweak().read_provenance(lib["finish"] / "DSC_0001_helper.jpg")
    assert prov["recipe"] is None and prov["modified_from"] == "blown-sky" and prov["params"]["sky_pull"] == 0.2
    assert prov["params"]["highlights"] == 0.2                       # the rest still came from the recipe
    run(lib, "accept", FRAME, "--apply", capsys=capsys)
    (row,) = rows(lib)
    assert row["recipe"] is None and row["modified_from"] == "blown-sky" and row["accepted"] is True
    code, out = run(lib, "suggest", FRAME, "--note", "sky blown", capsys=capsys)
    assert "no decisions yet" in out                                 # blown-sky earned nothing from it


def test_a_plain_recipe_apply_has_no_modified_from(lib, capsys):
    make_candidate(lib, capsys)
    run(lib, "accept", FRAME, "--apply", capsys=capsys)
    (row,) = rows(lib)
    assert row["recipe"] == "blown-sky" and "modified_from" not in row


def test_apply_says_when_it_replaces_an_undecided_candidate(lib, capsys):
    code, out = run(lib, "apply", FRAME, "--params", "sky_pull=0.3", "--apply", capsys=capsys)
    assert "replacing an undecided candidate" not in out
    code, out = run(lib, "apply", FRAME, "--params", "sky_pull=0.2", "--apply", capsys=capsys)
    assert code == 0 and "replacing an undecided candidate" in out


def test_suggest_json_carries_the_export_and_the_exact_candidate_path(lib, capsys):
    save_recipe(lib, capsys)
    code, out = run(lib, "suggest", FRAME, "--json", capsys=capsys)
    j = json.loads(out)
    assert code == 0
    assert set(j) == {"frame", "measure", "export", "candidate", "candidate_exists", "matches"}
    assert j["export"] == str(lib["export"]) and j["candidate"] == str(lib["finish"] / "DSC_0001_helper.jpg")
    assert j["candidate_exists"] is False
    run(lib, "apply", FRAME, "--params", "sky_pull=0.3", "--apply", capsys=capsys)
    code, out = run(lib, "suggest", FRAME, "--json", capsys=capsys)
    assert json.loads(out)["candidate_exists"] is True
    assert Path(json.loads(out)["candidate"]).is_file()


def test_suggest_json_export_and_candidate_are_null_before_there_is_an_export(lib, capsys):
    lib["export"].unlink()
    code, out = run(lib, "suggest", FRAME, "--json", capsys=capsys)
    j = json.loads(out)
    assert code == 0 and j["export"] is None and j["candidate"] is None and j["candidate_exists"] is False
    assert j["matches"] == []


def test_suggest_json_error_is_an_error_object_with_a_nonzero_exit(lib, capsys):
    code, out = run(lib, "suggest", "Hiking/../top.jpg", "--json", capsys=capsys)
    assert code == 3 and "error" in json.loads(out) and "matches" not in json.loads(out)
    code, out = run(lib, "suggest", "No/Such/Frame.jpg", "--json", capsys=capsys)
    assert code == 3 and "error" in json.loads(out)
    lib["settings"].write_text("{ not json", encoding="utf-8")
    code, out = run(lib, "suggest", FRAME, "--json", capsys=capsys)
    assert code == 2 and "error" in json.loads(out)
    with pytest.raises(SystemExit) as e:                              # a usage error is JSON too
        tweak().main(["suggest", "--json"])
    assert e.value.code == 2 and "error" in json.loads(capsys.readouterr().out)


def test_a_no_match_is_not_an_error(lib, capsys):
    code, out = run(lib, "suggest", FRAME, "--json", capsys=capsys)
    j = json.loads(out)
    assert code == 0 and "error" not in j and j["matches"] == []


@pytest.mark.parametrize("style", ["equals", "separate"])
def test_note_text_that_starts_with_a_dash_is_a_value(lib, capsys, style):
    save_recipe(lib, capsys, extra=("--trigger", "gap=0.0:0.1"))      # trigger misses: only words can match
    note = "-sky is blown"
    args = ["--note=" + note] if style == "equals" else ["--note", note]
    code, out = run(lib, "suggest", FRAME, *args, capsys=capsys)
    assert code == 0 and "blown-sky" in out and "words: blown, sky" in out


def test_symptom_and_step_text_that_starts_with_a_dash_is_a_value(lib, capsys):
    code, out = run(lib, "recipe", "save", "dashy", "--frame", FRAME, "--symptom", "-sky blown",
                    "--step", "-1 stop on the sky", "--apply", capsys=capsys)
    assert code == 0, out
    r = saved(lib)[0]
    assert r["symptom"] == "-sky blown" and r["lightroom_steps"] == ["-1 stop on the sky"]
