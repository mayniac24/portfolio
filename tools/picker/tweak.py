"""Finish helper tools and recipes (DESIGN-2026-10-17 D16, 2.3 Help).

A helper for ONE frame he is stuck on, run only when he asks on that frame (D5). It is
not a tone pass and it never runs on a schedule. No model, no network: PIL and numpy.

    tweak.py measure <frame> [--json]
    tweak.py suggest <frame> [--note "his words"] [--json]
    tweak.py apply   <frame> (--recipe NAME | --params k=v ...) [--apply]
    tweak.py accept  <frame> [--apply] [--force]
    tweak.py reject  <frame> [--apply]
    tweak.py recipe save NAME --frame F --symptom "..." --step "..." [--params k=v ...] [--apply]
    tweak.py recipe edit NAME [...] [--apply]
    tweak.py recipe retire NAME [--apply]
    tweak.py recipe list [--json]

`<frame>` is a frame id (path relative to photos_root, as the bus names it), or an absolute
path to the export inside the finish folder. The image the helper works on is his EXPORT in
`<photos_root>/<finish_dir>/<outing>/`; `apply` writes only `<export stem>_helper.jpg` beside
it. The master, the outing folder, `Portfolio Selections` and `Portfolio Corrected` are never
touched (every write is guarded to sit under the finish folder).

`accept` replaces the export with the candidate and keeps the replaced export as
`<stem>_pre-helper.<ext>` (`-2`, `-3` ... when that name is taken; nothing is ever deleted).
`reject` deletes the candidate and nothing else. Both write one `tweak` event to
`studio-picker.jsonl` through studio_link:

    {"kind": "tweak", "frame": <id>, "recipe": <name or null>, "accepted": true|false,
     "export": <path>, "outing": <id>, ...bus envelope}

A recipe earns a hit rate from those rows (accepted / decided). `recipes.yaml` has exactly one
writer: `recipe save|edit|retire`. Every verb that writes is a dry run without `--apply`.

Exit codes: 0 ok, 2 usage or environment, 3 refused (nothing found, stale, name taken).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import studio_link as link  # noqa: E402
from finish_watch import find_export  # noqa: E402
from stage_picks import read_manifest  # noqa: E402

try:
    import numpy as np
    from PIL import Image, ImageOps
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tweak.py needs pillow, numpy and pyyaml (pip install -r requirements.txt): %s" % exc)

RECIPES_FILE = HERE / "recipes.yaml"
HEADER = ("# Finish-helper recipes (DESIGN-2026-10-17 D16). Data, not code.\n"
          "# Exactly one writer: `python tools/picker/tweak.py recipe save|edit|retire --apply`.\n"
          "# Do not hand-edit; comments are rewritten on every save.\n")
CANDIDATE_SUFFIX = "_helper"
PRE_SUFFIX = "_pre-helper"
METRICS = ("sky", "land", "gap", "highlight_clip", "shadow_clip")
SKY_FRAC, LAND_FRAC = 0.35, 0.40          # top third is sky, bottom 40 percent is land, for measuring
CLIP_HI, CLIP_LO = 0.98, 0.02             # a channel at or above / luma at or below these is clipped

#: Every script parameter, its (min, max) and the value that means "off". Out of range is refused.
BOUNDS = {
    "highlights": (0.0, 0.30),    # luminosity-masked pull of the brightest tones
    "shadows": (0.0, 0.30),       # luminosity-masked lift of the darkest tones
    "dehaze": (0.0, 0.30),        # black-point lift toward the frame's own darkest percentile
    "sky_pull": (0.0, 0.40),      # graduated, luminosity-weighted darkening of the top of the frame
    "sky_height": (0.20, 0.80),   # where the graduated mask ends, as a fraction of frame height
    "sky_feather": (0.05, 0.50),  # how softly it ends
}
DEFAULT = {"highlights": 0.0, "shadows": 0.0, "dehaze": 0.0, "sky_pull": 0.0,
           "sky_height": 0.45, "sky_feather": 0.20}
EFFECTS = ("highlights", "shadows", "dehaze", "sky_pull")
STOP = set("the and for with that this are was but not its from has have too very just more some into than "
           "then them they his her out all can get got one off over under about frame photo picture image".split())


# ---------------------------------------------------------------- pixels

def _load_rgb(path: Path):
    """8-bit RGB float array in 0..1 (EXIF orientation applied) plus the icc profile and exif."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        icc = im.info.get("icc_profile")
        try:
            exif = im.getexif()
            if 0x0112 in exif:
                exif[0x0112] = 1
            exif_bytes = exif.tobytes() if len(exif) else None
        except Exception:  # noqa: BLE001
            exif_bytes = None
        rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return arr, icc, exif_bytes


def _luma(a):
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def measure_array(a) -> dict:
    h = a.shape[0]
    lum = _luma(a)
    sky_rows = max(1, int(h * SKY_FRAC))
    land_rows = max(1, int(h * LAND_FRAC))
    sky = float(lum[:sky_rows].mean())
    land = float(lum[h - land_rows:].mean())
    return {"sky": round(sky, 4), "land": round(land, 4), "gap": round(sky - land, 4),
            "highlight_clip": round(float((a.max(axis=2) >= CLIP_HI).mean()), 4),
            "shadow_clip": round(float((lum <= CLIP_LO).mean()), 4)}


def measure_file(path: Path) -> dict:
    return measure_array(_load_rgb(path)[0])


def apply_params(a, p: dict):
    """The conservative edit. Each term is bounded (see BOUNDS) and monotone in luminance
    where it is a tone curve, so it can lift or pull but never invert tones."""
    out = a.copy()
    if p["dehaze"] > 0:
        black = float(np.percentile(_luma(out), 1.0)) * p["dehaze"]
        out = np.clip((out - black) / max(1.0 - black, 1e-6), 0.0, 1.0)
    if p["highlights"] > 0:
        m = _smooth(0.5, 1.0, _luma(out))[..., None]
        out = out * (1.0 - p["highlights"] * m)
    if p["shadows"] > 0:
        w = (1.0 - _smooth(0.0, 0.5, _luma(out)))[..., None]
        out = out + p["shadows"] * w * (1.0 - out)
    if p["sky_pull"] > 0:
        h = out.shape[0]
        t = (np.arange(h, dtype=np.float32) + 0.5) / h
        end = p["sky_height"]
        grad = (1.0 - _smooth(max(0.0, end - p["sky_feather"]), end, t))[:, None]
        w = _smooth(0.25, 0.75, _luma(out))
        out = out * (1.0 - p["sky_pull"] * grad * w)[..., None]
    return np.clip(out, 0.0, 1.0)


def write_candidate(export: Path, cand: Path, params: dict, recipe) -> None:
    a, icc, exif = _load_rgb(export)
    out = apply_params(a, params)
    img = Image.fromarray(np.rint(out * 255.0).astype(np.uint8), "RGB")
    note = json.dumps({"tweak": 1, "recipe": recipe, "params": params,
                       "export_sha1": link.sha1_file(export)}, sort_keys=True).encode("ascii")
    part = cand.with_name(cand.name + ".part")
    kw = {"quality": 95, "subsampling": 0, "comment": note}
    if icc:
        kw["icc_profile"] = icc
    if exif:
        kw["exif"] = exif
    try:
        img.save(part, "JPEG", **kw)
        os.replace(part, cand)
    finally:
        if part.exists():
            part.unlink()


def read_provenance(cand: Path) -> dict:
    try:
        with Image.open(cand) as im:
            raw = im.info.get("comment") or b""
        data = json.loads(raw.decode("ascii", "replace"))
        return data if isinstance(data, dict) and data.get("tweak") == 1 else {}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------- params and triggers

def parse_params(pairs) -> dict:
    out = {}
    for pair in pairs or []:
        for token in str(pair).replace(",", " ").split():
            if "=" not in token:
                raise ValueError("parameter %r is not k=v" % token)
            k, v = token.split("=", 1)
            if k not in BOUNDS:
                raise ValueError("unknown parameter %r (known: %s)" % (k, ", ".join(BOUNDS)))
            try:
                val = float(v)
            except ValueError:
                raise ValueError("parameter %s=%r is not a number" % (k, v))
            lo, hi = BOUNDS[k]
            if not lo <= val <= hi:
                raise ValueError("parameter %s=%s is out of bounds [%s, %s]" % (k, v, lo, hi))
            out[k] = val
    return out


def parse_trigger(pairs) -> dict:
    out = {}
    for pair in pairs or []:
        m = re.fullmatch(r"(\w+)=(-?[\d.]*):(-?[\d.]*)", str(pair).strip())
        if not m or m.group(1) not in METRICS:
            raise ValueError("trigger %r is not metric=lo:hi (metrics: %s)" % (pair, ", ".join(METRICS)))
        lo = float(m.group(2)) if m.group(2) else None
        hi = float(m.group(3)) if m.group(3) else None
        out[m.group(1)] = [lo, hi]
    return out


def trigger_from_measure(meas: dict, margin: float) -> dict:
    out = {}
    for k in METRICS:
        v = meas[k]
        pad = max(0.02, 0.5 * v) if k.endswith("_clip") else margin
        lo = max(0.0 if k != "gap" else -1.0, v - pad)
        out[k] = [round(lo, 4), round(min(1.0, v + pad), 4)]
    return out


def trigger_fraction(trigger: dict, meas: dict):
    """Share of the trigger's metrics the measurement falls inside; None when there is no trigger."""
    if not trigger:
        return None
    hit = 0
    for k, (lo, hi) in trigger.items():
        v = meas.get(k)
        if v is not None and (lo is None or v >= lo) and (hi is None or v <= hi):
            hit += 1
    return hit / len(trigger)


def words(text) -> set:
    out = set()
    for w in re.findall(r"[a-z']+", str(text or "").lower()):
        if len(w) < 3 or w in STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


# ---------------------------------------------------------------- recipes (the one writer)

def recipes_path(settings, explicit=None) -> Path:
    if explicit:
        return Path(explicit)
    return Path(settings["picker_tools"].get("recipes") or RECIPES_FILE)


def load_recipes(path: Path) -> list:
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = data.get("recipes") if isinstance(data, dict) else None
    return [r for r in (rows or []) if isinstance(r, dict) and r.get("name")]


def write_recipes(path: Path, recipes: list) -> None:
    text = HEADER + yaml.safe_dump({"recipes": recipes}, sort_keys=False, allow_unicode=True, width=100)
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8", newline="\n")
    os.replace(part, path)


def hit_rates(settings, events_dir) -> dict:
    """{recipe: [accepted, decided]} from the tweak events."""
    out = {}
    try:
        rows = link.read_own_events(settings, events_dir)
    except RuntimeError:
        rows = []
    for r in rows:
        if r.get("kind") == "tweak" and r.get("recipe"):
            acc = out.setdefault(r["recipe"], [0, 0])
            acc[1] += 1
            acc[0] += 1 if r.get("accepted") else 0
    return out


def rate_text(rate) -> str:
    if not rate or not rate[1]:
        return "no decisions yet"
    return "%d/%d accepted (%d%%)" % (rate[0], rate[1], round(100 * rate[0] / rate[1]))


# ---------------------------------------------------------------- finding the frame

class Refused(Exception):
    """Nothing found, stale, name taken: exit 3."""


def finish_base(settings, root: Path) -> Path:
    base = Path(settings.get("finish_dir") or "_Finish")
    return base if base.is_absolute() else root / base


def _under(path: Path, base: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(base)))
        return True
    except ValueError:
        return False


class Found:
    def __init__(self, frame, master, folder, export, manifest_outing):
        self.frame, self.master, self.folder = frame, master, folder
        self.export, self.manifest_outing = export, manifest_outing


def locate(a, settings, root: Path) -> Found:
    base = finish_base(settings, root)
    arg = str(a.frame)
    explicit = Path(a.export) if getattr(a, "export", None) else None
    as_path = Path(arg)
    if explicit is None and as_path.is_absolute() and as_path.is_file() and _under(as_path, base):
        explicit = as_path
    if explicit is not None:
        if not explicit.is_file():
            raise Refused("export not found: %s" % explicit)
        if not _under(explicit, base):
            raise Refused("refusing: %s is not inside the finish folder %s" % (explicit, base))
        folder = explicit.parent
        man = read_manifest(folder)
        fid = arg.replace("\\", "/")
        if as_path.is_absolute():
            fid = link.rel_or_abs(explicit, root)
            for k, m in man["frames"].items():
                stem = Path(m.get("name", "")).stem.lower()
                s = explicit.stem.lower()
                if s == stem or s.startswith(stem + "-"):
                    fid = k
                    break
        master = root / fid
        return Found(fid, master if master.is_file() else None, folder, explicit, man["outing"])

    fid = arg.replace("\\", "/").strip("/")
    master = root / fid
    folder, entry, man_outing = None, None, None
    if base.is_dir():
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            man = read_manifest(sub)
            if fid in man["frames"]:
                folder, entry, man_outing = sub, man["frames"][fid], man["outing"]
                break
    if folder is None:
        for part in Path(fid).parent.parts[::-1]:
            if (base / part).is_dir():
                folder = base / part
                break
    if folder is None:
        raise Refused("no finish folder for %s (stage it first: stage_picks.py)" % fid)
    export = None
    if entry:
        export = find_export(folder, entry["name"], entry["src_mtime_ns"], entry.get("sha1"))
    elif master.is_file():
        export = find_export(folder, master.name, master.stat().st_mtime_ns, None)
    return Found(fid, master if master.is_file() else None, folder, export[0] if export else None, man_outing)


def need_export(f: Found) -> Path:
    if f.export is None:
        raise Refused("no export yet for %s in %s: finish it in Lightroom first (D5: the helper "
                      "works on his export, never the master)" % (f.frame, f.folder))
    return f.export


def candidate_of(export: Path) -> Path:
    return export.with_name(export.stem + CANDIDATE_SUFFIX + ".jpg")


def first_pre(export: Path):
    for n in range(1, 50):
        p = export.with_name(export.stem + PRE_SUFFIX + ("" if n == 1 else "-%d" % n) + export.suffix)
        if p.is_file():
            return p
    return None


def free_pre(export: Path) -> Path:
    n = 1
    while True:
        p = export.with_name(export.stem + PRE_SUFFIX + ("" if n == 1 else "-%d" % n) + export.suffix)
        if not p.exists():
            return p
        n += 1


def measure_source(f: Found, prefer_pre=False) -> Path:
    if prefer_pre and f.export is not None:
        pre = first_pre(f.export)
        if pre:
            return pre
    if f.export is not None:
        return f.export
    if f.master is not None:
        try:
            with Image.open(f.master):
                return f.master
        except Exception:  # noqa: BLE001
            pass
    raise Refused("nothing to measure for %s: no export, and the master is not a readable still" % f.frame)


# ---------------------------------------------------------------- verbs

def _env(a):
    settings = link.load_settings(a.settings)
    root = link.photos_root(settings, a.photos_root)
    return settings, root


def cmd_measure(a):
    settings, root = _env(a)
    f = locate(a, settings, root)
    src = measure_source(f)
    meas = measure_file(src)
    if a.json:
        print(json.dumps({"frame": f.frame, "source": str(src), **meas}))
        return 0
    print("frame %s  (%s)" % (f.frame, "export" if src == f.export else "master"))
    print("  sky luminance        %.3f   (top %d%% of the frame)" % (meas["sky"], SKY_FRAC * 100))
    print("  land luminance       %.3f   (bottom %d%%)" % (meas["land"], LAND_FRAC * 100))
    print("  gap (sky - land)     %+.3f" % meas["gap"])
    print("  highlight clipping   %.2f%%  (a channel at or above %.0f/255)" % (100 * meas["highlight_clip"], CLIP_HI * 255))
    print("  shadow clipping      %.2f%%  (luma at or below %.0f/255)" % (100 * meas["shadow_clip"], CLIP_LO * 255))
    return 0


def cmd_suggest(a):
    settings, root = _env(a)
    f = locate(a, settings, root)
    meas = measure_file(measure_source(f))
    rates = hit_rates(settings, a.events_dir)
    note = words(a.note)
    found = []
    for r in load_recipes(recipes_path(settings, a.recipes)):
        if r.get("retired"):
            continue
        frac = trigger_fraction(r.get("trigger") or {}, meas)
        sym = words(r.get("symptom"))
        shared = sorted(note & sym)
        trig_ok = frac == 1.0
        if not (trig_ok or shared):
            continue
        score = (1.0 if trig_ok else 0.0) + (len(shared) / len(sym) if sym else 0.0)
        rate = rates.get(r["name"])
        found.append((-score, -(rate[0] / rate[1] if rate and rate[1] else 0.0), r["name"], r, frac, shared, rate))
    found.sort(key=lambda t: t[:3])
    if a.json:
        print(json.dumps({"frame": f.frame, "measure": meas, "matches": [
            {"name": r["name"], "trigger_match": frac == 1.0 if frac is not None else None,
             "words": shared, "hits": rate[0] if rate else 0, "decided": rate[1] if rate else 0,
             "symptom": r.get("symptom"), "lightroom_steps": r.get("lightroom_steps") or [],
             "params": r.get("params") or {}, "origin_frame": r.get("origin_frame")}
            for _, _, _, r, frac, shared, rate in found]}))
        return 0
    if not found:
        print("no recipe matches %s (sky %.3f, land %.3f, gap %+.3f)" % (f.frame, meas["sky"], meas["land"], meas["gap"]))
        return 0
    print("%d recipe(s) match %s (sky %.3f, land %.3f, gap %+.3f):" % (len(found), f.frame, meas["sky"], meas["land"], meas["gap"]))
    for _, _, name, r, frac, shared, rate in found:
        why = []
        if frac is not None:
            why.append("trigger %s" % ("matches" if frac == 1.0 else "%d%% of ranges" % round(frac * 100)))
        if shared:
            why.append("words: " + ", ".join(shared))
        print("- %s  [%s]  %s" % (name, "; ".join(why), rate_text(rate)))
        print("    symptom: %s" % r.get("symptom", ""))
        for i, s in enumerate(r.get("lightroom_steps") or [], 1):
            print("    %d. %s" % (i, s))
        if r.get("params"):
            print("    params: " + " ".join("%s=%s" % kv for kv in r["params"].items()))
    return 0


def cmd_apply(a):
    settings, root = _env(a)
    f = locate(a, settings, root)
    export = need_export(f)
    params, recipe = dict(DEFAULT), None
    if a.recipe:
        rows = {r["name"]: r for r in load_recipes(recipes_path(settings, a.recipes))}
        if a.recipe not in rows:
            raise Refused("no recipe named %r" % a.recipe)
        if rows[a.recipe].get("retired"):
            raise Refused("recipe %r is retired" % a.recipe)
        params.update(parse_params(["%s=%s" % kv for kv in (rows[a.recipe].get("params") or {}).items()]))
        recipe = a.recipe
    params.update(parse_params(a.params))
    if not any(params[k] > 0 for k in EFFECTS):
        raise ValueError("no effect requested: set at least one of %s above 0" % ", ".join(EFFECTS))
    cand = candidate_of(export)
    shown = " ".join("%s=%s" % (k, params[k]) for k in BOUNDS)
    print("candidate: %s" % cand)
    print("from export: %s" % export)
    print("params: %s%s" % (shown, "  (recipe %s)" % recipe if recipe else ""))
    if not a.apply:
        print("DRY RUN - pass --apply to write the candidate (the export is never changed)")
        return 0
    if not _under(cand, finish_base(settings, root)):
        raise Refused("refusing: candidate would sit outside the finish folder")
    write_candidate(export, cand, params, recipe)
    before, after = measure_file(export), measure_file(cand)
    print("wrote %s" % cand)
    print("  gap %+.3f -> %+.3f   sky %.3f -> %.3f   land %.3f -> %.3f   highlight clip %.2f%% -> %.2f%%" % (
        before["gap"], after["gap"], before["sky"], after["sky"], before["land"], after["land"],
        100 * before["highlight_clip"], 100 * after["highlight_clip"]))
    print("accept: tweak.py accept %s --apply   reject: tweak.py reject %s --apply" % (f.frame, f.frame))
    return 0


def _outing_id(a, settings, root, f: Found):
    if a.outing_id:
        return a.outing_id
    if f.manifest_outing:
        return f.manifest_outing
    od = link.outing_dir_of_frame(f.master, f.folder.name) if f.master else None
    return link.outing_id_for(settings, root, od, a.events_dir) if od else None


def _event(a, settings, root, f, recipe, accepted, export):
    link.append_event(settings, "tweak", outing=_outing_id(a, settings, root, f), via=a.via,
                      member=a.by or None, events_dir=a.events_dir, frame=f.frame,
                      recipe=recipe, accepted=accepted, export=link.rel_or_abs(export, root))


def cmd_accept(a):
    settings, root = _env(a)
    f = locate(a, settings, root)
    export = need_export(f)
    cand = candidate_of(export)
    if not cand.is_file():
        raise Refused("no candidate %s: run apply first" % cand.name)
    prov = read_provenance(cand)
    recipe = a.recipe or prov.get("recipe")
    if prov.get("export_sha1") and prov["export_sha1"] != link.sha1_file(export) and not a.force:
        raise Refused("the export changed since the candidate was made; re-run apply (or --force)")
    new = export.with_name(export.stem + ".jpg")
    if new != export and new.exists():
        raise Refused("refusing: %s already exists" % new.name)
    pre = free_pre(export)
    print("export %s -> kept as %s; candidate becomes %s" % (export.name, pre.name, new.name))
    if not a.apply:
        print("DRY RUN - pass --apply to swap the export and write the accept event")
        return 0
    os.rename(export, pre)
    try:
        os.rename(cand, new)
    except OSError:
        os.rename(pre, export)
        raise
    try:
        _event(a, settings, root, f, recipe, True, new)
    except (RuntimeError, OSError, KeyError, FileNotFoundError):
        os.rename(new, cand)
        os.rename(pre, export)
        raise
    print("accepted: %s is the export; the replaced one is %s" % (new.name, pre.name))
    return 0


def cmd_reject(a):
    settings, root = _env(a)
    f = locate(a, settings, root)
    export = need_export(f)
    cand = candidate_of(export)
    if not cand.is_file():
        raise Refused("no candidate %s to reject" % cand.name)
    recipe = a.recipe or read_provenance(cand).get("recipe")
    print("delete candidate %s (the export stays)" % cand.name)
    if not a.apply:
        print("DRY RUN - pass --apply to delete it and write the reject event")
        return 0
    _event(a, settings, root, f, recipe, False, export)
    cand.unlink()
    print("rejected")
    return 0


def _recipe_fields(a, settings, root, current=None):
    """The fields a save or edit sets, validated; keys not given are left out."""
    out = {}
    if a.symptom is not None:
        if not words(a.symptom):
            raise ValueError("--symptom needs his words for what is wrong")
        out["symptom"] = a.symptom.strip()
    if a.step:
        out["lightroom_steps"] = [s.strip() for s in a.step if s.strip()]
    if a.params:
        out["params"] = parse_params(a.params)
    trig = parse_trigger(a.trigger) if a.trigger else {}
    if a.frame:
        f = locate(argparse.Namespace(frame=a.frame, export=None), settings, root)
        out["origin_frame"] = f.frame
        measured = trigger_from_measure(measure_file(measure_source(f, prefer_pre=True)), a.margin)
        out["trigger"] = {**measured, **trig}
    elif trig:
        out["trigger"] = {**((current or {}).get("trigger") or {}), **trig}
    return out


ORDER = ("name", "symptom", "trigger", "lightroom_steps", "params", "origin_frame", "created", "retired")


def cmd_recipe(a):
    settings, root = _env(a)
    path = recipes_path(settings, a.recipes)
    recipes = load_recipes(path)
    if a.verb == "list":
        rates = hit_rates(settings, a.events_dir)
        if a.json:
            print(json.dumps([{**r, "hits": (rates.get(r["name"]) or [0, 0])[0],
                               "decided": (rates.get(r["name"]) or [0, 0])[1]} for r in recipes]))
            return 0
        for r in recipes:
            print("%s%s  %s  | %s" % (r["name"], " (retired)" if r.get("retired") else "",
                                      rate_text(rates.get(r["name"])), r.get("symptom", "")))
        if not recipes:
            print("no recipes yet")
        return 0
    by_name = {r["name"]: r for r in recipes}
    name = a.name
    if a.verb == "save":
        if name in by_name:
            raise Refused("recipe %r exists: use `recipe edit`" % name)
        fields = _recipe_fields(a, settings, root)
        for need in ("symptom", "lightroom_steps"):
            if need not in fields:
                raise ValueError("recipe save needs --symptom and at least one --step")
        if "trigger" not in fields:
            raise ValueError("recipe save needs --frame (to measure the trigger) or --trigger metric=lo:hi")
        row = {"name": name, **fields, "created": a.created or date.today().isoformat()}
        row.setdefault("params", {})
        row.setdefault("origin_frame", None)
        recipes.append({k: row[k] for k in ORDER if k in row})
        what = "saved"
    else:
        if name not in by_name:
            raise Refused("no recipe named %r" % name)
        row = by_name[name]
        if a.verb == "edit":
            fields = _recipe_fields(a, settings, root, row)
            if not fields:
                raise ValueError("nothing to edit: give --symptom, --step, --params, --trigger or --frame")
            row.update(fields)
            what = "edited"
        else:
            if row.get("retired"):
                raise Refused("recipe %r is already retired" % name)
            row["retired"] = a.created or date.today().isoformat()
            what = "retired"
        recipes[:] = [{k: r[k] for k in ORDER if k in r} for r in recipes]
    print("recipe %s %s -> %s" % (name, what, path))
    if not a.apply:
        print("DRY RUN - pass --apply to write recipes.yaml")
        return 0
    write_recipes(path, recipes)
    return 0


# ---------------------------------------------------------------- cli

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--settings")
    common.add_argument("--photos-root")
    common.add_argument("--events-dir")
    common.add_argument("--recipes", help="recipes file (default picker_tools.recipes, else tools/picker/recipes.yaml)")
    common.add_argument("--outing-id")
    common.add_argument("--via", default="cli")
    common.add_argument("--by", default="")
    common.add_argument("--apply", action="store_true", help="write (default: dry run)")
    common.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def frame_cmd(name, fn, **kw):
        p = sub.add_parser(name, parents=[common], **kw)
        p.add_argument("frame", help="frame id (relative to photos_root) or an export path in the finish folder")
        p.add_argument("--export", help="the export file, when it cannot be found by the manifest")
        p.set_defaults(fn=fn)
        return p

    frame_cmd("measure", cmd_measure)
    p = frame_cmd("suggest", cmd_suggest)
    p.add_argument("--note", default="", help="his words for what is wrong")
    p = frame_cmd("apply", cmd_apply)
    p.add_argument("--recipe")
    p.add_argument("--params", nargs="*", default=[], metavar="k=v",
                   help="; ".join("%s %s..%s" % (k, *BOUNDS[k]) for k in BOUNDS))
    for n, fn in (("accept", cmd_accept), ("reject", cmd_reject)):
        p = frame_cmd(n, fn)
        p.add_argument("--recipe", help="recipe name when the candidate does not carry it")
        if n == "accept":
            p.add_argument("--force", action="store_true")

    r = sub.add_parser("recipe", parents=[common], help="the one writer of recipes.yaml")
    r.add_argument("verb", choices=("save", "edit", "retire", "list"))
    r.add_argument("name", nargs="?")
    r.add_argument("--frame", help="origin frame: its (pre-helper) export is measured for the trigger")
    r.add_argument("--symptom", help="his words for what is wrong")
    r.add_argument("--step", action="append", help="one Lightroom step (repeat)")
    r.add_argument("--params", nargs="*", default=[], metavar="k=v")
    r.add_argument("--trigger", action="append", metavar="metric=lo:hi", help="override one measured range")
    r.add_argument("--margin", type=float, default=0.12, help="trigger half-width around the measured value")
    r.add_argument("--created", help=argparse.SUPPRESS)
    r.set_defaults(fn=cmd_recipe)
    return ap


def main(argv=None) -> int:
    link.stdout_utf8()
    a = build_parser().parse_args(argv)
    if a.cmd == "recipe" and a.verb != "list" and not a.name:
        print("error: recipe %s needs a NAME" % a.verb)
        return 2
    try:
        return a.fn(a)
    except Refused as exc:
        print("refused: %s" % exc)
        return 3
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print("error: %s" % exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
