"""The picker's link to the studio bus: settings, ids, and event rows.

Shared by stage_picks.py, finish_watch.py, index_library.py, app.py (the star
route) and tools/publish_external.py, so none of them hardcodes a path, a
person, a camera or an account. Everything comes from `_studio_settings.json`
(read as written) or from arguments.

Settings file, in order: the explicit argument, the env var
MAYNIAC_STUDIO_SETTINGS, then `$VAULT_PATH/_studio_settings.json`. A missing
file means the documented defaults (DEFAULTS below).

Events go to `studio-picker.jsonl` in the bus directory. When
`settings.events_module` names an existing file it is loaded by path and its
`append()` is the one writer; otherwise this module writes the documented line
itself:

    {"v": 2, "ts": ..., "day": ..., "source": "studio", "actor": "studio-picker",
     "via": ..., "kind": ..., "outing": <id>, ...payload}

The bus directory is, in order: the `events_dir` argument, the env var
MAYNIAC_STUDIO_EVENTS, then the module's own resolver or `$VAULT_PATH`'s
`_vault_paths.json` key `events`. Under pytest an explicit directory is
required, so a test can never write the real bus.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WRITER = "picker"
SETTINGS_ENV = "MAYNIAC_STUDIO_SETTINGS"
EVENTS_ENV = "MAYNIAC_STUDIO_EVENTS"
STILL_SUFFIXES = {".jpg", ".jpeg", ".nef", ".dng", ".tif", ".tiff", ".png"}

#: Documented defaults, used for keys the settings file omits or when it is absent.
DEFAULTS = {
    "photos_root": "",
    "finish_dir": "_Finish",
    "camera_match": "D3200",
    "events_module": "",
    "picker_tools": {},
}


# ---------------------------------------------------------------- settings

def settings_path(explicit=None):
    if explicit:
        return Path(explicit)
    if os.environ.get(SETTINGS_ENV):
        return Path(os.environ[SETTINGS_ENV])
    if os.environ.get("VAULT_PATH"):
        return Path(os.environ["VAULT_PATH"]) / "_studio_settings.json"
    return None


def load_settings(explicit=None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    path = settings_path(explicit)
    if path is None or not path.is_file():
        return out
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError("settings file %s is not valid JSON: %s" % (path, exc))
    if isinstance(data, dict):
        out.update(data)
    if not isinstance(out.get("picker_tools"), dict):
        out["picker_tools"] = {}
    return out


def selections_path(settings) -> Path:
    return Path(settings["picker_tools"].get("selections") or HERE / "selections.json")


def index_path(settings) -> Path:
    return Path(settings["picker_tools"].get("index_file") or HERE / "picker_index.json")


def photos_root(settings, override=None) -> Path:
    """The library root. Argument, then settings, then the picker index's own root."""
    value = override or settings.get("photos_root")
    if not value:
        idx = index_path(settings)
        if idx.is_file():
            value = json.loads(idx.read_text(encoding="utf-8")).get("root")
    if not value:
        raise ValueError("photos_root is not set: pass --photos-root or set it in _studio_settings.json")
    return Path(value)


def finish_folder(settings, root: Path, outing_dir: Path) -> Path:
    base = Path(settings.get("finish_dir") or "_Finish")
    base = base if base.is_absolute() else root / base
    return base / outing_dir.name


def resolve_outing_dir(arg, root: Path) -> Path:
    """An outing folder given absolute, relative to the cwd, or relative to photos_root."""
    p = Path(arg)
    if p.is_dir():
        return p.resolve()
    cand = root / arg
    if cand.is_dir():
        return cand.resolve()
    raise FileNotFoundError("outing folder not found: %s" % arg)


# ---------------------------------------------------------------- events module

_MODULES = {}


def _events_module(settings):
    path = settings.get("events_module") or ""
    if not path or not Path(path).is_file():
        return None
    if path not in _MODULES:
        try:
            spec = importlib.util.spec_from_file_location("studio_events_ext", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:
            mod = None
        _MODULES[path] = mod
    return _MODULES[path]


def resolve_events_dir(settings, events_dir=None) -> Path:
    if events_dir:
        return Path(events_dir)
    if os.environ.get(EVENTS_ENV):
        return Path(os.environ[EVENTS_ENV])
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError("a test must pass events_dir or set %s; it never writes the real bus" % EVENTS_ENV)
    mod = _events_module(settings)
    if mod is not None:
        return Path(mod.resolve_events_dir())
    vault = os.environ.get("VAULT_PATH")
    if not vault:
        raise FileNotFoundError("no events directory: set VAULT_PATH or pass --events-dir")
    reg = json.loads((Path(vault) / "_vault_paths.json").read_text(encoding="utf-8"))
    rel = (reg.get("paths") or {}).get("events")
    if not rel:
        raise KeyError("_vault_paths.json has no 'events' key")
    return Path(vault) / rel


def events_file(settings, events_dir=None) -> Path:
    return resolve_events_dir(settings, events_dir) / ("studio-%s.jsonl" % WRITER)


def append_event(settings, kind, *, outing=None, via="cli", member=None, events_dir=None,
                 now=None, **payload) -> dict:
    """Append one row to studio-picker.jsonl and return it."""
    base = resolve_events_dir(settings, events_dir)
    extra = dict(payload)
    if member:
        extra["member"] = member
    mod = _events_module(settings)
    if mod is not None:
        return mod.append(WRITER, kind, events_dir=base, now=now, via=via, outing=outing, **extra)
    if isinstance(now, str):
        ts = now
    else:
        ts = (now or datetime.now()).astimezone().isoformat(timespec="seconds")
    row = {"v": 2, "ts": ts, "day": ts[:10], "source": "studio",
           "actor": "studio-" + WRITER, "via": via, "kind": kind}
    if outing is not None:
        row["outing"] = outing
    row.update(extra)
    base.mkdir(parents=True, exist_ok=True)
    with open(base / ("studio-%s.jsonl" % WRITER), "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def read_own_events(settings, events_dir=None) -> list:
    """Rows already in studio-picker.jsonl ([] when the file or directory is absent)."""
    try:
        path = events_file(settings, events_dir)
    except (FileNotFoundError, KeyError):
        return []
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


# ---------------------------------------------------------------- ids

def frame_id(path, root) -> str:
    """The frame's path relative to photos_root, `/`-separated, case kept."""
    return Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root))).as_posix()


def rel_or_abs(path, root) -> str:
    try:
        return frame_id(path, root)
    except ValueError:
        return Path(os.path.abspath(path)).as_posix()


def _sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stills_in(outing_dir: Path):
    return sorted(p for p in Path(outing_dir).rglob("*")
                  if p.is_file() and p.suffix.lower() in STILL_SUFFIXES)


_OUTING_IDS = {}


def outing_id_for(outing_dir) -> str | None:
    """`out-` + 10 hex of SHA-1("outing:" + SHA-1 of the outing's earliest frame).

    The earliest frame is the lowest mtime (the sorter's copy preserves it), name
    order on a tie: the same rule as lib/studio_events.py. None when the folder
    holds no still. Cached per process: the star route asks once per outing.
    """
    key = str(outing_dir)
    if key not in _OUTING_IDS:
        files = stills_in(Path(outing_dir))
        if not files:
            _OUTING_IDS[key] = None
        else:
            first = min(files, key=lambda p: (p.stat().st_mtime, p.as_posix()))
            digest = hashlib.sha1(("outing:" + _sha1(first)).encode("ascii")).hexdigest()
            _OUTING_IDS[key] = "out-" + digest[:10]
    return _OUTING_IDS[key]


def outing_dir_of_frame(frame_path: Path, outing_name: str) -> Path | None:
    """The ancestor folder of a frame named `outing_name` (the index's `outing`)."""
    for parent in Path(frame_path).parents:
        if parent.name == outing_name:
            return parent
    return None
