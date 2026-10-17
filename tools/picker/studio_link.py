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
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WRITER = "picker"
SETTINGS_ENV = "MAYNIAC_STUDIO_SETTINGS"
EVENTS_ENV = "MAYNIAC_STUDIO_EVENTS"
#: The sorter's default `still_ext` (life-timeline/scanners/outing_sort.py); settings.still_ext wins.
STILL_SUFFIXES = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".nef", ".dng", ".cr2", ".cr3",
                  ".arw", ".tif", ".tiff", ".webp")

#: Documented defaults, used for keys the settings file omits or when it is absent.
DEFAULTS = {
    "photos_root": "",
    "finish_dir": "_Finish",
    "camera_match": "",          # "" = no camera filter; the seeded settings file names one
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
    out["_settings_file"] = str(path) if path is not None else ""   # not a setting: where it was read
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


def sha1_file(path) -> str:
    return _sha1(path)


def still_suffixes(settings=None) -> tuple:
    got = (settings or {}).get("still_ext")
    return tuple(e.lower() for e in got) if got else STILL_SUFFIXES


def stills_in(outing_dir: Path, settings=None):
    exts = still_suffixes(settings)
    return sorted(p for p in Path(outing_dir).rglob("*")
                  if p.is_file() and p.suffix.lower() in exts)


def stdout_utf8() -> None:
    """Entry points call this so a cp1252 console cannot crash a print of a file name."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


# ---- outing ids: the bus first, hashing only when the bus has never heard of the folder

def _folder_key(rel: str) -> str:
    return str(rel).replace("\\", "/").strip("/").lower()


def _read_rows(path):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def vault_root_of(settings):
    f = settings.get("_settings_file")
    if f:
        return Path(f).parent
    return Path(os.environ["VAULT_PATH"]) if os.environ.get("VAULT_PATH") else None


def id_from_projection(settings, key: str):
    """`<vault>/_studio_outings.json`: the projection already folds renames and merges."""
    vault = vault_root_of(settings)
    if vault is None:
        return None
    try:
        data = json.loads((vault / "_studio_outings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for rec in (data.get("outings") or []) if isinstance(data, dict) else []:
        if isinstance(rec, dict) and rec.get("id") and _folder_key(rec.get("folder") or "") == key:
            return rec["id"]
    return None


def id_from_events(events_dir: Path, key: str):
    """Replay the `outing` rows of every studio-*.jsonl: create, rename, merge, split."""
    rows = []
    for path in sorted(Path(events_dir).glob("studio-*.jsonl")):
        rows += [r for r in _read_rows(path) if r.get("kind") == "outing" and r.get("outing")]
    rows.sort(key=lambda r: str(r.get("ts") or ""))
    folder_to_id = {}
    for r in rows:
        oid, op = r["outing"], r.get("op")
        if op == "create" and isinstance(r.get("folder"), str):
            folder_to_id[_folder_key(r["folder"])] = oid
        elif op == "rename":
            if isinstance(r.get("from"), str):
                folder_to_id.pop(_folder_key(r["from"]), None)
            new = r.get("to") or r.get("folder")
            if isinstance(new, str):
                folder_to_id[_folder_key(new)] = oid
        elif op == "merge":
            gone = set(r.get("from") or [])
            for f, i in list(folder_to_id.items()):
                if i in gone:
                    folder_to_id[f] = oid
        elif op == "split" and r.get("to") and isinstance(r.get("folder"), str):
            folder_to_id[_folder_key(r["folder"])] = r["to"]
    return folder_to_id.get(key)


def _capture_time(path: Path):
    """EXIF DateTimeOriginal, else mtime: the sorter's order (date_ladder), local naive."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            raw = im.getexif().get_ifd(0x8769).get(0x9003)
        if raw:
            return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:  # noqa: BLE001
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def hash_outing_id(outing_dir, settings=None):
    """`out-` + 10 hex of SHA-1("outing:" + SHA-1 of the earliest still); None with no still.

    Mirrors the sorter's rule for an outing that has stills: the earliest still by
    capture time (EXIF first, mtime when there is none), over the sorter's still_ext set.
    An outing with only video has no fallback id: the bus has to name it.
    """
    files = stills_in(Path(outing_dir), settings)
    if not files:
        return None
    first = min(files, key=lambda p: (_capture_time(p), p.as_posix()))
    return "out-" + hashlib.sha1(("outing:" + _sha1(first)).encode("ascii")).hexdigest()[:10]


_ID_CACHE = {}


def _stat_sig(paths):
    sig = []
    for p in paths:
        try:
            st = Path(p).stat()
            sig.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append((str(p), 0, 0))
    return tuple(sig)


def outing_id_for(settings, root, outing_dir, events_dir=None):
    """The outing's id. Order: the projection (by folder), the events (by folder), then
    hashing the earliest still. The cache key carries the stat of every file consulted, so a
    rename or a new create row is seen on the next call."""
    try:
        key = _folder_key(frame_id(outing_dir, root))
    except ValueError:
        key = _folder_key(Path(outing_dir).name)
    watched = []
    vault = vault_root_of(settings)
    if vault is not None:
        watched.append(vault / "_studio_outings.json")
    ev = None
    try:
        ev = resolve_events_dir(settings, events_dir)
        if ev.is_dir():
            watched += [p for p in sorted(ev.glob("studio-*.jsonl")) if p.name != "studio-%s.jsonl" % WRITER]
    except (RuntimeError, FileNotFoundError, KeyError, ValueError, OSError):
        ev = None
    ck = (key, str(outing_dir), _stat_sig(watched))
    if ck in _ID_CACHE:
        return _ID_CACHE[ck]
    oid = id_from_projection(settings, key)
    if oid is None and ev is not None and ev.is_dir():
        oid = id_from_events(ev, key)
    if oid is None:
        oid = hash_outing_id(outing_dir, settings)
    _ID_CACHE[ck] = oid
    return oid


def outing_dir_of_frame(frame_path: Path, outing_name: str) -> Path | None:
    """The ancestor folder of a frame named `outing_name` (the index's `outing`)."""
    for parent in Path(frame_path).parents:
        if parent.name == outing_name:
            return parent
    return None
