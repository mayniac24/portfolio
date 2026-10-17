#!/usr/bin/env python3
"""Stage an outing's starred frames for the hand finish.

COPIES every starred frame of one outing (selections.json) into
`<photos_root>/<finish_dir>/<outing name>/` and writes one `pick` event per frame
to `studio-picker.jsonl`. Nothing is moved, deleted or overwritten: the originals
stay where they are, and a file already staged (he may be editing it in place) is
left alone. Prints the folder to import into Lightroom.

    python tools/picker/stage_picks.py <outing folder> [--apply]

Dry run by default; `--apply` copies and writes events. Settings come from
`_studio_settings.json` (see studio_link.py); `--settings`, `--photos-root`,
`--events-dir`, `--selections`, `--outing-id`, `--via`, `--by` override.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import studio_link as link  # noqa: E402

MANIFEST = ".staged.json"      # beside the staged copies; finish_watch reads it


def starred_in_outing(selections: dict, root: Path, outing_dir: Path) -> list:
    """[(frame id, source path, meta)] for starred frames under outing_dir, in id order."""
    out = []
    for fid in sorted(selections):
        src = root / fid
        try:
            src.resolve().relative_to(outing_dir.resolve())
        except ValueError:
            continue
        out.append((fid, src, selections[fid] or {}))
    return out


def load_manifest(dest: Path) -> dict:
    f = dest / MANIFEST
    if f.is_file():
        try:
            return json.loads(f.read_text(encoding="utf-8")).get("frames", {})
        except ValueError:
            pass
    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outing", help="outing folder (absolute, or relative to photos_root)")
    ap.add_argument("--apply", action="store_true", help="copy and write events (default: dry run)")
    ap.add_argument("--settings", help="_studio_settings.json (default: env / $VAULT_PATH)")
    ap.add_argument("--photos-root")
    ap.add_argument("--selections", help="selections.json (default: settings picker_tools.selections)")
    ap.add_argument("--events-dir")
    ap.add_argument("--outing-id", help="override the computed outing id")
    ap.add_argument("--via", default="cli", help="cli | horizons | horizons-web | phone")
    ap.add_argument("--by", default="", help="member who acted (recorded as `member`)")
    a = ap.parse_args(argv)

    try:
        settings = link.load_settings(a.settings)
        root = link.photos_root(settings, a.photos_root)
        outing_dir = link.resolve_outing_dir(a.outing, root)
        sel_file = Path(a.selections) if a.selections else link.selections_path(settings)
        selections = json.loads(sel_file.read_text(encoding="utf-8")) if sel_file.is_file() else {}
    except (ValueError, OSError) as exc:
        print("error: %s" % exc)
        return 2

    picks = starred_in_outing(selections, root, outing_dir)
    dest = link.finish_folder(settings, root, outing_dir)
    print("outing     : %s" % outing_dir)
    print("starred    : %d (of %d total starred)" % (len(picks), len(selections)))
    print("finish dir : %s" % dest)
    if not picks:
        print("nothing starred in this outing")
        return 0

    names = {}
    for fid, src, _ in picks:
        names.setdefault(src.name.lower(), []).append(fid)
    clash = {k: v for k, v in names.items() if len(v) > 1}
    if clash:
        print("error: starred frames share a file name, the finish folder is flat:")
        for v in clash.values():
            print("  " + "  ".join(v))
        return 1
    missing = [fid for fid, src, _ in picks if not src.is_file()]
    if missing:
        print("error: starred frame(s) missing on disk:")
        for fid in missing:
            print("  " + fid)
        return 1

    manifest = load_manifest(dest)
    todo = [p for p in picks if p[0] not in manifest]
    for fid, src, _ in picks:
        mark = "already staged" if fid in manifest else "copy"
        print("  %-14s %s" % (mark, fid))

    if not a.apply:
        print("\nDRY RUN - pass --apply to copy %d frame(s) and write %d pick event(s)" % (len(todo), len(todo)))
        return 0

    outing = a.outing_id or link.outing_id_for(outing_dir)
    dest.mkdir(parents=True, exist_ok=True)
    kept = 0
    for fid, src, meta in todo:
        target = dest / src.name
        if target.exists():
            kept += 1                   # never overwrite a file that is already there
        else:
            shutil.copy2(src, target)
        manifest[fid] = {"name": src.name, "src_mtime_ns": src.stat().st_mtime_ns,
                         "bytes": src.stat().st_size}
        try:
            link.append_event(settings, "pick", outing=outing, via=a.via, member=a.by or None,
                              events_dir=a.events_dir, frame=fid, state="starred",
                              category=meta.get("category", ""), alt=meta.get("alt", ""))
        except (RuntimeError, OSError, KeyError, FileNotFoundError) as exc:
            print("error: could not write the pick event for %s: %s" % (fid, exc))
            (dest / MANIFEST).write_text(json.dumps({"outing": outing, "frames": {
                k: v for k, v in manifest.items() if k != fid}}, indent=1), encoding="utf-8", newline="\n")
            return 2
    (dest / MANIFEST).write_text(json.dumps({"outing": outing, "frames": manifest}, indent=1),
                                 encoding="utf-8", newline="\n")
    print("\nstaged %d frame(s), %d pick event(s)%s" % (
        len(todo), len(todo), (", %d already in place, left alone" % kept) if kept else ""))
    print("Import into Lightroom: %s" % dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
