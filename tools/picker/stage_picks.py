#!/usr/bin/env python3
"""Stage an outing's starred frames for the hand finish.

COPIES every starred frame of one outing (selections.json) into
`<photos_root>/<finish_dir>/<outing name>/` and writes one `pick` event per frame
to `studio-picker.jsonl`. Nothing is moved, deleted or overwritten. Each copy goes to
`<name>.part`, is checked against the source (size and SHA-1), then renamed into place,
so a crash never leaves a half file that looks staged. A file already in the folder
that is not a copy of the source (he may be editing it in place) is refused, not
adopted. Prints the folder to import into Lightroom.

    python tools/picker/stage_picks.py <outing folder> [--apply]

Dry run by default; `--apply` copies and writes events. Settings come from
`_studio_settings.json` (see studio_link.py); `--settings`, `--photos-root`,
`--events-dir`, `--selections`, `--outing-id`, `--via`, `--by` override.
"""
from __future__ import annotations

import argparse
import json
import os
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


def read_manifest(dest: Path) -> dict:
    """{'outing': id or None, 'frames': {frame id: {name, src_mtime_ns, bytes, sha1}}}"""
    f = dest / MANIFEST
    if f.is_file():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            return {"outing": data.get("outing"), "frames": data.get("frames") or {}}
        except ValueError:
            pass
    return {"outing": None, "frames": {}}


def load_manifest(dest: Path) -> dict:
    return read_manifest(dest)["frames"]


def write_manifest(dest: Path, outing, frames: dict) -> None:
    tmp = dest / (MANIFEST + ".part")
    tmp.write_text(json.dumps({"outing": outing, "frames": frames}, indent=1), encoding="utf-8", newline="\n")
    os.replace(tmp, dest / MANIFEST)


def copy_verified(src: Path, target: Path, sha1: str) -> None:
    """Copy via `<target>.part`, verify size and SHA-1 against the source, then rename."""
    part = target.with_name(target.name + ".part")
    try:
        if part.exists():
            part.unlink()
        shutil.copy2(src, part)
        if part.stat().st_size != src.stat().st_size or link.sha1_file(part) != sha1:
            raise OSError("copy does not match the source (size or SHA-1)")
        os.replace(part, target)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise


def main(argv=None) -> int:
    link.stdout_utf8()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outing", help="outing folder (absolute, or relative to photos_root)")
    ap.add_argument("--apply", action="store_true", help="copy and write events (default: dry run)")
    ap.add_argument("--settings", help="_studio_settings.json (default: env / $VAULT_PATH)")
    ap.add_argument("--photos-root")
    ap.add_argument("--selections", help="selections.json (default: settings picker_tools.selections)")
    ap.add_argument("--events-dir")
    ap.add_argument("--outing-id", help="override the resolved outing id")
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

    outing = a.outing_id or link.outing_id_for(settings, root, outing_dir, a.events_dir)
    mf = read_manifest(dest)
    manifest = mf["frames"]
    if manifest and mf["outing"] and outing and mf["outing"] != outing:
        print("error: %s was staged for outing %s, this outing is %s; two outings cannot share a "
              "finish folder (rename one)" % (dest, mf["outing"], outing))
        return 1
    todo = [p for p in picks if p[0] not in manifest]
    for fid, src, _ in picks:
        print("  %-14s %s" % ("already staged" if fid in manifest else "copy", fid))

    if not a.apply:
        print("\nDRY RUN - pass --apply to copy %d frame(s) and write %d pick event(s)" % (len(todo), len(todo)))
        return 0

    try:                                   # the bus has to be reachable BEFORE anything is copied
        link.resolve_events_dir(settings, a.events_dir)
    except (RuntimeError, OSError, KeyError, FileNotFoundError) as exc:
        print("error: no event directory, nothing copied: %s" % exc)
        return 2

    dest.mkdir(parents=True, exist_ok=True)
    staged, adopted, failed = 0, 0, []
    try:
        for fid, src, meta in todo:
            target = dest / src.name
            try:
                sha = link.sha1_file(src)
                if target.exists():
                    if link.sha1_file(target) != sha:
                        raise OSError("%s already exists and is not a copy of the source; refusing to "
                                      "overwrite or adopt it" % target.name)
                    adopted += 1           # an identical copy from an interrupted run
                else:
                    copy_verified(src, target, sha)
                st = src.stat()
                manifest[fid] = {"name": src.name, "src_mtime_ns": st.st_mtime_ns,
                                 "bytes": st.st_size, "sha1": sha}
                link.append_event(settings, "pick", outing=outing, via=a.via, member=a.by or None,
                                  events_dir=a.events_dir, frame=fid, state="starred",
                                  category=meta.get("category", ""), alt=meta.get("alt", ""))
                staged += 1
            except OSError as exc:
                failed.append(fid)
                manifest.pop(fid, None)
                print("error: %s: %s" % (fid, exc))
    finally:
        write_manifest(dest, outing, manifest)
    print("\nstaged %d frame(s), %d pick event(s)%s" % (
        staged, staged, (", %d already in place and identical" % adopted) if adopted else ""))
    if failed:
        print("FAILED %d frame(s); fix and rerun" % len(failed))
        return 1
    print("Import into Lightroom: %s" % dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
