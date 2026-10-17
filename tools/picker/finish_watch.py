#!/usr/bin/env python3
"""Detect finished edits in an outing's finish folder.

He finishes frames by hand (Lightroom or Camera Raw) and exports full size into
`<photos_root>/<finish_dir>/<outing name>/`. This looks, never edits. A staged
frame is FINISHED when that folder holds a file with its stem (or `<stem>-...`,
Lightroom's `-Edit`) whose mtime is newer than the original's capture file. The
unchanged staged copy keeps the original's mtime, so it never counts.

    python tools/picker/finish_watch.py <outing folder> [--apply]

Prints `finished n of m`. Dry run by default; `--apply` writes one `finish` event
(`frame, export, bytes`) per newly finished frame to `studio-picker.jsonl`. A frame
already recorded with the same export and size is not written twice; a re-export with
different bytes is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import studio_link as link  # noqa: E402
from stage_picks import MANIFEST, load_manifest, starred_in_outing  # noqa: E402

#: exFAT stores mtimes at 2 s resolution; an edit is days later, a copy is within rounding.
SLACK_NS = 2_000_000_000


def find_export(finish_dir: Path, name: str, src_mtime_ns: int):
    """The newest file in finish_dir for staged file `name` that is newer than the original."""
    stem = Path(name).stem.lower()
    best = None
    for p in finish_dir.iterdir():
        if not p.is_file() or p.name == MANIFEST or p.suffix.lower() not in link.STILL_SUFFIXES:
            continue
        s = p.stem.lower()
        if s != stem and not s.startswith(stem + "-"):
            continue
        st = p.stat()
        if st.st_mtime_ns <= src_mtime_ns + SLACK_NS:
            continue
        if best is None or st.st_mtime_ns > best[1]:
            best = (p, st.st_mtime_ns, st.st_size)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outing", help="outing folder (absolute, or relative to photos_root)")
    ap.add_argument("--apply", action="store_true", help="write finish events (default: dry run)")
    ap.add_argument("--settings")
    ap.add_argument("--photos-root")
    ap.add_argument("--selections")
    ap.add_argument("--events-dir")
    ap.add_argument("--outing-id")
    ap.add_argument("--via", default="cli")
    ap.add_argument("--by", default="")
    a = ap.parse_args(argv)

    try:
        settings = link.load_settings(a.settings)
        root = link.photos_root(settings, a.photos_root)
        outing_dir = link.resolve_outing_dir(a.outing, root)
    except (ValueError, OSError) as exc:
        print("error: %s" % exc)
        return 2
    finish_dir = link.finish_folder(settings, root, outing_dir)
    manifest = load_manifest(finish_dir)
    if not manifest:
        # No manifest (staged by hand, or before this tool): fall back to the stars.
        sel_file = Path(a.selections) if a.selections else link.selections_path(settings)
        selections = json.loads(sel_file.read_text(encoding="utf-8")) if sel_file.is_file() else {}
        for fid, src, _ in starred_in_outing(selections, root, outing_dir):
            if src.is_file() and (finish_dir / src.name).is_file():
                manifest[fid] = {"name": src.name, "src_mtime_ns": src.stat().st_mtime_ns}
    if not finish_dir.is_dir() or not manifest:
        print("nothing staged for %s (run stage_picks.py first)" % outing_dir.name)
        print("finished 0 of 0")
        return 0

    seen = {}
    for row in link.read_own_events(settings, a.events_dir):
        if row.get("kind") == "finish":
            seen[(row.get("frame"), row.get("export"), row.get("bytes"))] = True

    finished, fresh = [], []
    for fid in sorted(manifest):
        m = manifest[fid]
        found = find_export(finish_dir, m["name"], m["src_mtime_ns"])
        if not found:
            continue
        path, _, size = found
        finished.append(fid)
        export = link.rel_or_abs(path, root)
        if (fid, export, size) not in seen:
            fresh.append((fid, export, size))

    outing = a.outing_id or (link.outing_id_for(outing_dir) if a.apply and fresh else None)
    for fid, export, size in fresh:
        print("  finished  %s  <- %s" % (fid, export))
        if a.apply:
            try:
                link.append_event(settings, "finish", outing=outing, via=a.via, member=a.by or None,
                                  events_dir=a.events_dir, frame=fid, export=export, bytes=size)
            except (RuntimeError, OSError, KeyError, FileNotFoundError) as exc:
                print("error: could not write the finish event for %s: %s" % (fid, exc))
                return 2
    print("finished %d of %d" % (len(finished), len(manifest)))
    if fresh and not a.apply:
        print("DRY RUN - pass --apply to write %d finish event(s)" % len(fresh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
