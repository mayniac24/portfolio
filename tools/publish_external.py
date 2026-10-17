#!/usr/bin/env python3
"""Publish starred photos from EXTERNALLY corrected files (e.g. a Lightroom export).

The picker's own publish path runs correct_colour.py on the library original.
When the photographer has already corrected a frame by hand, running it again
stacks an automated pass on top of deliberate work, so this route copies the
supplied file through untouched and does everything else identically.

Every new entry carries `captured`, the real EXIF date from the picker index.
optimize_images.py derives the copyright year from the filename, and a Lightroom
export names files by EXPORT date -- that is how the maternity set ended up
prefixed 20260202 for photographs taken in August 2025.

    python tools/publish_external.py <export-dir> [--outing <outing>] [--apply]

Matching is on the DSC stem, which is unique within a starred set of this size;
the script refuses to run if it is not. `--outing` limits the starred set to one
outing folder, so the stem check (and the match) runs within that outing: two
outings can both hold a DSC_0001. Each NEWLY published frame appends a `post` event
(channel portfolio) to studio-picker.jsonl; a re-published frame is not a new post.
Settings, ids and events: tools/picker/studio_link.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PICKER = HERE / "picker"
sys.path.insert(0, str(PICKER))
import studio_link as link  # noqa: E402
CORRECTED = REPO.parent / "Portfolio Corrected"
GALLERY = REPO / "data" / "gallery.json"


def post_events(settings, a, root, outing_dir, photos, plan, new_ids) -> None:
    """One `post` event (channel portfolio) per newly published frame. Never blocks publishing."""
    when = datetime.now().astimezone().isoformat(timespec="seconds")
    for f, pid, gid, date in plan:
        if gid not in new_ids:
            continue
        try:
            src = root / pid
            od = outing_dir or link.outing_dir_of_frame(src, photos[pid].get("outing", ""))
            link.append_event(settings, "post", outing=link.outing_id_for(od) if od else None,
                              via=a.via, member=a.by or None, events_dir=a.events_dir,
                              frames=[pid], channel="portfolio", when=when, slug=gid)
        except Exception as exc:  # noqa: BLE001
            print("post event not written for %s: %s" % (gid, exc))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--outing", help="limit to one outing folder (absolute, or relative to photos_root)")
    ap.add_argument("--settings", help="_studio_settings.json (default: env / $VAULT_PATH)")
    ap.add_argument("--events-dir")
    ap.add_argument("--via", default="cli")
    ap.add_argument("--by", default="")
    a = ap.parse_args(argv)

    settings = link.load_settings(a.settings)
    sel_file, idx_file = link.selections_path(settings), link.index_path(settings)
    sel = json.loads(sel_file.read_text(encoding="utf-8"))
    index = json.loads(idx_file.read_text(encoding="utf-8"))
    photos = {p["id"]: p for p in index["photos"]}
    root = link.photos_root(settings) if settings.get("photos_root") else Path(index["root"])
    outing_dir = None
    if a.outing:
        outing_dir = link.resolve_outing_dir(a.outing, root)
        sel = {i: m for i, m in sel.items()
               if outing_dir in (root / i).resolve().parents}
        print("outing     : %s" % outing_dir)
    if not sel:
        print("nothing starred"); return 1

    missing = [i for i, m in sel.items() if not (m.get("alt") or "").strip()]
    if missing:
        print("%d starred photo(s) have no alt text:" % len(missing))
        for m in missing[:10]:
            print("   ", m)
        return 1

    by_stem: dict[str, list[str]] = {}
    for pid in sel:
        by_stem.setdefault(Path(pid).stem.upper(), []).append(pid)
    dupes = {k: v for k, v in by_stem.items() if len(v) > 1}
    if dupes:
        print("ambiguous stems among the starred set - cannot match by filename:")
        for k, v in dupes.items():
            print("  %s -> %s" % (k, v))
        return 1

    exports = sorted(p for p in a.export_dir.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg"))
    if not exports:
        print("no JPEGs in %s" % a.export_dir); return 1

    gallery = json.loads(GALLERY.read_text(encoding="utf-8"))
    known = {p["id"] for p in gallery["photos"]}

    plan, unmatched = [], []
    for f in exports:
        stem = f.stem.upper()
        # tolerate a Lightroom suffix like DSC_0034-Edit
        pid = next((by_stem[k][0] for k in by_stem if stem == k or stem.startswith(k + "-")), None)
        if pid is None:
            unmatched.append(f.name); continue
        date = photos[pid]["date"]
        gid = "%s-%s" % (date.replace("-", ""), Path(pid).stem)
        plan.append((f, pid, gid, date))

    print("export dir : %s" % a.export_dir)
    print("starred    : %d   exported files: %d   matched: %d"
          % (len(sel), len(exports), len(plan)))
    if unmatched:
        print("UNMATCHED (skipped): %s" % ", ".join(unmatched))
    not_exported = [p for p in sel if not any(x[1] == p for x in plan)]
    if not_exported:
        print("starred but not exported (skipped): %d" % len(not_exported))
    print()
    for f, pid, gid, date in plan:
        mark = "  [already published]" if gid in known else ""
        print("  %-34s -> %s.jpg   %s%s" % (f.name, gid, date, mark))

    if not a.apply:
        print("\nDRY RUN - pass --apply to write"); return 0

    CORRECTED.mkdir(parents=True, exist_ok=True)
    added = []
    for f, pid, gid, date in plan:
        shutil.copy2(f, CORRECTED / ("%s.jpg" % gid))     # copied, NOT corrected
        if gid in known:
            continue
        added.append({"id": gid, "category": sel[pid]["category"],
                      "attributes": [], "alt": sel[pid]["alt"].strip(),
                      "captured": date})
    if added:
        land = [x for x in added if x["category"] == "landscape"]
        rest = [x for x in added if x["category"] != "landscape"]
        gallery["photos"] = land + gallery["photos"] + rest
        GALLERY.write_text(json.dumps(gallery, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    print("\nstaged %d master(s); %d new gallery entr(ies)" % (len(plan), len(added)))
    for cmd in (["optimize_images.py"], ["build.py"]):
        r = subprocess.run([sys.executable, *cmd], cwd=REPO,
                           capture_output=True, text=True)
        print("%s -> exit %d" % (cmd[0], r.returncode))
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-2000:]); return 1
    post_events(settings, a, root, outing_dir, photos, plan, {x["id"] for x in added})
    print("\nNothing committed. Review with `git diff` / `git status`, "
          "run `python -m pytest tests/ -q`, then commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
