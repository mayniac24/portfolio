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

    python tools/publish_external.py <export-dir> [--apply]

Matching is on the DSC stem, which is unique within a starred set of this size;
the script refuses to run if it is not.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PICKER = HERE / "picker"
CORRECTED = REPO.parent / "Portfolio Corrected"
GALLERY = REPO / "data" / "gallery.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    sel = json.loads((PICKER / "selections.json").read_text(encoding="utf-8"))
    if not sel:
        print("nothing starred"); return 1
    index = json.loads((PICKER / "picker_index.json").read_text(encoding="utf-8"))
    photos = {p["id"]: p for p in index["photos"]}

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
    print("\nNothing committed. Review with `git diff` / `git status`, "
          "run `python -m pytest tests/ -q`, then commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
