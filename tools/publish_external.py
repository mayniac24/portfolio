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

Face guard (Evan's rule: our kids' faces are never posted): every frame goes through
`faces.gate` before it can reach the gallery. A refused frame is skipped with its reason, no
post event is written for it, and the exit code is 3. A frame he has masked is published by
exporting/placing the `<stem>_masked.jpg` that tools/picker/faces.py wrote next to the export;
the plain export is then ignored. See tools/picker/faces.py.
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
import faces  # noqa: E402  (the face guard: no child's face is published)
CORRECTED = REPO.parent / "Portfolio Corrected"
GALLERY = REPO / "data" / "gallery.json"


def post_events(settings, a, root, outing_dir, photos, plan) -> None:
    """One `post` event (channel portfolio) per published frame that has none yet.

    "New" means no post event with that slug exists in studio-picker.jsonl: the gallery
    can hold a photo published long before the bus, and a re-export is not a second post.
    Never blocks publishing."""
    when = datetime.now().astimezone().isoformat(timespec="seconds")
    posted = {r.get("slug") for r in link.read_own_events(settings, a.events_dir)
              if r.get("kind") == "post" and r.get("channel") == "portfolio"}
    for f, pid, gid, date in plan:
        if gid in posted:
            continue
        try:
            src = root / pid
            od = outing_dir or link.outing_dir_of_frame(src, photos[pid].get("outing", ""))
            link.append_event(settings, "post",
                              outing=link.outing_id_for(settings, root, od, a.events_dir) if od else None,
                              via=a.via, member=a.by or None, events_dir=a.events_dir,
                              frames=[pid], channel="portfolio", when=when, slug=gid)
            posted.add(gid)
        except Exception as exc:  # noqa: BLE001
            print("post event not written for %s: %s" % (gid, exc))


def gate_plan(settings, a, root, outing_dir, photos, plan):
    """Run the face gate on every matched export. -> (kept plan, [(file, pid, reason)], faces rows).

    When a frame has both its plain export and a `_masked` one, the masked one is tried first
    and the plain one only if that fails. A gate error is a refusal."""
    by_pid = {}
    for item in plan:
        by_pid.setdefault(item[1], []).append(item)
    chosen, refused, cleared = {}, [], {}
    for pid, items in by_pid.items():
        items.sort(key=lambda it: not it[0].stem.lower().endswith("_masked"))
        try:
            src = root / pid
            od = outing_dir or link.outing_dir_of_frame(src, photos[pid].get("outing", ""))
            oid = link.outing_id_for(settings, root, od, a.events_dir) if od else None
        except Exception as exc:  # noqa: BLE001
            refused += [(it[0], pid, "error, so refused: cannot resolve the outing (%s)" % exc) for it in items[:1]]
            continue
        res = None
        for it in items:
            res = faces.gate(it[0], frame=pid, outing=oid, settings=settings, events_dir=a.events_dir)
            if res["allowed"]:
                chosen[pid] = it
                cleared[pid] = (oid, res.get("state"), res.get("by"))
                break
        else:
            refused.append((items[0][0], pid, res["reason"]))
    return [it for it in plan if chosen.get(it[1]) is it], refused, cleared


def record_clearances(settings, a, kept, cleared) -> None:
    """A published master is looked up by its id (optimize_images' guard): leave a `faces`
    event for the id when the frame is masked or cleared."""
    for f, pid, gid, date in kept:
        oid, state, by = cleared[pid]
        if state not in ("masked", "cleared"):
            continue
        # a copied `cleared` row keeps who cleared it (`cleared_by`): it is honoured only while
        # that person is in face_clearers; a masked one holds the sha1 of the file published
        extra = ({"sha1": link.sha1_file(f), "image": link.rel_or_abs(f, CORRECTED)} if state == "masked"
                 else {"cleared_by": by})
        faces.record(settings, "key", gid, state, outing=oid, by=a.by, via=a.via, events_dir=a.events_dir,
                     of=pid, **extra)


def main(argv=None) -> int:
    link.stdout_utf8()
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
        if stem.endswith("_MASKED"):            # what faces.py mask writes beside an export
            stem = stem[:-len("_MASKED")]
        # tolerate a Lightroom suffix like DSC_0034-Edit
        pid = next((by_stem[k][0] for k in by_stem if stem == k or stem.startswith(k + "-")), None)
        if pid is None:
            unmatched.append(f.name); continue
        date = photos[pid]["date"]
        gid = "%s-%s" % (date.replace("-", ""), Path(pid).stem)
        plan.append((f, pid, gid, date))

    plan, refused, cleared = gate_plan(settings, a, root, outing_dir, photos, plan)
    for f, pid, why in refused:
        print("REFUSED by the face guard: %s: %s" % (f.name, why))

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

    if not plan:
        print("nothing left to publish" if refused else "nothing matched: nothing to publish")
        return faces.EXIT_REFUSED if refused else 0

    if not a.apply:
        print("\nDRY RUN - pass --apply to write"); return faces.EXIT_REFUSED if refused else 0

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
    try:
        record_clearances(settings, a, plan, cleared)
    except Exception as exc:  # noqa: BLE001 - without it optimize_images refuses the masked master
        print("faces event not written (%s); optimize_images may refuse a masked or cleared frame" % exc)
    for cmd in (["optimize_images.py", "--only", *[g for _, _, g, _ in plan]], ["build.py"]):
        r = subprocess.run([sys.executable, *cmd], cwd=REPO,
                           capture_output=True, text=True)
        print("%s -> exit %d" % (cmd[0], r.returncode))
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-2000:]); return 1
    post_events(settings, a, root, outing_dir, photos, plan)
    if refused:
        print("\n%d frame(s) REFUSED by the face guard and not published (exit 3)." % len(refused))
        return faces.EXIT_REFUSED
    print("\nNothing committed. Review with `git diff` / `git status`, "
          "run `python -m pytest tests/ -q`, then commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
