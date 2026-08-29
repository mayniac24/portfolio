#!/usr/bin/env python3
"""Local photo picker — browse the library, star frames, publish them to the site.

Runs on localhost only. There is no auth because there is no network exposure
and no reason to add one; do not bind this to 0.0.0.0.

Why local and not a web CMS: the candidate photos live on M: and are hundreds of
gigabytes. No hosted admin can see them. A web CMS (Decap at /admin) is the right
tool for editing *published* photos, and is a separate piece of work.

    pip install flask
    python tools/picker/index_library.py     # once, or after adding shoots
    python tools/picker/app.py               # -> http://127.0.0.1:5000

Publishing a starred photo runs the full pipeline: colour-correct into
../Portfolio Corrected, generate variants, append to data/gallery.json, rebuild
index.html. Nothing is committed — review the diff and commit yourself.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import correct_colour  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INDEX = HERE / "picker_index.json"
SELECTIONS = HERE / "selections.json"
CACHE = HERE / ".thumbs"
CORRECTED = REPO.parent / "Portfolio Corrected"
GALLERY = REPO / "data" / "gallery.json"

app = Flask(__name__)
CACHE.mkdir(exist_ok=True)

_index = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else None
if _index is None:
    sys.exit("no picker_index.json — run tools/picker/index_library.py first")
ROOT = Path(_index["root"])
PHOTOS = {p["id"]: p for p in _index["photos"]}


def load_selections() -> dict:
    if SELECTIONS.exists():
        return json.loads(SELECTIONS.read_text(encoding="utf-8"))
    return {}


def save_selections(sel: dict) -> None:
    SELECTIONS.write_text(json.dumps(sel, indent=1), encoding="utf-8", newline="\n")


def thumb(photo_id: str, size: int) -> Path:
    """Cache thumbnails on disk; the library is far too large to re-decode."""
    key = photo_id.replace("/", "_")
    out = CACHE / f"{size}_{key}"
    if not out.exists():
        src = ROOT / photo_id
        if not src.exists():
            abort(404)
        im = Image.open(src)
        im.draft("RGB", (size * 2, size * 2))          # fast partial decode
        im = ImageOps.exif_transpose(im).convert("RGB")  # or portraits show sideways
        im.thumbnail((size, size), Image.LANCZOS)
        im.save(out, "JPEG", quality=80)
    return out


@app.route("/thumb/<path:photo_id>")
def route_thumb(photo_id: str):
    return send_file(thumb(photo_id, int(request.args.get("s", 320))),
                     mimetype="image/jpeg")


@app.route("/api/photos")
def api_photos():
    sel = load_selections()
    out = _index["photos"]
    if (q := request.args.get("outing")):
        out = [p for p in out if p["outing"] == q]
    if (q := request.args.get("light")):
        out = [p for p in out if p["light"] == q]
    if (q := request.args.get("camera")):
        out = [p for p in out if p["camera"] == q]
    if request.args.get("starred") == "1":
        out = [p for p in out if p["id"] in sel]
    return jsonify({
        "photos": [{**p, "starred": p["id"] in sel, **sel.get(p["id"], {})}
                   for p in out],
        "total": len(out),
    })


@app.route("/api/facets")
def api_facets():
    from collections import Counter
    ps = _index["photos"]
    return jsonify({
        "outings": [{"name": k, "n": v} for k, v in
                    sorted(Counter(p["outing"] for p in ps).items())],
        "lights": [{"name": k, "n": v} for k, v in
                   Counter(p["light"] for p in ps).most_common()],
        "cameras": [{"name": k, "n": v} for k, v in
                    Counter(p["camera"] for p in ps).most_common()],
        "starred": len(load_selections()),
        "published": len(json.loads(GALLERY.read_text(encoding="utf-8"))["photos"]),
    })


@app.route("/api/star/<path:photo_id>", methods=["POST"])
def api_star(photo_id: str):
    if photo_id not in PHOTOS:
        abort(404)
    sel = load_selections()
    body = request.get_json(silent=True) or {}
    if body.get("starred") is False:
        sel.pop(photo_id, None)
    else:
        entry = sel.get(photo_id, {"category": "landscape", "alt": ""})
        entry.update({k: v for k, v in body.items()
                      if k in ("category", "alt")})
        sel[photo_id] = entry
    save_selections(sel)
    return jsonify({"ok": True, "starred": len(sel)})


def publish_one(photo_id: str, meta: dict) -> str:
    """Correct -> stage as a master -> return the gallery id."""
    src = ROOT / photo_id
    when = PHOTOS[photo_id]["date"].replace("-", "")
    stem = f"{when}-{src.stem}"
    CORRECTED.mkdir(parents=True, exist_ok=True)
    correct_colour.process(src, CORRECTED / f"{stem}.jpg", None)
    return stem


@app.route("/api/publish", methods=["POST"])
def api_publish():
    sel = load_selections()
    if not sel:
        return jsonify({"ok": False, "error": "nothing starred"}), 400
    missing = [i for i, m in sel.items() if not m.get("alt", "").strip()]
    if missing:
        return jsonify({"ok": False,
                        "error": f"{len(missing)} starred photo(s) need alt text",
                        "missing": missing[:10]}), 400

    gallery = json.loads(GALLERY.read_text(encoding="utf-8"))
    known = {p["id"] for p in gallery["photos"]}
    log, added = [], []

    for photo_id, meta in sel.items():
        stem = publish_one(photo_id, meta)
        if stem in known:
            log.append(f"skip {stem} (already published)")
            continue
        added.append({"id": stem, "category": meta["category"],
                      "attributes": [], "alt": meta["alt"].strip()})
        log.append(f"corrected {stem}")

    if added:
        # Landscapes lead, so new landscapes go to the front and the rest append.
        land = [a for a in added if a["category"] == "landscape"]
        rest = [a for a in added if a["category"] != "landscape"]
        gallery["photos"] = land + gallery["photos"] + rest
        GALLERY.write_text(json.dumps(gallery, indent=2) + "\n",
                           encoding="utf-8", newline="\n")

    for cmd in (["optimize_images.py"], ["build.py"]):
        r = subprocess.run([sys.executable, *cmd], cwd=REPO,
                           capture_output=True, text=True)
        log.append(f"$ python {cmd[0]} -> exit {r.returncode}")
        if r.returncode:
            log.append(r.stderr[-800:])
            return jsonify({"ok": False, "log": log}), 500
        log.append(r.stdout.strip()[-400:])

    save_selections({})
    return jsonify({"ok": True, "added": len(added), "log": log})


@app.route("/")
def index():
    return (HERE / "ui.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    print(f"library root : {ROOT}")
    print(f"indexed      : {len(PHOTOS)} photos")
    print(f"starred      : {len(load_selections())}")
    app.run(host="127.0.0.1", port=5000, debug=False)
