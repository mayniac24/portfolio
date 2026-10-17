#!/usr/bin/env python3
"""The face guard: no child's face is published unless it is covered or Evan cleared it.

Evan's rule (2026-10-17, binding): our kids' faces do not get posted anywhere, ever. If a
photo with them is shared, the faces are hidden behind a hard emoji or something equally
unrecoverable and still tasteful. This one tool is what every publisher calls. Unknown means
refuse: a detector that cannot run, a companion nobody can identify, an outing with no hike
log, an unreadable file.

    python tools/picker/faces.py check <image> [--json]
    python tools/picker/faces.py mask  <image> [--auto] [--box x,y,w,h]... [--out PATH]
                                       [--frame ID --outing OID | --key SLUG] [--by M] [--via V]
                                       [--checked-all-faces] [--apply]
    python tools/picker/faces.py clear <frame id | --key SLUG> --outing OID --by NAME
                                       [--i-am-at-the-desk] [--apply]
    python tools/picker/faces.py status <frame id | --key SLUG> --outing OID [--json]
    python tools/picker/faces.py gate  <image> (--frame ID --outing OID [--companions-file NOTE]
                                               | --no-frame [--key SLUG]) [--json]

Exit codes: 0 fine; 3 refused or failed (a detector error is a refusal: the guard fails
CLOSED, so a caller that only checks the exit code is safe); 2 bad usage.

DETECTION is LOCAL ONLY: no network, no model download. The detectors that are available on
the machine all run and their boxes are unioned:
  * the Windows Runtime FaceDetector (a component of Windows; no model file to fetch), on four
    rotations (0/90/180/270). This is the only one on Evan's PC today;
  * OpenCV Haar cascades (frontal default, alt2, profile and mirrored profile), only if this
    OpenCV build has a CascadeClassifier AND the XML files are present (OpenCV 5 has neither);
  * OpenCV YuNet, only if a face_detection_yunet*.onnx already sits in cv2's data dir or
    `face_models_dir`.
With none available the guard refuses everything that is not already cleared.

Scan: the whole image downscaled to ~1600 px; for an image over 1600 px also full-resolution
~1600 px tiles with 25% overlap (so a face under ~400 px lands whole in some tile); for an
image under 1200 px also an upscaled copy. Boxes are mapped back and merged.

MEASURED LIMITS (Windows FaceDetector, two adult faces from the live site pasted on a grey
canvas at the scan scale): it finds faces about 70 px wide and up; below ~56-64 px it misses.
It is frontal-oriented: in-plane tilts up to ~35 degrees off any of the four axes were found,
tilts of about 40-55 degrees (45 above all) were missed on both faces. Profiles, heads
turned away, faces behind sunglasses or hands, and heavy shadow are not reliably found. This is
NOT a guarantee. The backstops are the companion rule (a frame from an outing with children,
or with companions nobody can identify, is refused even when no face is found), the
masked-file re-check, and Evan's own review of every masked frame. Non-hike outings (client
sessions) have no hike log, so every frame needs an owner `clear` or a mask.

MASKING: boxes are in the coordinates of the image as DISPLAYED (EXIF orientation applied),
because the masked output carries no orientation tag. Each box is padded 35% per side, an
opaque ellipse in the region's mean colour is drawn over it and an emoji on top: no blur or
pixelation, no layers, nothing of the face survives. After writing, the detector runs on the
OUTPUT; if any face is still found nothing is kept or recorded. `mask` also refuses when
fewer boxes were given than faces were detected on the source.

STATE lives on the studio bus as `faces` events (`studio-picker.jsonl`); the last honoured
event for a frame or key wins. A `masked` row allows publishing only the file whose sha1 it
holds, and only if the detector finds no face in it; with children (or unidentifiable
companions) aboard it must also have been made with `--auto --checked-all-faces`. A `cleared`
row counts only if the person who made it (`member`, or `cleared_by` on a key row copied by
publish_external) is in `face_clearers` now. `clear` needs an interactive terminal, or
`--i-am-at-the-desk` for the Horizons desktop door; a script or worker cannot clear.
Settings (studio_link.py): face_clearers, child_relationships, face_emoji, face_models_dir.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import studio_link as link  # noqa: E402

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat
except ImportError:  # the guard cannot work without Pillow: every entry point then refuses
    Image = None

VERSION = 1
EXIT_REFUSED = 3
PAD = 0.35                       # each side, as a fraction of the box
LONG_EDGE = 1600                 # detection copy and tile size
TILE_OVERLAP = 0.25
JPEG_QUALITY = 92
EMOJI_FONT = "seguiemj.ttf"
MINOR_YEARS = 18


class FaceError(Exception):
    """Anything that stops the guard from being sure. Callers treat it as a refusal."""


# ------------------------------------------------------------------ pixels

def load_pixels(path):
    """-> (RGB image with EXIF orientation applied, source Exif, source ICC bytes)."""
    if Image is None:
        raise FaceError("Pillow is not installed")
    try:
        with Image.open(path) as im:
            im.load()
            exif, icc = im.getexif(), im.info.get("icc_profile")
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
                rgba = im.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, (255, 255, 255))
                rgb.paste(rgba, mask=rgba.getchannel("A"))
            else:
                rgb = im.convert("RGB")
    except FaceError:
        raise
    except Exception as exc:  # noqa: BLE001 - unreadable means unknown means refuse
        raise FaceError("cannot read image %s: %s" % (Path(path).name, exc))
    return rgb, exif, icc


# ------------------------------------------------------------------ boxes

def merge_boxes(boxes, overlap=0.5):
    """Union boxes whose intersection covers more than `overlap` of the smaller one."""
    out = [list(map(float, b[:4])) + [float(b[4]) if len(b) > 4 else 1.0] for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                if iw <= 0 or ih <= 0:
                    continue
                if iw * ih > overlap * min(a[2] * a[3], b[2] * b[3]):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    out[i] = [x0, y0, x1 - x0, y1 - y0, max(a[4], b[4])]
                    del out[j]
                    changed = True
                    break
            if changed:
                break
    return out


def _clamp_box(b, W, H):
    x0, y0 = max(0.0, b[0]), max(0.0, b[1])
    x1, y1 = min(float(W), b[0] + b[2]), min(float(H), b[1] + b[3])
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0), b[4] if len(b) > 4 else 1.0]


def unrotate_box(k, box, W, H):
    """A box found in the image rotated k*90 degrees CCW -> the same box in the original
    (W x H). Maps both corners and takes the bounding box."""
    x, y, w, h = box[:4]

    def back(u, v):
        if k == 1:
            return W - v, u
        if k == 2:
            return W - u, H - v
        if k == 3:
            return v, H - u
        return u, v
    (a, b), (c, d) = back(x, y), back(x + w, y + h)
    return [min(a, c), min(b, d), abs(c - a), abs(d - b)] + list(box[4:5])


def parse_box(text):
    parts = [p for p in re.split(r"[,\s]+", str(text).strip()) if p]
    if len(parts) != 4:
        raise ValueError("--box wants x,y,w,h; got %r" % (text,))
    x, y, w, h = (float(p) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError("--box width and height must be positive")
    return [x, y, w, h, 1.0]


def pad_box(b, pad=PAD):
    return [b[0] - pad * b[2], b[1] - pad * b[3], b[2] * (1 + 2 * pad), b[3] * (1 + 2 * pad)]


# ------------------------------------------------------------------ detectors

_WINFACE_PS = r"""
param([Parameter(Mandatory=$true)][string]$ListFile)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Media.FaceAnalysis.FaceDetector, Windows.Media.FaceAnalysis, ContentType=WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
  $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$type) {
  $t = $asTask.MakeGenericMethod($type).Invoke($null, @($op)); $null = $t.Wait(-1); $t.Result
}
if (-not [Windows.Media.FaceAnalysis.FaceDetector]::IsSupported) { Write-Output '{"error":"FaceDetector not supported"}'; exit 4 }
$det = Await ([Windows.Media.FaceAnalysis.FaceDetector]::CreateAsync()) ([Windows.Media.FaceAnalysis.FaceDetector])
$out = @{}
foreach ($p in (Get-Content -LiteralPath $ListFile -Encoding UTF8)) {
  if (-not $p) { continue }
  $name = [System.IO.Path]::GetFileName($p)
  $boxes = @()
  try {
    $f = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($p)) ([Windows.Storage.StorageFile])
    $s = Await ($f.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $dec = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($s)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bmp = Await ($dec.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $g = [Windows.Graphics.Imaging.SoftwareBitmap]::Convert($bmp, [Windows.Graphics.Imaging.BitmapPixelFormat]::Gray8)
    $faces = Await ($det.DetectFacesAsync($g)) ([System.Collections.Generic.IList[Windows.Media.FaceAnalysis.DetectedFace]])
    foreach ($fc in $faces) { $b = $fc.FaceBox; $boxes += ,@([int]$b.X, [int]$b.Y, [int]$b.Width, [int]$b.Height) }
    $out[$name] = @{ ok = $true; boxes = $boxes }
  } catch { $out[$name] = @{ ok = $false; error = $_.Exception.Message } }
}
$out | ConvertTo-Json -Depth 6 -Compress
"""


def _powershell():
    root = os.environ.get("SystemRoot") or r"C:\Windows"
    exe = Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    return str(exe) if exe.is_file() else shutil.which("powershell")


class WinRTDetector:
    """Windows.Media.FaceAnalysis.FaceDetector through a hidden PowerShell: local, no model
    file. Frontal and upright only, so each image also goes in rotated 90/180/270."""
    name = "winrt-facedetector"

    @staticmethod
    def available():
        return sys.platform == "win32" and bool(_powershell())

    def detect_many(self, images):
        if not self.available():
            raise FaceError("Windows FaceDetector is not available on this machine")
        with tempfile.TemporaryDirectory(prefix="faces-") as tmp:
            tmp = Path(tmp)
            jobs, lines = [], []
            for i, im in enumerate(images):
                W, H = im.size
                for k in range(4):
                    rot = im if k == 0 else im.rotate(90 * k, expand=True)
                    name = "%d_%d.png" % (i, k)
                    rot.save(tmp / name, "PNG", compress_level=1)
                    jobs.append((i, k, name, W, H))
                    lines.append(str(tmp / name))
            (tmp / "list.txt").write_text("\n".join(lines), encoding="utf-8")
            (tmp / "winface.ps1").write_text(_WINFACE_PS, encoding="utf-8-sig")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                r = subprocess.run(
                    [_powershell(), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                     "-File", str(tmp / "winface.ps1"), str(tmp / "list.txt")],
                    capture_output=True, timeout=300, creationflags=flags)
            except (OSError, subprocess.SubprocessError) as exc:
                raise FaceError("Windows FaceDetector failed to run: %s" % exc)
            text = r.stdout.decode("utf-8", errors="replace").strip()
            try:
                data = json.loads(text)
            except ValueError:
                raise FaceError("Windows FaceDetector gave no result (exit %s)" % r.returncode)
            if r.returncode or not isinstance(data, dict) or "error" in data:
                raise FaceError("Windows FaceDetector refused: %s" % (data.get("error") if isinstance(data, dict) else text[:200]))
            out = [[] for _ in images]
            for i, k, name, W, H in jobs:
                res = data.get(name)
                if not isinstance(res, dict) or not res.get("ok"):
                    raise FaceError("Windows FaceDetector could not read a working copy (%s)"
                                    % ((res or {}).get("error", "missing")))
                boxes = res.get("boxes") or []
                if boxes and not isinstance(boxes[0], (list, tuple)):
                    boxes = [boxes]
                for b in boxes:
                    out[i].append(unrotate_box(k, [b[0], b[1], b[2], b[3], 1.0], W, H))
            return out

    def detect(self, image):
        return self.detect_many([image])[0]


class HaarDetector:
    """OpenCV Haar cascades; the profile cascade also runs on the mirrored image."""
    name = "opencv-haar"
    FRONTAL = ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml")
    PROFILE = ("haarcascade_profileface.xml",)

    def __init__(self, cv2mod, folders):
        self.cv2, self.folders = cv2mod, [Path(f) for f in folders if f]

    @classmethod
    def locate(cls, settings=None):
        try:
            import cv2
        except Exception:  # noqa: BLE001
            return None
        if not hasattr(cv2, "CascadeClassifier"):
            return None
        folders = [(settings or {}).get("face_models_dir"), getattr(getattr(cv2, "data", None), "haarcascades", None)]
        det = cls(cv2, folders)
        return det if det._path(cls.FRONTAL[0]) else None

    def _path(self, name):
        for f in self.folders:
            if (f / name).is_file():
                return str(f / name)
        return None

    def detect(self, image):
        import numpy as np
        cv2 = self.cv2
        gray = cv2.equalizeHist(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY))
        W = gray.shape[1]
        flipped = cv2.flip(gray, 1)
        out = []
        for name in self.FRONTAL + self.PROFILE:
            path = self._path(name)
            if not path:
                continue
            cascade = cv2.CascadeClassifier(path)
            if cascade.empty():
                raise FaceError("cannot load cascade %s" % name)
            kw = dict(scaleFactor=1.08, minNeighbors=3, minSize=(24, 24))
            for x, y, w, h in cascade.detectMultiScale(gray, **kw):
                out.append([float(x), float(y), float(w), float(h), 1.0])
            if name in self.PROFILE:
                for x, y, w, h in cascade.detectMultiScale(flipped, **kw):
                    out.append([float(W - x - w), float(y), float(w), float(h), 1.0])
        return out


class YuNetDetector:
    """OpenCV's YuNet face detector, only when its onnx is already on this machine."""
    name = "opencv-yunet"

    def __init__(self, cv2mod, model):
        self.cv2, self.model = cv2mod, model

    @classmethod
    def locate(cls, settings=None):
        try:
            import cv2
        except Exception:  # noqa: BLE001
            return None
        if not hasattr(cv2, "FaceDetectorYN"):
            return None
        folders = [(settings or {}).get("face_models_dir"), getattr(getattr(cv2, "data", None), "__path__", [None])[0]]
        for f in folders:
            if f and Path(f).is_dir():
                for m in sorted(Path(f).glob("face_detection_yunet*.onnx")):
                    return cls(cv2, str(m))
        return None

    def detect(self, image):
        import numpy as np
        W, H = image.size
        det = self.cv2.FaceDetectorYN.create(self.model, "", (W, H), score_threshold=0.5)
        _, faces = det.detect(np.asarray(image)[:, :, ::-1].copy())
        return [] if faces is None else [[float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])]
                                         for f in faces]


def default_detectors(settings=None):
    found = []
    if WinRTDetector.available():
        found.append(WinRTDetector())
    for cls in (HaarDetector, YuNetDetector):
        det = cls.locate(settings)
        if det is not None:
            found.append(det)
    if not found:
        raise FaceError("no face detector is available (need the Windows FaceDetector, or OpenCV "
                        "Haar cascades / a YuNet onnx in face_models_dir); refusing")
    return found


def _detect_many(det, images):
    if hasattr(det, "detect_many"):
        return det.detect_many(images)
    fn = det.detect if hasattr(det, "detect") else det
    return [fn(im) for im in images]


def tile_origins(length, tile=LONG_EDGE, overlap=TILE_OVERLAP):
    """Start offsets of tiles along one axis: `tile` long, evenly spread so neighbours overlap by
    at least `overlap` and the last ends exactly at `length`. One tile when it already fits."""
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    n = -(-(length - tile) // step) + 1             # tiles needed at <= step spacing
    return [round(i * (length - tile) / (n - 1)) for i in range(n)]


def scan_passes(img):
    """[(copy, scale, ox, oy)]: every copy the detectors see. A copy's box maps back to the
    original as (x/scale + ox, y/scale + oy). The downscaled whole always runs; an image over
    LONG_EDGE also runs full-resolution ~1600 px tiles with 25% overlap (a face is a small
    part of a big frame), and a small one (under 1200 px) also runs an upscaled copy."""
    W, H = img.size
    long_edge = max(W, H)
    s = min(1.0, LONG_EDGE / long_edge)
    passes = [(img if s == 1.0 else img.resize((max(1, round(W * s)), max(1, round(H * s))), Image.LANCZOS),
               s, 0, 0)]
    if long_edge < 1200:
        up = LONG_EDGE / long_edge                   # a 400 px variant: faces are tiny
        passes.append((img.resize((max(1, round(W * up)), max(1, round(H * up))), Image.LANCZOS), up, 0, 0))
    elif long_edge > LONG_EDGE:
        for oy in tile_origins(H):
            for ox in tile_origins(W):
                passes.append((img.crop((ox, oy, min(W, ox + LONG_EDGE), min(H, oy + LONG_EDGE))), 1.0, ox, oy))
    return passes


def detect_faces(img, detectors):
    """Faces in an RGB image: the union over detectors of every scan pass (see scan_passes),
    boxes mapped back and merged. Returns (boxes in the image's own coordinates, names)."""
    W, H = img.size
    passes = scan_passes(img)
    copies = [p[0] for p in passes]
    found, names = [], []
    for det in detectors:
        names.append(getattr(det, "name", det.__class__.__name__))
        for (_, s, ox, oy), boxes in zip(passes, _detect_many(det, copies)):
            for b in boxes:
                found.append(_clamp_box([b[0] / s + ox, b[1] / s + oy, b[2] / s, b[3] / s,
                                         b[4] if len(b) > 4 else 1.0], W, H))
    found = [b for b in found if b[2] > 0 and b[3] > 0]
    return merge_boxes(found), names


def check_image(path, settings=None, detectors=None):
    """-> {"faces": [...], "detector": "...", "version": 1}. Raises FaceError."""
    img, _, _ = load_pixels(path)
    dets = detectors if detectors is not None else default_detectors(settings)
    if not dets:
        raise FaceError("no face detector supplied")
    boxes, names = detect_faces(img, dets)
    return {"faces": [{"x": int(round(b[0])), "y": int(round(b[1])), "w": int(round(b[2])),
                       "h": int(round(b[3])), "score": round(float(b[4]), 3)} for b in boxes],
            "detector": "+".join(names), "version": VERSION}


# ------------------------------------------------------------------ masking

def _emoji_font_path():
    root = os.environ.get("SystemRoot") or r"C:\Windows"
    p = Path(root) / "Fonts" / EMOJI_FONT
    return p if p.is_file() else None


def render_emoji(emoji, size=256):
    """A square RGBA tile of the emoji, or None when the font is missing or cannot draw it."""
    path = _emoji_font_path()
    if path is None or not emoji:
        return None
    try:
        font = ImageFont.truetype(str(path), 109)
        layer = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((40, 40), emoji, font=font, embedded_color=True)
        bbox = layer.getbbox()
        if not bbox:
            return None
        glyph = layer.crop(bbox)
        side = max(glyph.size)
        tile = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        tile.paste(glyph, ((side - glyph.width) // 2, (side - glyph.height) // 2))
        return tile.resize((size, size), Image.LANCZOS)
    except Exception:  # noqa: BLE001 - the ellipse alone is acceptable
        return None


def mask_pixels(img, boxes, emoji):
    """A copy of `img` with every box hidden: padded 35%, an opaque ellipse in the region's
    mean colour, an emoji centred on top. Runs on the full-resolution pixels."""
    out = img.copy()
    out.info = {}
    W, H = out.size
    tile = render_emoji(emoji)
    for b in boxes:
        px, py, pw, ph = pad_box(b)
        x0, y0, x1, y1 = (int(round(v)) for v in (px, py, px + pw, py + ph))
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        mean = tuple(int(round(v)) for v in ImageStat.Stat(out.crop((cx0, cy0, cx1, cy1))).mean[:3])
        ImageDraw.Draw(out).ellipse([x0, y0, x1 - 1, y1 - 1], fill=mean)
        if tile is not None:
            side = max(x1 - x0, y1 - y0)
            small = tile.resize((side, side), Image.LANCZOS)
            ox, oy = (x0 + x1) // 2 - side // 2, (y0 + y1) // 2 - side // 2
            sx0, sy0 = max(0, -ox), max(0, -oy)
            sx1, sy1 = min(side, W - ox), min(side, H - oy)
            if sx1 > sx0 and sy1 > sy0:
                part = small.crop((sx0, sy0, sx1, sy1))
                out.paste(part.convert("RGB"), (ox + sx0, oy + sy0), part.getchannel("A"))
    return out


def _optimize_images():
    path = REPO / "optimize_images.py"
    spec = importlib.util.spec_from_file_location("optimize_images_faces", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def share_metadata_for(stem, exif, icc, settings):
    """(exif bytes|None, icc bytes|None) exactly as optimize_images publishes them. If that
    module cannot be loaded, NO metadata is written (the strictest reading)."""
    try:
        oi = _optimize_images()
        return oi.published_metadata(stem, exif, icc, oi.share_allowlist(settings))
    except Exception as exc:  # noqa: BLE001
        print("  share_metadata unavailable (%s); writing no metadata" % exc, file=sys.stderr)
        return None, None


def write_masked(img, out_path, exif_bytes, icc, verify=None):
    """Encode to `<out>.part`, let `verify(part)` veto it (it raises), then rename into place."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")
    meta = {}
    if exif_bytes:
        meta["exif"] = exif_bytes
    if icc:
        meta["icc_profile"] = icc
    img.info = {}
    img.convert("RGB").save(tmp, "JPEG", quality=JPEG_QUALITY, optimize=True, **meta)
    if verify is not None:
        try:
            verify(tmp)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, out_path)
    return out_path


def default_out(image):
    image = Path(image)
    return image.with_name(image.stem + "_masked.jpg")


# ------------------------------------------------------------------ frame state (the bus)

def _ident_matches(row, ident, outing):
    if row.get("kind") != "faces" or ident not in (row.get("frame"), row.get("key")):
        return False
    return not (outing and row.get("outing") and row.get("outing") != outing)


def clearers(settings):
    allowed = (settings or {}).get("face_clearers")
    return {str(x).casefold() for x in allowed} if isinstance(allowed, list) else set()


def may_clear(settings, who):
    return bool(who) and str(who).casefold() in clearers(settings)


def _honoured(row, settings):
    """A masked row always counts (its sha is checked where it is used). A cleared row counts
    only when the person who cleared it is in face_clearers NOW: `cleared_by` on a key row
    copied by publish_external, else the row's `member`."""
    if row.get("state") == "masked":
        return True
    return row.get("state") == "cleared" and may_clear(settings, row.get("cleared_by") or row.get("member"))


def status_of(settings, ident, outing=None, events_dir=None):
    """Last honoured `faces` event for the frame or key: {"state": masked|cleared|none,
    image, sha1, by, auto, checked, detected, boxes}. A `cleared` row by anyone not in
    face_clearers is ignored, as if it were not there."""
    last = None
    for row in link.read_own_events(settings, events_dir):
        if _ident_matches(row, ident, outing) and _honoured(row, settings):
            last = row
    if last is None:
        return {"state": "none", "image": None, "sha1": None, "by": None}
    return {"state": last["state"], "image": last.get("image"), "sha1": last.get("sha1"),
            "by": last.get("cleared_by") or last.get("member"), "auto": bool(last.get("auto")),
            "checked": bool(last.get("checked")), "detected": last.get("detected"),
            "boxes": last.get("boxes")}


def has_clearance(ident, settings=None, events_dir=None, image=None):
    """True when the frame/key is cleared by a clearer, or masked AND `image` is the very file
    that was recorded (its sha1 matches). Any error is False."""
    try:
        st = status_of(settings or link.load_settings(), ident, None, events_dir)
        if st["state"] == "cleared":
            return True
        return bool(st["state"] == "masked" and st["sha1"] and image
                    and link.sha1_file(image) == st["sha1"])
    except Exception:  # noqa: BLE001
        return False


def record(settings, ident_kind, ident, state, *, outing=None, by="", via="cli", events_dir=None, **extra):
    payload = {ident_kind: ident, "state": state}
    payload.update({k: v for k, v in extra.items() if v is not None})
    return link.append_event(settings, "faces", outing=outing or None, via=via, member=by or None,
                             events_dir=events_dir, **payload)


# ------------------------------------------------------------------ companions

_WIKI = re.compile(r"\[\[([^\]|]*)(?:\|([^\]]*))?\]\]")


def _flatten(raw):
    if raw is None:
        return
    if isinstance(raw, (list, tuple)):
        for item in raw:
            yield from _flatten(item)
    else:
        text = str(raw).strip()
        if text:
            yield text


def split_names(raw):
    out = []
    for text in _flatten(raw):
        for part in re.split(r"\s*(?:,|;|&|\band\b)\s*", text):
            part = part.strip()
            m = _WIKI.fullmatch(part) if part else None
            if m:
                part = (m.group(2) or m.group(1).rsplit("/", 1)[-1]).strip()
            if part:
                out.append(part)
    return out


def _raw_tags(block):
    """Tags read straight from the frontmatter text, so `- #child` (which YAML takes as a
    comment and drops) and `'#child'` survive. Leading `#` is stripped."""
    out, inside = [], False
    for ln in block.splitlines():
        m = re.match(r"tags\s*:(.*)$", ln)
        if m:
            inside, rest = True, m.group(1)
        elif inside and re.match(r"[A-Za-z0-9_-]+\s*:", ln):
            inside, rest = False, ""
        elif inside:
            rest = re.sub(r"^\s*-\s*", " ", ln)
        else:
            continue
        out += [x.lstrip("#") for x in re.findall(r"#?[\w/-]+", rest)]
    return [x for x in out if x]


def read_frontmatter(path):
    """The note's frontmatter as a dict. `tags` is the union of what YAML parsed and what the
    raw text holds (see _raw_tags). Works without PyYAML (a minimal reader)."""
    text = Path(path).read_text(encoding="utf-8")
    m = re.match("﻿?---\r?\n(.*?)\r?\n---", text, re.S)
    if not m:
        return {}
    block = m.group(1)
    try:
        import yaml
        data = yaml.safe_load(block) or {}
        data = data if isinstance(data, dict) else {}
    except ImportError:
        data, key = {}, None                       # scalars, [a, b], and - lists (indented or not)
        for ln in block.splitlines():
            item = re.match(r"\s*-\s+(.*)", ln)
            kv = re.match(r"([A-Za-z0-9_-]+):\s*(.*)", ln)
            if kv:
                key, val = kv.group(1), kv.group(2).strip()
                if val.startswith("[") and val.endswith("]"):
                    data[key] = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
                else:
                    data[key] = val.strip("'\"") if val else []
            elif item and key and isinstance(data.get(key), list):
                data[key].append(item.group(1).strip().strip("'\""))
    tags = [str(x).lstrip("#") for x in _flatten(data.get("tags"))] + _raw_tags(block)
    if tags or "tags" in data:
        data["tags"] = sorted(set(t for t in tags if t))
    return data


def _vault_folder(settings, key):
    vault = link.vault_root_of(settings)
    if vault is None:
        raise FaceError("no vault root: cannot find the People notes")
    reg = json.loads((vault / "_vault_paths.json").read_text(encoding="utf-8"))
    rel = (reg.get("paths") or {}).get(key)
    if not rel:
        raise FaceError("_vault_paths.json has no %r key" % key)
    return vault / rel


def _dob_age(value, today=None):
    if isinstance(value, datetime):
        value = value.date()
    if not isinstance(value, date):
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(value or ""))
        if not m:
            return None
        try:
            value = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    today = today or date.today()
    return today.year - value.year - ((today.month, today.day) < (value.month, value.day))


class People:
    """The People notes by stem, alias and (when unique) first word; as the collage tool does."""

    def __init__(self, folder):
        self.by_name, self.count = {}, 0
        first = {}
        for path in sorted(Path(folder).rglob("*.md")):
            fm = read_frontmatter(path)
            if fm.get("type") != "person":
                continue
            rec = {"relationship": list(_flatten(fm.get("relationship"))),
                   "tags": list(_flatten(fm.get("tags"))),
                   "ages": [a for a in (_dob_age(fm.get(k)) for k in ("date_of_birth", "birthday")) if a is not None]}
            self.count += 1
            self.by_name.setdefault(path.stem.casefold(), rec)
            for a in _flatten(fm.get("aliases")):
                self.by_name.setdefault(a.casefold(), rec)
            first.setdefault(path.stem.split()[0].casefold(), []).append(rec)
        for word, recs in first.items():
            if len(recs) == 1:
                self.by_name.setdefault(word, recs[0])

    def find(self, name):
        return self.by_name.get(name.casefold())


def _word(w):
    return str(w).strip().lstrip("#").casefold()


def is_child(rec, child_words):
    """A child by `relationship` token or tag matching child_relationships (a leading # is
    ignored on both sides), or by a date_of_birth / birthday under 18."""
    words = {_word(w) for w in child_words} - {""}
    tokens = {_word(t) for r in rec["relationship"] for t in re.split(r"[/,;|\s]+", r) if t}
    tokens |= {_word(t) for t in rec["tags"]}
    return bool(tokens & words) or any(a < MINOR_YEARS for a in rec["ages"])


SOLO_WORDS = {"solo", "alone", "myself", "by myself", "just me", "none"}


def companion_names(settings, outing, companions_file):
    """-> (names, why). `names` is None when nothing says who was along; `why` then says why
    (never a name). An empty list means a hike log with blank companions: unknown, not solo.
    A note that cannot be read raises."""
    if companions_file:
        fm = read_frontmatter(companions_file)
        names, source = split_names(fm.get("companions")), "the companions note"
    else:
        vault = link.vault_root_of(settings)
        if not outing:
            return None, "no outing id"
        if vault is None:
            return None, "no vault root"
        try:
            data = json.loads((vault / "_studio_outings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None, "no outing projection"
        rec = next((r for r in (data.get("outings") or []) if isinstance(r, dict) and r.get("id") == outing), None)             if isinstance(data, dict) else None
        if rec is None:
            return None, "the outing is not in the projection"
        logs = rec.get("hike_logs") or ([rec["hike_log"]] if rec.get("hike_log") else [])
        if not logs:
            return None, "the outing has no hike log"
        names, source = [], "the hike log"
        for lg in logs:
            names += split_names((lg or {}).get("companions"))
    if not names:
        return None, "%s lists no companions (blank is unknown, not solo)" % source
    return names, source


def companion_check(settings, outing, companions_file):
    """{"known", "why", "companions", "children", "unresolved", "unset"}; never names.
    known=False: nobody can say who was along. unresolved: names with no People note.
    unset: there are companions but child_relationships is empty, so nobody can be ruled out.
    Raises FaceError when the People notes cannot be consulted."""
    names, why = companion_names(settings, outing, companions_file)
    out = {"known": names is not None, "why": why if names is None else None,
           "companions": 0, "children": 0, "unresolved": 0, "unset": False}
    if not names:
        return out
    named = [n for n in names if _word(n) not in SOLO_WORDS]
    if not named:
        return out                                    # every entry says "solo": nobody to look up
    people = People(_vault_folder(settings, "people"))
    words = settings.get("child_relationships") or []
    for n in named:
        rec = people.find(n)
        if rec is None:
            out["unresolved"] += 1
            out["companions"] += 1
        elif {_word(t) for r in rec["relationship"] for t in re.split(r"[/,;|\s]+", r) if t} == {"self"}:
            continue                                  # the owner
        else:
            out["companions"] += 1
            if is_child(rec, words):
                out["children"] += 1
    out["unset"] = bool(out["companions"] and not [w for w in words if _word(w)])
    return out


# ------------------------------------------------------------------ the gate

def _refuse(reason, **more):
    return dict({"allowed": False, "reason": reason}, **more)


def _allow(reason, **more):
    return dict({"allowed": True, "reason": reason}, **more)


def gate(image, *, frame=None, key=None, outing=None, settings=None, events_dir=None,
         companions_file=None, detectors=None):
    """May this image be published? -> {"allowed", "reason", ...}. Never raises: any error
    is a refusal, and so is anything unknown. `frame` selects the frame rules (a frame
    needs an outing, so the companions can be read), otherwise the no-frame rules (`key`)."""
    try:
        return _gate(image, frame, key, outing, settings, events_dir, companions_file, detectors)
    except Exception as exc:  # noqa: BLE001 - fail CLOSED
        return _refuse("error, so refused: %s" % exc)


def _masked_file_check(image, st, settings, detectors):
    """None when `image` is the recorded masked file AND the detector finds no face left in it,
    else the refusal."""
    if not (st["sha1"] and link.sha1_file(image) == st["sha1"]):
        return _refuse("this frame is masked: publish the masked file (%s), not this one"
                       % (st["image"] or "<stem>_masked.jpg"), state="masked")
    left = check_image(image, settings, detectors)
    if left["faces"]:
        return _refuse("%d face(s) are still detectable in the masked file (%s): mask the misses"
                       % (len(left["faces"]), left["detector"]), faces=len(left["faces"]), state="masked")
    if st.get("detected") is not None and st.get("boxes") is not None and st["boxes"] < st["detected"]:
        return _refuse("the mask was recorded with fewer boxes (%s) than faces detected (%s)"
                       % (st["boxes"], st["detected"]), state="masked")
    return None


def _gate(image, frame, key, outing, settings, events_dir, companions_file, detectors):
    if settings is None or not settings.get("_settings_file") or not Path(settings["_settings_file"]).is_file():
        return _refuse("the studio settings file is missing, so the guard cannot know the rules")
    if not Path(image).is_file():
        return _refuse("image not found: %s" % image)
    ident = frame or key
    st = status_of(settings, ident, outing, events_dir) if ident else {"state": "none"}
    if st["state"] == "cleared":
        return _allow("cleared by %s" % st.get("by"), state="cleared", by=st.get("by"))
    if st["state"] == "masked":
        bad = _masked_file_check(image, st, settings, detectors)
        if bad:
            return bad
        if not frame:
            return _allow("the masked file, no face left in it", state="masked")
        try:
            comp = companion_check(settings, outing, companions_file)
        except Exception:  # noqa: BLE001 - cannot read the People notes: assume the worst
            comp = {"known": True, "children": 0, "unresolved": 1, "unset": False, "error": True}
        if (comp["children"] or comp["unresolved"] or comp["unset"]) and not (st["auto"] and st["checked"]):
            return _refuse("children (or companions that cannot be ruled out) are on this outing: the "
                           "mask must come from `faces.py mask --auto --checked-all-faces`",
                           state="masked", companions=comp)
        return _allow("the masked file, no face left in it", state="masked", companions=comp)
    result = check_image(image, settings, detectors)
    n = len(result["faces"])
    if n:
        return _refuse("%d face(s) detected (%s): mask them with faces.py mask --auto, or have a "
                       "clearer clear the frame" % (n, result["detector"]),
                       faces=n, detector=result["detector"], state="none")
    if not frame:
        return _allow("no face detected", faces=0, detector=result["detector"], state="none")
    comp = companion_check(settings, outing, companions_file)
    more = dict(faces=0, detector=result["detector"], companions=comp, state="none")
    if not comp["known"]:
        return _refuse("companions unknown (%s), and detectors miss profiles and small faces: mask or "
                       "clear the frame (an outing without a hike log needs a clear per frame)" % comp["why"], **more)
    if comp["unresolved"]:
        return _refuse("%d companion(s) are not in the People notes, so nobody can be ruled out a child"
                       % comp["unresolved"], **more)
    if comp["unset"]:
        return _refuse("child_relationships is not configured, so nobody can be ruled out a child", **more)
    if comp["children"]:
        return _refuse("%d child(ren) on this outing's companion list and no face was found: detectors "
                       "miss profiles and small faces, so mask or clear it" % comp["children"], **more)
    return _allow("no face detected, and every companion is known and not a child", **more)


# ------------------------------------------------------------------ CLI

def _settings(a, strict):
    try:
        return link.load_settings(a.settings)
    except Exception:  # noqa: BLE001
        if strict:
            raise
        return {k: v for k, v in link.DEFAULTS.items()}


def _emit(a, obj, text=None):
    if getattr(a, "json", False) or text is None:
        print(json.dumps(obj, ensure_ascii=False))
    else:
        print(text)


def _interactive():
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def cmd_check(a):
    result = check_image(a.image, _settings(a, False))
    if a.json:
        print(json.dumps(result))
    else:
        print("%d face(s) by %s" % (len(result["faces"]), result["detector"]))
        for f in result["faces"]:
            print("  x=%(x)d y=%(y)d w=%(w)d h=%(h)d score=%(score)s" % f)
    return 0


def cmd_mask(a):
    settings = _settings(a, False)
    image = Path(a.image)
    img, exif, icc = load_pixels(image)
    dets = default_detectors(settings)            # no detector: cannot verify, so cannot mask
    found, names = detect_faces(img, dets)
    print("detected on the source (%s): %d" % ("+".join(names), len(found)))
    manual = [parse_box(b) for b in a.box or []]
    boxes = (list(found) if a.auto else []) + manual
    print("boxes: %d auto + %d manual" % (len(found) if a.auto else 0, len(manual)))
    if not boxes:
        print("nothing to mask: pass --auto or --box x,y,w,h")
        return 2
    for b in boxes:
        print("  x=%d y=%d w=%d h=%d" % tuple(round(v) for v in b[:4]))
    if len(boxes) < len(found):
        print("REFUSED: %d face(s) detected but only %d box(es): use --auto, or add a --box for each"
              % (len(found), len(boxes)))
        return EXIT_REFUSED
    out = Path(a.out) if a.out else default_out(image)
    if out.resolve() == image.resolve():
        print("refusing: the output would overwrite the source")
        return 2
    print("output: %s   emoji: %s" % (out, settings.get("face_emoji") or "(ellipse only)"))
    if not a.apply:
        print("DRY RUN - pass --apply to write")
        return 0
    masked = mask_pixels(img, boxes, settings.get("face_emoji") or "")
    ex, ic = share_metadata_for(image.stem, exif, icc, settings)

    def verify(part):
        left, _ = detect_faces(load_pixels(part)[0], dets)
        if left:
            raise FaceError("%d face(s) are still detectable in the masked output, so nothing was "
                            "written or recorded: add a --box for each miss" % len(left))
    write_masked(masked, out, ex, ic, verify=verify)
    sha = link.sha1_file(out)
    print("wrote %s  sha1 %s  (re-checked: no face detectable)" % (out, sha))
    ident = a.frame or a.key
    if ident:
        try:
            root = link.photos_root(settings)
        except Exception:  # noqa: BLE001
            root = None
        rec = link.rel_or_abs(out, root) if root else Path(os.path.abspath(out)).as_posix()
        record(settings, "frame" if a.frame else "key", ident, "masked", outing=a.outing, by=a.by,
               via=a.via, events_dir=a.events_dir, image=rec, sha1=sha, boxes=len(boxes),
               detected=len(found), auto=bool(a.auto), checked=bool(a.checked_all_faces))
        print("recorded: faces/masked for %s" % ident)
    return 0


def cmd_clear(a):
    settings = _settings(a, False)
    ident = a.ident or a.key
    if not ident:
        print("clear wants a frame id or --key")
        return 2
    if not may_clear(settings, a.by):
        print("refused: %r is not in face_clearers; only a listed person can clear a frame" % a.by)
        return EXIT_REFUSED
    if not (a.i_am_at_the_desk or _interactive()):
        print("refused: clearing a frame needs Evan at an interactive terminal (or the desktop door's "
              "--i-am-at-the-desk); a script or worker cannot clear")
        return EXIT_REFUSED
    if not a.apply:
        print("DRY RUN - would record faces/cleared for %s by %s" % (ident, a.by))
        return 0
    record(settings, "frame" if a.ident else "key", ident, "cleared", outing=a.outing, by=a.by,
           via=a.via, events_dir=a.events_dir)
    print("recorded: faces/cleared for %s" % ident)
    return 0


def cmd_status(a):
    ident = a.ident or a.key
    st = status_of(_settings(a, False), ident, a.outing, a.events_dir)
    _emit(a, st, st["state"] + ("   " + str(st["image"]) if st["image"] else ""))
    return 0


def cmd_gate(a):
    try:
        settings = link.load_settings(a.settings)
    except Exception as exc:  # noqa: BLE001
        res = _refuse("settings unreadable, so refused: %s" % exc)
    else:
        if a.no_frame:
            res = gate(a.image, key=a.key, outing=a.outing, settings=settings, events_dir=a.events_dir)
        elif a.frame:
            res = gate(a.image, frame=a.frame, outing=a.outing, settings=settings, events_dir=a.events_dir,
                       companions_file=a.companions_file)
        else:
            res = _refuse("pass --frame <id> --outing <id>, or --no-frame")
    _emit(a, res, ("ALLOWED: " if res["allowed"] else "REFUSED: ") + res["reason"])
    return 0 if res["allowed"] else EXIT_REFUSED


def build_parser():
    ap = argparse.ArgumentParser(description="The face guard (see the module docstring).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, outing=True):
        p.add_argument("--settings", help="_studio_settings.json (default: env / $VAULT_PATH)")
        p.add_argument("--events-dir")
        p.add_argument("--json", action="store_true")
        if outing:
            p.add_argument("--outing", help="the outing id")

    p = sub.add_parser("check"); p.add_argument("image"); common(p, False); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("mask"); p.add_argument("image"); common(p)
    p.add_argument("--auto", action="store_true"); p.add_argument("--box", action="append")
    p.add_argument("--out"); p.add_argument("--frame"); p.add_argument("--key")
    p.add_argument("--by", default=""); p.add_argument("--via", default="cli"); p.add_argument("--apply", action="store_true")
    p.add_argument("--checked-all-faces", action="store_true",
                   help="a person looked at the masked file and every face is covered; needed to "
                        "publish a masked frame from an outing with children")
    p.set_defaults(fn=cmd_mask)
    p = sub.add_parser("clear"); p.add_argument("ident", nargs="?"); common(p)
    p.add_argument("--key"); p.add_argument("--by", default=""); p.add_argument("--via", default="cli")
    p.add_argument("--i-am-at-the-desk", action="store_true",
                   help="for the Horizons desktop door only: a human pressed the button")
    p.add_argument("--apply", action="store_true"); p.set_defaults(fn=cmd_clear)
    p = sub.add_parser("status"); p.add_argument("ident", nargs="?"); common(p); p.add_argument("--key")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("gate"); p.add_argument("image"); common(p)
    p.add_argument("--frame"); p.add_argument("--no-frame", action="store_true"); p.add_argument("--key")
    p.add_argument("--companions-file"); p.set_defaults(fn=cmd_gate)
    return ap


def main(argv=None) -> int:
    link.stdout_utf8()
    a = build_parser().parse_args(argv)
    if a.cmd in ("clear", "status") and a.ident and not a.outing:
        print("%s of a frame wants --outing <id>" % a.cmd)
        return 2
    if a.cmd in ("mask", "gate") and a.frame and not a.outing:
        print("--frame wants --outing <id>")
        return 2
    try:
        return a.fn(a)
    except FaceError as exc:
        print("REFUSED: %s" % exc if a.cmd == "gate" else "failed: %s" % exc)
        return EXIT_REFUSED
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print("failed: %s" % exc)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
