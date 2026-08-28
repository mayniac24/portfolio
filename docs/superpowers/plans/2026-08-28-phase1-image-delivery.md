# Phase 1: Image Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the site serving multi-megabyte full-resolution originals, and make sure returning visitors actually receive the fix.

**Architecture:** Add a 2000w variant tier so the responsive ladder no longer has a gap that forces browsers up to the original. Rewrite the gallery markup with a one-off script to cap every `srcset` at 2000w, add intrinsic `width`/`height`, and add explicit `data-lightbox-src` targets. Repoint the lightbox JS at those attributes instead of `img.src`. Bump the service worker cache and make HTML network-first so the deploy is not masked. Finally relocate the originals out of the deployed tree.

**Tech Stack:** Python 3.12, Pillow 12.1, pytest 9.0, vanilla JS, static HTML.

**Branch:** `portfolio-overhaul` (already checked out)

**Spec:** `docs/superpowers/specs/2026-08-28-portfolio-overhaul-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `tests/test_image_delivery.py` | Create. Regression guard — the contract that keeps F1/F2 dead. |
| `tools/rewrite_gallery_markup.py` | Create. One-off markup rewrite. Superseded by `build.py` in phase 3. |
| `optimize_images.py` | Modify. Add 2000w tier; later repoint input at the originals directory. |
| `index.html` | Modify. Gallery markup (by script) and lightbox JS (by hand). |
| `service-worker.js` | Modify. Cache version, network-first HTML, variant-only image caching. |
| `requirements.txt` | Create. `pillow`, `pytest`. |

---

## Task 1: Regression guard tests

These tests define "fixed". They must fail now and stay passing forever after.

**Files:**
- Create: `tests/test_image_delivery.py`
- Create: `requirements.txt`

- [ ] **Step 1: Create `requirements.txt`**

```
pillow>=11
pytest>=8
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_image_delivery.py`:

```python
"""Regression guards for image delivery (spec findings F1, F2).

These encode the contract that no full-resolution original is ever
referenced by the deployed page.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "index.html"
MAX_WIDTH = 2000

SRCSET_RE = re.compile(r'srcset="([^"]+)"')
CANDIDATE_RE = re.compile(r"(\S+)\s+(\d+)w")
IMG_SRC_RE = re.compile(r'<img[^>]*\ssrc="(images/[^"]+)"')
LIGHTBOX_RE = re.compile(r'data-lightbox-(?:src|fallback)="(images/[^"]+)"')
# Must list the known tiers explicitly. A loose `-(\d+)\.jpg$` would treat
# originals like `20260202-DSC_0026-1.jpg` as variants and silently pass.
VARIANT_RE = re.compile(r"-(?:400|800|1200|2000)\.(?:jpg|webp)$")
ASSET_RE = re.compile(r"images/[\w.-]+\.(?:jpg|webp)")


def _html():
    return INDEX.read_text(encoding="utf-8")


def _srcset_candidates():
    for srcset in SRCSET_RE.findall(_html()):
        for url, width in CANDIDATE_RE.findall(srcset):
            yield url, int(width)


def test_no_srcset_candidate_exceeds_max_width():
    oversized = [(u, w) for u, w in _srcset_candidates() if w > MAX_WIDTH]
    assert oversized == [], f"srcset candidates above {MAX_WIDTH}w: {oversized}"


def test_img_src_never_points_at_an_original():
    bad = [s for s in IMG_SRC_RE.findall(_html()) if not VARIANT_RE.search(s)]
    assert bad == [], f"<img src> pointing at full-resolution originals: {bad}"


def test_lightbox_targets_are_variants():
    targets = LIGHTBOX_RE.findall(_html())
    assert targets, "no data-lightbox-src attributes found in index.html"
    bad = [t for t in targets if not VARIANT_RE.search(t)]
    assert bad == [], f"lightbox targets pointing at originals: {bad}"


def test_lightbox_js_does_not_read_img_src():
    assert "lightboxImg.src = img.src" not in _html(), (
        "lightbox is still reading the src attribute (F1)"
    )


def test_every_referenced_asset_exists_on_disk():
    refs = sorted(set(ASSET_RE.findall(_html())))
    missing = [r for r in refs if not (REPO / r).exists()]
    assert missing == [], f"referenced but missing from disk: {missing}"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_image_delivery.py -v`

Expected: 4 failures and 1 pass. `test_no_srcset_candidate_exceeds_max_width` fails listing 6016w/4000w candidates; `test_img_src_never_points_at_an_original` fails listing 23 originals; `test_lightbox_targets_are_variants` fails with "no data-lightbox-src attributes found"; `test_lightbox_js_does_not_read_img_src` fails. `test_every_referenced_asset_exists_on_disk` passes already.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt tests/test_image_delivery.py
git commit -m "test: regression guards for full-resolution image delivery"
```

---

## Task 2: Add the 2000w variant tier

**Files:**
- Modify: `optimize_images.py:29` and `optimize_images.py:36-37`

- [ ] **Step 1: Widen `SIZES`**

In `optimize_images.py`, change line 29:

```python
SIZES = [400, 800, 1200]  # Width breakpoints for srcset
```

to:

```python
SIZES = [400, 800, 1200, 2000]  # Width breakpoints for srcset
```

- [ ] **Step 2: Stop the new tier being treated as a source image**

`get_image_files()` excludes existing variants by suffix. Without `-2000` in that
tuple the script would treat every `-2000.jpg` as a new original and generate
`-2000-400.jpg` and friends. Change:

```python
    return [f for f in INPUT_DIR.iterdir()
            if f.suffix.lower() in extensions and not f.stem.endswith(('-400', '-800', '-1200'))]
```

to:

```python
    return [f for f in INPUT_DIR.iterdir()
            if f.suffix.lower() in extensions
            and not any(f.stem.endswith(f"-{s}") for s in SIZES)]
```

- [ ] **Step 3: Generate the new variants**

Run: `python optimize_images.py`

Expected: for each of the 23 originals, `Created: <stem>-2000.jpg` and
`Created: <stem>-2000.webp`. Existing 400/800/1200 files are rewritten identically.
The script skips any width `>= original_width`, so nothing is upscaled.

- [ ] **Step 4: Confirm the tier exists**

Run: `ls images/*-2000.jpg | wc -l && ls images/*-2000.webp | wc -l`

Expected: `23` and `23`. (Every original is at least 4000px wide, so all 23 qualify.)

- [ ] **Step 5: Commit**

```bash
git add optimize_images.py images/*-2000.jpg images/*-2000.webp
git commit -m "feat: add 2000w variant tier to close the responsive ladder gap"
```

---

## Task 3: Rewrite the gallery markup

**Files:**
- Create: `tools/rewrite_gallery_markup.py`
- Modify: `index.html` (via the script)

- [ ] **Step 1: Write the rewrite script**

Create `tools/rewrite_gallery_markup.py`:

```python
#!/usr/bin/env python3
"""One-off rewrite of the gallery markup in index.html.

Caps every srcset at 2000w, repoints the src fallback away from the original,
adds intrinsic width/height (fixes layout shift), and adds explicit lightbox
targets so the lightbox stops reading the src attribute.

Superseded by build.py in phase 3, which must reproduce this exact markup.
"""

import re
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "index.html"
IMAGES = REPO / "images"
SIZES = [400, 800, 1200, 2000]

SOURCE_RE = re.compile(
    r'<source type="image/webp" srcset="images/(?P<stem>[^"]+)-400\.webp [^"]*" '
    r'sizes="(?P<sizes>[^"]*)">'
)
IMG_RE = re.compile(
    r'<img src="images/(?P<stem>[^"]+)\.jpg" srcset="[^"]*" sizes="(?P<sizes>[^"]*)" '
    r'alt="(?P<alt>[^"]*)" loading="lazy">'
)


def available(stem: str, ext: str) -> list[int]:
    return [s for s in SIZES if (IMAGES / f"{stem}-{s}.{ext}").exists()]


def srcset(stem: str, ext: str) -> str:
    return ", ".join(f"images/{stem}-{s}.{ext} {s}w" for s in available(stem, ext))


def rebuild_source(m: re.Match) -> str:
    stem = m.group("stem")
    return (
        f'<source type="image/webp" srcset="{srcset(stem, "webp")}" '
        f'sizes="{m.group("sizes")}">'
    )


def rebuild_img(m: re.Match) -> str:
    stem, sizes_attr, alt = m.group("stem"), m.group("sizes"), m.group("alt")
    jpgs = available(stem, "jpg")
    webps = available(stem, "webp")
    if not jpgs or not webps:
        raise SystemExit(f"no variants on disk for {stem}; run optimize_images.py first")

    largest_jpg = max(jpgs)
    largest_webp = max(webps)
    fallback = max([s for s in jpgs if s <= 1200], default=largest_jpg)

    with Image.open(IMAGES / f"{stem}-{largest_jpg}.jpg") as im:
        width, height = im.size

    return (
        f'<img src="images/{stem}-{fallback}.jpg" srcset="{srcset(stem, "jpg")}" '
        f'sizes="{sizes_attr}" width="{width}" height="{height}" '
        f'data-lightbox-src="images/{stem}-{largest_webp}.webp" '
        f'data-lightbox-fallback="images/{stem}-{largest_jpg}.jpg" '
        f'alt="{alt}" loading="lazy">'
    )


def main() -> None:
    html = INDEX.read_text(encoding="utf-8")
    html, n_source = SOURCE_RE.subn(rebuild_source, html)
    html, n_img = IMG_RE.subn(rebuild_img, html)
    INDEX.write_text(html, encoding="utf-8", newline="\n")
    print(f"rewrote {n_source} <source> and {n_img} <img> tags")
    if n_source != n_img:
        raise SystemExit(f"mismatch: {n_source} sources vs {n_img} images")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python tools/rewrite_gallery_markup.py`

Expected: `rewrote 23 <source> and 23 <img> tags`

If either count is not 23, stop — the markup is not as uniform as assumed. Do not
hand-patch; fix the regex and re-run against a clean `git checkout index.html`.

- [ ] **Step 3: Spot-check one entry**

Run: `grep -m1 -o 'data-lightbox-src="[^"]*"' index.html`

Expected: something of the form `data-lightbox-src="images/20250927-DSC_0209-2000.webp"`

- [ ] **Step 4: Run the guard tests**

Run: `python -m pytest tests/test_image_delivery.py -v`

Expected: 4 pass, 1 fail. Only `test_lightbox_js_does_not_read_img_src` still fails —
that is Task 4.

- [ ] **Step 5: Commit**

```bash
git add tools/rewrite_gallery_markup.py index.html
git commit -m "fix: cap gallery srcset at 2000w and add lightbox targets (F2)"
```

---

## Task 4: Point the lightbox at the variant

**Files:**
- Modify: `index.html` — `openLightbox()`, `changeImage()`, and the image error handler

- [ ] **Step 1: Add a resolver and update `openLightbox()`**

Find:

```javascript
        function openLightbox(index) {
            lastFocusedElement = document.activeElement;
            currentIndex = index;
            const img = visibleItems[index].querySelector('img');
            showLoading();
            lightboxImg.src = img.src;
            lightboxImg.alt = img.alt;
```

Replace with:

```javascript
        // Resolve the display-sized variant. Never img.src -- that attribute is the
        // JPEG fallback, and reading it pulled the full-resolution original (F1).
        function lightboxTarget(img) {
            return img.dataset.lightboxSrc || img.currentSrc || img.src;
        }

        function openLightbox(index) {
            lastFocusedElement = document.activeElement;
            currentIndex = index;
            const img = visibleItems[index].querySelector('img');
            showLoading();
            lightboxImg.dataset.fallback = img.dataset.lightboxFallback || '';
            lightboxImg.src = lightboxTarget(img);
            lightboxImg.alt = img.alt;
```

- [ ] **Step 2: Update `changeImage()`**

Find:

```javascript
            const img = visibleItems[currentIndex].querySelector('img');
            showLoading();
            lightboxImg.src = img.src;
            lightboxImg.alt = img.alt;
```

Replace with:

```javascript
            const img = visibleItems[currentIndex].querySelector('img');
            showLoading();
            lightboxImg.dataset.fallback = img.dataset.lightboxFallback || '';
            lightboxImg.src = lightboxTarget(img);
            lightboxImg.alt = img.alt;
```

- [ ] **Step 3: Add a one-shot JPEG fallback on the error handler**

Find:

```javascript
        lightboxImg.addEventListener('load', hideLoading);
        lightboxImg.addEventListener('error', hideLoading);
```

Replace with:

```javascript
        lightboxImg.addEventListener('load', hideLoading);
        lightboxImg.addEventListener('error', () => {
            // One-shot fallback for a browser without WebP support.
            const fb = lightboxImg.dataset.fallback;
            if (fb && !lightboxImg.getAttribute('src').endsWith(fb)) {
                lightboxImg.src = fb;
                return;
            }
            hideLoading();
        });
```

- [ ] **Step 4: Run the full guard suite**

Run: `python -m pytest tests/test_image_delivery.py -v`

Expected: 5 passed.

- [ ] **Step 5: Verify by hand in a browser**

Run: `python -m http.server 8000`

Open `http://localhost:8000`, open DevTools → Network → filter Img, click a gallery
image, then arrow through three more.

Expected: each lightbox open fetches a `-2000.webp` of roughly 150-400KB. No request
exceeds 1MB. Before this task the same interaction fetched 6-8MB per image.

Stop the server with Ctrl-C.

- [ ] **Step 6: Commit**

```bash
git add index.html
git commit -m "fix: lightbox loads the 2000w variant instead of the original (F1)"
```

---

## Task 5: Service worker — unmask the deploy

Without this, every returning visitor keeps the cached old page and sees none of
Tasks 1-4. It ships with them, not after.

**Files:**
- Modify: `service-worker.js`

- [ ] **Step 1: Bump the cache name**

Change:

```javascript
const CACHE_NAME = 'portfolio-v1';
```

to:

```javascript
const CACHE_NAME = 'portfolio-v2';
```

- [ ] **Step 2: Make HTML network-first and restrict image caching**

Replace the entire `fetch` listener (from `self.addEventListener('fetch'` to the end
of the file) with:

```javascript
// Only ever cache derived variants. Originals are no longer deployed, but this
// keeps a stray full-resolution request from being written to a visitor device.
const CACHEABLE_IMAGE = /-(?:400|800|1200|2000)\.(?:jpg|webp)$/i;

function isHTML(request) {
  return request.mode === 'navigate' ||
    (request.headers.get('accept') || '').includes('text/html');
}

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;

  // HTML is network-first: a deploy must never be masked by a stale cache.
  if (isHTML(event.request)) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => caches.match(event.request).then((c) => c || caches.match('/')))
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) return cachedResponse;

      return fetch(event.request).then((response) => {
        if (response.ok && CACHEABLE_IMAGE.test(new URL(event.request.url).pathname)) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return response;
      }).catch(() => {
        if (/\.(jpg|jpeg|png|webp)$/i.test(event.request.url)) {
          return new Response(
            '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300"><rect fill="#1a1a1a" width="400" height="300"/><text fill="#555" x="50%" y="50%" text-anchor="middle" dy=".3em" font-family="system-ui">Offline</text></svg>',
            { headers: { 'Content-Type': 'image/svg+xml' } }
          );
        }
      });
    })
  );
});
```

Note the stale-while-revalidate background refetch is gone deliberately: it re-fetched
every cached asset on every hit, which doubled image traffic for no benefit on a site
whose filenames are already content-versioned.

- [ ] **Step 2b: Verify the syntax parses**

Run: `node --check service-worker.js`

Expected: no output (success). If `node` is unavailable, skip — step 3 catches errors.

- [ ] **Step 3: Verify the upgrade path in a browser**

Run: `python -m http.server 8000`

1. Open `http://localhost:8000`, DevTools → Application → Service Workers. Confirm a
   worker is active and Cache Storage lists `portfolio-v2`.
2. Confirm `portfolio-v1` is gone (the existing `activate` handler deletes it).
3. Hard-reload, then normal-reload. Confirm the page still renders offline-capable:
   DevTools → Network → Offline, reload — the shell should still appear.

Stop the server with Ctrl-C.

- [ ] **Step 4: Commit**

```bash
git add service-worker.js
git commit -m "fix: network-first HTML and variant-only image caching (F4)"
```

---

## Task 6: Relocate the originals

Only after Tasks 3-5 are committed. These are what stop the site referencing
originals; moving them first would break the live page.

**Files:**
- Modify: `optimize_images.py:27-28` and `optimize_images.py:79-118`
- Move: `images/*.jpg` originals → `../Portfolio Selections/`
- Delete: `images/*.webp` full-size derivatives

- [ ] **Step 1: Confirm nothing references an original**

Run: `python -m pytest tests/test_image_delivery.py -v`

Expected: 5 passed. Do not proceed otherwise.

- [ ] **Step 2: Move the original JPEGs**

```bash
mkdir -p "../Portfolio Selections"
for f in images/*.jpg; do
  case "$f" in *-400.jpg|*-800.jpg|*-1200.jpg|*-2000.jpg) continue;; esac
  git mv "$f" "../Portfolio Selections/$(basename "$f")" 2>/dev/null || mv "$f" "../Portfolio Selections/"
done
ls "../Portfolio Selections" | wc -l
```

Expected: `23`

Note `../Portfolio Selections` is outside the repo, so `git mv` will refuse; the `mv`
fallback handles it and the deletion is staged in the next step.

- [ ] **Step 3: Delete the full-size WebP derivatives**

These are regenerable output, not source, so they are deleted rather than moved.

```bash
for f in images/*.webp; do
  case "$f" in *-400.webp|*-800.webp|*-1200.webp|*-2000.webp) continue;; esac
  rm "$f"
done
ls images/ | wc -l
```

Expected: `184` (23 images x 4 sizes x 2 formats). Was 230 after Task 2.

- [ ] **Step 4: Repoint the script at the new source directory**

In `optimize_images.py`, change:

```python
INPUT_DIR = Path("images")
OUTPUT_DIR = Path("images")  # Output to same directory
```

to:

```python
INPUT_DIR = Path("../Portfolio Selections")  # full-resolution masters, not deployed
OUTPUT_DIR = Path("images")  # derived variants, deployed
```

- [ ] **Step 5: Stop generating the full-size WebP**

In `optimize_image()`, delete these four lines:

```python
            # Create full-size WebP
            webp_full = OUTPUT_DIR / f"{stem}.webp"
            img.save(webp_full, 'WEBP', quality=WEBP_QUALITY)
            print(f"  Created: {webp_full.name}")
```

- [ ] **Step 6: Remove the dead snippet generator**

`generate_html_snippet()` emits markup with the original in the srcset — exactly the
bug Task 3 removed. Delete the whole function (`def generate_html_snippet` through the
closing `""")`), and delete its call in `main()`:

```python
    generate_html_snippet(image_files)
```

Also delete the now-inaccurate closing instructions in `main()`:

```python
    print("\nNext steps:")
    print("1. Review the generated HTML snippets above")
    print("2. Update index.html with <picture> elements")
    print("3. Test on different screen sizes")
```

- [ ] **Step 7: Verify the pipeline still works end to end**

Run: `python optimize_images.py`

Expected: `Found 23 images to process.` and, since every variant already exists, it
rewrites them in place with no errors. No `-2000-400.jpg` style files appear.

Run: `ls images/ | wc -l`

Expected: `184` — unchanged.

- [ ] **Step 8: Confirm the guards still pass and the tree shrank**

```bash
python -m pytest tests/test_image_delivery.py -v
du -sh images/
```

Expected: 5 passed, and `images/` roughly 40-60MB (was 221MB). The 400/800/1200 tiers
measure 1.7MB/5.4MB/11MB, and the new 2000w tier is the largest single contributor.
Record the actual figure in the commit message rather than trusting this estimate.

- [ ] **Step 9: Commit**

```bash
git add -A images optimize_images.py
git commit -m "refactor: move originals out of the deployed tree (spec section 3)"
```

---

## Task 7: Verify the phase

**Files:** none modified.

- [ ] **Step 1: Full suite**

Run: `python -m pytest tests/ -v`

Expected: 5 passed.

- [ ] **Step 2: Confirm no original is reachable from the deployed tree**

Run: `grep -oE 'images/[A-Za-z0-9_.-]+\.(jpg|webp)' index.html | sort -u | grep -vE '\-(400|800|1200|2000)\.'`

Expected: no output. Any line printed is a reference to a file that no longer exists.

- [ ] **Step 3: Measure the result**

Run: `python -m http.server 8000`

In DevTools → Network, hard-reload the page and record total transferred. Then open
the lightbox and arrow through five images and record again.

Expected: initial page load well under 2MB; each lightbox navigation 150-400KB. Record
both numbers in the commit message for the phase-2 Lighthouse comparison.

Stop the server with Ctrl-C.

- [ ] **Step 4: Tag the phase**

```bash
git tag -a phase1-image-delivery -m "Phase 1: image delivery, service worker, originals relocated"
git log --oneline f7acff4..HEAD
```

---

## Deferred to later phases

Recorded here so they are not mistaken for oversights:

- Category filters remain non-functional (F3) — phase 3, needs the taxonomy split.
- `auto_categorize.py` still holds dead `M:\Photography\...` paths (F5) — phase 3,
  where it is rewritten to target `gallery.json`.
- `<title>`, `rel="canonical"`, Netlify cutover (F6) — phase 2.
- CSS/JS extraction and the focus trap — phase 4.
- `tools/rewrite_gallery_markup.py` is a one-off. Phase 3's `build.py` must reproduce
  its output exactly; the post-Task-3 `index.html` is the fixture for that test.
