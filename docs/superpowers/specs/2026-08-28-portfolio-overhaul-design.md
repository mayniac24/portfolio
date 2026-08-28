# Portfolio Site Overhaul — Design

**Date:** 2026-08-28
**Repo:** `M:\Photos & Videos\Portfolio Website` → `github.com/mayniac24/portfolio`
**Status:** Approved, pending implementation plan

## Context

The site is a static photography portfolio. Two decisions frame this work:

- **Purpose:** showcase now, client-facing later. Fix the foundation and get identity
  right so it can become a business site without a rewrite. Do not build lead-gen
  machinery yet.
- **Canonical host:** Netlify (`https://mayniac-portfolio.netlify.app/`). GitHub Pages
  is retired.

## Findings

Verified by reading the source, not inferred.

### F1 — The lightbox always downloads the full-resolution original

`openLightbox()` and `changeImage()` both do:

```js
lightboxImg.src = img.src;
```

`img.src` reflects the `src` **attribute**, not the srcset-selected candidate (that is
`currentSrc`). The `src` attribute points at the original — e.g.
`images/20250927-DSC_0209.jpg` at 6016px, 6-8MB. So every lightbox open fetches a
multi-megabyte JPEG, and arrow-keying the gallery pulls roughly 160MB. It also bypasses
WebP entirely: the lightbox is always JPEG. This fires on every device, every time.

### F2 — srcset can select the original on large displays

Every `srcset` lists the original as its largest candidate:

```
srcset="...-400.jpg 400w, ...-800.jpg 800w, ...-1200.jpg 1200w, ...DSC_0209.jpg 6016w"
sizes="(max-width: 480px) 100vw, (max-width: 768px) 50vw, 25vw"
```

On a 2560px viewport at DPR 2, a 25vw slot needs ~1280px. The largest real variant is
1200w — just short — so the browser jumps to the 6016w original. Narrow or non-retina
screens are unaffected. Root cause is shared with F1: `optimize_images.py` defines
`SIZES = [400, 800, 1200]`, so nothing exists between 1200px and the original.

### F3 — Category filters are non-functional

Every image carries both `portrait` and `landscape`:

```
9 x data-category="portrait landscape maternity"
9 x data-category="portrait landscape wedding"
3 x data-category="portrait landscape maternity family"
1 x data-category="portrait landscape wedding bw"
1 x data-category="portrait wedding"
```

"Portraits" matches all 23; "Landscapes" matches 22. The filter bar looks functional and
does nothing.

Cause: `landscape` means two different things in the same file — a CSS **orientation**
class (`class="gallery-item landscape"`, 15 images) and a **session type**. The
categorizer was given a vocabulary mixing both senses and applied both. A `bw` tag exists
with no matching filter button.

### F4 — Service worker will hide this entire overhaul from returning visitors

`CACHE_NAME` is pinned to `portfolio-v1` and never bumped, and the fetch handler is
cache-first for HTML. A returning visitor gets the cached `index.html` and never sees the
rebuild.

The same handler caches any URL matching an image extension, so every original pulled by
F1 is written to the visitor device permanently.

### F5 — Tooling is broken by the April 2026 folder rename

Uncommitted changes to `auto_categorize.py` hardcode paths under `M:\Photography\...`,
which has not existed since 2026-04-01 (renamed to `M:\Photos & Videos`). The script
cannot run as written.

### F6 — Metadata identity is generic; canonical tag missing

`<title>` is `Photography Portfolio` — no name, no location — while the JSON-LD
correctly describes *Mayniac Creations*, Colorado, weddings/elopements/maternity.
`og:url`, `og:image`, and the JSON-LD `url` already point at Netlify and are correct as
written; only `rel="canonical"` is absent.

## Non-goals

- No visual redesign. Layout, palette, and typography stay as they are.
- No lead-gen features (booking, pricing pages, CRM, analytics funnels).
- **No forms migration.** Formspree stays as-is. It works today, and replacing a working
  integration is not worth the churn in this pass.
- No custom domain purchase. The design must not obstruct adding one later.
- No git history rewrite.

## Design

### 1. Image delivery

The highest-value change, and it ships first.

- `optimize_images.py`: `SIZES = [400, 800, 1200, 2000]`. Regenerate variants. Never
  upscale — an original below a breakpoint simply does not get that variant.
- **Lightbox (F1):** replace `lightboxImg.src = img.src` with a read of a new
  `data-lightbox-src` attribute on each thumbnail, pointing at the largest available
  variant at or below 2000w, WebP preferred with a JPEG sibling for fallback.

  This attribute is the interim source of truth because `gallery.json` does not exist
  until section 3. When it does, `build.py` generates the identical attribute from the
  `variants` list, so the lightbox JS never changes again.
- **Grid (F2):** remove the original from every `srcset`; largest candidate becomes
  2000w. The `src` fallback points at `-1200.jpg`, never the original.
- Add `width`/`height` to every `<img>` to fix cumulative layout shift. Until
  `gallery.json` exists these come from reading the files with Pillow.
- Ship with the service worker changes in section 2, or returning visitors see none of it.

Worst-case image fetch drops from ~8MB to roughly 300-500KB.

**Cost of ordering this first:** the srcset, `width`/`height`, and `data-lightbox-src`
edits land in `index.html` via a one-off script, and section 4 later regenerates that
markup from a template. This is not wasted work — the hand-edited output defines the
exact target the template must reproduce, and gives the section 4 tests something real
to diff against.

### 2. Service worker

- Bump the cache name to `portfolio-v2`, and make HTML **network-first** so this deploy
  and every future one are never masked by a stale cache (F4).
- Restrict image caching to the derived variants. Section 5 removes originals from the
  deploy anyway, so this is defense-in-depth: it keeps a stray full-resolution request
  from ever being written to a visitor device again.

### 3. Originals relocation

Originals (46 files, ~200MB of the 221MB `images/` folder) move to
`M:\Photos & Videos\Portfolio Selections`. That directory already exists, is currently
empty, and CLAUDE.md already documents it as the intended source location.

Must land **after** sections 1 and 2, since those are what stop the site referencing
originals. `images/` then retains only derived variants and the working tree drops to
roughly 35MB. Git history is left intact by decision, so a fresh clone stays ~410MB —
acceptable for a static site and non-destructive.

### 4. Hosting cutover

- Add `netlify.toml`: no build command, `publish = "."`, long-lived cache headers for
  `images/*`, short cache for `index.html`.
- Delete `.github/workflows/deploy.yml`; disable GitHub Pages on the repo. This also
  clears the two `pages-build-deployment` runs stuck queued since 2026-02-02.
- Add `<link rel="canonical" href="https://mayniac-portfolio.netlify.app/">`.

Verify Netlify is building before disabling Pages.

### 5. Taxonomy

Two orthogonal axes, no shared vocabulary:

| Axis | Values | Purpose |
|---|---|---|
| Orientation | `is-wide`, `is-tall` | CSS grid spanning only. Renamed to end the collision. |
| Session | wedding, engagement, maternity, family, portrait, event | The filter bar. Exactly one per image. |
| Attributes | `bw` | Metadata only. No filter button until the count justifies one. |

All 23 images are re-tagged. The vision prompt in `auto_categorize.py` is rewritten to
select exactly one session type from an explicit list and to never emit orientation
words — that missing constraint is what produced F3. **The generated mapping is reviewed
by the owner before it lands.**

### 6. Data model and build pipeline

`data/gallery.json` becomes the single source of truth:

```json
{
  "id": "20250927-DSC_0209",
  "session": "wedding",
  "orientation": "is-wide",
  "attributes": [],
  "alt": "Wedding couple poses by a stone chapel with mountains behind.",
  "width": 6016,
  "height": 4016,
  "variants": [400, 800, 1200, 2000]
}
```

Responsibilities split cleanly:

- `optimize_images.py` — writes derived variants, records true `width`/`height`
- `auto_categorize.py` — writes **only** `gallery.json`, never touches HTML. Its dead
  `M:\Photography\...` paths are repaired here (F5).
- `build.py` — renders `index.html` from a Jinja2 template, reproducing exactly the
  markup section 1 established by hand

This ends the regex-surgery-on-HTML approach, which is fragile and is what makes the
current script risky to run.

Adding a photo becomes: drop the file, run `python build.py`, commit, Netlify deploys.

Add a `requirements.txt` (`pillow`, `requests`, `jinja2`); the repo currently has none.

### 7. Structure and metadata

- Extract `css/styles.css` and `js/main.js`. `index.html` becomes ~150 lines of structure
  plus a generated gallery block.
- `<title>` becomes `Mayniac Creations — Colorado Mountain & Elopement Photography`;
  align the OG/Twitter titles.
- Add a focus trap to the lightbox. Escape, arrow keys, `role="dialog"`, `aria-modal`,
  the `aria-live` counter, and focus save/restore are **already implemented** and need no
  work.
- Add a `prefers-reduced-motion` guard.
- Update CLAUDE.md: it documents the stale four-category scheme
  (landscape/portrait/nature/urban), omits `optimize_images.py`, and its deployment notes
  describe steps already completed (Formspree ID, PWA icons, `og:url`).

## Testing

- **Regression guard (highest value):** assert no `srcset` entry exceeds 2000w, and that
  no `src` or `data-lightbox-src` resolves to an original. This is the test that keeps
  F1/F2 from returning, and it must be written in phase 1 against the hand-edited HTML.
- Asset checker: every variant referenced by the HTML exists on disk.
- `build.py` unit tests: `gallery.json` to expected HTML fragment. The phase 1 output is
  the fixture — the template must reproduce it byte-for-byte modulo whitespace.
- Taxonomy check: every entry has exactly one session value; no entry uses an orientation
  word as a session.
- Lighthouse before/after on the deployed Netlify URL.

## Phasing

1. **Image delivery + service worker (F1, F2, F4)** — the largest user-visible win, and
   the service worker fix must ride along or returning visitors see nothing change.
   Then relocate originals.
2. **Hosting cutover (F6 canonical)** — retire Pages, add `netlify.toml`.
3. **Data model, build pipeline, taxonomy re-tag (F3, F5).**
4. **Structure split, metadata, CLAUDE.md (F6).**

## Risks

- The service worker fix must ship in phase 1. If image fixes land without it, returning
  visitors keep the cached old page and the work looks like it did nothing.
- Relocating originals before the srcset and lightbox fixes are deployed would break the
  live site. Order within phase 1 matters.
- Disabling Pages before confirming Netlify builds would take the site offline. Verify first.
- The re-tagging pass is model-generated and needs owner review before it lands.
- Moving originals out of `images/` breaks any external link to a full-resolution file.
  No such links are known in-repo; worth a check before the move.
