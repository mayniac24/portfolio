# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this site is

A photography portfolio for **Mayniac Creations** (Colorado). As of 2026-08-29
the stated identity is **landscapes leading, client work alongside** — not a
wedding site with landscapes bolted on. Title, tagline, metadata and gallery
order all encode that. If you change one, change all of them.

Static site, no server, no build step at deploy time. `python build.py`
regenerates `index.html` locally and the result is committed.

## Architecture

```
data/gallery.json     SOURCE OF TRUTH. Photo list, categories, alt text, order.
build.py              Renders gallery + filter bar into index.html markers.
index.html            Generated between markers; CSS/JS live here by hand.
optimize_images.py    ../Portfolio Corrected -> images/ derived variants.
service-worker.js     Offline cache. Network-first HTML, variant-only images.
tests/                Regression guards. Run them.
tools/                One-off migrations. Historical; do not run again.
```

**Never hand-edit the gallery markup in `index.html`.** It lives between
`<!-- GALLERY:START -->` / `<!-- GALLERY:END -->` and
`<!-- FILTERS:START -->` / `<!-- FILTERS:END -->`, and `build.py` overwrites both.
Edit `data/gallery.json` and rebuild. `tests/test_build.py` fails if you forget.

## Image directories

| Directory | Contents | Deployed |
|---|---|---|
| `../Portfolio Selections/` | Uncorrected original JPEGs | No |
| `../Portfolio Corrected/` | Colour-corrected masters. **Input to `optimize_images.py`.** | No |
| `images/` | Derived variants only: 400/800/1200/2000 in jpg + webp | Yes |

**The RAW files no longer exist.** Searched C:, D:, E:, M: and every recycle bin
on 2026-08-29 — gone. The JPEGs in `Portfolio Selections` are the only masters,
so nothing may overwrite that directory. (Lightroom *CC* is installed, which is
cloud-backed; originals may still exist at lightroom.adobe.com. Unverified.)

## Adding a photo

```bash
cp new-photo.jpg "../Portfolio Corrected/"
python optimize_images.py          # generates 8 variants
# add an entry to data/gallery.json: id, category, attributes, alt
python build.py
python -m pytest tests/ -q
```

The `id` is the filename stem. Alt text is required — a test enforces it.

## Taxonomy

Two **orthogonal** axes. Keeping them separate is the whole point.

| Axis | Values | Where |
|---|---|---|
| Orientation | `is-wide`, `is-tall` | CSS class, derived from real pixels |
| Category | `landscape`, `wedding`, `maternity` (+ engagement, family, portrait, event reserved) | `data-category`, one per photo |
| Attributes | `bw` | `data-attributes`, metadata only |

Historical bug (F3): `landscape` was used as *both* a CSS orientation class and
a category, so the categoriser tagged everything `portrait landscape ...`. The
filter bar showed "Portraits (23)" out of 23 images. Orientation was renamed to
`is-wide`/`is-tall`, which frees `landscape` to be a real category.
`tests/test_taxonomy.py` guards this.

## Things that will bite you

- **`sizes` differs by orientation.** Wide tiles span two grid columns, tall
  span one, so tall images use half the widths (`SIZES_TALL`). Getting this
  wrong makes portrait images fetch a 2× oversized variant. `build.py` shipped
  this bug for one commit; `test_build.py` now catches it.
- **The lightbox must never read `img.src`.** That attribute is the JPEG
  fallback. Use `data-lightbox-src`. Reading `src` pulled full-resolution
  originals on every open (finding F1, ~8MB per image).
- **Bump `CACHE_NAME` in `service-worker.js`** when changing HTML or JS, or
  returning visitors keep the old page.
- **`auto_categorize.py` is stale** — it still hardcodes `M:\Photography\...`,
  a path that stopped existing in April 2026, and it rewrites HTML directly.
  It has uncommitted local changes. Do not run it. It should be rewritten to
  emit `gallery.json` and nothing else.

## Deployment

Netlify is intended to be canonical (`https://mayniac-portfolio.netlify.app/`);
GitHub Pages is still live and not yet retired. See the spec's phase 2.
Contact form is Formspree and works — leave it alone.

## Specs and plans

- `docs/superpowers/specs/2026-08-28-portfolio-overhaul-design.md` — design
- `docs/superpowers/plans/` — per-phase implementation plans
- `docs/DECISIONS.md` — what was decided and why, including approaches that
  failed. Read this before proposing photo-selection automation.
