# Portfolio Picker

A local tool for choosing photos from the library and publishing them to the site.

## Why this exists

The candidate photos live on `M:` and run to hundreds of gigabytes. No hosted CMS
can see them, so selection has to happen on this machine.

It is deliberately **not** a ranker. Two attempts at automatically scoring photo
quality were built and both failed — one on sharpness, one on solar elevation plus
aesthetic proxies. The second's top results were a toddler in a pumpkin costume, a
dog on a bed, and sunsets shot over rooflines. File-level statistics cannot see
composition or subject. See `docs/DECISIONS.md`.

So this tool does the parts a machine is good at — indexing, thumbnailing,
filtering, colour correction, variant generation, wiring up the data — and leaves
the judgement to you.

## Setup

```bash
pip install -r requirements.txt
python tools/picker/index_library.py      # once; re-run after adding shoots
python tools/picker/app.py                # http://127.0.0.1:5000
```

The index covers real-camera (Nikon) frames only. Add `--all-cameras` to include
phone photos, or `--root <path>` to index a subtree.

## Using it

- Filter by outing, light band, or camera in the header.
- Click a thumbnail to view it large; click again to dismiss.
- Click ☆ to star. Starring reveals a category dropdown and an alt text field.
- **Alt text is required** — publishing refuses without it, and a test enforces it.
- Click **Publish**.

Publishing runs the whole pipeline for each starred photo:

1. colour-corrects it into `../Portfolio Corrected/` (shadow-based white balance,
   black point, selective desaturation — see `tools/correct_colour.py`)
2. runs `optimize_images.py` to generate the eight variants
3. appends an entry to `data/gallery.json`, landscapes to the front
4. runs `build.py` to regenerate `index.html`

**Nothing is committed.** Review with `git diff` and `git status`, run
`python -m pytest tests/ -q`, then commit yourself.

## Notes

- Binds to `127.0.0.1` only. There is no auth and it should not be exposed.
- The light band is descriptive, not a quality signal. Overcast defeats it: a low
  sun behind cloud still reads "golden" while the light is flat grey. That is
  precisely why it filters rather than ranks.
- Thumbnails cache to `.thumbs/`; first load of a large outing is slow, then fast.
- `picker_index.json`, `selections.json` and `.thumbs/` are gitignored — they hold
  machine-specific paths and in-progress work.
