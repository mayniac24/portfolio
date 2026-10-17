# Editing photos on the live site — by hand, no Claude

Written 2026-09-25. Everything below is plain Python + git, run from this folder.

## How the site is built

| What | Where | Edit it? |
|---|---|---|
| Colour-corrected masters — what the site is built from | `M:\Photos & Videos\Portfolio Corrected\` | **Yes — this is where edits go** |
| Uncorrected originals — the only surviving masters (the RAWs are gone) | `M:\Photos & Videos\Portfolio Selections\` | **Never overwrite or export into it** |
| Web sizes (400/800/1200/2000 px, jpg + webp) | `images\` in this repo | No — generated |
| Photo list, order, category, alt text | `data\gallery.json` | Yes |
| The gallery part of the page | `index.html` | No — `build.py` writes it |

A photo's **id is its filename without the extension**, e.g. `20240326-DSC_0482`.
To find one from the site, right-click the photo → open image in new tab: the
file is `<id>-1200.jpg`.

**Going live:** pushing to `master` on GitHub deploys the site automatically,
to **Netlify** (the real site, <https://mayniac-portfolio.netlify.app/>) and to
the old GitHub Pages copy. It takes a minute or two.

## One-time setup

Open PowerShell in this folder:

```powershell
cd "M:\Photos & Videos\Portfolio Website"
pip install -r requirements.txt
```

## Re-edit a photo that's already on the site

1. **Edit the master** `Portfolio Corrected\<id>.jpg` in Lightroom or Photoshop.
2. **Export back over the same file, with the same filename**, as a full-size JPEG.
   - Don't let Lightroom rename it by export date. The id comes from the
     filename (a new name is a new photo), and the copyright year is read from
     its date prefix.
   - Vertical shots are fine either way: since 2026-09-25 the script applies
     the camera's rotation tag before resizing.
3. **Make the web sizes:**
   ```powershell
   python optimize_images.py
   ```
   This redoes all 8 sizes for every master, which takes a few minutes.
4. **Rebuild the page** (needed if your crop changed the shape; harmless otherwise):
   ```powershell
   python build.py
   ```
5. **Bump the cache** in `service-worker.js`: change
   `const CACHE_NAME = 'portfolio-v3';` to the next number (`'portfolio-v4'`).
   The site caches images by filename. Without this, anyone who has visited
   before keeps seeing the old version of the photo.
6. **Run the checks:**
   ```powershell
   python -m pytest tests/ -q
   ```
   All must pass.
7. **Look at it before it goes live:**
   ```powershell
   python -m http.server 8000
   ```
   Open <http://localhost:8000>, check the photo (upright, colours right,
   lightbox opens it), then press Ctrl+C.
8. **Commit and push only your photo's files:**
   ```powershell
   git status
   git add images/<id>-* index.html service-worker.js
   git commit -m "Re-edit <id>"
   git push
   ```
   Or use **GitHub Desktop**: open this repo, tick only `images/<id>-*`,
   `index.html` and `service-worker.js`, commit to `master`, then **Push origin**.
   - If `git status` lists images you didn't touch, leave them unstaged.
   - Leave the uncommitted `.gitignore`, `auto_categorize.py` and
     `tools/picker/*` changes out unless you mean to publish them.
9. **Check the live site** a couple of minutes later at
   <https://mayniac-portfolio.netlify.app/>. Press Ctrl+F5 to skip your
   browser's cache.

## Other changes

- **Alt text, category, order, or removing a photo:** edit `data\gallery.json`,
  then run steps 4–9 (skip `optimize_images.py`). To remove a photo, delete
  its entry; its `images\<id>-*` files can go too.
- **Adding new photos:** use the picker (`tools\picker\README.md`). For shots you
  corrected yourself in Lightroom, use `python tools\publish_external.py <export-dir>`
  to preview, then add `--apply` to publish.
  The repo's `CLAUDE.md` ("Adding a photo") has the bare commands.

## If something goes wrong on the live site

- **Put the previous version back immediately:** Netlify → your site → **Deploys**
  → click the last good deploy → **Publish deploy**. Afterwards, check the
  Deploys page. If it says auto publishing is stopped or locked, turn it back
  on after you've pushed a fix, or your next push won't go live.
- **Undo your last commit in git:** `git revert HEAD`, then `git push`.
