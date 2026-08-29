# Decisions & Lessons

Running log of what was decided, why, and what was tried and failed. Read before
proposing work — several obvious-looking approaches here have already been tested
and rejected with evidence.

Last updated 2026-08-29.

---

## Site identity — landscapes lead

**Decided 2026-08-29 by the owner:** the site shows **both** landscape and client
work, **with landscapes leading**.

This overrides the earlier framing. On 2026-08-28 the site was scoped as
"showcase now, clients later" with a wedding/maternity taxonomy, and an
assistant pass then optimised the whole portfolio toward client work — ranking
images partly on whether they contained people, and recommending against adding
landscapes at all. The owner corrected this: *"I mainly photograph landscapes and
animals, not people, so why would I want the work I share to be imbalanced?"*

Consequences, all implemented:
- `<title>`, tagline, `<h1>`, OG/Twitter titles and JSON-LD lead with landscape
- `landscape` is a first-class category, listed first in the filter bar
- `data/gallery.json` orders landscapes first; `test_build.py` enforces it

**A photograph does not need a person in it to be portfolio-worthy.** The real
criteria are a compositional anchor and light with direction. Those are
genre-neutral.

---

## Photo selection cannot be automated. Two attempts failed.

This is the most important lesson here. **Do not build a third scorer without
reading this.**

### Attempt 1 — technical scoring (pre-existing, 2026-08-19)

`_Portfolio Cull *` folders ranked 23,787 images by sharpness, mean brightness
and blown-highlight fraction. Output: 220 landscape candidates.

**Failed.** Those metrics measure whether a photo is correctly exposed and in
focus, not whether it is any good. The top-ranked frame was the sharpest image in
the library, not the best. Nearly all picks were midday record shots with flat
light and no subject.

### Attempt 2 — solar elevation + aesthetic proxies (2026-08-29)

Scanned 28,624 JPEGs, computed sun elevation at capture time from EXIF, kept the
2,257 shot in golden or blue hour, then scored those on sky-cloud structure,
warm/cool spread, tonal range and near/far depth.

**Also failed.** The top of that ranking was a toddler in a pumpkin costume on a
deck, a dog on a bed, a backyard birch tree, sunsets shot over rooflines and
through power lines, and images already published on the site. Reasons:
- "Sky interest" measured luminance variance in the top third — a cluttered
  roofline scores identically to a dramatic cloudbank
- Overcast defeats the solar filter entirely: the sun is low, so the frame scores
  1.0, but the light is flat grey
- No file-level metric sees composition or subject

### What actually worked

Generating labelled contact sheets and looking at them. Every frame worth adding
was found by eye. `tools/` has no scorer in it for this reason.

**If selection needs to scale, build a browsing/starring tool for the owner —
not another ranker.**

---

## Colour correction — balance shadows, never highlights

`optimize_images.py` consumes `../Portfolio Corrected/`, whose contents were
produced by a correction pass with three steps: white balance, black point,
selective desaturation.

**White balance is taken from the shadows**, specifically the darkest ~22% of
pixels excluding near-black. Outdoor shadow is lit by skylight and should sit
near neutral, so a warm or teal shadow is a genuine cast. **Highlights carry the
real colour of the sun** — an earlier version balanced against highlights and
could not distinguish a yellow-green cast from a real golden hour. It turned a
sunset (`20260202-DSC_0385-1`) purple-mauve.

Guard rails that matter:
- Gains clamped to ±14%. The unclamped run over-corrected.
- Deadband: shadows within ±15 R−B are left alone. Real sunsets measure 6–11
  (warm bounce); genuine casts measure 21–30.
- Neutral protection: tint scaled by existing chroma, so snow and white fabric
  stay neutral. Without it, snow went cream.

Measured result across the set: shadow-tint spread sd 20.1 → 12.8, warmth sd
0.34 → 0.26, pixels at true black 0.03% → 0.38%.

**Not fixable from JPEG:** baked-in vignetting (`DSC_0574` teal corner halo,
`DSC_0646` cyan arc) and clipped skies (`DSC_0668`). Recommendation on the table
but not actioned: drop `DSC_0574`.

---

## The RAW files are gone

Searched C:, D:, E:, M: and every recycle bin on 2026-08-29. The only NEFs on the
machine are 7 moon frames in `M:\Photos & Videos\Nature\_process\moon\`.

`../Portfolio Selections/` JPEGs are therefore the only masters. **Nothing may
write to that directory.**

One lead not yet chased: **Lightroom CC** (not Classic) is installed. It is
cloud-backed and uploads originals. If those NEFs were ever imported, they may
still be in the Adobe account at lightroom.adobe.com. Requires the owner's login.

---

## Equipment and library facts

- **Nikon D3200** (2012) — the entire published portfolio. Weak above ISO 800.
- **Nikon D7500** (2017, "Big Camera") — 543 golden-hour frames, all personal
  family hiking. Better body, not used for client work.
- **`Animals/`** is 388 frames, **all from Samsung phones**, zero Nikon. It's pet
  snapshots, not a wildlife body of work. Wildlife photography would need to be
  shot, not curated.
- **`Adventure/`** is 254 Snapchat `.mp4` files, no stills.
- Landscape stills live in `Hiking/`, organised by date.

Shooting note worth passing on: the 2025-09-27 sunrise session ran 06:46–08:03
and the ISO was left at 3200 until 07:14. `DSC_0209` was shot at ISO 3200,
1/1000s, f/3.5 in good light — at ISO 100 it would have been 1/30s at 18mm. Its
noisy sky was avoidable in camera.

---

## Open / not done

1. **Private admin backend** (requested 2026-08-29). Split in two because the
   candidate photos live on `M:` and no hosted CMS can see them:
   - **Local picker — BUILT.** `tools/picker/`. Indexes 11,860 real-camera
     frames across 114 outings, browse/filter/star/publish. See its README.
   - **Web admin — not started.** Decap CMS at `/admin`, git-backed, for editing
     tags, captions and order of already-published photos from anywhere.
2. **Phase 2 — hosting cutover.** Netlify canonical, `netlify.toml`,
   `rel="canonical"`, retire GitHub Pages, clear two `pages-build-deployment`
   runs queued since 2026-02-02. Verify Netlify builds *before* disabling Pages.
3. **More landscape frames.** 8 added of ~2,200 golden-hour candidates reviewed
   only in part. Needs the picker.
4. **`auto_categorize.py`** still has dead `M:\Photography\...` paths and
   uncommitted local changes. Should be rewritten to emit `gallery.json` only.
6. **Three lightbox images exceed 1MB** (`0026`, `0034`, `0069`). Dropping WebP
   quality to 80 on the 2000w tier would fix it.


---

## GitHub Actions `schedule` is not a scheduler

Diagnosed 2026-08-29 while chasing repeated healthchecks.io down/up alerts.

The alerts were **not** about this website. They come from a dead man's switch
pinged by `publish.yml` in `mayniac24/mayniac-creations` (the Instagram bot).
healthchecks.io is passive — it never probes anything, it waits for pings — so
"the site is down for 90 minutes" was never the right reading.

Measured over 2026-08-09..29, against a `7,22,37,52 * * * *` cron requesting 96
runs/day:

| Period | Runs/day |
|---|---|
| Aug 9–14 | ~20 |
| Aug 15–23 | 36–48 |
| Aug 24–26 | 33, 34, 21 |
| Aug 27–29 | **3, 2, 3** |

Every run that fired succeeded (100/100). Runs simply were not being created,
with dark periods up to 11 hours. **Not a billing problem** — the repo is public,
so Actions minutes are unlimited; that was my first hypothesis and it was wrong.

**`workflow_dispatch` is not throttled.** A test dispatch started in one second.

Fix in place: scheduled task **"Mayniac publish trigger"** on this machine runs
`M:\_repos\mayniac-creations\scripts	rigger-publish.ps1` every 15 minutes at
:03/:18/:33/:48, which dispatches the workflow. The machine is already up 24/7
for Plex and Syncthing. The workflow's own cron is left as a harmless fallback;
`concurrency: publish` prevents overlap.

**Generalise:** never put anything time-sensitive on a GitHub `schedule` trigger.
Drive it externally and dispatch.


---

## Watermarks removed; rights embedded instead

**Decided 2026-08-29 by the owner**, after asking how to stop people stealing
the work. Watermarks are gone from all 31 published photos.

The reasoning, not just the outcome. A corner wordmark deters casual reposting
and nothing else -- generative fill erases one in seconds and it does nothing
against a screenshot -- while being the most common visual tell of an amateur
portfolio. What actually protects the work, in order:

1. **Resolution limiting.** Already in place: nothing above 2000px wide is
   deployed and originals are not on the server. A web copy cannot be printed
   large or licensed.
2. **Embedded rights.** Now on all 248 deployed files: EXIF `Copyright` and
   `Artist`, plus XMP `dc:creator`, `dc:rights` and `xmpRights:WebStatement` on
   WebP. Invisible, survives reposting, gives provenance.
3. **Copyright registration.** Not done, and the only thing with real teeth --
   without it you are limited to actual damages. ~$65 for a batch.
4. Reverse-image monitoring.

`optimize_images.py` writes a **minimal fresh** metadata block rather than
copying source EXIF, so GPS never reaches the web. `tests/test_rights.py`
guards all of this, including that no GPS is published.

### How the watermarks came off

Not by inpainting. Unwatermarked originals existed in the library
(`_Professional Work/` for the elopement, `Hiking/2025/Estes Park, August 2025/`
for the maternity set) but were *less graded* -- using them directly would have
discarded the owner's colour work.

`tools/dewatermark.py` instead fits a per-channel LUT from the clean original to
the graded export using only pixels **outside** the watermark region, then
applies it to the clean file. The result carries the grade and never had a mark.
Verified per photo by residual: the maternity set recovered 27-82% of the grade
(residual 28 -> 4), the elopement set needed almost nothing (residual already
0.5-2.7, i.e. JPEG noise). Sources recorded in `data/watermark_sources.json`.

Filename matching alone was not enough -- `DSC_0087` matches 44 files in this
library. Matching used basename + capture timestamp + dimensions, then content
correlation.

---

## The 20260202- filenames are wrong, and capture dates were recovered

The maternity photos carry a `20260202-` prefix taken from a **Lightroom export
timestamp**, five months after the shoot. Their EXIF had been stripped of
`DateTimeOriginal` entirely.

Content-matching each export against the camera originals in
`Estes Park, August 2025 - Big Camera/` recovered exact capture times for 7 of
12, all on **2025-08-20** between 19:15 and 19:57. The other 5 could not be
matched above 0.85 correlation and were left as year-only rather than guessed.

Two traps worth remembering:
- The same-numbered original is often a **different photograph**. `DSC_0056` and
  `DSC_0139` correlated at 0.28-0.36. A naive filename lookup also dated
  `DSC_0292` to 08-18 06:08 when the content-matched frame is 08-20 19:32.
- Portrait frames need rotation-invariant comparison; the stripped exports carry
  no orientation tag.

True dates now live in `data/gallery.json` as `captured`, and
`optimize_images.py` prefers that over the filename when building the copyright
notice -- otherwise it would assert 2026 for photographs taken in 2025, which on
a registered work is a real problem.

**Still open:** the filenames themselves remain wrong. Renaming would change 96
variant filenames and every `id` in `gallery.json`. Harmless while the site is
unpublished, worth doing before it goes live.
