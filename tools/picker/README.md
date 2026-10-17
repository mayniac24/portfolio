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

The index keeps frames whose EXIF Model or Make contains `camera_match` from
`_studio_settings.json`; empty or absent means no camera filter, and the finish folder
(`finish_dir`) is never indexed. The seeded settings name the camera, and nothing in code
does. Before 2026-10-17 the filter was any Nikon, which also indexed a second shooter's
D7500 frames: rebuilt with a D3200 match the index went from 11,860 to 10,796 frames, the
2,246 D7500 frames removed (the library had also gained about 1,180 D3200 frames since the
old index was built). `--camera-match X` overrides, `--all-cameras` includes phone photos,
`--root <path>` indexes a subtree. `--add <outing folder>` merges ONE outing into the
existing index instead of replacing it (ids stay relative to
photos_root; it drops that outing's old entries first, so it is safe to repeat).

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

## Stage picks, detect finished edits (the studio lane)

Finishing a frame is a hand pass (Lightroom or Camera Raw, "a little more than a
minute per frame"). The machine stages and detects; it never edits. Design:
`DESIGN-2026-10-17-photos-and-content.md` D4/D5, 2.3.

```bash
python tools/picker/stage_picks.py  "<outing folder>" [--apply]   # copy starred -> _Finish/<outing>/
python tools/picker/finish_watch.py "<outing folder>" [--apply]   # "finished n of m"
```

- `stage_picks.py` COPIES (never moves, never overwrites) every starred frame of the
  outing to `<photos_root>/<finish_dir>/<outing name>/`, writes one `pick` event per
  frame, and prints the folder to import into Lightroom. Two starred frames sharing a
  file name in one outing are refused (the finish folder is flat). It records what it
  staged in `.staged.json` in that folder.
- Export each finished frame into the same folder under the same stem (`DSC_0123.jpg`,
  or Lightroom's `DSC_0123-Edit.jpg`).
- `finish_watch.py` counts a staged frame as finished when the folder holds a file with
  its stem that is newer than the original, writes one `finish` event
  (`frame, export, bytes`), and prints `finished n of m`. Re-running does not repeat
  events; a re-export with new bytes writes a new one.
- Both are dry runs without `--apply`. Common options: `--settings`, `--photos-root`,
  `--selections`, `--events-dir`, `--via cli|horizons|...`, `--by <member>`.
- `publish_external.py <exports> --outing <outing folder>` scopes the starred set (and
  its ambiguous-stem check) to one outing and appends a `post` event (channel
  `portfolio`) per newly published frame.
- The star route in `app.py` appends a `pick` event on star and on unstar (not on alt or
  category edits); a bus failure never blocks the star.

Settings are read from `_studio_settings.json` at the vault root: the env var
`MAYNIAC_STUDIO_SETTINGS`, else `$VAULT_PATH/_studio_settings.json`; absent = defaults.
Keys used: `photos_root, finish_dir, camera_match, events_module,
picker_tools.{selections,index_file}`. Events go to `studio-picker.jsonl` in the bus
directory (`--events-dir`, env `MAYNIAC_STUDIO_EVENTS`, or the vault's `events` path)
through `events_module` when it exists, else the documented line written by
`studio_link.py`. Ids: a frame id is its path relative to photos_root. An outing id is
looked up on the bus by folder: `_studio_outings.json` (the projection) first, then the
`outing` rows (create, rename, merge, split) of `studio-*.jsonl`; only when neither knows
the folder is it hashed, the sorter's way: `out-` + 10 hex of SHA-1("outing:" + SHA-1 of the
earliest still), still extensions from the sorter's `still_ext`, EXIF capture time first, mtime
when there is none.
Stage copies through `<name>.part`, verifies size and SHA-1, then renames; it refuses a file
already in the finish folder that is not a copy of the source, and a finish folder staged
for a different outing. A finish needs a newer file (by more than 2 s) whose bytes differ
from the staged copy.
Tests: `python -m pytest tests/test_picker_stage.py -q -p no:cacheprovider --basetemp .pt`.

## Finish helper: tweak and recipes (the help lane)

For ONE frame he is stuck on, run only when he asks on that frame (D5; nothing here is a
watcher or a schedule, and nothing is headless). Not a tone pass. No model, no network: PIL and
numpy. Design: `DESIGN-2026-10-17-photos-and-content.md` D16, 2.3 Help.

```bash
python tools/picker/tweak.py measure <frame>                       # sky/land luminance, gap, clipping
python tools/picker/tweak.py suggest <frame> [--note "his words"]  # matching recipes + hit rates, no model
python tools/picker/tweak.py apply   <frame> --recipe NAME | --params k=v ... [--apply]
python tools/picker/tweak.py accept  <frame> [--apply]             # candidate becomes the export
python tools/picker/tweak.py reject  <frame> [--apply]             # candidate deleted, nothing else
python tools/picker/tweak.py recipe save NAME --frame <frame> --symptom "his words" --step "..." [--params k=v ...] [--apply]
python tools/picker/tweak.py recipe edit|retire NAME [...] [--apply]
python tools/picker/tweak.py recipe list [--json]
```

- `<frame>` is a frame id (path relative to photos_root, what the bus calls it) or an absolute
  path to the export inside the finish folder. The helper works on his EXPORT in
  `<finish_dir>/<outing>/` (found through `.staged.json`); with no export yet it refuses
  (`apply`) or measures the master (`measure`, `suggest`). Every write is guarded to sit under
  the finish folder: the master, the outing folder, `Portfolio Selections` and
  `Portfolio Corrected` are never written.
- Dry run is the default for every verb that writes; `--apply` writes. `--json` on `measure`,
  `suggest`, `recipe list` for the Horizons finish station.
- `measure`: luminance (Rec.709 on the stored values, 0..1) of the top 35% (sky) and bottom 40%
  (land), their gap, and the share of pixels with a channel at or above 250/255 (highlight
  clip) and luma at or below 5/255 (shadow clip).
- `apply` writes `<export stem>_helper.jpg` beside the export (quality 95, ICC and EXIF kept,
  orientation baked in) and nothing else; the candidate carries its recipe, params and the
  SHA-1 of the export it was made from in the JPEG comment. Parameters are bounded and out of
  range is refused: `highlights 0..0.30`, `shadows 0..0.30` (luminosity-masked pull and lift,
  monotone so tones never invert), `dehaze 0..0.30` (black-point lift), `sky_pull 0..0.40`
  with `sky_height 0.20..0.80` and `sky_feather 0.05..0.50` (a graduated mask weighted by
  luminosity, so only the bright sky comes down).
- `accept` renames the export to `<stem>_pre-helper.<ext>` (`-2`, `-3` when taken; nothing is
  ever deleted), renames the candidate to `<stem>.jpg`, then writes the event; if the event
  cannot be written the swap is rolled back. It refuses a candidate made from an export that has
  since changed (`--force` overrides). `reject` writes the event, then deletes the candidate.
- Event, `studio-picker.jsonl`, through `studio_link.append_event` (`events_module` when set,
  else the documented line): `{"kind": "tweak", "frame": <id>, "recipe": <name|null>,
  "accepted": true|false, "export": <path>, "outing": <id>, ...envelope}`. A recipe's hit rate is
  accepted / decided over those rows.
- `recipes.yaml` has exactly one writer, `recipe save|edit|retire`. A recipe holds: `name`,
  `symptom` (his words), `trigger` (`{sky|land|gap|highlight_clip|shadow_clip: [lo, hi]}`,
  measured on the origin frame's pre-helper export, plus `--margin`, or set with
  `--trigger metric=lo:hi`), `lightroom_steps`, `params`, `origin_frame`, `created`, and
  `retired` once retired (kept so its hit rate survives). `suggest` offers a recipe when every
  trigger range holds or a symptom word is in `--note`; ranked by both, no model call.
- Exit codes: 0 ok, 2 usage or environment, 3 refused.
- Tests: `python -m pytest tests/test_tweak.py -q -p no:cacheprovider --basetemp .pt`.

## Notes

- Binds to `127.0.0.1` only. There is no auth and it should not be exposed.
- The light band is descriptive, not a quality signal. Overcast defeats it: a low
  sun behind cloud still reads "golden" while the light is flat grey. That is
  precisely why it filters rather than ranks.
- Thumbnails cache to `.thumbs/`; first load of a large outing is slow, then fast.
- `picker_index.json`, `selections.json` and `.thumbs/` are gitignored — they hold
  machine-specific paths and in-progress work.
