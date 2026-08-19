# Visual testing

Thalos diagnoses and iterates on graphics with deterministic, full-resolution
headless captures. ADR-20260721T192218Z-persistent-visual-iteration keeps the
one-camera design while making the normal loop persistent; its experimental
Rust hot-patch mechanism was superseded by
ADR-20260724T153619Z-retire-hotpatch-single-stable-capture-lane.
A live multi-camera split screen is still rejected: it would
duplicate the one-view renderer and change LOD, SSAO, shadow, antialiasing, and
other viewport-dependent inputs while supposedly holding them constant.

## Agent quickstart

Use the smallest workflow that answers the question:

| Need | Command | Read |
|------|---------|------|
| Choose a relevant viewpoint without rendering | `cargo run -p thalos_capture -- list viewpoints --gallery` | `artifacts/visual/catalog/viewpoints/contact_sheet.png` + `index.json`; thumbnails are composition hints, not evidence |
| One canonical view | `just screenshot spaceport-aerial` | `artifacts/visual/latest/spaceport_aerial.png` + `.capture.json` |
| A player’s saved view | `just screenshot <viewpoint-id>` | the matching latest PNG + receipt; the current 1080p fit is temporary safety scaffolding |
| Same camera, another epoch | `cargo run -p thalos_capture -- shot <viewpoint-id> --time 72000` | the requested output/receipt; use `--out` to preserve both epochs |
| One shot with custom graphics | `just screenshot <scene> --graphics clouds=off,grass=on` | PNG + receipt recording the effective graphics settings |
| Intentional output extent today | `cargo run -p thalos_capture -- shot <viewpoint-id> --size 3840x2160` | temporary expert override until typed fidelity profiles land |
| Several scenes | `just capture <scene>...` | each scene’s latest PNG; compatible scenes reuse one world |
| A/B or N-way diagnosis | `just compare <scene> <axis>` | `contact_sheet.png`, then full variants, `wipe_*`, `diff_*`, `manifest.json` |
| Final isolated proof | `just compare-cold <scene> <axis>` | the same evidence bundle, one clean process per variant |
| Isolate terrain shading | `just compare <scene> terrain-lighting` | lit / fullbright / geometric-normal |
| Isolate screen-space effects | `just compare <scene> ssao` or `shadow` | off / applied / raw field |
| Isolate cloud internals | `just compare <scene> cloud-tier` or `cloud-reconstruction` | registered N-way variants |
| Check whether your edits are present | read `<image>.capture.json` | `source_floor_guaranteed: true` proves the invocation floor is included; `workspace_matches` says whether the checkout stayed exact |
| Hand framing to/from a human | human presses F9; agent runs `just screenshot <id>` | exact body-fixed camera, lens, scene, and saved time |
| Hand the result back to a human | show or link the PNG files in the normal chat response | Build an optional HTML report only when the user requests one |

### Framing versus fidelity

Do not use output resolution to define a viewpoint. The camera bookmark owns
pose, lens, sensor gate/filmback, and crop window: these determine the rays and
composition. A smaller sensor at the same focal length is a crop-factor zoom
and intentionally frames a smaller part of the world. The output pixel grid is
chosen later by capture fidelity and may change without moving or zooming the
camera.

The planned capture tiers are `draft`, `standard`, `high`, and `reference`.
Agents should normally start at `standard`, use `draft` only for questions that
cannot depend on fine detail, promote to `high` when a frame is soft or
inconclusive, and finish accepted visual work at `reference`. A comparison keeps
one tier across every variant. Its manifest records the requested tier and
effective output/internal renderer settings so “higher fidelity” is a
reproducible choice, not an unexplained collection of overrides.

The current catalog `viewport` field and automatic 4K→1080p fit are migration
scaffolding. When the physical-camera system lands, catalog migration retains
only the viewport's aspect as sensor-gate information and discards its pixel
dimensions. Until then, use `--size` consciously when `standard`-class detail is
needed; do not revise a viewpoint merely to get a sharper PNG.

For a visual defect, do not begin with an arbitrary beauty shot:

1. choose the existing preset/viewpoint that reproduces it;
2. state two or more plausible causes;
3. choose one registered axis or inspection channel that separates them;
4. run the matrix and inspect stderr → manifest → contact sheet → full frames →
   wipe/diff;
5. fix only after the evidence pins the cause;
6. rerun the same matrix cold for the final before/after.

`wipe_01_vs_02.png` is the canonical 50/50 split view. It is assembled from two
independent full-resolution renders so both variants keep identical viewport
inputs. For more than two variants, use the labelled contact sheet; pairwise
wipes/diffs always compare each variant with variant 1.

If no registered axis or debug view distinguishes the hypotheses, add one typed
factor to the capture registry and its runtime override. Do not take two
hand-configured screenshots and call them an A/B: the manifest would be unable
to prove what stayed fixed.

## Canonical workflows

- `just screenshot <preset>` captures one in-context beauty frame through the
  persistent renderer. Compatible subsequent captures reuse its world and GPU.
- `just screenshot <preset> --graphics clouds=off,grass=on` applies a typed
  partial graphics profile to that request. Each request starts from capture
  defaults, so settings never leak from the player's file or an earlier shot.
- `just capture <preset>...` captures several scenes in one invocation.
  Presets sharing body + spawn + hub mode + effective capture extent + startup overrides
  amortize one world boot; other boot contexts restart automatically.
- Concurrent agents queue behind the current capture owner. Compatible queued
  scenes are camera poses rendered sequentially through the same world and GPU
  host; they are not simultaneous renderer instances.
- F8 opens the shared viewpoint manager. Developers can create, inspect, apply,
  update, rename, and delete exact saved poses and agent-scripted views stored
  in `assets/viewpoints.json`; agents edit the same file directly.
- `just screenshot <viewpoint-id>` replays a named catalog point. `just
  screenshot latest` remains a compatibility alias for the newest entry.
- `thalos_capture list viewpoints --gallery` builds the catalog contact sheet
  and individual thumbnails entirely offline from cached canonical captures.
  It labels stale, unattributed, unreadable, and missing entries rather than
  silently rendering them; consult `index.json` before choosing a view.
- F2 desktop screenshots show saved/error toasts only after the capture outcome
  is known, so feedback is never baked into its own image. Toasts do not appear
  in F1 photo mode, whose clean frame hides the shared toast container.
- `just compare <preset> <axis>` captures every live-compatible typed variant in
  that same renderer, then assembles one comparison artifact set.
- `just screenshot-cold` and `just compare-cold` are the clean-process,
  full-warm-up acceptance lanes.
- `just preview` remains the supplemental isolated-asset path. It does not replace
  the in-context game capture.
- `cargo run --release -p thalos_terrain_baker -- diagnose <Body>` answers height
  questions **numerically**, without a renderer in the loop. Alongside the
  adjacent-sample profiles it runs an **alias audit**: the point sample a tile
  bake would store for a texel, against the box mean of that same span sampled at
  1 m. The residual is the unresolved energy in metres. Use it before blaming a
  speckle/stipple artifact on terrain geometry — on 2026-07-22 it falsified
  exactly that assumption for BL-33 in one run, where three renderer-side guesses
  would each have taken a capture cycle to disprove.

`artifacts/visual/latest/` is the curated latest-view surface: every named preset
owns one stable filename there and overwrites it on the next canonical capture.
Ad-hoc framings set `THALOS_SCREENSHOT_OUT` to a path under
`artifacts/visual/runs/`; they do not mint new names beside the canonical
views. Comparison matrices always use the run tree automatically.

`artifacts/visual/catalog/viewpoints/` is a disposable discovery cache derived
from that latest-view surface. Its contact sheet and thumbnails may be stale
and therefore cannot close a visual verification gate; the adjacent
`index.json` says which original and receipt each card came from.

## Shared viewpoint catalog

### F9 — save this view

**F9, Enter** saves what is on screen right now under a name derived from the
view (`Thalos 340 m`, `Mira 412 km`, numbered when that name is taken); **F9,
type, Enter** overrides the name, because the suggestion starts fully selected.
The toast reports the id the entry landed under — that string is what
`just screenshot <id>` takes. The pose is frozen at the keypress, not at Enter,
so the world may keep moving (or warping) while the name is typed.

Collisions never block a save: a typed name whose slug is taken gets the next
free `-2` / `-3` id, and the toast says so. Use F8 when you want to browse,
re-view, replace, or delete instead.

### F8 — the catalog manager

Press F8 while a 3-D viewer is active. The shared native manager reads the
application's viewpoint catalog. “Save current as new” captures the current
typed frame, position, orientation, lens, and application-specific replay
context under a stable id. “View” applies a selected entry in the running
world; “Replace from current” updates its pose. Reload rereads agent edits
without restarting the application.

“View” always enters freecam and seeds freecam's own body-fixed anchor from the
resolved pose, including the authored lens. Do not write only the rendered
`ShipCamera` transform: the normal orbit-camera system owns that transform and
will restore its focus-derived pose on the next frame.

The catalog is also the public registry for the existing agent views:
spaceport, atmosphere, ocean, coastline, Mira, clouds, plume, interstage, and massif framings appear
in the same F8 list with an `[agent]` badge. Their JSON record selects a
validated procedural driver so search-driven targets and diagnostics retain
their behavior. Viewing one live applies its focus and camera framing;
capture-only state such as false-colour output, forced plume pressure, or a
temporal slew is applied only by `just screenshot` and is called out in the
manager status. Replacing an agent view from the current camera converts that
entry into an exact saved pose.

The manager is ordinary `thalos_ui` attached to the application's existing UI
camera; it does not own a second world or window camera.

An agent can add or adjust the JSON with the same versioned schema, then ask the
developer to press F8, reload, and view it. To capture a named point headlessly:

```text
just screenshot mira-ridge-dawn
```

Saved viewpoints replay at their recorded `sim_time_s`, so the same body-fixed
camera returns to the same lighting epoch. Override only the time while keeping
the camera and lens fixed with either the capture CLI:

```text
cargo run -p thalos_capture --bin thalos_capture -- shot mira-ridge-dawn --time 72000
```

or the compatibility environment knob used by `just screenshot`
(`THALOS_SCREENSHOT_TIME=72000`; canonical seconds). The caller value wins over
the catalog metadata. Persistent-host requests reapply it independently, so
successive captures may move backward or forward in time without a restart.

The CLI also accepts `viewpoint:mira-ridge-dawn` when an explicit namespace is
useful. The result overwrites
`artifacts/visual/latest/mira_ridge_dawn.png`. `just screenshot latest` selects
the catalog entry with the newest `saved_unix_ms` and keeps writing
`latest_perspective.png`.

The v3 catalog tags every camera as either authored-body-fixed or
projected-local. The game projects authored poses through the fresh body's
current transform and uses the real `ShipCamera`, so floating-origin shifts do
not change the view. Kòrsou applies projected-local poses through its own
planar/ellipsoid adapter. Neither application guesses the other's coordinate
space.

This is a viewpoint catalog, not a save game. Replay applies the recorded time
to the canonical world clock, but still boots the named canonical spawn instead
of partially restoring craft dynamics. It reproduces geographic framing,
lighting, and time-driven environment state; a one-off moving-craft state or
non-deterministic weather moment still needs a normal screenshot or a dedicated
deterministic preset.

Procedural drivers remain code when they search for a site, drive time, or
install a diagnostic mode. Their public identity and metadata do not: those
live only in the shared catalog.

Atmosphere has two complementary canonical framings: `earth-reference` for the
space/orbital limb and `runway-atmosphere` for the near-surface sky, long
slant-path haze, and terrain/structure recession. Both capture the sole custom
atmosphere renderer; ADR-20260721T185221Z removed the resolved Bevy/custom
comparison axis and all backend-selection state.

One comparison run changes exactly one declared axis. Any normal screenshot
framing overrides (`THALOS_SCREENSHOT_SIZE`, `_AZIMUTH`, `_ELEVATION`,
`_DISTANCE`, `_TIME`, `_WARMUP`, `_HUD`) are inherited identically by all variants.
The runner owns `THALOS_SCREENSHOT_OUT` and the axis-specific override so those
cannot drift between captures.

The first persistent capture launches the host with plain `cargo run` on the
shared stable `dev-renderer` fingerprint. File-backed and embedded WGSL changes
reload through Bevy's asset watcher; the client waits for the reload event.
Any Rust/manifest edit triggers a managed rebuild/restart on the next request.
Preset changes reuse the process when their boot context matches; body, spawn,
hub-mode, or effective capture-extent changes restart it. Output extent is a
capture-resource boundary, not part of viewpoint identity. Manual
`just capture-stop` is optional hygiene.

When no request is active, the resident host parks its camera and polls at
10 Hz; it retains the expensive world/GPU setup without continuously rendering
at 60 Hz. **Capture operations from parallel agent sessions and worktrees
serialize through one machine-wide lock.** Within a checkout, requests reuse
its singleton host. On Windows the operation lock is a kernel mutex, not a PID
file that can be stolen during stale-owner cleanup. A second client reports the
owning PID/command and waits without starting another active renderer or
overwriting the shared control files.
Verification can still queue behind a long comparison; that is a scheduling
problem, not a broken change. Say so in the report rather than marking a visual
change verified.

Every successful persistent shot writes `<image>.capture.json`. Its
`source.fingerprint` is the invocation's **source floor**:
`source_floor_guaranteed: true` means the renderer was prepared from that state
or a state reached later while the build was in flight. `workspace_matches:
true` is the stronger exact-equality case. When it is false, the receipt says
`workspace_relation: advanced-since-source-floor`; the PNG remains valid for
the floor, but it cannot prove whether any particular later edit reached the
frame. Recapture only when that later edit is itself under verification.
Comparison `manifest.json` carries the same source-before/after contract and
points to each variant receipt.

Every request reapplies live diagnostic resources and invalidates cloud temporal
history. The first frame uses the preset's full warm-up; subsequent requests use
60 settle frames unless `_WARMUP` is explicitly set. This fast lane intentionally
retains streamed terrain and renderer caches. Use the cold lane for final
regression proof; its orchestrator reproduces Cargo's complete dynamic-library
search path (INC-0008) and starts one clean process per variant.

Cold-process shutdown is completion-driven: after requesting the image, the
headless driver waits for Bevy's `Capturing` marker to clear before counting its
readback-flush tail. A fixed frame delay is not a readback-completion contract
(INC-20260722T172947Z-cold-capture-exited-before-readback).

## Initial axes

| Axis | Variants | Intended diagnosis |
|---|---|---|
| `ssao` | `off`, `on`, `raw` | No AO, normal AO application, and the raw AO field |
| `shadow` | `cascade-only`, `contact`, `raw` | Isolate the **contact tier** (W18a) from the cascade rig: does the screen-space contact march contribute, and is a defect in the march or in how receivers apply it (ADR-20260722T111848Z) |
| `terrain-lighting` | `lit`, `fullbright`, `geometric-normal` | Separate raster coverage from lighting, then isolate the terrain normal stack. Honoured by **both** ground renderers (udlod's material and `tile_terrain.wgsl`'s `TileShadingParams::inspect`), so the axis means the same thing whichever one owns the body |
| `renderer` | `tiles`, `udlod` | The same scene through the **default** standard-path tile renderer and the **legacy** UDLOD baseline. Structural, so it always runs cold; the capture client automatically adds the default-off `legacy-udlod` feature only for the UDLOD variant. Ordinary game/capture builds never compile it |
| `terrain-culling` | `backface`, `two-sided` | Test whether grazing holes are missing back-facing raster coverage |
| `terrain-regolith-filter` | `legacy-unfiltered`, `footprint-filtered` | Matched before/after for airless procedural-detail Nyquist filtering |
| `shadow-quality` | `high-4`, `medium-3`, `low-2`, `off-0` | Player-facing shadow range tiers at fixed 4096² active-map detail; structural, so every variant runs cold |
| `shadow-cascades` | `4-full`, `3-no-farthest`, `2`, `1`, `0-off` | Raw diagnostic cost/coverage ladder underlying the named tiers |

Axes are intentionally typed in `tools/capture/src/compare.rs`, inside the
single `thalos_capture` binary. Add a new one only when every variant can be
selected by a capture-only override that
does not persist user settings. A multi-test may have N variants, but they must
all remain values of the same factor.

The axis environment key must also be present in the headless runtime's
`CAPTURE_OVERRIDE_KEYS`; otherwise the runner labels distinct variants while
the game silently renders the default each time. Any secondary diagnostic held
fixed during an A/B belongs in `INVARIANT_ENV_KEYS` so the manifest proves it did
not drift (INC-20260722T182934Z).

`terrain-culling`, `renderer`, `shadow-quality`, and `shadow-cascades` are
structural — they specialize a pipeline or select a renderer/camera budget at
boot — so the normal comparison command automatically sends them through the
cold lane. New axes
should be runtime resources/material inputs when possible; otherwise mark them
cold rather than pretending an existing pipeline changed.

The scene argument accepts a **saved viewpoint id** as well as a scripted
preset: the player hands over a framing with F9/F8 (ADR-20260724T211627Z), and
`just compare <viewpoint-slug> <axis>` runs the matrix at exactly the framing
they were looking at. That is usually the right scene for a "why does this look
wrong" A/B — no preset has to be invented to chase a report.

## Artifact contract

Each run writes to:

```text
artifacts/visual/runs/comparisons/<preset>/<axis>/
  01_<variant>.png
  02_<variant>.png
  ...
  contact_sheet.png
  diff_01_vs_02.png
  wipe_01_vs_02.png
  manifest.json
```

This directory is disposable working evidence, separate from the curated
latest views. The original variant PNGs remain full-resolution. The labelled
contact sheet is for rapid inspection; each diff amplifies absolute RGB error
against variant 1; each wipe preserves the baseline on the left and the
compared variant on the right. The manifest records the Git revision/dirty
state, capture lane, invariant screenshot overrides, per-variant environment override, image
paths, and numerical diff metrics. Copy a result to an explicitly named
evidence location only when a revision-to-revision artifact must be retained.

Pixel differences are evidence, not a verdict. Stochastic or temporal effects
can produce a non-zero diff without a meaningful visual regression. Use the
contact sheet/wipe to interpret the numbers and keep the diagnosis tied to a
specific hypothesis.

BL-20 landed 2026-07-24: the controller scans the current request's host log
(including first-boot initialization), promotes shader/pipeline/device failures
to a failed request, validates that the output exists and decodes, and refuses
to assemble comparisons from invalid variants. Cold variants receive the same
log validation.

## Optional report artifacts

Do not build an HTML report after each change. Show or link captures in the
normal chat response. Build a self-contained report only when the user
explicitly requests one. The report template and publisher remain available for
that future use.

The page is a self-contained HTML file in `artifacts/reports/`
(`<date>-<slug>.html`, gitignored like all evidence), handed back by showing it
in the agent's in-app browser pane. Its editable token input is
`<date>-<slug>.html.in`. The input deliberately does not have an HTML suffix,
so the token-bearing file cannot masquerade as the finished browser page.
Its report body also lives in a non-rendering source wrapper: opening the input
shows only an **Unpublished report input** warning, never a plausible report
with visible tokens. Never open or hand it back as the result.
Images are inlined as data URIs even though a
local file could reference them live: `artifacts/visual/latest/` and the
comparison dirs are overwritten on every rerun, so a page linking them would
silently change under the reader — embedding freezes the evidence at hand-back
time. It also keeps the file portable: the same page can be published as a
claude.ai Artifact (whose CSP blocks every external load) when it needs to
travel off the machine. `scripts/present_embed.py` does the inlining.

### Design language

**Start from `scripts/present_template.html`: copy it, keep the stylesheet
untouched, replace the content.** Every report reads the same way, so the user
never has to re-learn a page. The look is a research paper, not a dashboard:

- One centered serif column (~46rem). A title block with a one-line meta row
  (`date · preset · axis`), plain sentence-case headings, figures numbered
  and captioned (`Figure 1. …`).
- The page is white in the viewer's light theme and black in dark, through the
  template's theme tokens. There is no accent color.
- Hairline rules and thin image borders are the only decoration. No cards,
  shadows, gradients, background tints, or hero sections.
- Never `text-transform: uppercase` and never letter-spacing. No em-dashes in
  page copy: use commas, colons, or the meta row's middle dot.
- **Text budget is a slide deck's.** The user reads the agent's chat reply
  regardless, so the page carries only what the visuals need: a few bullets per
  section, one sentence per figure caption saying what to look at and where.
  Prose paragraphs on the page are a smell.
- Before/after pairs use the template's `.pair` grid: two matched captures side
  by side, short labels under each half, one shared caption for the pair.

### Handing it back

Write `<report>.html.in` in `artifacts/reports/` with tokens where the
captures go:

```html
<h2>Ridge stipple, before and after</h2>
{{img:artifacts/visual/runs/comparisons/mira-eva/terrain-regolith-filter/01_legacy.png|legacy}}
{{img:artifacts/visual/runs/comparisons/mira-eva/terrain-regolith-filter/02_filtered.png|footprint-filtered}}
```

Then embed it:

```bash
just publish-report artifacts/reports/report.html.in
```

and present the result by navigating the agent browser pane to the canonical
`report.html` `file://` path (pass `force: true` if the pane is showing a stale
snapshot of an earlier file). In a session with no browser pane, publish the
same file as a claude.ai Artifact or fall back to markdown with PNG paths.

The command writes `report.html` with every token replaced by an `<img>` carrying
a data URI. It fails instead of publishing if a token is malformed or unresolved,
if no image was embedded, if any live image reference remains, or if the input
lacks the template's unpublished-input wrapper. On success it prints `OPEN ONLY:`
followed by the canonical file URL. With
Pillow installed it downscales to JPEG q82, or PNG when the source has alpha;
without Pillow it embeds the original files so publication still works. Open
`report.html`; keep `report.html.in` beside it, so a recapture only costs a
rerun of the command. Token paths resolve relative to the input, then to the
repository root. Aim for a page in the low single-digit MB; `--width` is the
underlying script's size knob when Pillow is available.

What belongs on it: the matched before/after pair from one comparison run at
identical framing and labelled (never two differently-framed stills), what you
see in plain language, what you changed, what you deliberately left alone, what
is still unverified, and the preset/axis names so the user can rerun it.
Mermaid renders natively (```mermaid fence or `<pre class="mermaid">`) for flow
and ownership diagrams; no external charting library will load.

The page is a courtesy to the reader, not a record. It does not replace the
backlog row, an ADR, or the `.capture.json` receipt beside each PNG, and nothing
in `docs/` should link to it.

## Root-cause loop

1. Choose or add a preset that reproduces the symptom.
2. State the competing hypotheses.
3. Pick one axis that distinguishes at least two of them.
4. Run the comparison with all framing and warm-up inputs held fixed.
5. Inspect the contact sheet, full captures, wipe, and diff metrics.
6. Eliminate hypotheses, then repeat on the next single axis until the cause is
   pinned.
7. Rerun the matched before/after with `just compare-cold` and keep that result as
   verification of the fix.

Future debug channels—normals, depth, terrain LOD/tile IDs, shadow factor and
cascade, material IDs, atmosphere transmittance, and lighting lobes—join this
same runner. They do not get their own camera or comparison framework.

## Verification

Runtime-verified 2026-07-21 with the comparison runner's two- and three-variant
axis shapes:

- The now-removed `earth-reference / atmosphere` axis produced two aligned
  1800×1200 captures and resolved the renderer decision recorded by
  ADR-20260721T185221Z.
- `just compare spaceport-aerial ssao` produced three aligned 1920×1080
  captures (off/on/raw), a labelled 2×2 sheet, two baseline diffs/wipes,
  metrics, and manifest through the direct dynamic-game launcher.
- `THALOS_TILE_CACHE=0 just compare mira-eva terrain-regolith-filter` produced
  aligned 2048×1280 legacy/filtered captures at the live canonical EVA site;
  the legacy ridge stipples while the footprint-filtered ridge is stable and
  nearby regolith texture remains resolved (INC-0009).
