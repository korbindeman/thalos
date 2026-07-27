# Visual testing

Thalos diagnoses and iterates on graphics with deterministic, full-resolution
headless captures. ADR-20260721T192218Z-persistent-visual-iteration keeps the
one-camera design while making the normal loop persistent; its experimental
Rust hot-patch mechanism was superseded by
ADR-20260724T153619Z-retire-hotpatch-single-stable-capture-lane.
A live multi-camera split screen is still rejected: it would
duplicate the one-view renderer and change LOD, SSAO, shadow, antialiasing, and
other viewport-dependent inputs while supposedly holding them constant.

## Canonical workflows

- `just screenshot <preset>` captures one in-context beauty frame through the
  persistent renderer. Compatible subsequent captures reuse its world and GPU.
- `just capture <preset>...` captures several scenes in one invocation.
  Presets sharing body + spawn + hub mode + viewport + startup overrides
  amortize one world boot; other boot contexts restart automatically.
- F8 opens the shared viewpoint manager. Developers can create, inspect, apply,
  update, rename, and delete exact saved poses and agent-scripted views stored
  in `assets/viewpoints.json`; agents edit the same file directly.
- `just screenshot <viewpoint-id>` replays a named catalog point. `just
  screenshot latest` remains a compatibility alias for the newest entry.
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

Press F8 while the ship/free camera, space-center hub, or another 3-D god-view
is active. The egui manager reads `assets/viewpoints.json`. “Save current as
new” captures the current body-fixed position, orientation, lens, viewport,
target body, and canonical boot scene under a stable id. “View” applies a
selected entry in the running world; “Replace from current” updates its pose.
Reload rereads agent edits without restarting the game.

“View” always enters freecam and seeds freecam's own body-fixed anchor from the
resolved pose, including the authored lens. Do not write only the rendered
`ShipCamera` transform: the normal orbit-camera system owns that transform and
will restore its focus-derived pose on the next frame.

The catalog is also the public registry for the existing agent views:
spaceport, atmosphere, ocean, Mira, clouds, plume, and massif framings appear
in the same F8 list with an `[agent]` badge. Their JSON record selects a
validated procedural driver so search-driven targets and diagnostics retain
their behavior. Viewing one live applies its focus and camera framing;
capture-only state such as false-colour output, forced plume pressure, or a
temporal slew is applied only by `just screenshot` and is called out in the
manager status. Replacing an agent view from the current camera converts that
entry into an exact saved pose.

The manager's primary egui context is attached explicitly to the canonical
`ShipCamera`; it does not own a second window camera. Do not use egui's implicit
“first camera” attachment here: Thalos creates the inactive map camera before
the ship camera. Do not add a dedicated same-window overlay camera either: even
with a load-preserving clear mode, that extra presentation pass blacked out the
world on the live renderer.

An agent can add or adjust the JSON with the same versioned schema, then ask the
developer to press F8, reload, and view it. To capture a named point headlessly:

```text
just screenshot mira-ridge-dawn
```

The CLI also accepts `viewpoint:mira-ridge-dawn` when an explicit namespace is
useful. The result overwrites
`artifacts/visual/latest/mira_ridge_dawn.png`. `just screenshot latest` selects
the catalog entry with the newest `saved_unix_ms` and keeps writing
`latest_perspective.png`.

The catalog stores cameras in each body's authored surface-fixed frame.
Headless replay projects the pose through the fresh body's current transform and
uses the real `ShipCamera`, so floating-origin shifts do not change the view and
all normal render passes remain coupled.

This is a viewpoint catalog, not a save game. The recorded simulation time is kept
as provenance, but replay deliberately boots the named canonical spawn instead
of partially restoring craft dynamics. It therefore reproduces geographic
framing and scene configuration; a one-off moving-craft state or weather moment
still needs a normal screenshot or a dedicated deterministic preset.

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
`_DISTANCE`, `_WARMUP`, `_HUD`) are inherited identically by all variants.
The runner owns `THALOS_SCREENSHOT_OUT` and the axis-specific override so those
cannot drift between captures.

The first persistent capture launches the host with plain `cargo run` on the
shared stable `dev-renderer` fingerprint. File-backed and embedded WGSL changes
reload through Bevy's asset watcher; the client waits for the reload event.
Any Rust/manifest edit triggers a managed rebuild/restart on the next request.
Preset changes reuse the process when their boot context matches; body, spawn,
hub-mode, or viewport changes restart it. Manual `just capture-stop` is optional
hygiene.

**The host is a singleton, so parallel agent sessions contend for it.** Two
sessions asking for presets with different boot contexts will restart it out
from under each other, and the loser reports `capture launcher exited` (or
`capture server exited`) with no defect of its own — check `just capture-status`
for whose `preset` is loaded before diagnosing a capture failure as a bug.
Verification can be blocked this way for a long stretch; that is a scheduling
problem, not a broken change. Say so in the report rather than marking a visual
change verified.

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
| `renderer` | `tiles`, `udlod` | The keystone question — the same scene through the **default** standard-path tile renderer and through the **legacy** udlod spine (`THALOS_TILE_RENDERER=0`, an A/B baseline only). Structural (the gate is a boot `OnceLock` deciding which ground streams at all), so it always runs cold |
| `terrain-culling` | `backface`, `two-sided` | Test whether grazing holes are missing back-facing raster coverage |
| `terrain-regolith-filter` | `legacy-unfiltered`, `footprint-filtered` | Matched before/after for airless procedural-detail Nyquist filtering |

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

`terrain-culling` and `renderer` are structural — one specializes a pipeline at
first use, the other decides at boot which ground streams — so the normal
comparison command automatically sends them through the cold lane. New axes
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
