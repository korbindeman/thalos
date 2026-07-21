# Visual testing

Thalos diagnoses and iterates on graphics with deterministic, full-resolution
headless captures. ADR-20260721T032344Z-isolated-headless-visual-comparisons rejects a live multi-camera split screen: it would
duplicate the one-view renderer and change LOD, SSAO, shadow, antialiasing, and
other viewport-dependent inputs while supposedly holding them constant.

## Canonical workflows

- `just screenshot <preset>` captures one in-context beauty frame.
- F8 in a live 3-D view + `just screenshot latest` hands the player's exact
  body-fixed camera position, orientation, lens, and viewport to an agent.
- F2 desktop screenshots and F8 perspective handoffs show saved/error toasts
  only after the capture outcome is known, so F2 feedback is never baked into
  its own image. Neither appears in F1 photo mode, whose clean frame hides the
  shared toast container.
- `just compare <preset> <axis>` captures every typed variant for one axis in
  isolated game processes, then assembles one comparison artifact set.
- `just preview` remains the supplemental isolated-asset path. It does not replace
  the in-context game capture.

`tools/screenshots/` is the curated latest-view surface: every named preset
owns one stable filename there and overwrites it on the next canonical capture.
Ad-hoc framings set `THALOS_SCREENSHOT_OUT` to a path under
`tools/agent_scratch/screenshots/`; they do not mint new names beside the
canonical views. Comparison matrices always use the scratch tree automatically.

## Player-saved perspective handoff

Press F8 while the ship/free camera, space-center hub, or another 3-D god-view
is active. Thalos writes a versioned handoff to
`tools/diagnostics/latest_perspective.json` and confirms it with an in-game
toast. An agent responding to “check my latest perspective” runs:

```text
just screenshot latest
```

The result overwrites `tools/screenshots/latest_perspective.png`. The handoff
stores the camera in the nearest terrain body's fixed frame, plus vertical FOV,
viewport, target body, and canonical spawn scene. Headless replay projects that
pose through the fresh body's current transform and uses the real `ShipCamera`,
so floating-origin shifts do not change the view and all normal render passes
remain coupled.

This is a camera handoff, not a save game. The recorded simulation time is kept
as provenance, but replay deliberately boots the named canonical spawn instead
of partially restoring craft dynamics. It therefore reproduces geographic
framing and scene configuration; a one-off moving-craft state or weather moment
still needs a normal screenshot or a dedicated deterministic preset.

Atmosphere has two complementary canonical framings: `earth-reference` for the
space/orbital limb and `runway-atmosphere` for the near-surface sky, long
slant-path haze, and terrain/structure recession. Both support the typed
`atmosphere` comparison axis. The legacy renderer is capture-only: there is no
live or persisted gameplay atmosphere selector, because sequentially switching
renderer-global resources inside one process invalidates the isolation contract.

One comparison run changes exactly one declared axis. Any normal screenshot
framing overrides (`THALOS_SCREENSHOT_SIZE`, `_AZIMUTH`, `_ELEVATION`,
`_DISTANCE`, `_WARMUP`, `_HUD`) are inherited identically by all variants.
The runner owns `THALOS_SCREENSHOT_OUT` and the axis-specific override so those
cannot drift between captures.

The recipe performs one Cargo build for both the game and orchestrator, then
launches the built orchestrator directly. It must not invoke nested or repeated
Cargo processes against the shared workspace target directory. The orchestrator
reproduces Cargo's platform dynamic-library search path when spawning the game;
this includes the profile, profile/deps, and `rustc --print target-libdir`
(dynamic `std` may live there). Directly launching a `bevy/dynamic_linking`
binary without that complete environment is the INC-0008 pre-main crash.

## Initial axes

| Axis | Variants | Intended diagnosis |
|---|---|---|
| `atmosphere` | `custom`, `bevy` | Legacy `BodySky` against the canonical Bevy raymarch |
| `ssao` | `off`, `on`, `raw` | No AO, normal AO application, and the raw AO field |
| `terrain-lighting` | `lit`, `fullbright`, `geometric-normal` | Separate raster coverage from lighting, then isolate the terrain normal stack |
| `terrain-culling` | `backface`, `two-sided` | Test whether grazing holes are missing back-facing raster coverage |
| `terrain-regolith-filter` | `legacy-unfiltered`, `footprint-filtered` | Matched before/after for airless procedural-detail Nyquist filtering |

Axes are intentionally typed in `crates/game/examples/visual_compare.rs`. Add a
new one only when every variant can be selected by a capture-only override that
does not persist user settings. A multi-test may have N variants, but they must
all remain values of the same factor.

## Artifact contract

Each run writes to:

```text
tools/agent_scratch/screenshots/comparisons/<preset>/<axis>/
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
state, invariant screenshot overrides, per-variant environment override, image
paths, and numerical diff metrics. Copy a result to an explicitly named
evidence location only when a revision-to-revision artifact must be retained.

Pixel differences are evidence, not a verdict. Stochastic or temporal effects
can produce a non-zero diff without a meaningful visual regression. Use the
contact sheet/wipe to interpret the numbers and keep the diagnosis tied to a
specific hypothesis.

Known harness gap (BL-20): a headless game process can currently exit zero even
after Bevy logs a shader/pipeline validation failure, leaving a PNG with missing
render layers. Until the runner promotes those errors to a failed variant,
inspect stderr and reject any comparison containing a pipeline-cache error.

## Root-cause loop

1. Choose or add a preset that reproduces the symptom.
2. State the competing hypotheses.
3. Pick one axis that distinguishes at least two of them.
4. Run the comparison with all framing and warm-up inputs held fixed.
5. Inspect the contact sheet, full captures, wipe, and diff metrics.
6. Eliminate hypotheses, then repeat on the next single axis until the cause is
   pinned.
7. Keep the matched before/after comparison as verification of the fix.

Future debug channels—normals, depth, terrain LOD/tile IDs, shadow factor and
cascade, material IDs, atmosphere transmittance, and lighting lobes—join this
same runner. They do not get their own camera or comparison framework.

## Verification

Runtime-verified 2026-07-21 with both initial axis shapes:

- `just compare earth-reference atmosphere` produced two aligned 1800×1200
  captures, a labelled A/B sheet, diff, wipe, metrics, and manifest.
- `just compare spaceport-aerial ssao` produced three aligned 1920×1080
  captures (off/on/raw), a labelled 2×2 sheet, two baseline diffs/wipes,
  metrics, and manifest through the direct dynamic-game launcher.
- `THALOS_TILE_CACHE=0 just compare mira-eva terrain-regolith-filter` produced
  aligned 2048×1280 legacy/filtered captures at the live canonical EVA site;
  the legacy ridge stipples while the footprint-filtered ridge is stable and
  nearby regolith texture remains resolved (INC-0009).
