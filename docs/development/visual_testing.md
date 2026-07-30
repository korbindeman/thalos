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
| Hand the result back to a human | write a page with `{{img:…}}` tokens, run `python3 scripts/present_embed.py <page>.html`, publish the `.embedded.html` | *Presenting results* below |

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

Press F8 while the ship/free camera, space-center hub, or another 3-D god-view
is active. The egui manager reads `assets/viewpoints.json`. “Save current as
new” captures the current body-fixed position, orientation, lens, legacy
viewport/aspect, target body, and canonical boot scene under a stable id. “View” applies a
selected entry in the running world; “Replace from current” updates its pose.
Reload rereads agent edits without restarting the game.

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

The catalog stores cameras in each body's authored surface-fixed frame.
Headless replay projects the pose through the fresh body's current transform and
uses the real `ShipCamera`, so floating-origin shifts do not change the view and
all normal render passes remain coupled.

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

## Presenting results

Captures exist to be looked at. When a change lands whose result is visual — or
whose shape is spatial or structural enough that a diagram beats prose — the
hand-back is a presentation page, not a paragraph describing the images
(CLAUDE.md · *Presenting your work*). Skip it for one-line fixes, doc edits, and
anything with no visible surface; a padded presentation costs more than it
returns.

The page is a self-contained HTML file published with the Artifact tool. It is
served under a strict CSP: no external host loads, and it cannot read the
developer's disk, so `<img src="artifacts/visual/…">` and `file://` both render
broken. Images have to be inlined as data URIs, which is what
`scripts/present_embed.py` is for.

Write the page with tokens where the captures go:

```html
<h2>Ridge stipple, before and after</h2>
{{img:artifacts/visual/runs/comparisons/mira-eva/terrain-regolith-filter/01_legacy.png|legacy}}
{{img:artifacts/visual/runs/comparisons/mira-eva/terrain-regolith-filter/02_filtered.png|footprint-filtered}}
```

Then embed and publish:

```bash
python3 scripts/present_embed.py report.html --width 1400
```

That writes `report.embedded.html` with every token replaced by an `<img>`
carrying a downscaled data URI (JPEG q82, or PNG when the source has alpha), and
prints the per-image and total payload size to stderr. Publish the
`.embedded.html`; keep the token source, so a recapture only costs a rerun of
the script. Token paths resolve relative to the page, then to the repository
root. Aim for a page in the low single-digit MB — `--width` is the knob.

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
