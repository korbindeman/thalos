# Handoff: the showcase patch — mountain texturing to reference quality

> Session-start prompt for the NTR-X4 showcase-patch effort. Read this whole
> document, then CLAUDE.md's "Current focus" section, then the backlog row
> NTR-X4 before writing code. Everything here was true as of 2026-07-24;
> reconcile against `docs/backlog.md` + `git log` at pickup.

## Mission

Make **one patch of Thalos — the size we already have (the 553 km diffusion
detail window around the spaceport, lat 7.6 / lon 178) — look great in
screenshots.** Not the whole planet, not runtime scalability work: a bounded
showcase area whose aerial and mid-altitude framings stand next to a
photogrammetry reference without embarrassment.

User verdict driving this (2026-07-24, after flying diffusion Thalos):

> "The base terrain itself looks great. Texturing on mountains is a bit weak.
> We need to get it more like the last reference image."

The **base geometry is accepted** — do not re-litigate the terrain content
pipeline. The gap is how mountains are *surfaced*.

## The reference (what "great" means, decomposed)

The reference image is an aerial photogrammetry-style render of an Alpine
massif (Wetterstein-like: a sharp central peak with snowfields, forested
valley flanks). What makes it read as real, in rough order of impact:

1. **Slope-driven material exposure.** Rock is where the terrain is steep —
   exposed faces, cliff bands, couloirs — and the transition follows local
   slope/curvature, not altitude alone. Vegetated ground holds on every bench
   and shoulder flat enough to keep soil.
2. **Lithological structure in the rock.** The faces aren't noise: they carry
   quasi-parallel **strata banding**, gully striations aligned with fall
   lines, and ledge systems. This is the single biggest difference from our
   current smooth gray masses.
3. **Talus/scree aprons.** Debris fans at the foot of every face, lighter and
   smoother than the parent rock, blending into vegetation below.
4. **Forest with grain.** At aerial distance the forest is a *texture* —
   per-canopy stippling, density varying with slope/aspect, crisp edges
   against rock and meadow — not a uniform green wash.
5. **Snow that follows the terrain.** Patchy snowfields in hollows and on
   benches with a soft but structured snowline; rock ribs poke through.
6. **Shading depth.** Strong AO in gullies and valley shadow; the material
   detail modulates lighting (normal maps), not just color.

Our current mountains (user screenshots, mid-altitude over the diffusion
window): believable *shapes* with snow caps, but surfaced as soft gray-brown
gradients — per-vertex macro albedo blobs + wrapped fine noise, no strata, no
scree structure, no canopy grain, ridge detail softened by the 90 m height
band + bilinear sampling.

## Where this work must happen (keystone constraints)

- **Target the standard-path tile renderer** (`thalos_body_render::tiles`,
  the NTR-X1 extraction), NOT udlod. udlod and the terrain WGSL stack are
  **end-of-life** (ADR-20260723T142945Z) — texture investment there is thrown
  away. The tile renderer already has the material seam:
  `tiles::material::TileTerrainMaterial` (`ExtendedMaterial<StandardMaterial,
  …>` + `assets/shaders/tile_terrain.wgsl`), with a Hapke branch for airless
  bodies and a stock-PBR branch that is the vegetated starting point. One
  lighting universe = Bevy's lights + `thalos::shadow` receive; keep it.
- Thalos currently renders through udlod by default; the tile renderer runs
  behind `THALOS_TILE_RENDERER=1` and installs on the first `ViewAnchor`
  body. For showcase work over Thalos you will effectively be doing a scoped
  slice of **NTR-X2b** (Thalos on tiles): for *terrain screenshots* the full
  composite re-coupling can be partial — atmosphere/aerial perspective
  matters for the framings (check what `BodySky`/the raymarched atmosphere
  need to composite over tile depth), ocean/clouds/vegetation-scatter can
  come later or be framed out. Do not silently break the Mira tile path
  while doing this; it shares the module.
- Terrain content comes from `thalos_terrain::DiffusionSurface`
  (`THALOS_TERRAIN=diffusion`): global chart (23 km/px) + 553 km 90 m window
  + conditioned filler bands. Landcover/albedo/moisture authority is
  canonical (`ProceduralSurface::macro_albedo_for`) — extend it, don't fork
  it: the material layers below should *consume* its bands/moisture, so the
  showcase patch and the whole planet keep one landcover model.

## Suggested attack plan (phased, each phase capture-gated)

**P0 — framing + baseline.** Add deterministic `ScreenshotPreset`s that frame
the showcase shots: (a) an aerial oblique over the window's big massif
(~5.9 km peaks NE of the site), reference-style; (b) a mid-altitude ridge
shot; (c) a valley-floor-toward-face shot. Capture baselines on both
renderers. Without pinned framings every later comparison is mush.
(`crates/runtime/game/src/screenshot.rs`; the diffusion window center /
massif coordinates are derivable from
`assets/terrain_packages/thalos_diffusion/thalos_site_detail_6144_90m.json`.)

**P1 — material layers (the core).** Per-material-class tiling PBR textures
(albedo + normal + roughness) blended in `tile_terrain.wgsl`'s vegetated
branch: rock, scree, alpine meadow/grass, forest-canopy, snow, bare soil.
- Producer: the offline **`thalos_texgen`** machinery (crates/offline/texgen)
  is the house pattern — versioned atlases, no runtime Bevy dependency; it
  already builds foliage/bark/grass atlases the same way. Painterly-but-
  detailed beats photo-sourced for coherence with the existing art.
- Selection inputs, per fragment/vertex: **slope** (from the mesh normal vs
  radial), **altitude + cold_lift** (reuse the canonical climate bands),
  **moisture/landcover** (vertex-carried from `SurfaceSample`), and
  **curvature/fall-line** where P3 needs it. Slope thresholds first — rock
  above ~35–40°, scree on the 25–35° shoulder below rock, else
  vegetation/snow by the canonical bands.
- Anti-tiling: hex/stochastic blending or two-scale octave mixing — the
  udlod shader has prior art to *read* (not port wholesale).
- Normal maps matter as much as albedo here (reference point 6): sample and
  perturb `pbr_input.N` before lighting.

**P2 — sharper relief under the texture.** The reference's crispness is part
geometry. Two independent levers, A/B them:
- **30 m export for the patch only**: the upstream project also released a
  30 m model (deferred at FT-0; see `terrain-diffusion/FT0_NOTES.md`). A
  patch-sized 30 m window (9× the pixels of 90 m over a *smaller* area —
  e.g. the massif quarter of the window) is cheap and slots in as another
  `DiffusionSurface` band/window. This most directly buys ridge definition.
- **Detail normals without geometry**: a height-derived detail-normal band in
  the material (the fine filler octaves evaluated in-shader or baked to a
  texture) so lighting shows sub-mesh relief the vertices can't carry.

**P3 — structure in the rock.** Strata banding (a stable pseudo-bedding
frame: dot the world position against a slowly-varying dip/strike direction
field, band the result, modulate albedo/roughness/normal), fall-line gully
striation (anisotropic noise stretched along the downhill tangent), and
scree fans (accumulate where slope transitions steep→moderate — a curvature/
slope function, no simulation needed). This is what kills "smooth gray mass".

**P4 — forest grain + snow line.** Canopy stippling in the forest layer
(cell-noise luminance + normal dimples at ~10–30 m cells so aerial forest has
per-tree grain long before the scatter's real trees load), density modulated
by the canonical moisture; snow as a material layer with a noise-broken,
slope-gated line (steep faces shed snow — the reference's rock ribs through
snowfields).

Order P1 → P2 → P3 → P4, screenshotting after each; stop and reassess against
the reference at each gate rather than building the full stack blind.

## Verification loop (mandatory)

- `THALOS_TERRAIN=diffusion THALOS_TILE_RENDERER=1 just screenshot <preset>`
  per iteration; `just compare` with a typed axis when attributing a change
  (one factor per axis — visual_testing.md rules). Read the PNGs yourself.
- Keep matched before/after evidence for the final writeup (the canonical
  comparison dir is overwritten — copy to `artifacts/visual/runs/`).
- The user judges "great" — batch the subjective checkpoints; don't ask per
  tweak. Their calibration anchor is the reference image described above.
- Known trap: a capture whose log shows a shader/pipeline error can still
  exit 0 and write PNGs (BL-20) — check stderr before trusting a frame.

## Rules of engagement

- WGSL work: read `.claude/skills/wgsl-bevy/SKILL.md` first; append new
  pitfalls there when hit.
- Backlog discipline: this effort is **NTR-X4** (rescoped to showcase-patch).
  Flip to `wip` on start; new discoveries become rows, never silent TODOs;
  land compile-clean slices with capture evidence → `verify`.
- One canonical path: material selection must read the canonical landcover /
  climate functions (`macro_albedo_for` inputs), never a parallel biome
  model. If a needed input isn't exposed, extend the canonical seam.
- Don't touch udlod's shader stack except to read prior art.
- Check for concurrent sessions in this checkout before heavy builds (`just
  check` failing on files you didn't touch = someone else mid-edit; one
  Cargo command at a time).
- Structural Rust changes require `just capture-stop` before the next
  persistent capture.

## Exit criteria

- Agent-verifiable: the three showcase presets render through the tile path
  with the material stack; per-phase A/B captures archived; no pipeline
  errors; Mira presets unregressed (`mira-rim` spot check).
- User-verifiable: the aerial-oblique showcase shot judged against the
  reference — "texturing on mountains" no longer the weak point.
