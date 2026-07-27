# Handoff: the showcase patch — mountain texturing to reference quality

> Session-start prompt for the NTR-X4 showcase-patch effort. Read this whole
> document, then CLAUDE.md's "Current focus" section, then the backlog row
> NTR-X4 before writing code. Everything here was true as of 2026-07-24;
> reconcile against `docs/backlog.md` + `git log` at pickup.
>
> **Start at "Status" (below) — rounds 1–4 are done and the effort is now
> blocked on a lighting-calibration defect, not on texturing.**

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

## Status (2026-07-24, rounds 1–4)

Evidence: `artifacts/visual/runs/ntr-x4-p1/{r2-bands,r3-lod-alpine,r4-final}/`
(the three presets per round) and `artifacts/visual/runs/ntr-x4-debug/`
(the attribution captures named below). `r4-final` is the current state.

### Measured facts about the showcase site

Derived offline from `thalos_site_detail_6144_90m.f32` over the 36 km window
around the 5799 m peak — quote these rather than re-measuring, and note that
**every slope threshold in the material shader is calibrated against them**:

| quantity | value |
|---|---|
| elevation | p10 2021 m · p50 4063 m · p90 5054 m · max 5799 m |
| slope | p50 17.6° · p75 26.1° · p90 32.9° · p99 42.9° |
| area steeper than 32° (the rock threshold) | 11.5 % |
| slope above 4600 m (the snowfields) | p50 12.4° · p75 19.3° |
| slope below 3400 m (the vegetated flanks) | p50 24.7° · p75 30.7° |
| detail retained below the 180 m / 360 m scale | 89.6 m / 111.7 m RMS |

Two consequences worth keeping: the alpine zone is *gentle* (so slope alone
never exposes it — hence the altitude coupling below), and the sub-treeline
flanks are the *steepest* ground in the window (so any slope rule tuned for
"cliffs" fires on the forest belt first).

### What landed

1. **The canonical altitude bands were re-anchored for the diffusion terrain.**
   The old bands (treeline 2400/3000, snowline 3000/3600) were authored when
   land topped out near 2–3 km; against 5.8 km massifs at the site's tropical
   latitude they buried **67 %** of the massif in permanent snow — the
   white-blob mountain the user flagged. Now treeline `ALPINE_LO/HI_M` =
   3400/4100 and snowline `SNOWLINE_LO/HI_M` = 4600/5300 (Earth's tropical
   lines), subalpine `upland` 2600→3400 so forest holds *to* the treeline
   instead of greying out a kilometre below it, and
   `CLIMATE_COLD_LIFT_MAX_M` 3600→5300 with its dependent thresholds scaled
   by the same factor so polar caps still reach sea level at the authored
   latitudes. Same massif, new bands: **14 %** snow. Mirrors updated in
   `landcover.wgsl` and (constants only) udlod's `body_terrain.wgsl`.
   `GENERATOR_VERSION` 18→19 — **this forces `just bake Mira`**, and the
   persistent capture host must be restarted (`just capture-stop`) after the
   bake or it keeps reporting the pre-bake package as stale.
   Planet-wide check (`just map`): land 35.3 %, forest 33 % of land,
   snow 1.6 %, ice polar-only, 75–90° top biome snow 76 %.
2. **`MaterialBands` carries ecological altitude, not a snow weight.** One
   interpolated scalar yields *every* altitude line (the shader needs the
   treeline as much as the snowline), it interpolates linearly across a
   triangle where a smoothstepped weight does not, and the thresholds were
   already mirrored shader-side. `thalos::landcover` gained
   `alpine_weight` / `snowline_weight` as the one shared definition.
3. **Rock exposure is altitude-coupled, not slope-only.** Above the treeline
   the surface is rock wherever it is steeper than ~8°, with talus on the
   true benches; below it, rock is still the steep-ground rule. Without this
   the alpine zone (median slope 15°) rendered as untextured macro albedo.
4. **Two structure bugs fixed by gating on the *steep* rock term rather than
   the alpine one**: strata banding drawn across the flat alpine zone made
   the upper massif terrace like a topographic map (the bedding frame is
   near-radial there, so its bands follow the contours), and the 22° scree
   shoulder swallowed every forested flank in a pale debris wash.
5. **`SPLIT_FACTOR` 3 → 6 — the geometry was the limiter, not the data.**
   A resident tile at distance `d` has sample spacing
   `d / (SPLIT_FACTOR × 64)`, so at 3.0 the god-view framings (22 km out)
   meshed the ground at 100–200 m while the source carries ~90 m RMS of
   detail below the 180 m scale. The mountains were soft because the *mesh*
   was. At 6.0 the near/mid field meshes at 50–100 m; costs ~4× the resident
   tiles. Visibly more ridge and drainage definition (`r3` vs `r2`).

P1's texgen half — versioned tiling PBR **atlases** from
`crates/offline/texgen` — is **not started**. Everything above is procedural
in-shader layering. That was deliberate: there is no point baking atlases
against a lighting response that is about to change (below).

### Blocking finding — RESOLVED 2026-07-24 (INC-20260724T204059Z)

Root cause: **nothing ever inserted a Bevy `Exposure` on the ship camera**, so
the whole `StandardMaterial` universe ran at `EV100_BLENDER` (9.7) while the
spine converted the same scene flux through its own constants — a factor of
2.77. `thalos_body_shading::spine_parity_exposure` now derives the exposure
from the spine's mirrored constants against `LUX_PER_SPINE_FLUX`, so the two
universes agree by construction; `AMBIENT_SKY_LUX_GAIN` 0.2 → 0.7 re-bridges
the sky fill that had been tuned against the wrong exposure (the env cubemap it
was deferring to is painted in scene-flux units and delivers ~nothing). After:
shadow fill p05 0.134 vs udlod's 0.125, saturation 0.259 vs 0.250. The frames in
`r5-exposure/` are the result — dark rock faces, real snowfields, green forest,
depth in the gullies.

**Still open from the same probe:** terrain receives no cast shadow at showcase
distances (backlog NTR-X6). And the hull/craft moved ~1.5 stops with everything
else Bevy-lit — that wants the user's eye in a flight scene.

The original writeup is kept below; it is the differential that got there.

### Blocking finding: the tile path is ~1 stop hot and flat

The dominant defect in every showcase frame is **not** material selection. Same
terrain, same framing, same epoch, tile path vs udlod
(`ntr-x4-debug/udlod_valley.png` vs `floor006_valley.png`):

| | ground mean luminance | mean saturation |
|---|---|---|
| udlod (spine shading) | 0.232 | 0.250 |
| tiles (Bevy stock PBR) | 0.454 | 0.160 |

The tile ground is twice as bright and a quarter less saturated — **in the
near field too**, so it is not aerial perspective. A washed, low-contrast
ground cannot read like the reference no matter how good the layers are, and
it makes every material judgement unreliable. Ruled out by capture, in order:

- **volumetric clouds** — `THALOS_SCREENSHOT_CLOUD_COVERAGE=0` is identical.
- **detail normals** — zeroing `normal_offset` is identical.
- **specular / roughness** — forcing `roughness = 1.0` is identical.
- **`SHADOW_FLOOR`** — 0.4 → 0.06 is *almost* identical, which is itself a
  finding: **the massif receives no cast shadow at showcase distances**
  (`shadow_f ≈ 1` across the frame). The reference's depth comes largely
  from terrain self-shadowing; the cascade does not reach here.
- **material albedo** — the layer-weight visualization
  (`ntr-x4-debug/layers_valley.png`) shows the pale regions are ordinary
  rock at 0.082 linear, i.e. *darker* than the vegetation it out-brightens.

That leaves the standard path's photometry: sun illuminance + ambient + the
`GeneratedEnvironmentMapLight` diffuse against `CameraExposure`, versus the
spine's own flux normalisation. This is the keystone's "two lighting
universes" debt (`gfx §3`) surfacing as the showcase blocker, and it is
filed as its own backlog row. **Fix it before tuning another material
constant.**

### Also worth knowing before the next round

- **The showcase site is dry.** udlod's own material stack renders it as
  dry-grass tan, i.e. canonical moisture there is low, so the reference's
  dark forested flanks will not appear at this site whatever the shader
  does. Either accept a dry massif as the showcase subject or move the site.
  Best reference-like blocks in the 90 m raster, scored by 6 km relief with
  a sub-treeline valley floor and an above-snowline summit:
  px(3316, 2378) relief 2699 m (1927→4626), px(3517, 435) relief 2668 m
  (2598→5265), px(1708, 1574) relief 2387 m (2953→5340).
  Note the ceiling: 2.7 km over 6 km is ~24° mean — a real mountain, but
  gentler than the reference's Alpine faces. That is terrain content, and
  the base geometry is accepted.
- **Talus fans need curvature**, not slope: debris collects where the slope
  goes steep→moderate downhill. The current model only has per-fragment
  slope, so scree is confined to alpine benches and the aprons the reference
  shows are missing (P3).
- **Stand-in detail needs a band, not a low-pass** (INC-20260724T221348Z,
  found by flying the fix's blind spot: 700 m AGL over the spaceport). Every
  layer here is gated by `footprint_fade`, which retires a wavelength the
  footprint can't resolve but returns 1.0 all the way *down* — so a term that
  stands in for geometry we don't have yet (the canopy stipple for scatter
  trees, the meadow mottle for ground cover, the rock/scree grain for
  micro-relief) sits at full strength precisely where the real thing would be
  resolved. The canopy dimple grew from 2 px/cell at 5 km to ~90 px/cell at
  150 m and the near field read as deep-fried. Stand-ins now ride
  `footprint_band` (near roll-off over 1→4 m per pixel); terms describing a
  real surface property at every scale — strata, striation, canopy colour —
  keep the one-sided fade. **Ask it of every new layer:** what is this a
  stand-in for, and what happens when the real thing is in reach?
- **These three framings only ever sample ≥ 2–4 m per pixel**, so nothing
  tuned against them says anything about how the material reads from a
  cockpit. Before choosing any distance threshold, false-colour
  `length(fwidth(p))` and read the range the framing actually samples
  (probe captures: `artifacts/visual/runs/ntr-x4-nearfield/fp_*.png`), and
  gate a near-field change on `runway-atmosphere` as well as the massif trio.

## Exit criteria

- Agent-verifiable: the three showcase presets render through the tile path
  with the material stack; per-phase A/B captures archived; no pipeline
  errors; Mira presets unregressed (`mira-rim` spot check).
- User-verifiable: the aerial-oblique showcase shot judged against the
  reference — "texturing on mountains" no longer the weak point.
