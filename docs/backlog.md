# Backlog

**Role:** the *execution layer* beneath the sprint plan docs. The plans —
[architecture_cleanup.md](architecture_cleanup.md) (`clean`) and
[graphics_fidelity.md](graphics_fidelity.md) (`gfx`) — are **strategy**: what to
build, why, in what order, with the full rationale. This file is the
**operational queue** that answers **"what's next?"** — concrete, scoped,
status-tracked items in a rolling near-term window. Agent-maintained: kept in
sync as work lands.

**How this file is driven** — see the `steer` skill
(`.claude/skills/steer/SKILL.md`) for the full procedure. In short:

- **"what's next?"** → the agent picks the top pickable item, scopes it into a
  brief, and stops for your go.
- **"add X / fix Y"** → filed here as a `next` item, then done as normal work.
- **vision / direction talk** → captured in the right plan doc (ADR if a fork
  resolves), then decomposed into items here.

**Status vocabulary:**

- `next` — ready, no unmet dependency; the pool "what's next?" picks from.
- `wip` — in progress.
- `verify` — **landed compile+clippy-clean but not runtime-verified.** The
  thalos-specific status: agents can't run the game, so landed work waits on
  either a headless probe (`just screenshot` / `just preview` / `just
  ui-preview` — agent-servable) or a user play session + screenshot. A `verify`
  item's row names which. Not `done` until observed working.
- `blocked` — waiting on another item or an external / user action (named in
  **Deps**).
- `done` — complete and verified (date in Notes).
- `later` — real but not now; listed for dependency context, not yet pickable.

**Conventions:**

- **IDs are stable.** Reuse the plan docs' own IDs where they exist (packages
  `CL-A`…`CL-G` from `clean §3`, `F1`–`F9` / `W`-numbers / `TM` / `C1` from
  `gfx`); mint `BL-n` for items with no plan-doc home. Never renumber; mark
  `done`, don't delete — the queue is also the record of what was done.
- **Rolling window.** Only the two active sprints are tracked at item
  granularity. The graphics later-sprint pool stays at doc granularity in
  `gfx §4`; pull items in here as they near.
- **One source of truth per fact.** Rationale and design stay in the plan docs /
  specs / ADRs; this file holds status + scope pointers, not re-explained
  reasoning. Flip the plan doc's checkbox and this row **in the same change**.
- **Discovered work becomes a row, never a silent TODO.**
- **Refs:** `clean §N` = architecture_cleanup.md · `gfx §N` =
  graphics_fidelity.md · `ADR-NNNN` = adr/ · `INC-NNNN` = incidents/ · other
  docs by filename.

> Statuses below were seeded 2026-07-18 from the plan docs (whose checkboxes
> date to 2026-07-05) plus auto-memory. First `steer` invocation should
> reconcile against `jj log` + the working tree.

---

## Track 1 — Architecture & code quality (primary sprint · `clean`)

Goal (`clean §1`): one canonical path per operation, N-by-default, in-flight
unifications finished. Packages land compile+clippy-clean, update the relevant
spec, and end with a user verification checklist.

| ID | Item | Status | Est | Deps | Refs |
|----|------|--------|-----|------|------|
| CL-B | GameContext Phase 3 — invert ownership (`NextState<GameContext>` sole writer, `ContextHistory` stack) | verify | — | **user checklist** in clean §3.B (menu↔hub↔VAB↔flight matrix) | clean §3.B · ui_flow.md |
| CL-C | One craft-placement core (`place_craft` in `spawn.rs`) | verify | — | **user checklist** in clean §3.C (all spawn/respawn/relaunch/launch paths; watch for teleport jitter) | clean §3.C |
| CL-C2 | Route map/debug teleports through `place_craft` — fixes the flagged candidate bug (map cmd-click teleport never clears the Avian bubble → phantom velocity) | next | S | CL-C verified first (the reorder needs runtime confidence that `transition_authority` doesn't snapshot state) | clean §3.C |
| CL-D | **Structures become the one placement layer**: split `runway.rs` (geometry → `structures/runway_geometry.rs`, placement → the C core, collider/anchoring/spaceport → `structures/`); promote connections into `StructureRegistry` (structurally fixes grass-under-taxiway); extract shared f64 `snap_to_body_surface` (align with `ViewAnchor`'s `cam_world`/`ground_world`, don't duplicate) | next | L | — | clean §3.D · base_building.md |
| CL-E1 | `ActiveCraft(Option<Entity>)` accessor seam + `track_active_craft` | verify | — | was blocked on an unrelated `thalos_ui` WIP — re-check it builds, then full scenario matrix (pure add, nothing observable should change) | clean §3.E |
| CL-E2 | Incremental per-craft-state migration (`GearState`/`ParkingBrake`/`EvaMode`/`RealizedControl`/`ManeuverPlan` → craft-entity components; `.single()` sites → `ActiveCraft`) | later | — | requirements-driven: do when a second craft actually exists | clean §3.E kept-singleton ledger |
| CL-G | UI kit runtime interaction pass (hover / drag / typing / hangar flows) | verify | — | **user session** — headless kitchen-sink PNG already checked | clean §3.G · ui.md |
| CL-A | Dead bake-pipeline purge — rescoped to function-level surgery on the `compile_terrain_config` chain | later | L | natural time = the terrain-generator rework | clean §3.A |
| CL-B2 | HUD hide/restore → `OnEnter`/`OnExit(GameContext)`; delete the `.open` mirrors once no reader needs them | later | M | CL-B verified | clean §3.B |
| CL-C3 | Placement helper dedups: one craft-clearance measurer, generic deferred-placement gate. (The shared cursor→pad raycast may already be done — `cursor_body_dir` landed 2026-07-05; re-verify at pickup) | later | S | — | clean §2.1 |
| CL-F | Small unifications, batch as touched: document the terrain-height three-mirror design at `terrain_registry.rs`; `camera.rs` submodule split only when camera work next happens | later | S | — | clean §3.F |

## Track 2 — Graphics fidelity (secondary sprint · `gfx`)

Goal (`gfx §2.3/§3`): the one-world principle via the F1–F9 unification
foundation. F1/F2 verified; F3–F6 landed awaiting calibration; F7–F9 next
sprint.

| ID | Item | Status | Est | Deps | Refs |
|----|------|--------|-----|------|------|
| GF-CAL | **Screenshot calibration & verification sweep** for everything landed-☑: noon exposure (`GLOBAL_EXPOSURE_STOPS`, W9 flux constant), SSAO tuning (F5 dials), one-shadow-world screenshots (F6/W5/W6 bias knobs), landcover palette, moonlight, clump cards. Start agent-side with `just screenshot` presets; queue what needs a live eye for the user | next | M | — | gfx §3, §4.1–4.3 · shadow_unification_prompt.md |
| TM-P1 | Macro landcover Phase 1 — moisture in f64 `ProceduralSurface` → albedo-attachment alpha | verify | — | landed 2026-07-18; screenshot pass | terrain_macro.md |
| TM-P2 | Climate MVP (terrain_macro Phase 2) — latitude cold-lift descends eco bands (polar ice caps, tundra), warmth-gated sand palette, moisture geography (equator/subtropic/mid-lat belts + continentality); shader latitude via `compute_local_position` | verify | — | landed 2026-07-18; world_map-verified, needs live-eye pass | terrain_macro.md §4 |
| TM-P2r.1 | Scatter/biome coupling — **trees & shrubs × moisture + treeline**: `woody_biome_gate` in `body_render::ground::scatter` multiplies the woody accept by a moisture dryness ramp (mirrors the ground's `vegetation_color` forest transfer) × a cold-lift-descended treeline term, so trees thin on dry steppe, vanish on bare desert, and stop at the poles instead of ignoring climate. Landed 2026-07-20, compile + scatter-tests clean | verify | — | **user:** orbit/cruise over a dry belt + a polar site — woody cover should follow the ground palette | terrain_macro.md §3, §4 |
| TM-P2r | Phase 2 remainder (post TM-P2r.1): grass profile per biome, explicit biome weights if needed, sea ice, regional relief character | later | M | TM-P2 verified | terrain_macro.md §4 |
| BL-11 | Whole-planet biome map export (`just map`): mercator (default) / equirect, true in-game macro palette render + flat `MacroBiome` class map + area-weighted per-biome coverage, dryness-histogram and per-latitude-band stats. `ProceduralSurface::sample_biome_d` classifies from the **same band weights the albedo blends** (`macro_band_ts`, output-identical refactor of `albedo_at`). *(Renumbered from a colliding BL-9 — the water-field work minted BL-9/BL-10 concurrently.)* | done | — | — | terrain_macro.md · world_map.rs · PNGs read 2026-07-20 |
| BL-12 | Celestial backdrop daylight suppression follows the render camera: `update_atmospheric_star_visibility` now consumes the canonical body-fixed `ViewAnchor`, so freecam, flight, god-view, and screenshot cameras evaluate twilight and the Kármán transition at the actual observer. Landed 2026-07-20; focused tests + `cargo check -p thalos_game` clean | verify | S | **user:** keep a craft on the daylight surface, freecam above the Kármán line (backdrop visible), then descend through it (backdrop fades according to the camera) | gfx §2.3 one-view-anchor invariant · `sky_render.rs` |
| BL-13 | **Earth-reference atmosphere convergence:** deterministic 3:2 ISS-like land-only orbital screenshot preset; matched custom `BodySky` vs Bevy `AtmosphereMode::Raymarched` captures; Bevy raymarch promoted as the one canonical rocky-body atmosphere through a camera-local `ViewAnchor` proxy; Earth aerosol/ozone projection and one 0.1 radiometric adapter replace the old broad white halo. Ocean and clouds are explicitly not part of this frame or acceptance test. In-atmosphere runway comparison framing is being added for the surface half of verification | wip | M | **agent:** inspect `just compare runway-atmosphere atmosphere`; **user:** run `just game orbit` and confirm the limb stays attached and stable through camera motion | atmosphere.md · gfx §2.3/§4.5 · ADR-0010 |
| BL-14 | **Fast, reliable Bevy dev loop:** route game, headless screenshots, object preview, and UI preview through one dynamic-link feature set; move temporary wgpu driver counters behind opt-in `gpu-counters` so preview/game reuse one dylib; fix the cross-platform screenshot env recipe; retire the unstable local `-Zthreads=8` experiment after its MIR ICE poisoned incremental objects; document one-target/one-Cargo-process hygiene | done | S | one-time feature/config fingerprint rebuild completed (196.9 s); default dynamic game graph compile-clean; all four recipes dry-run clean | tooling.md · justfile · local `.cargo/config.toml` · INC-0006 |
| BL-15 | **Photographic daylight star suppression:** the Earth-reference orbit frame is correctly above the Kármán line, but the exposed starfield is much brighter than the daylight reference. Extend BL-12's camera-anchored atmospheric visibility with the shared exposure/sky-luminance response; do not hide stars merely because a planet is in frame | later | S | after BL-13 live verification / F7 exposure calibration | gfx §2.3 · `sky_render.rs` |
| BL-16 | **Mira EVA transparent horizon stipple:** the canonical eye-level `mira-eva` probe reproduces the ridge artifact after a 1,200-frame warm-up. Fullbright proves complete opaque raster coverage; culling, provider, tile lookup, self-shadow, external-shadow, and height-normal-mip A/Bs do not remove it; geometric-normal rendering does. Root cause: Mira's metre-scale procedural regolith colour and micro-normal octaves were distance-faded but not filtered against the fragment's body-space pixel footprint, so grazing terrain compressed them below Nyquist and Hapke amplified the unresolved normal flips into bright/dark stipple. Regolith detail now uses the shared per-octave footprint fades; f64 crater distance also removes an independent compatibility-producer height quantisation defect. | done | M | matched `terrain-regolith-filter` A/B inspected; cache-disabled `mira-eva`, `mira-surface`, and `mira-orbit` captures | mira_airless_mvp.md · INC-0009 |
| BL-18 | **Deterministic visual comparison harness:** `just compare <preset> <axis>` runs typed variants as isolated headless game processes and emits full-resolution captures, labelled contact sheet, baseline diffs/wipes, metrics, and a provenance manifest. Its direct child launcher reproduces Cargo's complete dynamic-library path (profile, deps, Rust target-libdir; INC-0008). First axes: atmosphere A/B and SSAO off/on/raw multi-test | done | M | runtime-verified 2026-07-21: 2-way atmosphere + 3-way SSAO artifacts inspected | visual_testing.md · gfx §4.10 · ADR-0011 · INC-0008 |
| BL-19 | Extend visual comparison axes with shared debug channels as their shader instrumentation lands: normals, depth, terrain LOD/tile IDs, shadow factor/cascade, material IDs, atmosphere transmittance, direct/ambient/specular lobes. Consider an interactive image wipe only if the artifact loop proves too slow | later | M | BL-18 | visual_testing.md · gfx §4.10 · ADR-0011 |
| BL-20 | **Visual comparison must fail on render-pipeline errors:** `just compare runway-atmosphere atmosphere` accepted two images after Bevy logged a fatal WGSL pipeline validation error, producing atmosphere over missing terrain. Route render/shader compilation failures into a headless non-zero exit or machine-readable failure marker and refuse to assemble artifacts from invalid variants | next | S | BL-18 | visual_testing.md · 2026-07-21 runway probe |
| TM-P3 | **Biome rebalance — the planet reads as a real planet, authored to lore** (35 % land, geologically old/low-riding, lush with rust-red laterite where cover thins — lore §II). Landed 2026-07-20, `GENERATOR_VERSION` 12: (1) macro altitude bands re-aligned to the ground's eco bands (upland 1500–2400 eco, rock 2400–3000, snow 3000–3600 — was upland-from-**120 m**, which claimed 59 % of land and crushed the climate palette); (2) **INC-0005** — `.max(EPSILON)` smoothstep guard inverted the forest term (forest painted on the *driest* ground since TM-P1); (3) tundra moved between upland and rock in the chain; laterite soil step added (mirrors ground `C_SOIL`); (4) land lowered (platform 420→300 m, interior gain 650→400 m) + `CONTINENT_C0` 0.143 → land 35.2 %; (5) subtropic belt 0.40→0.70 + real continentality gate; (6) **ecotone mosaic gate** (v13, from the user's orbital screenshot — dry belt read as splotchy camo): 90 km/9 km moisture tiers fade to 35 % where the latitude+continental trend has committed, so belt cores are coherent and patchwork lives at climate transitions. Map-verified: forest 30/grass 26/steppe 19/desert 4.6/tundra 2/snow 3.6 % of land, coherent belts in the right latitudes, runway site LAND at 602 m. **Note:** coastlines reshaped by the C0 change — the pending water/coast live-eye (new BL-9) sees new coasts. **Next slice:** TM-P3b | verify | — | **user:** orbit + descent over a dry belt and the polar gradient; `just screenshot` presets agent-side first | terrain_macro.md §4 · INC-0005 · lore/solar_system.md §II |
| TM-P3b | Biome *identities* on top of the TM-P3 balance: erg/reg desert character (dunes vs gravel), savanna between forest and steppe, taiga vs temperate forest tone, softer polar rock ring (wider tundra), sea ice at the polar ocean; scatter coupling folds in via TM-P2r | later | M | TM-P3 verified | terrain_macro.md §4 |
| MIRA-MVP | **Playable offline-package Mira compatibility slice:** standalone `just bake Mira`; 31.4 MiB content-keyed deterministic airless package; one N-body `BodySurfaceRegistry` authority through ground/impostor/height/physics; package-aware tile-cache namespace; `mira`/`mira-eva` routes; crater-targeted `mira-orbit` + `mira-surface` headless probes; airless scatter gate + calibrated shared Hapke regolith | done | — | — | screenshots inspected 2026-07-20 · mira_airless_mvp.md §1 · ADR-0008 |
| MIRA-0 | **Production terrain-package tracer:** schema-v1 magic/manifest + producer/model identity; 32→512 five-level adaptive height pyramid with six R16 roots, quantized signed residuals, canonical half-open ownership, 2,047 logical nodes / 1,961 blobs, 86 ancestor fallbacks within a 256 m compatibility budget; predictor/reconstruction metrics; checksum/overlap/hierarchy/error/content validator; exact-artifact tile-cache fingerprint; `PackageSurface` reader. Deterministic SHA-256 and cache-disabled orbit/surface captures verified 2026-07-20; no seams/fallback holes | done | — | MIRA-MVP | mira_airless_mvp.md §5, §8–§9 · ADR-0008 |
| MIRA-1 | **Airless diffusion patch proof:** checksum/licence-pinned SLDEM2015 + aligned Kaguya DTM subsets, disjoint geographic holdouts, labelled synthetic crater/ejecta/gardening surfaces, physical-scale Laplacian S0–S3 cascade, airless conditioning, overlapping-window inference, fixed-seed determinism, model card and spectral/slope/SFD + VRAM/timing evidence | later | L–XL research | MIRA-0 schema | mira_airless_mvp.md §6, §8 |
| MIRA-2 | **Whole-sphere adaptive Mira bake:** 4096-face campaign starting point, sphere-native priors + direction seed, cross-face tangent windows/overgenerated seam consensus, canonical package borders, hierarchical diffusion, H0 occupancy/material channels, complexity/error-driven residual pruning and per-node codec selection; fixed Mira package + limb/face/equirect/rate-distortion/SFD/seam evidence | later | XL | MIRA-1 | mira_airless_mvp.md §5–§6, §8–§9 |
| MIRA-3 | **Client reconstruction + cache:** deterministic conditioned close-band terrain, collidable parity boundary, cosmetic micro + airless boulders; package/decode/reconstruction cache layers and cold/warm benchmarks; evaluate an optional small single-pass residual CNN only if procedural detail fails the low-approach gate, with Tier 0 independent of learned weights | later | L | MIRA-2 | mira_airless_mvp.md §5, §7–§9 |
| MIRA-4 | **Mira playability + acceptance:** parameterised target-body orbit/EVA routes, headless map/orbit/approach/surface matrix, height-authority evidence, tuning, and user play-session checklist | later | M | MIRA-3 | mira_airless_mvp.md §8–§9 |
| F4r | F4 remainder: SH-9 + spine (terrain) ambient port; env-cubemap crossfade; space ambient retires with F7 | next | M | — | gfx §3 F4 |
| F5r | AO onto veg/rock/grass/hull + `StandardMaterial` ambient; spatial blur; VBAO upgrade (W8) | next | M | — | gfx §3 F5 · §4.3 |
| W12r | Terrain horizon self-shadow, object-side v2: per-fragment horizon term for spine materials (trees/grass/rocks), longer-than-resident reach, max-mip acceleration | next | M–L | — | gfx §4.2 |
| W1 | Aggregate canopy/grass colour baked into terrain albedo (kills the orbit→ground pop; tint-out and geometry-in as one coupled curve) | next | S | — | gfx §4.7 |
| W2 | Cloud sun transmittance for all receivers — retained stable ID, now subsumed by CLOUD-5: global density-derived tail + near cascades, also drives atmosphere shafts and overcast ambient | later | L | CLOUD-4; cloud architecture decision | gfx §4.2 · cloud §3.5 |
| W3 | GoT grass shader tricks: view-space blade widening + curved normals + fractional-width LOD | next | S | — | gfx §4.7 |
| W4 | Two-sided foliage translucency + bake into the impostor; pull W21 (foliage atlas mips) forward with it | next | S | — | gfx §4.7 |
| C1 | Tonemapper A/B: AgX vs Khronos PBR Neutral (ACES rejected) | next | S | after GF-CAL settles exposure | gfx §4.9 · Open Q2 |
| VEG-R | One `VegLayer` driver folding `grass.rs`/`vegetation.rs`/`rocks.rs` (they triplicate the clipmap lifecycle, diverge on base-clearing) | next | M | Open Q5 call | gfx §4.7 |
| BL-1 | GPU grass slice 2: compute cull + indirect draw (slice 1 landed behind `GraphicsSettings::gpu_grass`, preview-verified) | next | M | — | vegetation.md |
| CLOUD-0 | Planet-cloud baseline, probes, and budgets: cloud-specific headless runway/cruise/interior/limb/sunset presets; current 1080p/1440p GPU+memory baseline; artifact inventory and acceptance captures | done | S–M | landed in `codex/cloud-0`; user verdict 2026-07-20: current renderer uniformly mediocre against Blackrack-class morphology/lighting bar, confirming structural replacement priorities | cloud §4 CLOUD-0 · cloud_baseline.md |
| CLOUD-1 | Canonical cloud ownership/schema: one authored `CloudClimate`, one per-body runtime `CloudWeatherField`, `None` authoritative, first weather-derived orbit layer; absorb render mechanism under `body_render` and delete legacy/reference ownership | next | M–L | CLOUD-0 done; ADR-0009 accepted | cloud §2–§4 · ADR-0009 |
| CLOUD-2 | Scalable cloud targets and reconstruction: viewport-relative interleaved march, robust body-fixed history validation/clamp/upscale, screenshot mode, and measured quality ladder | later | L | CLOUD-1 | cloud §3.3–§4 |
| CLOUD-3 | Multi-scale cloud density and range: weather cube type/base/top, true 3-D base/detail noise, profile/anvil shaping, anti-tiling, empty-space skip, and near-to-horizon regime LOD | later | L | CLOUD-2 | cloud §3.1–§3.3 |
| CLOUD-4 | Atmosphere-coupled cloud lighting: shared sun/eclipses/atmosphere/sky inputs, dark-core/powder/multiple-scatter response, and correct foreground/background media ordering | later | L | CLOUD-3 + F3/F4 substrate | cloud §3.4 |
| CLOUD-5 | One-world cloud interactions: density-derived global + near cloud-sun transmittance, all surface receivers, matched atmosphere shafts, and overcast ambient/IBL response | later | L | CLOUD-4 + F6 substrate | cloud §3.5 |
| CLOUD-6 | Orbital cloud projection and handoff: density-derived optical-depth/albedo/normal/height moments, reduced-detail limb regime, and invisible surface↔orbit↔map transition | later | M–L | CLOUD-3 | cloud §3.6 |
| CLOUD-7 | Living weather and authoring: simulation-time advection/growth/decay, front/cyclone stamps, per-body presets, and paint/import/debug tools | later | L | CLOUD-5 + CLOUD-6 | cloud §3.1 · cloud §4 |
| CLOUD-8 | Cloud-interior and storm polish: boundary wisps/extinction, precipitation or virga shafts, lightning/emissive path, and final reference-bar tuning | later | M–L | CLOUD-4 + CLOUD-7 | cloud §4 · cloud visual target |
| TM1 | Tiling-material detail: material-ID height-biased weight blend over a texgen material array | later | M | mind the Metal bind budget (F7 note) | gfx §4.6 |
| TM3 | Collapse the palette/mirror debt onto the `SurfaceQuery` seam — **largely overlaps TM-P1; re-scope against what Phase 1 landed before starting** | later | M | TM-P1 verified | gfx §4.6 · terrain_macro.md |
| F7 | Metallic branch in `surface_brdf` + one shared view-level scene/atmosphere bind group + prefiltered env from the F3 LUT | later | L | F3/F4 verified | gfx §3 |
| F8a | Structures onto `shade_surface` | later | M | F7 | gfx §3 |
| F8b | Hull onto `shade_surface`; retire the CPU reflection probe + magic constants | later | M–L | F8a | gfx §3 |
| F9 | `FOLIAGE`/`WATER` branches in `shade_surface`; retire `shade_foliage` + `body_water.wgsl`; re-enable ground-LOD water | later | M | F7 | gfx §3 · ADR-0002 |
| GF-pool | Later-sprint pool — W11, W13–W18, W20, TM4, 6b, SSR, bent normals… stays at doc granularity until pulled in | later | — | — | gfx §4 |
| BL-4 | Orbital black continents + coast speckle: sky-pass near/far classification → shell-segment membership; shelf ease-out shoulder; awash-reef fold (offshore relief never breaches — `GENERATOR_VERSION` 5); error-aware shoreline tie-band feather | done | — | — | INC-0003 · verified via `THALOS_SCREENSHOT_DISTANCE` bracket captures 2026-07-19 |
| BL-7 | Distant terrain grey/shiny/pixelated: material-mask bake divided its slope/curvature stencil by a step clamped to 250 m while coarse texels span km — rock+wetness masks saturated at every LOD coarser than 250 m/texel (fix: true `tile_lod_m` step + fractal-scale rock-slope compensation `(step/30 m)^0.35`; `GENERATOR_VERSION` 8) | done | — | — | INC-0004 · verified via `THALOS_SCREENSHOT_DISTANCE` brackets 8/100/400/1800 km 2026-07-20 |
| BL-8 | **Coastline as authored data** (ADR-0005): per-body coast/bathymetry cube baked from the `SurfaceQuery` surface at spawn; `BodySky` ocean crossfades authority by range — near = depth-compare (exact), far = atlas coverage/colour + height-based occlusion. Round 2 (user live-eye findings): foreshore deepened 4→15 m so archipelago seas stop reading as one vast translucent bank; awash reef 2→6 m (surface "scum" rings → pale submerged bathymetry); shallow-colour e-folding 14→8 m; near-feather error floor models mesh coastal sampling error (2e-5·t, kills mid-range coast "bites" + tile-seam leak-through); atlas 512→1024 so macro islands stay resolved at range (`GENERATOR_VERSION` 10) — **user fly-over failed verification 2026-07-20** (speckle + translucent wash regenerated); the depth-compare/crossfade half is superseded by BL-9 / ADR-0006, the coast atlas survives as the cascade tail | done | — | — | ADR-0005 · ADR-0006 · INC-0003 |
| BL-9 | **Water as a projection of the one signed sea field** (ADR-0006): `BodySky` ocean coverage/colour sampled directly from the resident udlod height-tile atlas (tile-tree walk at footprint LOD, mip-sampled) with the coast atlas as the coarse tail — same field, two resolutions, no authority crossfade; depth demoted to resolvable-geometry occlusion (field-assisted land test + footprint-scaled in-front margin for craft). Deletes the `2e-5·t` mesh-error feather, the error-aware tie band, `OCEAN_AUTHORITY_*`, and the reconstructed-height occlusion thresholds. `BodySkyMaterial` hand-implements `AsBindGroup` to bind the udlod atlas/tile-tree | done | — | — | ADR-0006 · body_sky.wgsl · sky_material.rs · screenshot brackets + **user fly-over verified stable 2026-07-20** |
| BL-10 | **Crisp beaches + shore-water interaction** (user directive 2026-07-20: no marshy semi-island coasts; beach biome now, other coastline types later). Phase A generator/landcover: coast-band detail-meander suppression (clean waterline arcs; keep the BL-6 domain warp; awash shoals pushed offshore), beach berm+foreshore profile, sand material band + veg/grass clearing, wet-sand darkening (0..~0.7 m), `GENERATOR_VERSION` bump. Phase B water optics: steeper shallow opacity (sand visible only ≤ ~1.5 m column — kills the soupy translucent shelf + tile-seam show-through), shallow tint from seabed material. Phase C tier 1 (MSFS-class, all normals/albedo in `shade_ocean` — ADR-0002 intact): shoreline + breaker foam bands from the signed field, wave shoaling (amplitude/λ shrink with depth) + refraction (align to −∇h); tier 2 later = advected breaker wavefronts (Sea-of-Thieves-class) | verify | M–L | user picked A+B+C1 2026-07-20; landed same day, 3 generator rounds (`GENERATOR_VERSION` 17): beach berm +4 m, thresholded islet clumps, shore band 60→9 m, **offshore shallow clearance** (relief may not park seabed in (−14 m, 0) away from the macro foreshore — the structural island-halo kill; deepening the awash fold alone did nothing, the halo was legal non-breaching tops), coverage AA band now uses the MEASURED local slope from the gradient taps (fixed-slope bands painted flat shoals as 40 % land at coarse footprints); sand strand + wet band in `eval_material_stack`; veg cleared ≤ +4 m (`VEG_BEACH_CLEAR_M`); turquoise shallows; shoaling + refracted breaker/swash foam in `shade_ocean`. Brackets clean (20 Mm/4 Mm/1.8 Mm/500 km); probes: 0 m wander × 8, 0 % breach, land 35.3 %, runway dry. **user:** beach walk + low flight over coast/archipelago | ADR-0006 · terrain_macro.md · water.wgsl |
| BL-5 | Limb dash streaks — **root-caused and fixed 2026-07-20** after two disproven premises (not mesh-silhouette pokes: flatten-all-terrain left them identical; not lagoon geography: a whole-planet fragmentation scan found zero lagoon fields). Magenta-debug traced them to the ocean branch's fixed occlusion thresholds letting the coarse mesh's broken sliver raster of beyond-horizon coasts punch through the mask coverage at high anisotropy. Fixed: coast atlas carries a mip chain sampled at an analytic footprint LOD, coverage band scales with footprint, and occlusion thresholds scale with footprint (geometry only overrides the mask when resolvable). Verified at 20 Mm: dense dash bands → faint sparse fringe; mainland coasts clean. Plus a deep-tau aerial-veil extension (physically-motivated limb haze) | done | — | — | INC-0003 · body_sky.wgsl · proc_impostor.rs |
| BL-6 | ~~Coastal lagoon-swarm geography~~ **premise corrected, rescoped, landed 2026-07-20**: fragmentation scan found NO lagoon fields (the "lagoon" impressions were rendering artifacts, since fixed) — the real problem was bland uniform coasts. Landed: coast-character refinement in `continentalness` (`GENERATOR_VERSION` 11) — a ruggedness-modulated fractal domain warp (15→85 km displacement) + rugged-coast islet sprinkle, cost-gated to the coastal band; depositional regions keep sweeping beach arcs, erosional regions get headlands/coves/rias/archipelagos. world_map-verified (regional variety, land fraction held 39.3%, runway site dry, spaceport coast calm); coastline_lod probe: waterline still moves 0 m across LODs | verify | M | **user:** live-eye pass over varied coasts (beach + rugged) at flight altitudes | procedural.rs `continentalness` · world_map.rs |

## Track 3 — Meta / infrastructure

| ID | Item | Status | Est | Deps | Refs |
|----|------|--------|-----|------|------|
| BL-2 | Adopt the steering harness: this backlog + `steer` skill + ADR log + incident log + docs README | done | — | — | ADR-0001 · 2026-07-18 |
| BL-3 | Polar orbit scenario (`just game polar` / start-screen + destruction picker) — same 200 km parking altitude as `orbit`, inclination 90° | verify | S | — | **user:** `just game polar`, confirm map/navball show polar path + ground track over poles | spawn.rs · debug_orbits.rs · boot.md |
| BL-17 | Codify headless screenshot + controlled A/B diagnosis as the required workflow for graphical fixes and visual iteration | done | XS | — | CLAUDE.md · documentation verified 2026-07-21 |
| BL-21 | Expand the CLAUDE visual-testing rule into an operational playbook: tool-selection boundary, controlled-axis procedure, artifact inspection order, failure caveats, and non-use cases | done | XS | — | CLAUDE.md · visual_testing.md · documentation verified 2026-07-21 |
| BL-22 | **Clean generated-artifact layout:** keep `tools/screenshots/` to the latest canonical preset views; route comparison/iteration captures through `tools/agent_scratch/`; route JSONL runtime diagnostics through `tools/diagnostics/`; prune superseded screenshot iterations | done | S | — | 164 stale direct captures pruned; 39 comparison artifacts moved; game + comparison example checks and clippy passed 2026-07-21 · tooling.md · visual_testing.md |

## Decisions pending

Forks that gate queued work. Resolving one = an ADR + flipping this row
(steer Mode 2).

| Fork | Gates | Where the options live |
|------|-------|------------------------|
| N-craft `Simulation` API shape (multi-craft records vs keyed map) | CL-E2 | clean §2.2 — decision needed, implementation deliberately deferred |
| Q2 — tonemapper base (AgX vs PBR Neutral) | C1 | gfx §7 |
| Q5 — VegLayer driver unification shape | VEG-R | gfx §7 |
| Q6 — Slice-6 distant-body path | W17 | gfx §7 |
| Q7 — TAA / motion vectors under big_space | W13 whole-scene consumers (cloud resolve is independent) | gfx §7 · cloud §2 |
| Q8 — sky-view LUT: sun-only vs per-dominant-light | F7/F9 polish | gfx §3 F3 caveat |
