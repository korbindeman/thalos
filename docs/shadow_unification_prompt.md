# Work order: F6 → one shadow world ("persistent, constant, proper shadows at any range")

> **STATUS 2026-07-02 — IMPLEMENTED (compile + clippy clean; screenshot
> verification pending).** The architecture target below landed in one pass:
> (1) one rig — new `ShadowedStandardMaterial`
> (`body_render::craft::ShadowReceiveExtension` +
> `assets/shaders/shadowed_standard.wgsl`) makes structures / runway / tarmac /
> plain craft parts / EVA receive `thalos::shadow`; runway top+skirt+posts cast;
> **stock Bevy CSM on the sun light is disabled** (`rendering/spawn.rs`).
> (2) stable CSM — `ShadowCascadeBlock` grew `sun_dir` + per-cascade texel size;
> `sun_shadow_factor_nrm` adds receiver normal-offset + slope-scaled bias
> (terrain/hull/structures use it; foliage keeps the legacy path at 2.5× base
> bias); base biases 0.25/1/4 m. Texel snapping was already
> floating-origin-relative + craft-centred — kept.
> (3) range — above 6 km AGL the rig now runs a **craft-local single-cascade
> mode** (hull self-shadow in orbit) instead of turning off.
> (4) W12 v1 — `body_render::horizon_sun_visibility` marches the `HeightSource`
> toward the sun at the craft and scales the sun `DirectionalLight` illuminance
> (`update_sun_light`), so a mountain shades the parked craft/base; the terrain
> keeps its own `terrain_self_shadow` atlas march.
> (5) the analytic `BodyTerrainShadow` craft proxy is **deleted**
> (shader + `ground_terrain.rs` driver + `THALOS_TERRAIN_CRAFT_SHADOW` env).
> **Round 2 (2026-07-02, after first screenshots — "shadows only at some
> angles/distances"):** three consistency fixes landed. (a) **Caster-pass
> scale-fade bug**: tree/impostor/rock clipmap fades reconstruct the craft
> anchor from `view.world_position`, which in the cascade caster pass is the
> *cascade* camera — casters collapsed to zero scale once the player camera
> moved away from the craft, so tree/rock shadows vanished with camera
> position. Fixed via `thalos::shadow::is_ortho_projection` — in an
> orthographic (caster/bake) pass the fades are bypassed (full-scale casting;
> the depth map is a silhouette union) and the tree impostor faces the
> **light axis** (`view.world_from_view[2]`) instead of the eye, sampling its
> octahedral atlas from the sun's angle. (b) **Altitude footprint scaling**:
> the ground cascade set no longer hard-cuts at 6 km camera AGL — extents,
> depth range, back distance, and metre bias all scale ∝ camera altitude above
> `SHADOW_REFERENCE_ALTITUDE_M` (1.5 km), clamped at 16× (far cascade 64 km),
> with the live camera `Projection`s updated in lockstep with the hand-built
> matrices; craft-local mode now starts at 50 km. (c) Impostor tiles already
> carried `SHADOW_CASTER_LAYER`, so distant (impostor-band) trees now actually
> cast.
>
> **Round 3 (2026-07-02, after "it's the same"):** the actual root cause of
> shadows-vanish-with-distance was the **bias model, not coverage**: per-cascade
> hand-authored metre biases (0.6/2.5/10 m, and round-2's footprint-scaled
> variants) plus the texel-proportional normal offset exceeded the *height of
> the casters* on the far/scaled cascades — a bias taller than a ~10 m tree
> makes its depth test always pass, erasing the shadow ("too large of an offset
> causes the depth test to erroneously pass", MS shadow-depth-maps guide). So
> coverage fixes changed nothing: the far cascades were covering but erasing.
> Fix: `shadow.wgsl` now derives bias + receiver offset **per cascade from the
> texel size with hard absolute caps** (`BIAS_MAX_M = 2.5`,
> `NORMAL_OFFSET_MAX_M = 1.5`, both well below tree height);
> `ShadowCascadeBlock.params.x` is now clip-per-metre (`1/(far−near)`), not a
> premultiplied bias. Acne stays controlled because the dominant receivers
> (terrain/grass) never render into the cascade maps at all, and
> caster-receivers are sub-pixel at the cascade scales where the caps bind.
> Verified end-to-end via `just preview` (tree shadow grounded, no acne).
>
> **Round 4 (2026-07-02, "great from certain angles, gone from others"):** the
> footprint was sized from camera altitude but CENTRED on the craft, so a
> camera orbited to the far side had its visible foreground outside the box.
> Footprint now = `(cam↔craft distance + 2×altitude) / far-cascade base`,
> capped ×32. The centre stays the this-frame craft anchor (camera transforms
> lag a frame in `SimStage::Sync` and crawl — don't centre on the camera).
> Tree rings reach 22 km from the craft, so this covers every extant caster
> from any vantage.
>
> **Round 5 (2026-07-02, "still inconsistent"):** screenshots showed a hard
> DIRECTIONAL boundary — shadows only on the down-sun side of the craft. Cause:
> the cascade eye sat a fixed `back = 150 m × footprint` up-sun with
> `near = 0.5 m`, so every ground point beyond `~back/cosθ` UP-SUN of the
> centre fell in front of the near plane — clipped out of the depth maps *and*
> the receiver test. The box is square in the light plane but its ground
> footprint runs `±half/tanθ` along the sun azimuth; at a low morning sun
> that's kilometres, of which only ~150 m up-sun was inside the depth range.
> Fix: per-cascade **up-sun ground slack** `half·cosθ/sinθ` added to the eye
> offset, `2×` added to the far plane (`SHADOW_MIN_SUN_SIN` clamps the
> divergence near the horizon, `SHADOW_SLACK_MAX_M` bounds the depth range).
>
> **Round 6 (2026-07-02, "much better; still disappear eventually, some fade"):**
> the fade is the far cascade's designed edge soft-fade — finally reachable.
> The remaining disappearance was coverage ending before the caster band: tree
> entities exist to 22 km from the craft, but with the camera near the base the
> footprint-scaled far cascade reached only 4–8 km. Fix: per-cascade minimum
> extents keyed to the vegetation rings (`CASCADE_MIN_HALF_M = [0, 6.5 km,
> 23.5 km]` — cascade 2 always spans the whole band), plus a
> `SHADOW_RELIEF_MARGIN_M = 4 km` depth margin (casters on hills above the
> centre's tangent plane were in front of the near plane at high sun — the
> vertical cousin of round 5). Far-cascade texel at baseline is now ~11.5 m, so
> far tree shadows are soft blobs by design.
>
> **Round 9 (2026-07-18, user: "overly pixelated, flicker while the sim runs;
> want MSFS-tier fidelity across all scales"):** three changes, preview- +
> headless-screenshot-verified (compile clean; interactive feel pending user).
> The guiding conclusion (from how MSFS/AAA structure this): **don't stretch
> one CSM across all scales** — MSFS splits "Shadow Maps" (a modest near-field
> object CSM) from "Terrain Shadows" (wide-area heightfield-derived shading),
> and stabilizes the rest with filtering + temporal accumulation.
> 1. **Filtered PCF (the pixelation):** `shadow.wgsl`'s `cascade_factor` now
>    uses a separable-tent kernel over a 4×4 `textureLoad` neighbourhood —
>    exactly equivalent to averaging 3×3 *hardware-bilinear* comparison taps —
>    so edges are smooth ~3-texel gradients instead of whole-texel staircases.
>    No comparison-sampler bindings needed in any material. Per-vertex grass
>    paths keep the cheap point 3×3 via new `sun_shadow_factor_vert`
>    (interpolation smooths them; grass is the heaviest vertex workload).
> 2. **Quantized sun stepping (the unpaused flicker):** round 8's body-fixed
>    snap stabilized *translation*, but the sun still moved relative to the
>    ground every simulated frame — every edge re-rasterized with fresh
>    sub-texel phase. The rig now HOLDS its sun direction fixed in the BODY
>    frame and steps only past `SUN_LOCK_STEP_RAD` (0.1°): between steps the
>    light co-rotates rigidly with ground + casters + snap grid, so cascade
>    content is frame-to-frame identical while the sim runs. Craft-local mode
>    keeps the continuous sun (no body frame to hold). Scene lighting stays
>    continuous — only the shadow rig quantizes (≤0.1° mismatch, invisible).
> 3. **Caster-band trim (the "all scales" split, step 1):** far coarse tree
>    rings (6–22 km, sub-pixel trees) no longer cast
>    (`TREE_SHADOW_CASTER_MAX_M = 6 km` in vegetation.rs), so
>    `CASCADE_MIN_HALF_M` shrank [0, 6.5 km, 23.5 km] → [0, 3 km, 6.5 km]:
>    mid/far cascade texels 3.2 m → 1.5 m and 11.5 m → 3.2 m. The far field
>    beyond the caster band is the heightfield horizon term's job (W12), per
>    the MSFS split.
>
> **Remaining — reordered 2026-07-22 by
> ADR-20260722T111848Z-shadows-three-tier-not-virtual-shadow-maps**, which
> formalises the MSFS split this file already anticipated into **three range
> regimes with three mechanisms**, and rejects virtual shadow maps (VSM serves
> only the mid-field, the one regime already adequate; the terrain — our
> dominant receiver — never renders into the cascade maps, so the caster set is
> small and a page table is pure overhead):
>
> 1. User interactive verification (crispness + unpaused stability) →
>    bias/strength tuning.
> 2. ~~**W18a contact shadows**~~ — **landed 2026-07-22** (check + clippy
>    clean, awaiting screenshot). `rendering::contact_shadow`: full-res
>    R16Float pass over `SceneDepthImage`, 12-step view-space march toward the
>    sun, thickness-tested occluders (depth is a heightfield, not a solid, so
>    without it a distant mountain shadows everything in front of it),
>    grazing-widened normal bias, 120 m fade-out. Full-res and unblurred
>    deliberately — SSAO's half-res + blur would erase the high-frequency edge
>    that is the whole point. The gate rides `ShadowCascadeBlock.gate.z`
>    (published by the rig, so every consumer inherits it) and
>    `thalos::shadow::contact_shadow_factor` is the one sampler. **Terrain is
>    the only receiver so far** — the remaining wiring (veg/rock/grass, hull,
>    and `ShadowedStandardMaterial` for runway/structures) is the follow-on,
>    and the runway matters most since parked-craft grounding is the point.
>    Diagnostics: `THALOS_CONTACT_SHADOW=off|show`, `just compare <preset>
>    shadow`.
> 3. **W12 v2** — per-fragment heightfield horizon term for all spine
>    materials (the far-field tier; cascades cannot reach here either).
> 4. **Hardware comparison samplers** — reclassified from "perf headroom if
>    needed" to a **prerequisite**: `cascade_factor` currently issues 16
>    `textureLoad`s and hand-weights a separable tent, where 9
>    `textureSampleCompare` taps reach equivalent filter quality with free
>    bilinear. This is what funds PCSS.
> 5. **W18c PCSS** contact-hardening penumbras + a cross-cascade blend (only
>    the outermost cascade fades today, so 0→1 and 1→2 hand off hard — a seam
>    that gets *more* visible once penumbra width varies). Consider
>    Vogel-disk + per-pixel rotation only if TAA lands; without temporal
>    accumulation rotated-disk noise shows raw.
> 6. Grass cast; W2 cloud shadows; F8's `shade_surface` port retiring the
>    `SHADOW_FLOOR` attenuation.
>
> **Cascade extents deliberately do not grow** — requests to "reach further"
> are answered by step 3, not by larger boxes, which is what forces every bias
> constant in `thalos::shadow` to be hard-capped today. Round-robin
> far-cascade updates remain available as perf headroom. Status also recorded
> in `docs/graphics_fidelity.md` §3 F6 + §4.2 W5/W6/W12/W18, and queued as
> backlog rows W18 / W12r / BL-37.

Self-contained prompt for the agent taking on the shadow sprint. Read
`docs/graphics_fidelity.md` §2.3 (one-world invariants #2 and #3), §3 (F6), and
§4.2 (W5/W6/W12/W18/W2) first — this file adds the current state, the user's
verdict, and the plan skeleton.

**User verdict (2026-07-02):** "shadows are quite buggy at the moment. I want
persistent constant proper shadows, at any range."

## Goal

Every solid object — terrain, trees, grass, rocks, craft, gear, buildings, pads,
tanks, runway — casts into and receives from **one** shadow world, **stable**
(no acne, no crawl/shimmer, no cascade popping), at **every range**: object
scale (a ship shadows the grass; a hangar shadows the ship) *and* planet scale
(a mountain shades the valley **and the ship parked in it**), robust through
time-of-day changes and warp.

## Current state (as of 2026-07-02)

Two shadow systems exist — the central F6 debt:

1. **Custom cascade rig** — `crates/runtime/game/src/rendering/sun_shadow.rs`:
   self-managed 3-cascade ortho depth maps (4096², `SHADOW_CASTER_LAYER = 8`,
   copy-node → `Depth32Float`), **craft-centred** framing (deliberate — dodges a
   frame-lagged crawl; keep it). Sampled via the `thalos::shadow` WGSL library
   (`crates/rendering/render/src/shading/shaders/shadow.wgsl`: `ShadowCascadeBlock`,
   `sun_shadow_factor`).
   - **Receive:** terrain (`body_terrain.wgsl`), trees, grass (per-vertex single
     tap), rocks, ground_patch (preview diorama), and the hull (F6b:
     `ship_part.wgsl` samples the rig, attenuating to a stylized
     `CRAFT_SHADOW_FLOOR` — proper BRDF-level shading waits on F8).
   - **Cast:** trees, rocks, craft + EVA (F6a, via `propagate_view_render_layers`
     adding `SHADOW_CASTER_LAYER`), structures (building/pad/tank stamped).
   - **Missing:** structures **receive** (StandardMaterial does not sample the
     rig), **runway cast**, grass cast.
2. **Bevy stock CSM** on the sun `DirectionalLight` still shadows the
   StandardMaterial universe (hull/gear/structures/runway receive *Bevy*
   shadows). Gotchas already learned the hard way (see memories):
   - The light must share `SHIP_LAYER` or craft casters are skipped
     (`rendering/spawn.rs`; memory `craft-shadow-caster-layer`).
   - **Every** game `DirectionalLight` must pin
     `crate::rendering::SHADOW_CASCADE_COUNT` (= 2) — Bevy 0.19 shares one
     thread-queue across directional lights; mismatched cascade counts = OOB
     panic (memory `directional-light-cascade-count-invariant`).

Additionally the terrain shader carries an **analytic craft-shadow proxy**
(`BodyTerrainShadow` capsule/quad casters ray-tested in `body_terrain.wgsl`,
penumbra `SHADOW_PENUMBRA_PER_M`) that predates the rig — a second definition of
the craft's shadow and a likely divergence source.

## Step 0 — bug inventory (do this before designing)

"Buggy" is unspecified. You cannot run the game — ask the user for
`just game runway` screenshots at morning/noon/dusk, plus: craft shadow on the
ground, ground under trees, building shadows, shadows at distance, and
during/after warp. Classify each defect: acne, peter-panning, crawl/shimmer
while the camera orbits, cascade-boundary popping, missing caster/receiver
pairs, staleness after warp (check how often the rig re-fits — the
reflection-probe arc in memory `f3-sky-view-lut` shows exactly how warp breaks
real-time-cadenced lighting state). Diagnose per the CLAUDE.md bug loop —
hypothesis set, falsifiable tests — before patching anything.

## Architecture target (proposal — verify against code; announce changes per CLAUDE.md)

1. **One rig.** Make the StandardMaterial path sample `thalos::shadow`
   (structures receive — the F6 remainder), add the runway as a caster, then
   **disable Bevy stock CSM on the sun light entirely** so exactly one shadow
   world exists. (F8's hull port later collapses the material split itself.)
2. **Stable CSM (W6).** Bounding-sphere cascade fit + **texel snapping computed
   in floating-origin-relative coordinates** — never planet-centric (at Mm
   radius, f32 light-space ULPs are texel-sized). Slope-scaled normal-offset
   bias per cascade (kills low-sun acne). Keep craft-centred framing.
   References: MJP / Valient / Tardif shadow-map articles.
3. **Range strategy.** Cascades cover the near field only (evaluate 3→4
   cascades / span vs. the 3×4096² budget). "Any range" beyond the last cascade
   is delivered by (4), not by stretching cascades to the horizon.
4. **Planet scale (W12).** A terrain **horizon-angle sun-visibility** term
   sampled by *all* surfaces — terrain and every object standing on it — so a
   mountain shades the valley and the parked ship at unlimited range. Raymarch
   the height atlas toward the sun (max-mip accelerated) or precompute per-tile
   horizon maps; body-local math, big_space-stable, no bake to disk (the
   no-bake invariant). This is the actual answer to "shadows at any range".
5. **Then retire the analytic terrain craft-shadow proxy** once craft cast into
   the rig on every receiver — one definition of the craft's shadow.
6. **Not optional polish** (reclassified 2026-07-22,
   ADR-20260722T111848Z-shadows-three-tier-not-virtual-shadow-maps): **W18a
   screen-space contact shadows** are the contact tier — the regime cascade 0
   cannot serve at ~0.2 m/texel — and land before PCSS, which in turn waits on
   the hardware-comparison-sampler refactor that pays for it. Cloud shadows
   (W2/CLOUD-5, integrating the canonical `CloudWeatherField`/density into
   `CloudSunTransmittance`) stay after those. Note W11's aerial-perspective
   froxel volume (ADR-20260722T111847Z) makes **volumetric shafts** nearly free
   — sampling `thalos::shadow` per froxel — so it is worth landing before
   further shadow-filter work.

## Constraints & tooling

- Mechanism → `thalos_body_render`, sim-reading drivers → `thalos_game`
  (graphics doc §5 crate rule). No material-local shadow math — everything
  imports `thalos::shadow`.
- Metal 16-vertex-buffer cap + `AsBindGroup` forcing vertex visibility on every
  `#[uniform]` — pack new per-frame state into existing blocks
  (`BodyTerrainExtras` precedent).
- WGSL: consult `.claude/skills/wgsl-bevy`; `naga` CLI is installed — validate
  standalone shaders; embedded/naga_oil shaders only compile in a real
  `just game` run (`cargo check` does not validate WGSL).
- `just preview` renders headless shadow dioramas (trimmed rig copy) — good for
  caster/receiver/bias iteration without the user; whole-scene verification
  still needs user screenshots.
- Every spawn starts paused; the runway scenario seats a morning epoch; the rig
  must stay correct through warp (add a sim-time-aware re-fit if it has a
  real-time cadence — same defect class as the reflection probe had).

## Verification (one-world checks, doc §6)

Ship shadows the grass; hangar shadows the ship; tree shadows the hull;
mountain shades the parked ship (W12); no acne at low morning sun; no crawl
while the camera orbits; no popping across cascade seams; shadows stay correct
through warp and settle instantly after.

## Pointers

`crates/runtime/game/src/rendering/sun_shadow.rs` · `crates/rendering/render/src/shading/shaders/shadow.wgsl`
· hull material `crates/rendering/render/src/craft/` · `crates/runtime/game/src/runway.rs`,
`structures.rs` · doc rows W5/W6/W12/W18/W2 in `docs/graphics_fidelity.md` §4.2
· memories: `craft-shadow-caster-layer`, `directional-light-cascade-count-invariant`,
`grass-receives-sun-shadows`, `f5-screen-space-ao` (the AO field is the ambient
complement of this work), `f3-sky-view-lut` (warp-staleness lesson).
