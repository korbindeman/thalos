# Neural terrain × standard-path renderer (`ntr`)

**Status:** primary sprint (keystone) · **Started:** 2026-07-23
**Decisions:** [ADR-20260723T142945Z-neural-terrain-standard-renderer-keystone](../adr/20260723T142945Z-neural-terrain-standard-renderer-keystone.md),
[ADR-20260723T143155Z-planetary-model-pytorch-finetune](../adr/20260723T143155Z-planetary-model-pytorch-finetune.md)
**Cross-ref prefix:** `ntr §N`

## §1 Thesis

**Make Thalos look good.** High-fidelity neural terrain
([terrain-diffusion](https://xandergos.github.io/terrain-diffusion/)-style,
fine-tuned for planetary terrain) and a renderer on Bevy's **standard render
path** (`Mesh` + `StandardMaterial`/`ExtendedMaterial`, Bevy lighting/shadows,
Solari as a measured option), designed **in harmony, paired for each other**.
Terrain generation and rendering advance as one vertical slice: the tile
contract is co-designed with the producer that fills it.

Thalos (earth-like) is the first target because terrain-diffusion already
proves the earth-like case — MIT-licensed, pretrained 30 m/90 m weights
released, trained on real ETOPO elevation + WorldClim climate, with
hierarchical, lazy, seed-deterministic O(1) random access. Airless (Mira) is
the second family; the end state is one unified model architecture for all
bodies (companion ADR).

This inverts the F-series direction of `gfx §3`: the one-world goal is
unchanged, but it is reached by pulling terrain — and every opaque surface —
onto **Bevy's** lighting universe (where crafts already live via
`ShadowedStandardMaterial`) rather than pulling crafts onto the custom spine.

## §2 Fixed decisions

Carried from the probe design (user, 2026-07-21) and the keystone ADR:

- **Standalone probe first.** A new, isolated Bevy application — not a Thalos
  branch — advances generation, meshing, LOD, lighting, and streaming in
  alternating vertical slices. No unverified Thalos stack beneath it; its M5
  milestone produces the measured extraction plan back into this repo.
- **`big_space` from the beginning**, Earth-scale radius (6,371,000 m — harder
  than Thalos's 3,186 km). Closed cube-sphere, six faces, cross-face
  addressing, explicit corner/seam tests. Meter-based heights and body radius
  preserved through the pipeline.
- **Ordinary Bevy meshes through the standard material and render paths.**
  Terrain-specific code may manage meshes, assets, visibility, and LOD; it does
  not replace the main opaque PBR pipeline. A material extension is acceptable
  only when it preserves Bevy's PBR and shadow passes. Terrain-only vertex
  displacement in a custom shader is not the default (shadow depth, normals,
  bounds, and collision diverge from the visible surface — exactly udlod's
  disease).
- **Neural generation behind a tile-provider contract** so analytic, live
  neural, and prebaked tiles are interchangeable. The renderer never knows
  which produced a tile. This is ADR-20260722T105147Z part 1
  (tile-as-surface-authority) in its first clean implementation.
- **Collision is a consumer of the same authoritative height tiles**, even
  while physics is deferred during the first visual milestones. (BL-34's
  lesson: a metre-scale CPU/GPU height disagreement buries a 1.7 m eye.)
- **Offline packages remain the shipping mode**
  (ADR-20260720T211046Z stands). Live neural synthesis is a benchmarked
  research mode; the likely endpoint is hybrid — planetary/regional bands
  prebaked, selected local residuals generated or procedural.
- **Scale consistency is an invariant** (ADR-20260722T105147Z part 3): every
  band is a conditional refinement of its parent, never additive content.
  Refinement adds bandwidth, not a new skyline.
- **Solari is evaluated, never assumed.** Baseline is standard PBR + CSM
  (+ 0.19 contact shadows / SSR where they fit). RT-hardware gating and BLAS
  rebuild cost on streamed tiles are open measurements, not settled costs.
  **Narrowed 2026-07-24** (ADR-20260724T224242Z): what is on the table is the
  raytracing *scene* used as a visibility service inside our own shading —
  never Solari's raytraced *lighting*, which would be a third lighting universe
  with no sky. RT stays permanently additive: the raster path is what every
  machine gets.

## §3 The probe (renderer workstream)

The probe repo owns its own detailed checklist; the milestone gates it must
report back are:

| Gate | Proves | Key acceptance |
|---|---|---|
| **M0** Earth-scale lit sphere | big_space precision + standard PBR at radius | orbit→1 m descent, no vertex jitter on a 1 m lit object; shadow casting on a raised feature; CPU/GPU/memory baseline captured |
| **M1** One generated patch end to end | the tile contract | analytic tile → displaced mesh → standard PBR; then the **same renderer code** consumes one neural tile |
| **M2** Surface-local LOD | streaming + reuse | quadtree selection, async tiles with cancellation, entity/mesh reuse, ground camera path at target frame time, no cracks/pops |
| **M3** Spherical continuity | cube-sphere seams | all edges/corners at low altitude, cross-face halos for analytic **and** neural providers, quantified seam height/normal error |
| **M4** Multiscale terrain | the cascade | coarse planetary band + ≥1 parent-conditioned residual band; macro features recognizable orbit→descent; live vs prebaked comparison |
| **M5** Performance decision | the extraction plan | scripted camera paths; frame-time percentiles, neural latency, queue depth, cache hit rate, upload bytes, VRAM, disk; which bands run live vs baked; whether GPU meshing / a PBR extension / virtualized geometry is justified **by measurement**; the Thalos extraction plan |

Initial budgets (experiment gates, not promises): 60 fps ground-level on the
dev GPU (4070 Ti); <1 ms average main-thread terrain work; 33² or 65²-vertex
patches; no visible blank tiles on the scripted path; no visible cracks
(shared-edge error recorded numerically); no jitter at 1 m; bounded
cancellable neural queue prioritized by screen error; deterministic prebaked
output for fixed (model, seed, tile key).

Non-goals: the final Thalos terrain crate inside the probe; accurate
geology/hydrology simulation; centimeter geometry globally; replacing Bevy PBR
before the standard path is measured; requiring live diffusion where prebaking
serves; multiplayer terrain authority.

## §4 Generation workstream (fine-tune)

Runs alongside the probe from M1 (the probe must meet neural output early, not
design around procedural assumptions).

1. **Reproduce upstream.** Pinned Python env (PyTorch/diffusers,
   infinite-tensor fork); run the released 90 m and 30 m models locally;
   verify seed-determinism and windowed-fusion behavior; record content hashes.
1b. **Direct reference, not imitation** (user direction 2026-07-23). The
   released pipeline is a *working earth-like implementation of the whole
   cascade* — coarse model (23 km/px elev+climate) → detail model (90 m) —
   so it is consumed **directly** as the terrain source and held as the
   **benchmark**: export its coarse band + detail regions into the tile
   contract (probe first, game after extraction); compute reference metrics
   (hypsometry, slope distribution, radial spectrum slope — MIRA-L-gate
   style) from its flat output; require the spherical composition to match
   those metrics as cube-sphere addressing wraps around it. Probe-side
   analytic octaves demote to gap-filler outside baked coverage. Only after
   the pipeline is proven against this reference does the recipe transfer to
   new datasets (Thalos-authored conditioning, airless) — the §4.3/companion
   -ADR path, unchanged in destination but now benchmark-anchored.
2. **Spherical adaptation.** Cube-sphere addressing `(face, level, x, y)` +
   halo; conditioning includes unit direction, scale level, body seed, physical
   sample spacing (so faces aren't six unrelated planar worlds); cross-face
   neighbors through one canonical adjacency transform; seams solved at
   generation time (shared context / deterministic reconciliation), not hidden
   by mesh skirts.
3. **Thalos conditioning.** The fine-tune must accept authored macro control:
   the lore constraints (35 % land, old low-relief continents,
   `lore/solar_system.md` §II), the authored continent layout, and eventually
   per-body climate targets. terrain-diffusion's coarse-map → detail split is
   the natural seam: Thalos authors the coarse map, the model details it.
   **Tectonic structure landed 2026-08-03 (NTR-X2f):** deterministic Euler-pole
   motion on an irregular grown cube-sphere plate graph labels convergent,
   divergent, and transform margins before `ConditioningChart` is exported.
   Weighted growth bias, preferred bearing, crustal noise, and boundary-born
   microplates replace the rejected nearest-cell/Voronoi geometry. Mountain
   belts are therefore an input to generation, not displacement pasted onto its
   output. Surface expression is intentionally less continuous than the process
   label: variable-width regional preservation yields separated old massifs,
   while subdued inherited terranes prevent empty plate interiors. This keeps a
   coherent contact for diffusion to elaborate without drawing it as a uniform
   ridge stroke in the analytic fallback.
   **Whole-contact response landed 2026-08-07:** convergent contacts now export
   asymmetric hinterland and foreland provinces around the preserved core, and
   divergent ocean contacts export a broad ridge swell. This gives the model a
   plateau/lowland/bathymetric hierarchy to elaborate instead of asking it to
   infer that hierarchy from narrow peak texture alone
   (ADR-20260807T175108Z).
   Generator 33 closes the remaining macro-organization defect: whole relief
   bands are footprint-gated, inherited terranes stay below active-range
   strength, and a low continuous collision spine connects regionally preserved
   massifs (ADR-20260807T193028Z). The checked-in learned chart is now explicitly
   stamped as conditioning generator 28, and maps/runtime warn on the 28→33
   mismatch. `target/thalos_macro` has been regenerated at 33; the external
   terrain-diffusion producer must consume it before the neural backing can
   claim the new relief.
4. **Climate channels → landcover (design-level).** Upstream co-generates
   WorldClim-style climate. Evaluate learned climate as the conditioning input
   to landcover/biomes, making the hand-built macro moisture/climate pipeline
   (TM-P1/P2/P3, `terrain_macro.md`) a *consumer* of generated conditioning
   rather than a parallel authored system.
5. **Package emission.** Fine-tuned output bakes into the terrain-package
   schema (MIRA-0 lineage); producer identity records base-model version +
   fine-tune dataset/config. Runtime-ready learned payloads too large for a
   normal Git blob live under `assets/terrain_packages/` through Git LFS, beside
   their tracked provenance/hash sidecars; `just terrain-assets` is the explicit
   deferred-checkout/repair path. Training runs, checkpoints, and optimizer
   state stay ignored because the runtime does not consume them. Q10 (pixel vs
   latent storage) is still the open schema fork and now gates this workstream
   too.
6. **Hydrology is downstream of the completed neural heightfield.** The final
   drainage solve consumes emitted neural DEM data, never a parallel analytic
   approximation or runtime point queries. The full-planet continuous band
   establishes watershed topology; any finer authored neural windows re-solve
   locally with conserved boundary inflow/outflow so detailed mountains and
   valleys move their tributaries without changing the continental water
   budget. Geometric catchment and climate-weighted discharge remain separate
   package channels. Their provenance records the exact hashes of every height
   and climate band consumed, so a terrain rebake cannot silently retain stale
   rivers. The current 2 km `SurfaceQuery` bake is a preview/validation adapter,
   not a shipping terrain authority. The complete work order, package contract,
   open routing forks, and acceptance gates live in
   [`hydro §1–§8`](../world/drainage.md).

**Candidate mechanism — one field kernel, two backends (CubeCL).** The model is
not the whole terrain. Downstream of it sits a *field cascade* every band
shares: parent upsample + conditioning, the sub-model-scale analytic detail
octaves, normal/slope derivation, material/albedo derivation, and structure
conditioning (flatten pads as a tile input, §6.5). That cascade has three
consumers which, written conventionally, become three implementations —

- the offline bakery (package emission, §4.5): CPU/CUDA, f64 available;
- runtime GPU tile synthesis: wgpu → WGSL/SPIR-V, f32 with translated origin;
- the runtime CPU height authority: colliders, spawn search, EVA, HUD (§6.3).

— and udlod's abandoned GPU tile production died on exactly that
(`terrain_lod_optimization.md` *What did not land, and why*): moving synthesis
to a compute shader creates a **second height authority** that drifts from the
CPU one the colliders read. [CubeCL](https://github.com/tracel-ai/cubecl)
writes such kernels once in Rust (`#[cube]`) and compiles them to
CUDA/ROCm/SPIR-V/WGSL/Metal plus a CPU runtime, so the *duplication* half of
that blocker stops existing by construction rather than by differential test.

What it does **not** solve: wgpu exposes no f64, so the runtime compilation is
f32-with-translated-origin either way; and one source is not bit-identical
across backends (FMA contraction, transcendentals, vectorization), so parity
stays a *tolerance* gate. It is also **not** the render path — standard PBR +
WGSL stands (§2), and CubeCL's own device/allocator makes render-graph interop
a cost with no benefit — and **not** the model runtime: training is PyTorch
(ADR-20260723T143155Z), with Burn (CubeCL underneath, already in-tree via
`tools/terrain_train`) the Rust inference candidate only if on-device decode
wins Q10.

**Why it is an early decision.** The cost is paid the moment the cascade is
written a second time. The probe's analytic provider is single-implementation
CPU through M2; the fork opens at M4 (parent-conditioned residual bands) and
closes at M5/extraction, when collision joins as a consumer. Evaluating after
that is a rewrite — which is precisely how udlod reached "needs a decision and
a GPU, not something to land blind."

Burn/MIRA-1 finishes its L2 gate evidence, then pauses (keystone ADR §4).
Nothing downstream of L2 starts.

## §5 What survives, freezes, and dies in Thalos

**Survives and continues** (couples to the standard path via the W11-style
`color·T + L` per-fragment seam):
- `BodySky` atmosphere (ADR-20260721T185221Z), volumetric clouds (in-flight
  CLOUD-4/6/BL-33 wip continues), analytic ocean, celestial sky, plumes.
- The capture/verification harness (CAP-*, `just screenshot`/`compare`) — it is
  also the probe's verification model.
- The tile cache concepts (memory>disk>synthesis, namespace-as-contract),
  package schema + validator, `big_space`.
- The `clean` track, as background — largely renderer-agnostic.

**Frozen** (no new investment; annotated in the backlog, not deleted):
- Spine-port and udlod-coupled gfx work: F4r, F5r, F7, F8a, F8b, F9, W12r,
  TM1; GF-CAL shrinks to calibrating survivors (clouds/atmosphere/ocean/
  exposure).
- udlod-side terrain work: the old BL-34 (tile-native seam *implementation
  against udlod* — the contract itself lives on in the probe), BL-35
  (Mira `ExtendedMaterial` prototype — superseded by the probe), BL-36's
  udlod half (the detail-noise deletion happens in the new renderer, where
  unconditioned noise simply never gets added).
- MIRA-V* airless visual calibration (shading knowledge retained; producer-side
  fixes like the crater depth law are renderer-independent and keep their
  value).

**Dies at extraction** (end-of-life; defect-driven fixes only until then):
- `thalos_udlod`, `BodyTerrainMaterial` + the terrain WGSL stack, the custom
  queue-ordering machinery.
- Pending probe measurement: the custom shadow rig (`thalos::shadow`) and SSAO
  node, in favor of Bevy CSM/contact-shadows/GTAO; the `thalos::lighting`
  surface-shading spine (its physical content — sky-view LUT, moonlight,
  eclipse — migrates into Bevy-path IBL/light terms, not away).

## §6 Thalos integration sequencing (post-M5, provisional)

> **Status 2026-08-09: the tile renderer is the default ground on every body.**
> Steps 1–3 are in, and step 5's scatter half landed. `thalos_udlod` is absent
> from default application graphs and available only through the optional
> `legacy-udlod` feature plus `THALOS_TILE_RENDERER=0` for matched A/B evidence.
> The remaining composites and open `NTR-X*` rows are what stand between here
> and step 6.

Order of re-entry, to be finalized by the M5 extraction plan:

1. **Tile contract lands in Thalos** (the probe's `TerrainTileProvider` shape
   replaces the raster→point→raster seam; `SurfaceQuery` becomes derived —
   ADR-20260722T105147Z part 1).
2. **One body flips renderers** — likely Thalos itself, since the fine-tune
   targets it; Mira follows on the compatibility package.
3. **Height-authority parity** — colliders, EVA, camera floor, HUD read the
   same tiles the mesh displays (BL-34/35's recurrence test: 1.7 m eye at a
   grazing view renders correctly).
4. **Composites re-couple** — atmosphere/ocean/clouds bind to the standard
   depth/visibility; aerial perspective applies via the froxel seam.
5. **Structures/flatten/vegetation re-anchor** onto the new tiles (flatten as a
   tile-conditioning input, not a shader override; scatter reads the same
   tiles).
6. **udlod deletion** in the same change as the last consumer moves — the
   `clean` sprint's delete-on-contact rule.

## §7 Open decisions

| Fork | Gates | Notes |
|---|---|---|
| Bevy/`big_space`/Solari revisions to pin together for the probe | probe M0 | probe-repo decision; report back |
| Patch resolution + screen-space-error rule | probe M2 distance rule → **relief-aware rule shipped 2026-07-25** → **package error shipped 2026-08-02** → **129² submission-density topology shipped 2026-08-14, runtime gate open** | The target rule is **relief-aware screen-space geometric error**: package metadata may remove refinement below the distance cap, never add it (scale-consistency invariant). Ruggedness (relief ÷ arc) remains the fallback where package error is unavailable, with a 45 m source-detail floor. Resolution is now 129² at L−1 in place of four 65² tiles at L. The selector preserves the established density products (384 base / 1,152 rugged samples across split distance), so the bare factors are 3/9 rather than 6/18; sample spacing and maximum detail stay fixed while entity, asset, main-pass, and shadow-twin submissions target one quarter. See §7.2 |
| Collision at M2 or after multiscale | probe M2/M4 | leaning after-M4 per probe non-goals |
| Q10 — package storage: pixel heights vs latent + on-device decode | §4.5 package emission; MIRA-2/3 schema freeze | carried over from *Decisions pending* |
| Unified model architecture across body classes | after the earth-like fine-tune produces accepted Thalos terrain | companion ADR end state; resist architecture (not just stack) divergence meanwhile |
| Learned climate channels as landcover conditioning | §4.4 evaluation | could retire TM-P2r/TM-P3b-style authored climate growth |
| Q11 — single-source field kernel (CubeCL `#[cube]`, one Rust source compiled to GPU + CPU) vs hand-maintained WGSL/Rust pairs for the post-model cascade | opens at probe M4 (residual bands), binding at M5/extraction when collision joins as a height consumer | §4 *Candidate mechanism* |
| Solari adoption — **shape resolved 2026-07-24** by [ADR-20260724T224242Z](../adr/20260724T224242Z-solari-scene-half-not-lighting-half.md): the *scene* half (`RaytracingScenePlugin`) yes, the *lighting* half (`SolariLightingPlugin`) never — it forces the opaque path deferred, extracts only `StandardMaterial`, and has **no sky/environment lighting at all**, which deletes the dominant surface ambient on an atmospheric body. Surfaces enter the RT scene through `Mesh3d`-less proxy entities so they keep their `ExtendedMaterial`/Hapke. **Refined 2026-07-29 by [ADR-20260729T070229Z](../adr/20260729T070229Z-rt-tiles-need-a-separate-mesh-not-a-shared-handle.md):** Solari's mesh gate is an *equality* on the attribute sequence, so craft parts share their visible mesh handle but terrain tiles (COLOR + UV_1 for the NTR-X4 layer stack) can never be eligible and need a separate RT-only twin at 1.90× a tile's VRAM — which makes near-`ViewAnchor` scoping mandatory, and makes **NTR-RT3 the first consumer**, ahead of RT2 | **NTR-RT1** — BLAS/TLAS cost against our tile streaming rate, in-game on Mira (not the probe: it has no crafts, so mirror steel is unmeasurable there). Deferred behind NTR-X4 | Consumers gated on it: NTR-RT2 (RT sun visibility → `thalos::shadow`, closing NTR-X6) and NTR-RT3 (stainless-hull reflections, the atmosphere.md mirror tier whose BLAS prerequisite the tile renderer now satisfies) |

### §7.1 Package error refinement (2026-08-02)

The compatibility package now supplies its retained residual displacement
bounds to the selector. The bound is projected with the live camera's physical
viewport and vertical focal length; a split in the ruggedness-only band is
omitted at or below one pixel. This preserves the scale-consistency rule:
package metadata cannot cross the distance selector's base-detail floor and
cannot add refinement.

The bound is complete rather than optimistic. Retained predictor and
reconstruction errors compose down the package hierarchy; pruned residuals are
zero because they are absent from the runtime surface. Airless regolith adds
its signed amplitude envelope. A compact cube-sphere mask rejects package
authority wherever runtime-only craters overlap, while dynamic layers and
unsupported runtime detail retain the previous heuristic. Missing metadata is
therefore exactly the old selector, not a guessed error.

`THALOS_PACKAGE_SCREEN_ERROR=0` restores that fallback path explicitly for
matched cold-capture performance and fidelity comparisons.

### §7.2 Tile submission density (2026-08-14)

The 65² topology spent most of the frame outside measured simulation stages:
one mesh asset and entity per quadtree leaf, plus one terrain caster twin per
shadow cascade. A settled low-altitude session held 4,722 terrain tiles and
submitted 985 visible terrain caster twins; average frame time was 119.3 ms
while physics, sync, and camera together accounted for 4.94 ms.

The first structural cut keeps the quadtree and provider contract but changes
one leaf from 65² at level L to 129² at L−1. `TILE_RES`, `MIN_LEVEL`, the base
and rugged split distances, and `max_level_for` move as one invariant: ground
sample spacing is unchanged, four former footprints become one, and aggregate
vertices remain approximately flat. Per-tile mesh and cache payloads grow by
about four, so byte-denominated residency/cache budgets remain the authority;
tile-count budgets would be invalid.

Acceptance is runtime evidence, not the arithmetic alone: matched framing and
settings, no silhouette/seam regression, terrain resident and caster-twin
counts near one quarter, and a material frame-time improvement. If submission
counts fall without frame time moving, the next discriminator is GPU timing,
not another LOD constant.

The first exact-current-source `forest-stand` capture settled with 1,542
resident = desired tiles at full split scale and no visible seam or missing
terrain. Its historical same-preset receipt held 5,838 tiles, a 3.79× reduction,
and accounted mesh residency fell from 1,979 to 1,518 MiB. The historical image
is not a matched source revision, so this closes the topology and visual sanity
gates, not the interactive frame-time gate. The first run also rebuilt every
shape-invalid 65² cache entry; that migration cost must not be confused with
steady-state frame time.

The first interactive 129² run confirms the structural reduction but not a
proportional frame win. In one stationary 70-second plateau, resident terrain
was 1,386 tiles and the four shadow views saw 302 terrain caster twins, both
about 29–31% of the 65² baseline. Mean frame time moved only 119.3 → 101.3 ms
(8.4 → 9.9 fps). Physics, sync, and camera occupied about 2 ms of that settled
frame, so tile streaming is not the floor and another selector constant is not
the next experiment.

The next discriminator is a cold foliage × custom-shadows matrix at identical
framing. `THALOS_PERF_FOLIAGE` and `THALOS_SHADOW_CASCADES` provide the two
independent session-only axes; `frame_gauge` records both with the window and
other graphics settings. If neither removal accounts for the missing render
time, the result escalates to a GPU/profile trace rather than another geometry
guess.
