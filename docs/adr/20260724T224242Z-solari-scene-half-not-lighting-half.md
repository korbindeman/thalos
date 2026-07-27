# ADR-20260724T224242Z-solari-scene-half-not-lighting-half: Thalos adopts Solari's raytracing *scene*, never its raytraced *lighting*

- **Status:** Accepted
- **Date:** 2026-07-24

## Context

[ADR-20260723T142945Z](20260723T142945Z-neural-terrain-standard-renderer-keystone.md)
rejected a Solari-first renderer and left "Solari adoption" as a fork gated on
measurement (`ntr §7`). Two of the three reasons it gave have since expired:

- **Terrain now has meshes.** The rejection partly rested on terrain being a
  GPU-procedural indirect draw with no `Mesh3d` — the state
  [atmosphere.md](../rendering/atmosphere.md) *Real-time GI / ray tracing*
  still described. Since NTR-X1/X2a, Mira and the diffusion Thalos render as
  ordinary meshes through `tiles::`. The BLAS door the "flawless mirror finish"
  tier explicitly needed is open.
- **A concrete deficiency now wants exactly this mechanism.** NTR-X6: across the
  whole `massif-valley` frame `shadow_f ≈ 1` — the cascades never reach terrain
  at showcase distances, so gully and valley shadow have no mechanism at all.

The third reason — BLAS build/refit cost against streamed tiles — remains
unmeasured, and is the only real gate left.

The user's stated targets for RT are **terrain shadows** and **mirror-finish
stainless ships**. Neither is a request for a new lighting model.

### What the 0.19 crate actually is

Read from `bevy_solari-0.19.0` source, not from its README:

| Fact | Source | Consequence for Thalos |
|---|---|---|
| Scene extraction queries `MeshMaterial3d<StandardMaterial>` only | `scene/extract.rs` | `TileTerrainMaterial`, `ShipPartMaterial`, `ShadowedStandardMaterial` are all `ExtendedMaterial` — **every Thalos surface is invisible to the RT scene as-is** |
| `SolariLightingPlugin` inserts `DefaultOpaqueRendererMethod::deferred()` app-wide | `realtime/mod.rs` | Custom BRDFs (the Hapke branch in `tile_terrain.wgsl`) either go deferred and lose the custom shading, or stay forward and receive no Solari lighting |
| **No sky / environment lighting.** Lights are directional + emissive triangles; rays that miss contribute nothing | `scene/binder.rs` (no environment sampling exists) | Atmospheric skylight and planetshine — the dominant surface ambient, and what NTR-X5 just calibrated — have no home in Solari 0.19 |
| `SolariLighting` requires `Hdr`, `DeferredPrepass`, `DepthPrepass`, `MotionVectorPrepass`, `Msaa::Off`, camera `STORAGE_BINDING` | `realtime/mod.rs` | A camera-stack rewrite; survivable, but only worth paying for the lighting half |
| BLAS keyed by `AssetId<Mesh>`, rebuilt on asset add/modify, over **every** compatible mesh asset app-wide | `scene/blas.rs` | Grass/rock/tree/impostor meshes must opt out via `Mesh::enable_raytracing = false` or we pay for BVHs nothing traces |
| Mesh gate is exact: attributes `[POSITION, NORMAL, UV_0, TANGENT]` in that order + `Indices::U32` | `scene/blas.rs:173` | Tile meshes ([tiles/mod.rs](../../crates/rendering/render/src/tiles/mod.rs)) write no TANGENT — they would be **silently skipped**, and "RT didn't help" would be the wrong conclusion drawn |
| TLAS recreated and rebuilt from scratch every frame at instance count | `scene/binder.rs` | ~1500 resident tiles + craft parts, per frame |
| BLAS compaction budgeted at 400k vertices/frame | `scene/blas.rs` | ~7 M-vertex cold fill ≈ 17+ frames of backlog |

Read together: **Solari's lighting half is not an addition to our renderer, it
is a replacement for our lighting universe — and a less capable one for planets
specifically, because it has no sky.** Adopting it on Thalos would trade a
physically-calibrated atmospheric ambient for black-sky sun-plus-bounce. That is
a regression on the showcase frame, not an improvement, and it would introduce a
*third* lighting universe on top of the keystone's existing two-universe debt.

Its scene half carries none of that: BLAS/TLAS construction plus scene bindings,
with no opinion about how anything is shaded.

## Decision

**Take `RaytracingScenePlugin`. Never add `SolariLightingPlugin`.**

Four parts.

**1. RT is a visibility service inside our shading, not a lighting system.** The
raytracing scene supplies ray queries; `thalos::lighting` / `thalos::shadow`
remain the one shading authority. Hapke, `BodySky`, the volumetric composites,
and the NTR-X5 exposure parity are untouched by anything in this ADR. This
preserves the shared-shader-library rule and the one-world principle rather than
forking a second answer to "how is this surface lit".

**2. Surfaces enter the RT scene through proxy entities, not by giving up their
materials.** `RaytracingMesh3d` is a separate component from `Mesh3d`, and
Solari's extraction wants `RaytracingMesh3d` + `MeshMaterial3d<StandardMaterial>`
+ transform. An entity holding only those three — **no `Mesh3d`** — is RT-only
and never rasterizes. So each tile and craft part gets a lightweight proxy
sharing the *same mesh handle* (one BLAS, no duplicated geometry) carrying a
plain `StandardMaterial` that approximates its albedo for ray hits. The visible
entity keeps its `ExtendedMaterial`. This is the seam that makes the scene half
usable at all given the `StandardMaterial`-only extraction.

**3. Two consumers, one enabler, in this order.** The enabler (NTR-RT1) is a
measurement gate, not a feature: BLAS/TLAS cost under our real streaming rate on
Mira's descent. If that fails its budget, the consumers die there and we have
spent one measurement instead of a renderer. The consumers are RT sun visibility
into a screen-space mask that `sun_shadow_factor` reads (NTR-RT2, closing
NTR-X6 — shadow rays need only hit/no-hit, so the material limitation does not
bite) and RT reflection rays for stainless hulls behind the existing reflection
source interface (NTR-RT3 — the "dream" tier of atmosphere.md, whose stated
prerequisite was terrain in a BLAS).

**4. Deferred behind NTR-X4.** The showcase patch is *currently* judging the
massif's look; changing the shadow mechanism mid-texturing confounds exactly the
comparison that row exists to make. Land the showcase patch under the shadow
situation it was scoped against, then change the mechanism and re-judge with
matched before/after evidence.

RT stays hardware-gated and therefore permanently additive: the raster path
(cascades, SSR + reflection probe) remains the baseline every machine gets, per
`ntr §2`'s "Solari is evaluated, never assumed". This ADR narrows *what would be
adopted*; it does not adopt it.

## Alternatives

- **Full Solari (scene + lighting).** Rejected on the sky: Solari 0.19 has no
  environment lighting, so every ray that misses returns nothing. On an
  atmospheric body that deletes the dominant ambient term. It also forces the
  opaque path deferred, which costs us the Hapke BRDF, and extracts only
  `StandardMaterial`, which excludes every surface we ship.
- **Solari on airless bodies only.** Genuinely tempting and recorded here so it
  is not re-derived: on Mira there *is* no sky, so the black-miss is physically
  correct, and sun plus crater-wall interreflection is the entire light
  transport. Rejected anyway, because the price is the Hapke BRDF — the thing
  that makes airless bodies read as airless (NTR-X1). Trading the correct
  regolith BRDF for correct bounce is not obviously a win, and it would fork the
  shading model per body class.
- **Evaluate in the probe repo.** Rejected as the vehicle. The probe has **no
  crafts at all**, so mirror steel is unmeasurable there by construction, and the
  showcase massif with its texturing now lives in this repo (NTR-X4) — porting
  realistic texturing and an atmosphere into the probe to judge a lighting
  feature is rebuilding the game inside the probe. The probe's unique value was
  isolating an unverified stack; that stack is now verified and extracted. Its
  one remaining advantage, streaming-rate BLAS measurement, the game reproduces
  on Mira with the same ported streaming code.
- **Wait for a Bevy release with environment lighting in Solari.** Not rejected
  — orthogonal. If upstream adds sky/environment sampling, part 1 of this ADR is
  worth reopening, and the proxy-entity seam (part 2) is what would make that
  cheap. Nothing here forecloses it.
- **Keep extending cascades / add a heightfield horizon term instead**
  (NTR-X6's other options). Not rejected by this ADR: the horizon term is
  view-distance-independent and runs on hardware without RT, so it stays the
  portable answer. RT sun visibility is the high-end tier above it, and NTR-RT2
  is explicitly measured against the cascade baseline, not assumed better.

## Consequences

- `Cargo.toml` gains `bevy_solari` only when NTR-RT1 starts, behind an off-by-
  default feature. That is a **second dev-renderer lane fingerprint** (the RT
  device features change wgpu device creation), so toggling it costs a full graph
  rebuild and it touches the capture host — budget for it, do not churn it.
- Tile meshes need TANGENT to be RT-eligible at all: ≈ +16 B/vertex on ~7 M
  resident vertices ≈ 110 MB VRAM. Measured in NTR-RT1, not assumed free.
- Scatter meshes (grass, rocks, trees, impostors) must set
  `Mesh::enable_raytracing = false` explicitly, or BLAS construction silently
  covers the whole scatter budget.
- [atmosphere.md](../rendering/atmosphere.md) *Real-time GI / ray tracing* and
  the mirror-finish tier are updated by this change: their premise ("the terrain
  has no mesh") is obsolete, and the BLAS source is tile meshes, not an extension
  of collider-patch trimesh extraction.
- `ntr §7`'s Solari fork narrows from "adopt or not" to "does the scene half fit
  the streaming budget", gated by NTR-RT1.
