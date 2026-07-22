# ADR-20260722T111847Z-aerial-perspective-froxel-volume-regime-split: Aerial perspective is a froxel volume inside the atmosphere shell; the raymarch stays authoritative outside it

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

Thalos currently has **two unrelated aerial-perspective models**, which is a direct
violation of the one-world principle (`gfx §2.3`):

- `object_aerial_recession` (`shading/shaders/lighting.wgsl:782`) — a distance
  `smoothstep(1 km, 35 km)` lerp toward `SurfaceSky.sky_radiance`, capped at
  `OBJECT_AERIAL_MAX = 0.32` with a `OBJECT_AERIAL_BRIGHTEN_CAP = 1.5` luminance
  clamp. Its callers are exactly five: `grass`, `gpu_grass`, `rock`, `tree`,
  `tree_impostor`.
- The `BodySky` depth-clipped `integrate_atmosphere_multiscatter` raymarch, which
  is what terrain actually recedes through.

So a tree at 5 km and the hill it stands on fade toward air by two different
models, and craft/structures/ocean get a third answer (approximately none).

The lerp is structurally incapable of three of the strongest distance cues, and
this is not a tuning problem:

1. **No transmittance.** `mix(color, haze, t)` rather than `color·T + L`, so
   distant dark objects cannot lose contrast correctly and bright objects can only
   be tinted, never extinguished. `OBJECT_AERIAL_BRIGHTEN_CAP` exists solely
   because the analytic sky radiance outruns object brightness — the constant is a
   symptom of not being energy-consistent with the raymarch next to it.
2. **No sun direction.** No Mie phase term, so looking into a low sun produces no
   forward-scatter glare and looking away produces no bluer, higher-contrast
   distance. This is the dominant MSFS-class cue and it is entirely absent.
3. **No altitude dependence.** The same curve applies at sea level and in the
   15,000 ft `cruise` scenario, where thinning air should nearly retire it.

Its four magic constants are also coupled to exposure, so they need retuning every
time `GLOBAL_EXPOSURE_STOPS` moves (F2/GF-CAL).

The standard fix (Hillaire 2020, as shipped in UE's `SkyAtmosphere`) is an
**aerial-perspective froxel volume**: a view-space 3-D LUT of in-scattered
radiance + transmittance that every surface samples once. The decision this ADR
records is *not* "should we build one" — it is **what range it covers**, because
Thalos's view distance spans cockpit-scale to ~1e9 m and a fixed-range volume
(UE's is ~96 km) cannot express that.

## Decision

**The froxel volume owns the atmospheric near-field; the existing raymarch owns
everything else.** Concretely:

- **Volume.** A view-space 32×32×32 RGBA16F 3-D texture (RGB = in-scattered
  radiance, A = transmittance), built by a compute pass, slices distributed
  exponentially. Sized in froxels, so cost is **independent of render
  resolution**.
- **Range.** The far plane is `min(atmosphere exit along the view axis, ~128 km)`
  — the volume covers only the portion of the frustum inside the active body's
  atmosphere shell. Beyond it, `BodySky`'s `integrate_atmosphere_multiscatter`
  remains the sole authority (this is W11's own "keep the raymarch as the
  space/upper-atmosphere fallback" note, promoted to a binding constraint).
- **Kármán gate.** Above the Kármán line the volume is **disabled entirely** and
  the raymarch is the only aerial authority. There is no near-field object detail
  to veil from orbit, and an exponential distribution would otherwise spend ~30 of
  32 slices in vacuum.
- **One model.** The volume is integrated from the **same**
  `integrate_atmosphere_multiscatter` primitives as the sky. It is a
  reorganisation of the existing model into a lookup, never a second model.
- **Consumers.** Every opaque/foreground surface applies `color · T + L` from one
  sample: the five current `object_aerial_recession` callers plus terrain, hull,
  structures, and ocean. **`object_aerial_recession` and its four constants are
  deleted** in the same change.
- **N-body composition order.** The volume only ever describes the body containing
  the camera; a second body's atmosphere reaches the frame through `BodySky`
  alone. Composition is **BodySky (including other bodies' limbs) → opaque
  surfaces with froxel `T`/`L` applied → ocean → clouds**.
- **Slice pairing.** Ships **with F7's** single shared view-level
  scene+atmosphere bind group, not before it.

## Alternatives

- **Keep the analytic lerp and tune it.** Rejected: no amount of tuning adds
  transmittance, a phase function, or altitude response, and it leaves two
  atmosphere models in the renderer permanently. The constants are also
  exposure-coupled, so "tuned" is not a stable state.
- **One volume covering the full view range**, slices parameterised by optical
  depth rather than distance, so orbit and surface share a single path. Rejected:
  materially more complex, and it buys nothing — from orbit there is no near-field
  geometry for the volume to veil, which is exactly the case the raymarch already
  handles well. Revisit only if the Kármán handoff proves visible.
- **Per-material raymarch** (each surface integrates its own view ray). Rejected:
  cost scales with overdraw and shader count, and N independent implementations is
  how the current two-model split happened in the first place.
- **Temporal accumulation from the start** (UE jitters and reprojects its AP
  volume). Rejected for the first slice: it would make W11 depend on W13's
  whole-scene temporal foundation, which is `later`. At 32³ = 32k froxels the
  volume affords real per-froxel sample counts without reprojection. Revisit after
  W13.
- **Wire the volume into materials first, F7 second.** Rejected: that means
  threading a binding through eight materials individually and then rewiring all
  eight when F7 lands.

## Consequences

- **One aerial model where there were three.** Terrain, vegetation, craft,
  structures, and ocean recede into the same air by construction — a real
  one-world violation closed, and `object_aerial_recession` plus four
  exposure-coupled constants deleted.
- **Two aerial regimes, therefore two handoffs** — the volume's far plane and the
  Kármán gate. Both are new artifact surfaces (a visible seam at the far plane, a
  pop at the gate) and neither is verifiable from a single beauty frame. The
  `aerial` comparison axis (BL-37) is a prerequisite for calling this done, run
  across `THALOS_SCREENSHOT_DISTANCE` brackets and an ascent through the gate.
- **Volumetric shafts become nearly free** (W11b): the march already visits every
  froxel, so sampling `thalos::shadow` per froxel yields crepuscular rays through
  terrain, and through clouds once W2/CLOUD-5 lands. This is the main reason to do
  W11 before further shadow-filtering work.
- **Precision improves.** The volume is built in view space, so it is
  origin-relative by construction — strictly better under big_space than today's
  world-space `distance(cam_pos, world_pos)`.
- **Cost decouples from resolution**, which matters for the 4K baseline W13/DLSS
  work wants to establish.
- **The froxel volume is energy-consistent but not free of calibration**: it will
  change apparent scene brightness at distance, so it must land against a settled
  exposure baseline (GF-CAL) or the two will be tuned against each other.
