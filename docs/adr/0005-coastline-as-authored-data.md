# ADR-0005: The coastline is authored data — scene depth is occlusion-only for water

- **Status:** Partially superseded by ADR-0006 (2026-07-20): the coast atlas
  survives as the coarse tail of the field cascade, but the depth-compare
  near authority and the range crossfade are replaced by direct sampling of
  the resident udlod height tiles — depth no longer decides coverage/colour
  at any range.
- **Date:** 2026-07-20

## Context

The analytic ocean (ADR-0002) decided per pixel whether a fragment is water by
comparing the ray-traced sea-sphere hit against the **scene-depth buffer** —
i.e. against whatever terrain mesh UDLOD happened to render this frame — and
derived the water's colour from the same signal (`column = scene_t − t_ocean`).
That made three distinct questions share one fragile input:

1. **Coverage** — is this pixel water or land?
2. **Colour** — how deep is the water column (shallow-cyan → deep-blue, and the
   see-through shoreline feather)?
3. **Occlusion** — is opaque geometry in front of the water surface?

Only (3) is a question scene depth can answer robustly. (1) and (2) inherited
every rendering property of the terrain: tile LOD selection and morph state,
streaming timing, texel resolution, and f32 depth-reconstruction error that
grows with view distance (~5·10⁻⁷ × range: centimetres in flight, metres from
orbit). INC-0003 hardened the *inputs* (deeper foreshore, no LOD-crossing
relief, an error-aware feather) but the symptoms kept regenerating from the
architecture: tile-grid seams visible as water-colour discontinuities,
kilometre-wide mushy shorelines over gentle foreshores, and coastal coverage
dithering wherever terrain sits inside the renderer's own error band. The
coastline was **emergent** — recomputed per pixel per frame from mesh geometry
— and at any finite tile resolution an emergent iso-contour is noise-shaped.

Constraint from gameplay: beaches, underwater, and water-surface gameplay
(ships, submarines) are wanted. So the f64 `ProceduralSurface` must remain the
sole gameplay height/bathymetry authority, gentle shallows must stay (no
cliff-edge band-gap coasts), and the near-field waterline must follow the real
terrain at human scale.

## Decision

**Far-field water coverage and colour come from a per-body baked coast atlas;
scene depth is demoted to what it can actually answer.**

- At world spawn, each ocean body bakes a low-res **coast/bathymetry cube**
  (R16Unorm-encoded height, metres about sea level) from the *same*
  `SurfaceQuery` surface the tiles bake from — one bake, no second authority,
  the SkyViewLut precedent. Because relief may never cross sea level
  (INC-0003's awash-reef invariant), the atlas's zero crossing *is* the
  LOD-invariant macro shoreline at any sampling resolution.
- In the `BodySky` ocean branch, authority **crossfades by range to the water
  hit** (`t_ocean`), each signal used only where it is trustworthy:
  - **Near** (< ~300 km): today's depth-compare path — exact at these ranges
    (error floor ~cm), and the waterline hugs the real streamed terrain at
    beach scale.
  - **Far** (> ~1.5 Mm): coverage = the atlas's signed height at the sphere
    hit (bilinear on a smooth field → crisp sub-texel shoreline, structurally
    independent of tile LOD); colour = atlas bathymetry (smooth → tile seams
    in water colour are impossible); occlusion = scene hit **height above
    sea** (mountains occlude ocean behind them; metre-noise in the height
    reconstruction is absorbed by a wide smoothstep and coastal plains are
    mask-land anyway).
- The near-field shoreline feather shrinks to ~1 m of column — a physical
  wet-edge, not an error-hiding device (the error-hiding job now belongs to
  the far-field mask).
- The generator gains a **foreshore drop** (a few metres of depth reached
  within ~the first kilometre of shore) so beaches read as beaches and near
  water clears the shallow see-through band promptly, while the wide gentle
  shelf beyond stays.

## Alternatives

- **Band-gap heights** (no terrain within ±~10 m of sea level, steep coast
  ramp): smallest change and fully consumer-consistent, but kills beaches,
  shallows, and tidal flats — exactly the aesthetics and gameplay surfaces
  wanted. Rejected.
- **Water folded into the terrain pipeline now** (F9: `WATER` branch in
  `shade_surface`, mask/SDF as a tile attachment): the long-term home, but it
  reintroduces the meshed-water facet-sag problem ADR-0002 exists to prevent
  unless the analytic sphere is kept anyway, and coarse tiles quantize the
  boundary unless an SDF channel is added — i.e. it *also* wants the authored
  coast data. Deferred, not rejected: the coast atlas built here becomes F9's
  attachment source. ADR-0002's analytic-sphere rule stands.
- **Evaluating the macro field analytically in WGSL** (no bake): crisp at
  infinite resolution, but it duplicates the continent field in a second
  language — the CPU/GPU dual-authority drift risk the tile pipeline already
  rejected for heights. A baked projection of the CPU field cannot drift.

## Consequences

- The coastline **shape** is now impossible to change from the rendering side:
  tiles, LODs, and streaming affect only near-field detail and far-field
  occlusion, not where the sea is.
- Water colour is smooth by construction at range; the INC-0003 limb-streak
  residual (BL-5) is largely subsumed — limb pokes below the occlusion height
  threshold get covered by mask water.
- The atlas is render-only. Gameplay (colliders, buoyancy at sea level 0,
  future submarine depth) keeps reading the f64 surface — a drift between the
  two is bounded by atlas texel size and only ever cosmetic at >300 km range.
- New invariant: **relief must never cross sea level** (from INC-0003) is now
  load-bearing for rendering correctness, not just aesthetics — the atlas's
  crisp zero crossing depends on it.
- The crossfade band (~300 km–1.5 Mm on `t_ocean`) is a tuning surface; a
  visible authority seam there is the recurrence tell.
