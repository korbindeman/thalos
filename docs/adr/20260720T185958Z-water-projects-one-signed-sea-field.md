# ADR-20260720T185958Z-water-projects-one-signed-sea-field: Water is a projection of one signed sea-height field — depth never decides coverage

- **Status:** Accepted (supersedes the near-authority/crossfade half of ADR-20260720T185957Z-coastline-as-authored-data)
- **Date:** 2026-07-20

## Context

ADR-20260720T185957Z-coastline-as-authored-data split water authority by range: near, coverage/colour from the exact
scene-depth compare (`t_ocean` vs `scene_t`); far, from the baked coast atlas;
a crossfade band (~300 km–1.5 Mm) between them. It landed with a stack of
error-hiding devices — a range-scaled mesh-error feather (`2e-5·t`), the
INC-0003 error-aware tie band, footprint-scaled occlusion thresholds on a
noisily-reconstructed scene height — and the user's verification fly-over
failed the same day: dotted coast speckle at range, mushy translucent washes
at aerial altitudes (the feathers *are* the wash), and instability whenever
the camera crossed the authority bands.

The root defect is architectural: "is this pixel water?" was still answered
by **comparing two independently-reconstructed distances** (analytic sphere
hit vs depth-buffer terrain hit). Both carry range-scaled f32 error, and at
the shoreline they are within metres of each other *by definition*, so the
comparison dithers at every range where the error exceeds the foreshore
depth. Every widened feather converts speckle into wash; the medicine became
the disease. Meanwhile the depth-compare's replacement data has been resident
on the GPU all along: the udlod height-tile atlas holds the *exact texels the
terrain mesh is displaced from*, streamed and mip-chained, and INC-0003
already made the field's sea-level crossings LOD-invariant.

## Decision

**Every water decision samples the one signed sea-height field directly; the
depth buffer never decides coverage or colour — only occlusion by resolvable
geometry.**

- The `BodySky` ocean branch samples signed height-about-sea at the sphere-hit
  direction from a **resolution cascade of the same field**:
  1. the resident **udlod height-tile atlas** (tile-tree walk capped at the
     pixel's footprint LOD, mip-sampled within the tile) — exact, the same
     data the visible mesh displaced from;
  2. the **coast/bathymetry cube** (ADR-20260720T185957Z-coastline-as-authored-data's atlas, mip-sampled at footprint)
     as the coarse tail — tiles not resident, terrain despawned, beyond the
     impostor swap.
  Near and far are the same quantity at different resolutions — a mip chain,
  not an authority handoff. There is no crossfade band and no seam to tune.
- **Coverage** = a smooth band around the field's zero crossing, sized by the
  *sampled texel* (a fixed wet-edge minimum near, texel-scale antialiasing
  far) — never by a range-scaled error model.
- **Colour / column** = the field's bathymetry over the slant path
  (`−h/μ`) — smooth by construction; tile seams in water colour are
  impossible.
- **Occlusion** (the depth buffer's one remaining job) is field-assisted:
  terrain occludes water when the *field's* height at the scene-hit direction
  is resolvable land at this footprint (exact heights, replacing the noisy
  radial reconstruction); non-terrain geometry (craft, structures) occludes
  when it stands in front of the sphere hit by more than a footprint-scaled
  margin. Unresolvable coarse-mesh slivers at the limb defer to the filtered
  mask — the BL-5 rule, with exact inputs.
- Deleted: the `OCEAN_AUTHORITY_NEAR/FAR_T_M` crossfade, the `2e-5·t`
  mesh-error feather, the error-aware tie band, and the reconstructed-height
  occlusion thresholds.

## Alternatives

- **Terrain height prepass** (re-render tiles into an R16F signed-height
  target; sky pass reads it like `scene_depth`): raster-exact silhouette
  match and inherits vertex-side effects (analytic flatten), but costs a new
  render pass + pipeline variant + target management (~3–4× the code) and
  re-renders all tiles every frame. The field sample achieves the same
  stability with zero extra passes; the mesh-vs-field morph mismatch is
  sub-texel and covered by the fixed texel-scale band. Rejected for scope,
  not principle — if a future consumer needs raster-exact terrain data in
  screen space, this is the way.
- **Water shaded inside the terrain fragment shader** (mini-F9): perfect
  coverage where tiles exist, but wrong parallax at depth (surface drawn at
  seabed geometry), breaks future submarines/underwater, duplicates
  `shade_ocean`, and still needs the analytic sphere + a mask where no tiles
  are streamed. Rejected; F9 remains the long-term shading home and will
  consume this same field input.
- **Keep hardening the depth-compare** (round 4): the error term is
  irreducible (f32 depth at planet ranges) and every feather that hides it
  produces the wash the user rejected. No.

## Consequences

- The waterline cannot move with camera distance, tile LOD, or streaming
  state: coverage is a pure function of a LOD-invariant field. During cold
  streaming the coast sits where the atlas says and *refines* as tiles land.
- `BodySkyMaterial` hand-implements `AsBindGroup` to bind udlod render-world
  resources (height atlas texture array, tile-tree + origins buffers) — the
  sky pass is now coupled to udlod internals through small public accessors.
  Both live in `body_render::ground`; the coupling is the point (one field).
- The sky material carries the terrain entity; the game's per-frame sky
  update refreshes it (terrain despawn/respawn safe — the material is
  re-prepared every frame already).
- `body_sky.wgsl` re-implements the tile-tree walk + cube-coordinate inverse
  (~100 lines) against its own binding slots — naga_oil import paths bind
  udlod's functions to udlod's bind groups, so they can't be reused directly.
  Kept byte-faithful to `thalos_udlod::math::Coordinate::from_world_position`
  / `functions.wgsl`; a comment on each side ties them.
- ADR-20260720T185954Z-analytic-planet-water-never-meshed stands: the water *surface* stays the analytic sphere. ADR-20260720T185957Z-coastline-as-authored-data's
  coast atlas survives as the cascade's coarse tail; its zero-crossing
  invariant (**relief never crosses sea level**) is now load-bearing for the
  whole cascade.
- The beach follow-up (sand material band + vegetation clearing from the same
  field) makes the stable coastline *read* as a coast; tracked separately.
