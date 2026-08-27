# INC-20260827T202201Z-level-bedding-painted-contour-rings: a level material frame traces contours

- **Date:** 2026-08-27 · **Surface:** any rock face on the tile renderer —
  `THALOS_TERRAIN=diffusion just screenshot mountain-close`

## Symptom

Broad lighter rings ran across every mountainside, concentric, following the
slope's contours, with no relief behind them. User-reported.

## Root cause

`tile_terrain.wgsl`'s rock layer bands its albedo on a strata coordinate,
`dot(p, DIP_K) / TILE_WRAP_M`. `DIP_K` was a single global bedding-plane normal
chosen to be **"≈ the local radial at the showcase window … so beds stack
near-horizontally there like real sedimentary strata"**. Measured against the
shipped viewpoints it sat within **3.6° of the local radial**.

A bedding plane whose normal is the radial *is a level surface*. Its
intersection with any landform is that landform's contour lines — so the term
was, by construction, a contour plotter. On the smooth cones this terrain
builds, the contours are closed and evenly spaced, which is what made it read
as a topographic map rather than as rock.

Two things made it hard to see:

- **It looks like terrain.** The obvious hypothesis is a height band. It is not:
  `THALOS_TERRAIN_INSPECTION=fullbright` (albedo, no lighting) shows the rings
  at full strength, and zeroing the `0.26 * strata` weight alone removes them
  while leaving the fall-line striation intact.
- **The file already recorded a wrong-target fix for it.** A round-3 note says
  the alpine zone "terraces like a topographic map" and moves the weighting from
  the alpine term to `rock_steep`. That changed *where* the term drew and not
  that what it drew was contours, so the artifact survived with a smaller
  footprint and a plausible explanation attached.

## Fix

Both halves, because either alone is incomplete:

1. `DIP_K` retilted to `(-57, 9, -27)` — same integer-lattice property, same
   128 m bed thickness, but **~30° of dip** over the shipped viewpoints instead
   of 3.6°. The traces now cut across the slope, which is what makes bedding
   read as bedding.
2. A `BEDDING_DIP_LO/HI` gate retires the strata albedo *and* its ledge normal
   as the dip approaches zero, wherever on the body that happens. Without it,
   retilting only moves the degenerate cap to a different longitude for someone
   else to rediscover.

Byproduct worth knowing: the ground spacing of the traces is
`thickness / sin(angle between bed and surface)`, which is **never finer than
the bed thickness**, so this term cannot alias however steep the face.

## Recurrence signal

**Concentric rings on a smooth landform mean a material frame has gone parallel
to the level surface.** The general rule is broader than bedding: any term keyed
on altitude, or on a plane whose normal is near-radial, plots contours. The file
already applies the same correction to its altitude-driven layer selection ("…
jittered by a low-frequency field so they follow local terrain instead of
drawing contour rings"), and the fine band's own strata needed the same break-up
for the same reason (INC-20260827T194228Z).

Separate rings from relief in one shot with
`THALOS_TERRAIN_INSPECTION=fullbright`: if the pattern survives with the
lighting removed, it is a material term and no amount of terrain work will
touch it.

Still open, and the thing to watch: under `fullbright` the retilted bands read
as regular parallel stripes. Lighting hides that at all three framings checked
(`mountain-close`, `small-valley`, `massif-aerial`), but the term is a
triangle wave of uniform amplitude — real sequences have a few resistant marker
beds and long quiet intervals. If corduroy ever shows in a lit view, modulate
the band amplitude by a low-frequency field rather than reaching for the
spacing.
