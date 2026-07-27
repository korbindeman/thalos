# INC-20260727T012304Z — the erosion band amplified the raster's bilinear creases into knife ridges

## Symptom

Close views of the massif (`massif-valley` at ~1.8 km, `THALOS_TERRAIN=diffusion`)
grew a field of thin, straight-edged blades standing out of the slope — "shark
fins", with a bright rim and a hard shadowed face. They got **worse the closer
the camera came**, and they arrived with the erosion band
(BL-20260726T045639Z-erosion-fine-band).

## Diagnosis path

1. **Attribution.** `EROSION_STRENGTH = 0` with `THALOS_TILE_CACHE=0`: every fin
   in the frame gone, slope smooth. (An earlier guess that these were LOD normal
   seams was wrong — this test is what corrected it.)
2. **Shape.** Resolution scaling of peak |second difference| along a fixed 3 km
   transect at 32 → 1 m: a band-limited feature quarters when the spacing
   halves, a C0 *fold* only halves. Measured ratios 0.46–0.71 ⇒ folds, with an
   implied slope jump of **~1.15 (≈49°)** — a knife, and scale-free, so
   refinement only sharpens it. That also explains why the round-1 retune from
   `EROSION_STRENGTH` 0.03 → 0.012 shrank the blades without removing them.
3. **First hypothesis — wrong, and worth recording.** The filter builds ridges
   by folding (`phacelle.y.abs()` / `sign()` in `bevy_erosion_filter::cpu`), so
   the creases looked like the filter's own. A rosette mollifier over the
   filter's **chart position** cut the fold only 1.155 → 1.075, and widening the
   radius to 105 m still only reached 0.868. A fix that barely moves the metric
   is falsifying its own diagnosis: the fold was mostly not in the chart
   coordinate.
4. **Elimination.** Pinning `rough_scale` to 1.0 made it *worse* (1.398);
   pinning the steering `slope` to a constant barely moved it (0.903). Neither
   was the carrier.
5. **The tell, found by printing the profile instead of reasoning about it.**
   Per-band slopes across the sharpest feature showed the **base** slope
   piecewise-constant — 0.0146 for fourteen consecutive samples, then 0.1311 —
   with the erosion band flipping sign at that same sample. That is a bilinear
   raster cell edge.

## Mechanism

`Raster::sample_px` reconstructed the 90 m detail raster with **bilinear**
interpolation, which is C0: the gradient jumps across every cell edge, so the
raster's own lattice is a grid of slope creases. In the height alone that is
mild (measured fold 0.096 — the faint blocky faceting long visible in hillshades
of the base). But the erosion band takes both its **steering slope** (finite
differences of `band_base`) and its **base height** from that field and responds
to them nonlinearly — through `gully_slope`, `combi_mask` and the fade anchor —
so each crease came back **~8× amplified** as a knife ridge in a carve up to
79 m deep.

The band was therefore not the origin of the defect, only its amplifier. Two
independent contributors, measured (implied slope jump, transect at h = 0.5 m):

| reconstruction | fold rounding | erosion band |
|---|---|---|
| bilinear | off | 1.155 |
| bilinear | on | 0.775 |
| Catmull-Rom | off | 0.789 |
| Catmull-Rom | on | **0.510** |

Base band alone: 0.096 → **0.002** under Catmull-Rom.

## Fix

- **`Raster::sample_px` is Catmull-Rom, not bilinear.** C1, no gradient
  flattening, so the reconstruction stops manufacturing creases for the band to
  amplify.
- **`EROSION_FOLD_ROUND`** keeps the rosette from step 3: it is not the main
  term, but the filter's own folds are real and it removes a further third,
  costed only where the mesh can resolve the rounding.

**Rejected: smootherstep-weighted bilinear**, the cheap C1 alternative. It forces
the gradient to *zero* at every cell edge, and the same amplification turned that
into a worse artifact than the original — the band's fold went to **3.58**, 4.5×
the bilinear baseline. A field consumed by a slope-sensitive nonlinearity needs
smoothness in the derivative, not mere continuity.

Cost: no measurable regression. Matched cache-off `massif-valley` runs measured
in `runtime.jsonl` (never a bench — [[tile-renderer-load-time]]): 20.9 ms/tile
bilinear vs 17.3–22.7 ms/tile Catmull-Rom, i.e. inside run-to-run variance. Tile
synthesis is dominated by the analytic octaves and the erosion filter, not by
raster reads, so 16 taps versus 4 does not show.

`GENERATOR_VERSION` 23 → 24.

## Recurrence tells

- **A fix that barely moves its own metric has falsified its diagnosis.** Step 3
  should have ended the "it's the filter's `abs()`" theory immediately.
- Any nonlinear consumer of a sampled raster inherits that raster's
  reconstruction class, amplified by its own gain. Before blaming the consumer,
  check what the field it reads is made of — and print the profile rather than
  reasoning about the shape from aggregate statistics.
- The scaling test is the general tool: peak |second difference| versus sample
  spacing separates band-limited features (h²), folds (h¹) and steps (h⁰).
