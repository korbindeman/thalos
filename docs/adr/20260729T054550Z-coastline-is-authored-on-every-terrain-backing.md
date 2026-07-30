# ADR-20260729T054550Z-coastline-is-authored-on-every-terrain-backing: The coastline is authored on every terrain backing — neural producers supply relief, not geography

- **Status:** Accepted
- **Date:** 2026-07-29
- **Decided by:** user, 2026-07-29 ("keep the waterline authored, start with layer C")

## Context

`DiffusionSurface` (NTR-X2a) took its geometry from the terrain-diffusion
reference pipeline and grew its *own* coast model to fill the gap the released
weights leave: the sea came from a binary landmask thresholded off the 23 km/px
chart, blurred to 46 km, with depth a monotone `smootherstep` of that blur and
the shore a ~46 km-wide blend between a land and an ocean branch.

That model could not express the thing it was responsible for. Anything smaller
than ~46 km was smoothed out of existence before the shore blend saw it, so
islands were not rare on that path — they were structurally impossible. It also
silently discarded the coastal system `ProceduralSurface` had accumulated over
INC-0003, BL-6 and BL-10: coast character (depositional beach arcs vs erosional
rias and archipelagos), the 15–85 km crenulation warp, the 10–45 km islet
clumps, the foreshore drop, the beach berm, the awash-reef cap and the offshore
shallow clearance. Thalos therefore had two coastlines that disagreed — the
authored land fraction was 0.30 while the diffusion body generated 0.32.

Three properties of the released model make it a poor coastline producer, and
none of them are tuning problems:

- it trains on MERIT DEM (**land only**) plus ETOPO1 for ocean, and ETOPO1 is
  ~1.8 km and largely altimetry-*predicted*, so below that scale there is no
  seafloor signal it could have learned;
- its data processing explicitly includes **coastline smoothing** (paper,
  Appendix D) — it was trained away from detailed shorelines;
- its binary mask channel is a **data-availability** mask, not a land/ocean
  semantic control, so there is no conditioning input that means "put a coast
  here".

Meanwhile the shoreline carries a hard invariant the neural bands cannot
satisfy. Because the relief cascade is LOD-aware, relief that defines a
waterline makes that waterline move with camera distance — INC-0003 measured
40 % → 13 % breach coverage across LODs on a shelf, and a gentle coast turns a
few metres of LOD height wobble into kilometres of horizontal shift.
ADR-20260720T185958Z-water-projects-one-signed-sea-field rests on the zero
crossing being LOD-invariant, and ADR-20260720T185957Z-coastline-as-authored-data's
baked coast atlas rests on it too.

## Decision

**Every terrain backing takes its waterline from one authored, LOD-invariant
signed sea field. A learned producer supplies relief above that field; it never
decides where the sea is.**

`ProceduralSurface::macro_signed_height_m` is that field, and it is now public
for exactly this reason. `DiffusionSurface::height` composes three layers:

- **A — authored signed sea field.** Waterline, shelf shoulder, continental
  slope, abyssal plain, foreshore drop, beach berm. Fixed-octave, no footprint
  argument, so its zero crossing is identical at every sampling resolution.
- **B — neural relief, measured against the same 0 m datum.** Inland the total
  is exactly the model's own elevation (`macro_h + (neural − macro_h)`), so the
  authored platform is a coastal profile, not a second continent riding under
  the first. Offshore the model is silent and relief is the depth-gated seabed
  band.
- **C — the canonical composition rules**, `combine_macro_and_relief`,
  unchanged: relief fades across `COAST_BAND_M` about sea level, macro land is
  floored at the waterline, macro seabed may never breach it.

The blurred landmask is deleted, not kept alongside.

## Alternatives

- **Bolt a foreshore and berm onto the existing blend.** This was the literal
  request ("start with layer C") and it does not work: a foreshore is a profile
  *in a signed field*, and the blend had no signed field to shape — its shore
  was an interpolation weight, not a zero crossing. Layer A had to replace the
  blend before layer C had anything to act on.
- **Crossfade between two absolute height fields** (authored near the coast,
  chart inland). Rejected: the two disagree by the authored crenulation, and the
  crossfade puts that disagreement as a step at the worst possible place.
- **Fine-tune for neural coastline now** (NTR-FT-3). Not rejected — deferred.
  It remains the end state for *coastal morphology* (dunes, sea cliffs,
  wave-cut platforms, spits), and it is worth noting that those features are
  mostly land-side, where high-resolution DEM coverage is abundant; NTR-FT-3
  currently scopes bathymetry as one XL problem gated on scarce multibeam and
  probably splits into a data-rich coastal-zone band and a data-poor deep band.
  But even a coastal fine-tune does not change this decision: a learned band is
  footprint-gated, so it still may not own the zero crossing.
- **Give the diffusion backing a higher-resolution landmask.** Does not help.
  The mask is a threshold of the chart, so its shoreline resolution is one chart
  pixel — 23 km — no matter how it is filtered. Authored coastline detail is
  analytic and unquantised.

## Consequences

- Thalos has **one coastline**, and the two backings now measure identical:
  land fraction 35.3 % on both (the lore target), against 0.30 authored vs 0.32
  generated before. The map images are pixel-identical in the sea.
- Islands, bays, rias and archipelagos appear on the diffusion path for the
  first time — they come from the authored islet sprinkle and crenulation, which
  that path had been discarding.
- The waterline is LOD-invariant on the diffusion backing:
  `cargo run -p thalos_terrain --example coastline_lod` reports **0 m** of
  horizontal movement across the full orbit→ground LOD ladder on all 8 sampled
  coasts, and `shelf_breach_probe` reports breach fractions identical to four
  significant figures at every LOD (11.2 % at LOD 0 and at 0.3 m — stable macro
  islets, not INC-0003 speckle). Both probes now honour `THALOS_TERRAIN` so they
  measure whichever backing the session renders.
- Bathymetry deepens from the old shelf's −3.45 km to the authored −4 km abyss;
  `height_range_m` accounts for it.
- **New constraint on the fine-tune**: a Thalos-conditioned model is trained and
  evaluated on relief, not on coastline shape. Coastline metrics do not belong
  in its gate.
- The recurrence tell is the two backings' land fractions diverging, or
  `coastline_lod` reporting non-zero movement on either.
