# ADR-20260725T222409Z — Cloud cells are an analytic surface field, and LOD never renders the mean

**Status:** accepted, 2026-07-25.
**Supersedes the homogenized-field half of** BL-20260724T003705Z (round 6's
`E[shaped | env]` band). Extends
[ADR-20260722T141000Z](20260722T141000Z-far-cloud-density-is-surface-parameterized.md)
(far cloud density is surface-parameterized) from the far tier to *all* tiers.

## Context

Nine rounds of cloud work converged the near field to a credible broken deck
and then stalled on the same verdict every time it was viewed at range:
"incoherent soupy blobby mess, with rough transitions between LOD"
(2026-07-25, against Blackrack/KSP and MSFS references).

Two measurements explain the whole family:

1. **Nothing in the pipeline could draw a 1–5 km cloud cell at range.** The
   planetary weather cube is 1024/face — 4.89 km per texel on Thalos
   (R = 3186 km), so ~10 km Nyquist — and its authored content is coarser
   still: the producer's finest terms, `fbm3(dir_raw · 128, 3)` and
   `fbm3(dir_raw · 211, 2)`, put the dominant cellular scale at **15–25 km**
   with a weak 6–7 km octave. Cell-scale morphology existed only in the 8 km
   periodic Cartesian volume, which the march retired past ~42 km and replaced
   outright past ~90 km.
2. **The LOD replaced the field with its mean.** Past band 2,
   `get_cloud_map_density` returned the derived `E[shaped | env]` — mean-
   preserving by construction, and that was believed sufficient. It is not:
   opacity and lighting are nonlinear in density, so `E[shade(σ)] ≠ shade(E[σ])`.
   A mean-preserving filter that destroys variance preserves the average
   brightness of a region while erasing every cloud in it. That is precisely
   what the captures showed — a flat beige sheet from ~40 km to the horizon,
   with a contrast step at the band edge.

Every prior attempt to get cells at this scale drove a **stored or periodic**
basis and died the same way: the spherical shell cuts a Cartesian repeat into
planet-visible rows or combs (ADR-20260722T102639Z, ADR-20260722T111036Z,
ADR-20260722T135123Z, ADR-20260722T141000Z, and round 9's rejected 0.42×
horizontal period).

## Decision

**Cell-scale cloud morphology is an analytic, aperiodic, direction-parameterized
column field — `cloud_cell_field` in `thalos::atmosphere` — shared verbatim by
the near march, the far projection, and the CPU calibration mirror.**

- **Aperiodic by construction.** It is evaluated from a hash lattice in the
  body-fixed *direction* domain, so there is no tile to repeat and no repeat for
  the shell to cut. This is the general form of ADR-20260722T141000Z's rule.
- **A column field.** One horizontal identity from base to top — what a
  convective column is. Vertical shape stays with the marcher's typed profiles
  and the round-7 dome threshold.
- **The periodic Cartesian volume is demoted to a sub-cell sculptor**
  (`SUBCELL_SHAPE_WEIGHT`), added mean-preservingly *inside* a cell the
  analytic field already placed, and retired at range where its lobes are
  sub-footprint anyway.
- **LOD is octave retirement inside one field, never a different field.** Each
  octave fades to its *own* mean once its period drops under the sampler
  (`filter_m` = refine step or projected pixel footprint, whichever is coarser).
  Range therefore degrades cells → coarser cells → strata, and no band or tier
  swaps representation.

Three constants are derived, not chosen, and are recorded here because they look
arbitrary in the code:

- `CLOUD_CELL_BILLOW_MEAN = 0.302816` — measured `E[|2v − 1|]` over 4M samples
  of the shared lattice noise. The billow octave is re-centred on it so its
  neutral value is exactly 0.5 and the octave fade cannot shift the field's
  mean, which would silently de-calibrate `fill_lut` at range.
- `CLOUD_CELL_GAIN = 3.2` — the raw octave mix has std **0.115** (measured over
  1.5M directions). At that spread the formation threshold is a cliff: 0.40 →
  0.60 swings areal coverage 80 % → 20 %, everything crosses by the same margin,
  and the deck renders as a carpet of identically-sized puffs (first capture).
  The gain restores std ≈ 0.20.
- `CLOUD_CELL_KNEE = 0.45` — the gain is applied through a soft saturation
  `x / (k + |x|)`, **not** a clamp. A hard clamp at gain 2.4 saturated **9.6 %**
  of the sky to exactly 1.0 (measured), and a constant region has no isosurface:
  the height-rising dome threshold cut every such core at one exact altitude and
  the deck rendered as flat-topped mesas with vertical sides (second capture).

## Consequences

- The derived `shape_response` LUT (`E[shaped | env]`) is **deleted** — from
  `fill_lut`, the compute uniform, `CloudsConfig`, and the disk cache schema.
  Its only consumer was the homogenized band. `FILL_LUT_VERSION` 5 → 6.
- The fill/opacity pairing (`fill_lut`) is otherwise **untouched and still
  authoritative**: it governs how much of the frame clouds cover, and it
  re-derives against the new density math at spawn, so tier parity survives by
  construction. The standing rule from round 5 holds — never hand-retune either
  tier's response.
- `cloud_strata_warp` is now applied **unconditionally** in both tiers. It was
  gated to coarse bands on the premise that the Cartesian sculptor hid the
  ~5 km texel lattice near the camera; that premise died here, because the cell
  field is thresholded against `env` directly and a texel edge cuts a vertical
  wall through a cloud. The warp is measure-preserving, so the calibration is
  unaffected.
- Range-keyed edge softening drops from `0.30` to `0.075`. It was keyed to the
  *broad* band step even though density is only ever integrated at the 0.2×
  refine cadence — ~5× over-conservative, and the prime remaining suspect for
  round 9's `massif-ridge` milky veil. Softening cannot survive in this design:
  a widening edge erases contrast with range just as effectively as rendering
  the mean did.
- **What this does not fix:** whole-disc framings. At ~14 km per pixel the cell
  field is correctly retired — a 450-pixel planet physically cannot resolve 3 km
  cells — so the full-disc view still reads as the weather producer's 15–25 km
  smears. Cell structure at that framing is not a rendering problem; it is the
  producer's, and is tracked separately.

## Standing rules

1. **A cloud LOD may reduce the field's bandwidth. It may never render the
   field's mean.** Mean-preserving is not appearance-preserving when the shading
   is nonlinear, and a variance-destroying filter deletes clouds while passing
   every average-brightness check.
2. **Cell-scale organization comes from an aperiodic surface-parameterized
   field.** If cells look wrong, change that field or the formation threshold —
   never shorten a stored volume's period, which is how the "rows" family
   arrives every time.
3. **A field that a threshold cuts must have spread.** Measure the marginal
   distribution before fitting anything to it; a narrow field turns any
   threshold into a cliff and any gain-with-clamp into flat-topped mesas.
