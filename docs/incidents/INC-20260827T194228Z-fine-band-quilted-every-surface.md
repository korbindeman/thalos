# INC-20260827T194228Z-fine-band-quilted-every-surface: sub-model relief read as a quilt, not as terrain

- **Date:** 2026-08-27 · **Surface:** any near-ground view on the diffusion backing —
  `THALOS_TERRAIN=diffusion just capture mountain-close small-valley`

## Symptom

Hillsides, valley floors and mountain flanks were covered in soft rounded swells
of one size — roughly 40–100 m across and a metre or two deep, evenly spaced,
with faint creases between them and no drainage organisation. It read as a
quilt or a hammered surface. Worst in the valley, where the light grazes the
slopes, but present on every landform.

Two hypotheses had to be ruled out before the real one was reachable:

- **Shading, not geometry.** `THALOS_TERRAIN_INSPECTION=geo-normal` (which zeroes
  the tile material's detail-normal offset) produced an essentially identical
  image. Not the fragment shader.
- **The erosion band.** Its cell-lattice signature was the obvious suspect given
  INC-20260729T010500Z. Ablating it (`THALOS_TERRAIN_BANDS=-erosion`, added for
  this investigation) left the lumps completely intact; ablating the fine band
  removed them. The erosion band alone drew smooth flowing rills.

Decisive evidence was **not** a screenshot. `band_roughness_probe` lights the
height field directly at 2 m/px with no renderer in the loop, and the lobed,
cloverleaf, closed-ring cells are unmistakable there.

## Root cause

Two independent defects in `fine_band`, both of which produce the same look.

**1. The octave ladder tilted the wrong way.** `FINE_OCTAVES` was
`(700, 7) (300, 4) (130, 2.4) (55, 1.5)`: amplitude fell ~0.6× per ~2.3× of
wavelength, i.e. a Hurst exponent near 0.60. Slope per octave goes as
`λ^(H−1)`, so slope *grew* ~35 % per octave and the finest octave was the
steepest by 2.7×. What you see in a ladder like that is its bottom rung — in
this case individual 55 m gradient-noise cells, which is exactly the size and
shape of the lumps. Real soil-mantled ground goes the other way: creep is a
diffusion, so a hillside is *smoother* at 5 m than a fractal extrapolation
would predict.

**2. The ridged mix left a slope discontinuity.** 20 % of each octave was
`(1 − |n|)`, a fold whose crease along every zero-contour of `n` is C0. A slope
discontinuity has unbounded bandwidth, so `footprint_gate` — which scales an
octave's amplitude — cannot attenuate it, and refining toward it only sharpens
it. Zero-contours of a random field are closed curves, which is why the creases
appeared as rings and cloverleaf outlines rather than as lines.

This is the same mechanism `EROSION_FOLD_ROUND` was written to explain, one
band over, three weeks earlier. Nobody applied it here because the fine band's
fold was small (20 %) and amplitude reasoning says a small term is a small
problem — which is true for amplitude and false for bandwidth.

## Fix

The erosion band is **deleted**, at the user's call: once the fine band became
regime-aware it had nothing left that the hillslope and rock regimes do not do
better, and three rounds of retuning had never got past its cell lattice.

`fine_band` is rebuilt as the whole terrain below the learned data's resolution
(90 m inside a detail window, 1.2 km outside), regime-selected on the base
terrain's own 90 m slope — depositional lowland, soil-mantled hillslope, bare
rock — each with its own amplitude, Hurst exponent and shaping. Sharpness now
comes from shaping, never from spectral tilt: every regime runs at `H ≥ 1.0`.

Three secondary traps were hit and closed while building it, each an instance
of the same rule:

- Bedding keyed on base height drew **concentric contour rings**, because the
  base is analytic and its contours are near-perfect closed curves. Broken with
  a bounded height offset (`BED_BREAK_M`), whose own bandwidth is folded into
  the strata's footprint gate.
- The ledge profile `sign(w)·|w|^0.6` has an **infinite derivative at every zero
  crossing** — the fold defect again, in a different costume — and drew hard
  zebra banding across crags at 2 m sampling. Replaced by `tanh(k·sin φ)`, whose
  `k` is an explicit ceiling on the riser slope.
- Ledge amplitude stated in metres was scale-dependent by construction: a bed's
  ground wavelength is `thickness / slope`, so the same height is a bench on a
  hillside and a near-vertical riser on a face. `STRATA_RISER_SLOPE` states the
  angle and the amplitude is derived from the wavelength.

Amplitudes are now RMS metres, divided by the noise's own measured spread
(`noise_moments`), so they can be checked against a real hillside rather than
against each other.

## Recurrence signal

`cargo run --release -p thalos_terrain --example band_roughness_probe`
(with `THALOS_TERRAIN=diffusion`) prints RMS amplitude **and RMS slope** per
footprint shell.

Each shell's slope is measured at its own lag, so the ladder is not expected to
be perfectly flat — the shipped band sits around 1–3° through the mid shells
and reaches ~4° at the finest, where the shells are wider in octaves and the
rock fold concentrates slope. **The tell is a shell whose slope is several
times its neighbours', or a rise that continues at the bottom of the ladder**:
that means the finest octave is the steepest thing on screen, which is the
defect stated as a number, before any of it is visible in a screenshot. The old
band read 0.0226 at the 10–25 m shell against 0.0019 and 0.0018 below it —
because it had no content below 55 m at all, so its bottom rung *was* the
character.

The probe also writes a 2 m/px shaded relief per site. Closed rings, cloverleaf
lobes, evenly sized cells or zebra banding in that image all mean a fold has
gone unrounded again. It runs three sites, and `far-field` is there on purpose:
it is outside every detail window, where the band has to fill 1.2 km rather
than 90 m downward, and a ladder tuned only on windowed ground ships untested
over most of the planet.

`fine_band_shaping_adds_no_dc` pins the other half: every shaping transform is
non-linear, so an uncentred one contributes a constant per octave, and each
octave is multiplied by a footprint gate — which turns that constant into a
**ground height that moves with camera distance**. That failure is invisible in
any single screenshot and is the LOD-invariance class INC-0003 belongs to.

Standing rule, and the reason this incident exists rather than a commit
message: **a fold, a `|x|`, an `abs`, or a fractional power is a bandwidth
decision, not an amplitude decision.** Lowering its weight does not make it
safer. Round it, bound its derivative, or do not use it.
