# INC-20260724T235437Z-plume-ended-on-a-lit-rim: the plume's length had a second authority the shader could not see

- **Date:** 2026-07-24 · **Surface:** `just screenshot plume` (atmosphere only — `THALOS_PLUME_PRESSURE=101325`)

## Symptom

In atmosphere the exhaust column stopped dead: a flat, still-bright rim hanging
in mid-air with nothing beyond it. In vacuum the same engine faded out correctly,
which is what made it look like an atmosphere-specific shading bug rather than a
geometry one.

The decisive number, evaluating the shader's own chain at the station where the
mesh ended (`s = 26·R0`, sea level): `er ≈ 0.79`, entrainment `e^-1.08 ≈ 0.34`,
so `T ≈ 0.34` and `band_emission ≈ 0.13`, with `1 - e^-tau ≈ 0.85`. That is
**~12 % of nozzle-exit radiance** — times a gain of 5.5 and the 1.8× afterburn
boost, comfortably above 1.0 in HDR. The geometry ended while the column was
still incandescent.

## Root cause

`docs/rendering/plume.md` states the design invariant plainly: brightness is
derived from shape, so "the mesh can simply stop". `resolve_params` then broke it
with a second, independent length authority:

```rust
let len_mixing = r0 * lerp(26.0, 400.0, vac);
let length_m = len_expansion.min(len_mixing) * …;
```

`len_expansion` was solved *from* the emission model, so it was safe. `len_mixing`
was not: it encoded "entrainment tears the jet apart after ~26 radii" as a cut,
while the fragment stage's entrainment cooling rate was an unrelated authored
constant (`0.016` per radius) that reached only `e^-0.42` over that same
distance. Two numbers describing one physical process, with nothing tying them
together — so the mesh ended a long way before the emission did. In vacuum
`len_expansion` was the smaller of the two, the model's own vanishing point won,
and the defect hid.

Throttle made it worse by the same mechanism: `length_m` was further scaled by
`lerp(0.7, 1.0, throttle)` and `lerp(0.45, 1.0, ignition)`, trims that no term in
the shader knew about, so a throttled or just-ignited engine was cut even deeper
into its hot column.

**An instructive wrong turn.** The first fix derived the entrainment rate from
the mixing length (so the tip *is* cold) and solved for the length at
`RADIANCE_FLOOR = 0.003` of peak — and the rim came back, now in the shear layer.
Two reasons, both worth remembering:

1. **A fraction-of-peak floor is not a visibility floor.** The core saturates at
   an HDR radiance of order 10, so 0.3 % of peak is ~0.04 linear, which exposure
   and tonemapping lift back to a clearly visible brown.
2. **The criterion watched only the core.** The sheath is wider, so its optical
   depth saturates long after the core has thinned, and in atmosphere
   afterburning makes it the *brighter* layer by the tail. Cutting on the core
   alone cuts through a glowing shear layer.

## Fix

One authority, and it is the rendered image:

- `visible_length_m` bisects `Column::radiance(s)` — the CPU twin of the
  fragment's full two-layer chain, gain included — for the station where it drops
  below `VISIBLE_RADIANCE`, an **absolute** HDR floor. The billboard is cut
  exactly there.
- Entrainment is no longer an authored rate: it is derived from the jet's mixing
  length so that emission genuinely dies within it.
- Everything that should shorten a plume now feeds that chain instead of trimming
  its result — throttle through mass flow (κ) and mixing length, ignition through
  exit temperature. No post-multipliers on `length_m`.

Two further changes were needed before the *silhouette* could feather, and they
belong to the same mechanism: a saturated top-hat has a razor edge in every
direction, because the only thing that can end it is the chord going to zero.
Both layers now use radial density kernels with compact support — `(1-(r/R)²)^½`
for the core and `(1-(r/R)²)²` for the shear layer, whose chord integrals are
`(π/2)·R·(1-(p/R)²)` and `(16/15)·R·(1-(p/R)²)^{5/2}` — so the profile reaches
exactly zero at the mesh boundary however optically thick the column is.

## Recurrence signal

An emissive volume that **ends on a straight edge** — flat across the tip, or a
clean line down the silhouette — while its interior is still bright. Look for a
length or width the fragment stage cannot compute, and for `1 - exp(-tau)`
saturating out to a geometric boundary. The standing rule lives with the model in
`docs/rendering/plume.md` ("One length authority") and on `visible_length_m`.

Corollary for any additive HDR effect: **visibility floors must be absolute.**
Relative-to-peak thresholds are meaningless downstream of a tonemapper, which is
in the business of making small fractions of a large peak visible again.
