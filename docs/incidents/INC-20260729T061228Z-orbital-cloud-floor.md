# INC-20260729T061228Z — a gated anvil term became a planet-wide cloud floor

**Symptom.** From orbit Thalos rendered as a near-white marbled disc with almost
no clear sky, while the same weather from low orbit showed scattered puffs with
real gaps between them. Authored coverage is 0.46; the full-disc impostor
rendered ~1.08× that as a *uniform veil* rather than 46% of the area as cloud.
User report 2026-07-29: "monotone cover of blobby looking clouds… too much cloud
overall".

**The tell.** `cloud_weather_probe`'s TRANSFER table. Binned by the producer's
own coverage channel, the clearest decile — 591,400 texels, 28% of the planet —
emitted strata 0.304 and rendered at 0.262 (far tier) / 0.340 (impostor). Only
4.8% of the planet came out at ≤0.02 opacity against 13.4% genuinely clear in
the coverage channel. **Clear sky was never clear**, and no climate trim could
make it so, because the information was destroyed upstream of every consumer.

**Mechanism.** In `cloud_surface_density_cpu` (`solar_system_state.rs`):

```rust
mass = mass.max((shape - (threshold - 0.06)) * anvil_profile * storm_w);
```

The anvil gate multiplies the *value* that is then used as a **floor**. Outside
a storm anvil the product is exactly `0.0`, so the line degrades to
`mass.max(0.0)` — `mass` can never go negative anywhere on the planet.
`anvil_profile = smoothstep(0.62, 0.76, h)` is zero for the lower strata, so it
fired at essentially every height outside storm columns.

Fourteen lines later the realization is a smoothstep **centred on zero**:

```rust
let areal_fraction = smoothstep(-SUB_TEXEL_RMS, SUB_TEXEL_RMS, mass);  // ±0.035
```

A floor of exactly 0.0 lands on that band's midpoint → `areal_fraction = 0.500`
for clear sky, planet-wide. Attribution table, before the fix:

| coverage | shape | threshold | mass | areal | frac>0.5 |
|---|---|---|---|---|---|
| 0.1–0.2 | 0.562 | 0.926 | **−0.000** | **0.500** | 0.000 |
| 0.3–0.4 | 0.568 | 0.786 | **0.000** | **0.500** | 0.000 |

True `mass` for the first row is `0.562 − 0.926 − 0.006 = −0.370`; the trace
reports `−0.000`. `frac>0.5 = 0.000` proves it was not a minority of texels
going fully cloudy — *every* texel was pinned at exactly half.

**Why it hid for so long.** The derivation has three mirrors that are required
to stay in lockstep. All three shared the bad `max`, but they realize `mass`
differently:

| mirror | realization | floor visible? |
|---|---|---|
| `solar_system_state.rs` (producer → far tiers) | `smoothstep(-0.035, +0.035, mass)` | **yes — 0.5** |
| `clouds_compute.wgsl` (near marcher) | `smoothstep(0.0, edge_softness, mass)` | no — 0 maps to 0 |
| `fill_lut.rs` (calibration) | `smoothstep(0.0, edge_softness, mass)`, and `max` was guarded | no |

So the identical defect was harmless in the two consumers a developer looks at
from the ground, and catastrophic in the one that feeds every orbital
projection. That is exactly the near/far disagreement it presented as.

Worse: `fill_lut.rs` — the *only* guarded mirror — is the one that derives the
`fill_response` LUT for the other two. The far tier's calibration was
Monte-Carlo-fitted against a floor-free field and then applied to a producer
that had a 0.5 pedestal, which is a plausible mechanism for the residual tier
disagreement left open in BL-20260723T214730Z.

**Fix.** Blend by the gate; never `max` against a gate-scaled value:

```rust
let anvil_gate = anvil_profile * storm_w;
let anvil_mass = shape - (threshold - 0.06);
mass += (mass.max(anvil_mass) - mass) * anvil_gate;
```

Gate 0 leaves `mass` untouched, gate 1 applies the full anvil floor, and no
intermediate value can clamp. Applied to all three mirrors. `fill_lut`'s
`if anvil_gate > 0.0` guard was also replaced — it stopped the exactly-zero case
but not a tiny positive gate, which still floored `mass` at ~0.
`FILL_LUT_VERSION` 9 → 10, because the disk-cached calibration is fitted against
the strata cube this changes.

**Second finding, forced by the first.** With the floor removed the planet came
out at 0.34× authored coverage. The formation threshold
`threshold = 1.03 - 0.70·cov` had been **co-tuned with the bug**: `shape` has a
narrow distribution (σ ≈ 0.09 about 0.562), so that line sweeps almost entirely
outside the field's own support. Nothing formed on its own merits — the 0.5
floor was supplying the planet's cloud. The line is now *derived*, not fitted by
eye: for occupancy to equal coverage `c` the threshold must be the `(1-c)`
quantile of the `shape - vertical_narrow` comparand, and
`cloud_weather_probe`'s THRESHOLD FIT table measures that per decile and
least-squares fits it. Re-derive there after any change to `surface_shape` or
the vertical terms.

**Third fix, forced by the second.** With the threshold refitted, the deck
rendered as flat pancakes: `vertical_narrow`'s dome coefficients (0.04 / 0.42 /
0.30) had been scaled against the *old* threshold span of 0.70, so at h→1 the
cumulus term reached 0.42 ≈ **4.7σ** of `shape` and drove top-stratum occupancy
to ~0 — every column lost its upper half. This too was survivable only while the
mass floor pinned all four strata at 0.5. Rescaled by 0.31 (0.012 / 0.130 /
0.093), which two independent routes agree on: the threshold-range ratio
(0.217 / 0.70) and the σ argument (want ~1.4σ at the top). Occupancy now tapers
~32% → ~6% base-to-top — a dome, not a cut. **The ratios between the three
coefficients are authored per-type shape; only the common scale is derived.**

**Verified.** `strata/coverage` 1.296 → 0.668; clear sky on the disc 4.8% →
56.6%. Captures: the planet reads as ocean + land + coherent frontal cloud
systems instead of a white marbled disc, and the cruise deck has rounded lobed
tops instead of flat plates.

**Recurrence tell.** Run `cargo run --release -p thalos_runtime --example
cloud_weather_probe`. The defect signature is a **nonzero value in the 0.0–0.1
row of the TRANSFER table** together with `areal = 0.500` and `frac>0.5 = 0.000`
in the ATTRIBUTION table — a field pinned at exactly half is a clamp, never a
weather pattern. A healthy field has the 0.0–0.1 row near zero and `mass`
tracking `shape - threshold - vertical_narrow`.

**Standing rule this earns.** A gate that multiplies a value must not then have
that value used as a floor or a ceiling — the gate's zero becomes the clamp's
identity, which is a legal number rather than "no effect". Blend by the gate
instead. Where two mirrors of one derivation realize the result through
transforms with different zero-crossings, a shared clamp bug will be visible in
one and invisible in the other, so "it looks fine in the near view" is not
evidence that a shared term is correct.
