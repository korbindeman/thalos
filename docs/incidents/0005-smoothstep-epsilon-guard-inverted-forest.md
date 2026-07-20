# INC-0005: Forest painted onto the driest ground — `.max(EPSILON)` denominator guard inverted every descending-edge smoothstep

- **Status:** Fixed
- **Date:** 2026-07-20 (observed) / 2026-07-20 (fixed)
- **Severity:** visual
- **Surface:** macro terrain albedo everywhere it is consumed — the baked tile
  albedo (ground macro tint), the distant-body impostor, the map view, and the
  `just map` export. Latent since the TM-P1/TM-P2 macro palette landed.

## Summary

The `just map` biome-class stats (BL-11) contradicted the moisture field they
were derived from: the subtropical belt had mean dryness 0.69 with half its
land above the steppe threshold, yet classified **85 % forest** — and the
planet had zero steppe/desert/tundra. Root cause: `procedural.rs`'s local
`smoothstep` guarded its denominator with `.max(f64::EPSILON)`, which turns
every **descending-edge** call (`edge0 > edge1`, the WGSL-style inverted ramp)
into a hard step that returns **1.0 above `edge0`** — the exact inverse of the
intent. The macro palette's forest term `smoothstep(0.42, 0.18, dryness)` had
therefore been painting closed-canopy forest onto all ground *drier* than
0.42 and none onto wet ground, ever since the palette landed. Fixed the
helper (sign-preserving denominator, WGSL parity), flipped the call site to
the house-style ascending form, and corrected three sibling copies of the
same broken idiom in `body_render` before they could bite.

## Symptoms

- `just map` biome stats: steppe / desert / tundra ≈ 0 % of land while the
  dryness histogram showed 18 % of land above the steppe threshold and 3.4 %
  above the desert threshold.
- Per-latitude table: the 15–30° dry belt (mean dryness 0.69) classified 85 %
  forest — dark canopy tint on the driest land, visible as dark green filling
  the subtropical belt in `target/world_biomes.png` / `world_map.png`.
- In-game (unnoticed until the map tool existed): from orbit the dry belts
  read *darker* green than the wet equator, i.e. the moisture → palette
  transfer ran backwards for the forest term.

## Evidence

`just map` (WORLD_PROJ=equirect) before the fix:

```
land dryness: p10 0.30  p50 0.49  p90 0.80  |  >0.55 39.2%  >0.72 18.4%  >0.88 3.4%
per-|lat| band (land): mean dryness | %>0.72 | top biome
  15–30°  0.69 | 49.3% | forest 85%      <-- forest weight is 0 above dryness 0.42; impossible
```

The classifier (`classify_macro`) and the albedo read the *same*
`macro_band_ts` evaluation of the *same* moisture value as the histogram —
so the contradiction could only live inside the band-weight computation
itself. Hand-tracing `macro_band_ts` at dryness 0.8 gave Steppe; the only
term whose sign depends on edge order was `forest: smoothstep(0.42, 0.18,
dryness)`. Reading the helper:

```rust
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0).max(f64::EPSILON)).clamp(0.0, 1.0);
    //                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ -0.24 → EPSILON
```

With descending edges the denominator is negative, `.max(EPSILON)` replaces
it with `2.2e-16`, and `t` explodes to ±huge → the clamp makes the function
`step(x > edge0)` — full-on exactly where the inverted ramp should be zero.

## Hypotheses considered

1. **Classifier weight-expansion bug** (my new `classify_macro` disagreeing
   with the albedo chain) — ruled out by hand-tracing both against the same
   `MacroBandTs`; they share the struct, and the albedo itself was also
   rendering green on dry land (the true-colour map showed no tan), so the
   inputs, not the expansion, were wrong.
2. **Moisture field too wet / belt missing the land** — ruled out by adding
   the dryness histogram + per-latitude table to `world_map`: the field was
   fine (18 % of land above 0.72, concentrated exactly in the 15–45° belts).
3. **Stats accounting error in the example** — ruled out: histogram and
   class counts read the same `Px` buffer.
4. **`fbm` sign convention** (unsigned noise biasing wet) — ruled out by
   reading `fbm`: signed Perlin, mean 0.
5. **Broken smoothstep on descending edges** — confirmed by reading the
   helper; the `.max(EPSILON)` guard only ever matters when the denominator
   is negative or zero, i.e. precisely on descending edges.

## Root cause

A defensive `.max(f64::EPSILON)` on the smoothstep denominator, presumably
added to avoid division by zero, silently breaks the WGSL/GLSL
descending-edge convention (`smoothstep(hi, lo, x)` = inverted ramp) by
collapsing the negative denominator to +EPSILON. Every descending-edge call
becomes an **inverted hard step**. `procedural.rs` had exactly one such call
— the macro forest term — so the whole macro moisture → forest transfer ran
backwards, and no palette/belt tuning could ever produce deserts (the forest
term claimed all dry ground first). The same broken idiom existed in three
more copies (`body_render`'s `pipeline.rs`, `scatter.rs`, `vegetation.rs`),
currently harmless because all their call sites use ascending edges; three
other copies (`landcover.rs`, `synthetic.rs`, `impostor/bake.rs`) were
correct — the codebase had both variants side by side.

## Fix

- `procedural.rs::smoothstep`: sign-preserving denominator with an explicit
  degenerate-edges branch (WGSL parity, descending edges work).
- The forest term rewritten in the house-style ascending form
  (`1.0 - smoothstep(0.28, 0.58, dryness)`) — edges also re-aligned to the
  ground shader's `vegetation_color` forest window (0.28–0.58; the old
  0.18–0.42 was a second, stricter forest threshold the ground never used).
- The three broken-but-not-yet-biting copies in `body_render` replaced with
  the correct variant (no behavioral change today — verified every call site
  in those files uses ascending edges).
- Landed inside the TM-P3 rebalance, `GENERATOR_VERSION` 12.

## Prevention & recurrence signals

- **Standing rule:** a WGSL-mirror `smoothstep` must support descending
  edges — never "fix" the denominator's sign. If a shared helper is ever
  extracted for these mirrors, this is why. (Code comments now sit on every
  corrected copy; grep `max(f32::EPSILON)` / `max(f64::EPSILON)` near a
  subtraction to find new occurrences.)
- **Tell:** a biome/mask/coverage field whose *statistics* look right while
  the *rendered classes* contradict them (here: dryness histogram vs class
  map). The `just map` dryness + per-latitude diagnostics were added
  precisely to expose this split — if classes and their driving scalar ever
  disagree again, suspect a transfer-curve helper before the field.
- Descending-edge smoothsteps silently misbehaving elsewhere would show as
  a band/threshold that acts as a hard on/off switch at `edge0` instead of
  a ramp between the edges.
