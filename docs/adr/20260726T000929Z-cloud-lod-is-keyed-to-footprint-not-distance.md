# ADR-20260726T000929Z — Cloud LOD is keyed to projected footprint, not camera distance

**Status:** accepted, 2026-07-26.
**Extends** [ADR-20260725T222409Z](20260725T222409Z-cloud-cells-are-an-analytic-surface-field-lod-never-renders-the-mean.md)
(cloud cells are an analytic surface field). **Supersedes** the banded
distance ladder and the entry-distance ownership window of BL-20260724T003705Z.

## Context

The user's ascent screenshots (2026-07-25) showed three defects that looked
unrelated and were one thing:

- mid-ascent cells carried a **horizontal comb** across them;
- above ~300 km the volumetric clouds **faded out entirely**;
- what replaced them was a smooth grey blanket that rendered **nothing at all**
  over regions the volumetric tier had just drawn as solid cells.

All three follow from the LOD ladder being keyed to **camera distance**. From
orbit, distance and footprint diverge completely:

- Bevy's default 45° vertical fov on a 1280×720 cloud target is 0.00115 rad per
  pixel. At 300 km a pixel covers **345 m**, so a 5.4 km cell is ~12 pixels
  across and the finest cell octave is still ~2 px.
- The old ladder read "300 km" and switched to 4800 m steps — roughly 10 samples
  through the whole deck, which is the comb — then handed the frame to the far
  estimator wholesale (`ray.start > CLOUD_MARCH_ENTRY_FADE_END_M`), which for a
  nadir view is just *altitude*.
- The far estimator renders `E[opacity | occupancy]` per texel. Over a
  10 %-occupancy region that is a ~0.09 alpha wash — visually nothing — while the
  near tier had correctly drawn a few *solid* cells there. This is the round-9
  "occupancy is not opacity" error arriving from the far side.

The geometry says the volumetric march was never the expensive part at altitude:
`get_ray` already skips clear air analytically to the shell entry, and a
near-nadir orbital ray crosses a 10.5 km shell in ~10–15 km of path. What is
expensive is *grazing* chords, which is a function of ray angle, not altitude.

## Decision

**Projected footprint (and the ray's own geometry) drives cloud LOD everywhere.
Camera distance does not appear in the density function at all.**

1. **Step law** — `cloud_march_step_m(t, pixel_angle, radial_rate)`. The
   footprint term is a FLOOR (never sample finer than a pixel, with a 600 m
   near-field minimum so runway cost is unchanged); two resolution terms are
   CAPS and win where they disagree: a radial cap (350 m of *vertical* travel,
   so a thin deck is never crossed in a handful of samples) and a cell cap
   (1800 m ≈ ⅓ of the coarsest cell period, because a grazing ray must resolve
   the cell, not the profile).
2. **Ownership** — the far projection owns a whole ray only once cell-scale
   morphology is genuinely SUB-PIXEL (`cloud_far_ownership`, 2.2 → 5.4 km of
   footprint). On Thalos that is ~1900–4700 km altitude, so **the entire flight
   envelope is volumetric**. The far tier keeps the per-distance tail past the
   marcher's probe frontier, which only grazing rays reach.
3. **The one filter scale.** `filter_m` = max(refine step, footprint) is the sole
   LOD input to `get_cloud_map_density`, driving the cell field's octave fade,
   the Cartesian sub-cell sculptor's retirement, erosion, and the formation-edge
   width. The `coarse` ladder is deleted.

## Rejected in the same change, by capture

**A chord-length budget floor on the step** (`step ≥ chord / budget`), with the
reach cap raised to 600 km so the marcher always completed its chord and no
frontier existed at all. It is the wrong shape: it makes the step a function of
how long the ray happens to be. On a ground-level horizon ray (chord ~600 km)
it produced **~9.7 km steps through a 1.15 km deck** and rendered the entire
distant band as horizontal slabs — worse than the artifact being fixed.

Budget belongs in the REACH, not in the step. The frontier is therefore
retained, now integrating the footprint law in closed form
(`cloud_march_stop_m`: uniform at the floor, geometric while the footprint
governs, uniform at the cell cap). The resolution caps are deliberately not
mirrored in it — they only ever make the real step smaller, and they bind
exactly where the frontier is irrelevant (steep rays finish their short in-shell
segments long before the budget runs out).

## Consequences

- Measured on the development RTX 4070 Ti, 1280×720 cloud target: cruise
  **1.98 ms mean / 3.33 ms p95**, and a 400 km nadir view **1.31 ms / 1.94 ms** —
  the orbital framing is *cheaper* than cruise, exactly as the chord geometry
  predicts. Both inside the provisional 3.5 ms target.
- Verified by capture at 150 km, 400 km and 900 km: individually resolved cells
  across the whole visible hemisphere out to the limb, no blanket, no fade-out,
  no ownership seam. Runway and cruise unchanged.
- Whole-disc framings are untouched: at ~14 km/pixel every cell octave is
  correctly below the sampler, so the disc shows the weather producer's
  15–25 km smears. That is a producer question, not a renderer one.

## Standing rule

**Distance is not footprint.** Any cloud LOD term keyed to camera distance is a
bug waiting for an orbital framing: at 300 km a pixel covers 345 m, and the deck
you are "far" from is the one you can see best. Key LOD to the projected
footprint and the ray's own geometry, and put the probe budget in the reach.
