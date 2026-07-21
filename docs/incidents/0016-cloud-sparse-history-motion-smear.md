# INC-0016: Sparse cloud history smeared during camera motion

- **Status:** Fixed
- **Date:** 2026-07-21 (observed) / 2026-07-21 (fixed)
- **Severity:** visual
- **Surface:** surface cloud views while rotating the camera, strongest toward the sun (`just game runway`)

## Summary

Clouds left coarse rectangular trails during a camera pan and filled back in
only over several frames. The production renderer refreshed one class of a
rotating 3×3 pattern and attempted to reconstruct the other eight pixels from
nearest-hit history. That is valid for a screen-static view but not for a
moving translucent volume: one stored distance cannot carry the radiance
integrated over an entire old ray onto a different current ray. The renderer
now keeps sparse refresh only while screen-static and raymarches every pixel
during camera motion; coherent body-fixed history can stabilize, but never
replace, the fresh moving result.

## Symptoms

- Large horizontal and rectangular pieces of cloud lagged behind camera motion.
- The holes and trails slowly filled as the rotating 3×3 schedule revisited
  them.
- The problem was most obvious around the sun-facing cloud lobe, where forward
  scattering and sky contrast amplify stale radiance; the lighting direction
  was an amplifier, not the source.
- User reproduction: `just game runway`, rotate the view across the sun and
  cloud deck.
- Deterministic reproduction: `just compare cloud-motion cloud-reconstruction`.

## Evidence

The user's two 2026-07-21 live captures showed the exact 3×3-history signature:
screen-aligned blocks and delayed fill after a camera turn. A deterministic
near-sun preset then slewed the camera by 18° over 36 frames and captured while
it was still moving.

After correcting old-camera distance handling but retaining one-in-nine moving
refresh, the raw and dense-history variants were clean while sparse history
remained severely smeared:

```text
raw -> dense-history:  MAE 0.93/255, RMS 1.47/255
raw -> sparse-history: MAE 4.75/255, RMS 10.07/255, max 101
```

A two-phase moving checkerboard removed the largest blocks but still produced
visible broad ghosting even though history was at most one frame old:

```text
raw -> sparse-history: MAE 3.01/255, RMS 6.11/255
```

With a fresh march for every moving pixel, the production variant is visually
aligned with dense reconstruction and has no block trail in the full-resolution
frame or wipe:

```text
1920x1080 Baseline, 1280x720 cloud target
raw -> dense-history:      MAE 1.70/255, RMS 2.92/255
raw -> production-history: MAE 1.74/255, RMS 3.00/255
production GPU: mean 1.62 ms over mixed static/moving samples,
                moving-frame p95 2.73 ms (3.5 ms budget)
```

All three isolated captures completed without a shader or render-pipeline
error. The manifest and images live under
`tools/agent_scratch/screenshots/comparisons/cloud-motion/cloud-reconstruction/`.

## Hypotheses considered

1. **Cloud density or lighting instability.** Ruled out: raw full-frame marches
   stayed coherent under the identical camera slew. The near-sun lobe increased
   visibility but did not create the stale blocks.
2. **Any temporal blending during motion is invalid.** Ruled out as the primary
   cause: dense history, applied only to freshly marched pixels with depth and
   opacity validation, stayed visually coherent.
3. **Bilinear history mixed valid hits with the clear-distance sentinel.**
   Confirmed as one defect. Colour and distance are now selected as one coherent
   2×2 tap, preferring the nearest distance compatible with the expected old
   camera range.
4. **Old-camera distances were being fed back as current-camera ranges.**
   Confirmed as one defect. Reprojection validation now compares stored range
   in the old camera's metric. Correcting it reduced false matches but did not
   make sparse substitution valid.
5. **Nine-frame history age alone caused the smear.** Partly confirmed, but a
   one-frame checkerboard still ghosted. This isolated the deeper problem:
   nearest-hit history cannot reconstruct a translucent moving ray integral.

## Root cause

The sparse marcher treated a cloud like an opaque surface with one reusable
motion depth. On eight of nine pixels it skipped the current raymarch and
reused radiance attached to a previous camera ray. A volumetric pixel contains
emission/in-scatter and transmittance accumulated across a finite shell
interval, potentially through several density features; its nearest-hit range
does not describe the rest of that integral. Rotation therefore warped old
radiance into unrelated current rays. The rotating schedule made those stale
regions persist for up to eight frames, producing the block-shaped delayed
fill.

## Fix

- Added a deterministic `cloud-motion` screenshot preset and a typed
  `cloud-reconstruction` axis with raw, dense-history, and production variants.
- Kept the rotating 3×3 sparse schedule only when the camera matrices are
  screen-static.
- Forced every pixel to raymarch its current ray during camera motion. Moving
  history is now a bounded, current-dominant stabilizer, never a current-sample
  substitute.
- Made history colour and distance selection coherent and hit-aware, and
  validates stored depth in the old camera's range metric.
- Made the temporal-off diagnostic truly disable moving history and recorded
  the actual reconstruction mode in each JSONL probe.
- Cleared stale per-variant reports before comparisons so timing evidence comes
  from the current run only.

## Prevention & recurrence signals

- Standing invariant: a single-depth temporal cloud history may amortize work
  only while the view is screen-static. During camera motion every current ray
  must be evaluated unless a future reconstruction stores conservative
  multi-layer/interval information capable of representing the translucent
  integral. See [clouds.md §3.3](../clouds.md#33-sampling-and-temporal-reconstruction).
- Keep `just compare cloud-motion cloud-reconstruction` as the first diagnostic
  for screen-aligned blocks, horizontal trails, slow fill, or artifacts that
  strengthen around the sun.
- Reject the comparison if any variant logs a shader/render-pipeline error;
  images alone are not proof of a valid run.
