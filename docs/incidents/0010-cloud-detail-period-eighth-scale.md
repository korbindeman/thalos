# INC-0010: Cloud detail collapsed into stipple

- **Status:** Fixed
- **Date:** 2026-07-21 (observed) / 2026-07-21 (fixed)
- **Severity:** visual
- **Surface:** `just screenshot cloud-runway`, `cloud-cruise`, and `cloud-interior`

## Summary

The first true-3-D cloud pass broke coherent cloud bodies into fine stipple and
micro-cloudlets, especially along grazing rays. Temporal accumulation and
shadow jitter made the artifact easier to notice but did not cause it. The
authored `detail_scale_m = 450` described one erosion cell, while the generated
volume's detail channel contained eight cells across its tile; sampling the
whole tile with a 450 m period therefore produced ~56 m cells, smaller than the
200–500 m view steps. Sampling that channel with a `450 m * 8` tile period
restored the authored feature size and coherent boundaries.

## Symptoms

- Cloud silhouettes appeared as ordered fine beads/stipple instead of
  cauliflower-scale erosion.
- Disabling temporal history exposed more noise but retained the same spatial
  structure.
- Replacing stochastic shadow offsets with deterministic centred samples did
  not remove it.
- Repro: capture `cloud-runway` or `cloud-cruise` with the true-3-D volume and
  a detail tile period equal to `detail_scale_m`.

## Evidence

Controlled captures changed one factor at a time: temporal history off,
deterministic shadow samples, deterministic view phase, reduced base octaves,
and additive base formation. None removed the minimum feature. The generated B
channel was a single Voronoi octave at frequency 8, making the actual cell size:

```text
450 m authored tile period / 8 cells = 56.25 m effective detail
grazing view step = 200–500 m
```

Changing only the sampled B-channel period to `detail_scale_m * 8` removed the
stipple while preserving the same weather, lighting, step schedule, temporal
path, and generated volume.

## Hypotheses considered

- **Temporal reprojection:** ruled out because temporal-off retained the same
  structure, only noisier.
- **Shadow jitter:** ruled out by deterministic centred shadow taps.
- **View-ray jitter/banding:** ruled out with a deterministic view phase.
- **Too many base octaves:** reducing them changed cloud occupancy but not the
  offending minimum scale.
- **Formation blend:** additive formation restored solid cores but the edge
  stipple remained until the physical period was corrected.
- **Detail unit mismatch:** confirmed by the one-factor `×8` period capture.

## Root cause

The runtime interpreted an authored feature size as a texture tile size. The
stored procedural channel is not one cell per tile: frequency 8 places eight
primary Voronoi cells along each axis. This silently divided the physical scale
by eight and pushed density variation below the marcher sampling theorem.

## Fix

The detail volume now wraps over `clouds_detail_scale_m * 8.0`, so one generated
cell measures the authored 450 m. Base shape and detail retain separate physical
periods; high-frequency detail is applied only near the density boundary, while
solid cores use the lower-frequency Perlin/Worley basis.

## Prevention & recurrence signals

- Authored noise scales name physical features, not implementation tile periods;
  generated channel frequency must be folded into the sampling transform.
- `docs/clouds.md` records this as a density-scale invariant.
- A recurrence presents as stable micro-cloudlets whose world size tracks the
  marcher step and survives temporal/shadow-jitter A/B tests.
