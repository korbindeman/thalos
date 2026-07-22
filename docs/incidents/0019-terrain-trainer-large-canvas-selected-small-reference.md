# INC-0019: Terrain trainer selected a patch smaller than its validation canvas

- **Status:** Fixed
- **Date:** 2026-07-21 (observed) / 2026-07-21 (fixed)
- **Severity:** crash
- **Surface:** `thalos_terrain_train smoke` after paid CUDA preflight

## Summary

The first corrected CUDA smoke trained successfully and then panicked before
writing its validation report. Validation had implicitly selected a 64² corpus
patch for a configured 128×96 overlap canvas, bypassing the existing synthetic
full-canvas generator. Validation references are now opt-in: a named source or
the explicitly requested first training patch is loaded and checked, while an
unspecified reference generates the configured synthetic canvas.

## Symptoms

- CUDA initialized and epoch checkpoints were written, but the command exited
  during validation.
- Repro: run `smoke` with `configs/smoke.toml` (64² patches, 128×96 canvas).

## Evidence

```text
thread 'main' panicked at tools/terrain_train/src/validate.rs:72:5:
assertion failed: sample.height.size >= full_size
```

The remote output directory contained both epoch checkpoints, proving CUDA
training had completed. The selected validation patch was 64² while
`full_size = max(128, 96)`.

## Hypotheses considered

1. **CUDA/NVRTC remained misconfigured.** Ruled out by successful backend
   initialization, GPU telemetry, and completed epoch checkpoints.
2. **The overlap sampler could not generate a canvas larger than one window.**
   Ruled out by the existing `reference: None` branch, which generates a
   full-size synthetic sample specifically for this case.
3. **Reference selection made that branch unreachable.** Confirmed:
   `validation_reference` silently fell back to the first validation corpus
   patch whenever no explicit reference option was configured.

## Root cause

Held-region evaluation added automatic fallback to a validation-split sample.
That changed the meaning of an omitted reference from “generate an independent
synthetic canvas” to “use one patch,” even when the requested canvas was larger
than every prepared patch.

## Fix

`validation_reference` now returns a sample only for
`reference_source_id` or `use_first_training_sample`. An omitted selector
returns `None` and preserves the synthetic overlap-canvas path. A missing named
source now returns an error instead of silently evaluating synthetic data. The
128×96 velocity-prediction CPU smoke completes end-to-end.

## Prevention & recurrence signals

- A validation source is always explicit; omission means synthetic canvas.
- “Training completed, then `sample.height.size >= full_size` panicked” is the
  recurrence tell.
- Keep the default smoke canvas larger than one patch so this contract remains
  exercised by the normal end-to-end probe.
