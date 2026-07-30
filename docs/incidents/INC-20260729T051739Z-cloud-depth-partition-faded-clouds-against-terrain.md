# INC-20260729T051739Z-cloud-depth-partition-faded-clouds-against-terrain: the near-tier depth partition rendered solid cloud as smoke wherever terrain sat behind it

- **Date:** 2026-07-29 · **Surface:** `just screenshot thin-clouds` (any low/mid-altitude view
  with cloud in front of ground; user-reported from a saved viewpoint)

## Symptom

Cumulus with open sky behind them read solid; the *same* clouds went see-through in every
part that had ground behind them, badly enough to read a distant mountain ridge through what
should have been a dense core. The tell is that the transparency tracked the **backdrop**,
not the cloud: a hard horizontal edge ran across the cloud bases exactly at the terrain
horizon, and clouds snapped back to solid past it. That is also why it was angle-dependent —
framings with nothing behind the deck looked fine.

## Root cause

The near volumetric march runs in `RenderGraphSystems::Begin`, before any per-view pass, so
it has no scene depth and integrates its **whole in-shell chord** — including cloud that is
genuinely behind a mountain. `cloud_composite.wgsl` therefore had to subtract that back out
from what the march *does* hand over, which was one scalar: the nearest cloud hit.

One distance cannot express a volume's occlusion against a depth buffer, so `near_visibility`
approximated it as a linear opacity ramp over a **constant 5.4 km** denominator (the coarsest
cell period). The constant is the defect. A 900 m deep cumulus with the ground a kilometre
behind it has *all* of its mass in front of that ground and should be fully opaque; the ramp
drew it at `1000/5400 ≈ 0.19` of its opacity. Every cloud with terrain within 5.4 km behind
it was faded in proportion to how close that terrain was — worst at the cloud base, which is
exactly where the ground behind is nearest.

The predecessor comment shows the ramp was already understood to be over-fading (it had been
narrowed from the full band chord to 5.4 km for the same "washed out horizon band"
complaint). Narrowing a constant that should not have been a constant only moved the range
over which the bug applies.

**Ruled out, instructively:** cloud density, coverage and march cadence were all suspected
first and are all innocent for this symptom. The decisive test was a one-line probe forcing
`near_visibility` to 1.0 and re-shooting the same viewpoint — the clouds went solid with no
other change.

**Not the whole story.** Forcing the partition off moved the *left* cloud in the reference
frame by ~10/255 mean, and the mid-distance cloud by **0.12/255** — i.e. the partition was
already inactive there (its terrain sits more than 5.4 km behind). That cloud's
see-through is a second, independent cause: the near tier's own integrated optical depth is
low in cloud bodies at that range. Tracked separately; do not expect this fix to address it.

## Fix

The march now reports the **span** its extinction actually occupies, not just where it
starts. `cloud_distance_texture` widens R32F → RG32F: `r` stays the nearest hit (unchanged,
so temporal reprojection keys on the same value), `g` carries `slab_far_distance` — the far
end of the uniform slab whose first distance-moment matches the ray's real extinction
profile, accumulated as `tau_total`/`tau_moment` in the march loop.

`near_visibility` then partitions in **optical depth** rather than opacity, over the span the
march measured on that ray:

```
frac    = clamp((scene_t - near) / (slab_far - near), 0, 1)
near_vis = (1 - exp(-tau * frac)) / (1 - exp(-tau))
```

This removes the class of bug rather than the instance: the limits are now exact by
construction — terrain beyond the cloud's own back face gives `frac == 1` and therefore
`near_vis == 1` for *any* transmittance, terrain in front gives 0 — and no constant is left
to be wrong at some other scale. Weighting the moment by optical depth (rather than taking
the last hit) also keeps a single wispy tail sample, or a second cloud further down the same
ray, from stretching the slab across empty air.

Cost: +7 MiB of cloud targets at 1280×720 (105.2 → 112.2 MiB, `thalos.cloud_probe.v1`
`memory.total_bytes`). No measurable GPU cost — 5.03 ms mean / 5.32 p95 after vs 5.06 / 5.39
before, on the same viewpoint and the development 4070 Ti.

## Recurrence signal

Cloud opacity that correlates with what is *behind* the cloud rather than with the cloud —
solid against sky, translucent against ground, with the transition pinned to the terrain
horizon. Any future summary the composite reads out of the march must keep the two limits
above exact; a fixed-length denominator anywhere in that partition is the same bug again.

Evidence for this fix: `artifacts/visual/runs/cloud-ghost-partition/` (`probe_no_partition`
is the diagnostic extreme, not a target — it draws cloud over terrain that occludes it).
