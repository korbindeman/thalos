# INC-20260809T190911Z-clouds-painted-over-analytic-ocean: the cloud compositor did not know the ocean existed

- **Date:** 2026-08-09 · **Surface:** `just screenshot ocean`

## Symptom

Small cloud volumes appeared across the water below the geometric horizon, as
though they were resting on the waves. Clouds above the horizon looked normal.

## Root cause

The ocean and clouds are separate fullscreen `Transparent3d` composites. Their
pinned order correctly draws ocean and then clouds, but order alone cannot decide
which cloud samples are physically in front of the water.

`CloudCompositeMaterial` clipped both cloud tiers only against
`SceneDepthImage`. That texture is copied after opaque rendering and before the
transparent phase, so it contains terrain, structures, and craft but cannot
contain the analytic `BodyOceanMaterial` pass. Over water it therefore reported
the seabed behind the sea surface. Cloud extinction between the ocean hit and
that deeper opaque hit survived the partition and was blended over the water.

The concurrent ocean-mechanism extraction also exposed a separate capture
validity defect: three new `BodySkyExtra` vectors were absent from the WGSL
`CloudCompositeParams` mirror, shifting every later field. Restoring those
fields changed cloud structure but did not remove the water-surface puffs,
which ruled the layout mismatch out as the occlusion cause.

## Fix

`thalos::ocean_waves::ocean_sphere_hit_distance_m` is now the one
cancellation-safe analytic-sphere intersection used by both the ocean shader
and later transparent consumers. The cloud compositor takes the nearer of the
opaque scene hit and this ocean hit before partitioning either cloud tier.
Opaque terrain naturally remains authoritative wherever land stands above sea
level; the ocean wins over the seabed behind it.

## Recurrence signal

Any cloud shape visible on water below the geometric horizon means a transparent
surface has again been omitted from the cloud pass's effective scene distance.
`just screenshot ocean` is the deterministic probe: the water must contain no
cloud fragments while the cloud bank above the horizon remains visible.

Matched evidence is in
`artifacts/visual/runs/ocean-cloud-occlusion/01-layout-baseline.png` and
`02-ocean-depth-fix.png`.
