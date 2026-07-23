# INC-20260723T221126Z — cloud tiers sampled diverged weather cubes (runtime cube re-upload scrambles GPU layout)

## Symptom

The volumetric near-cloud tier and the far/impostor cloud projections never
spatially agreed: ascending from a site with a visible deck, the impostor
showed clear sky (or an unrelated pattern); descending into an
impostor-clouded region found different volumetrics. Persisted through every
producer/threshold/opacity calibration of the 2026-07-23 cloud fidelity
rounds. User verdict: "the impostor and the actual clouds don't really line
up at all."

## Root cause

Two copies of the canonical weather/strata cubemaps existed on the GPU:

- **Spawn cubes** — created once per body at world spawn via
  `cloud_weather_image` with data attached at asset creation. Content
  correct. Read by the impostor (`SolidPlanetMaterial`) and the cloud
  composite's far tier.
- **Compute cubes** — created zero-filled at plugin setup, then filled at
  runtime by `sync_cloud_weather_map` mutating `image.data` in place. **The
  runtime re-upload scrambles the cube's face/mip layout on the GPU** — the
  resulting field is garbage that still looks like plausible weather. Read
  exclusively by the near-volume raymarcher.

The near tier therefore flew through a corrupted field while every far
consumer rendered the correct one. No calibration could reconcile them.

A wholesale-replacement variant (`images.insert` of a freshly built image on
the existing handle) also corrupts — it rendered the planet disc as
misaligned rectangular face blocks (user screenshot, the decisive evidence).
The exact Bevy/wgpu mechanism (re-upload path vs `TextureDataOrder`/mip
framing for cube textures) is still to be pinned with a minimal repro; it is
plausibly an upstream bug.

## Fix

`sync_cloud_weather_map` (data mutation) is replaced by
`sync_cloud_weather_binding` (pure handle rebind): body spawn registers each
body's spawn-uploaded cube handles in `BodyCloudCubes`, and the compute
pass's `CloudsImage` bindings are swapped to the active body's spawn cubes.
No cube data is ever re-uploaded at runtime. The compute texture bind group
is rebuilt from `CloudsImage` every frame, so the swap is immediate. One
correctly-uploaded field now serves every consumer.

Verified: same-framing tier A/B (`far-only` ownership-bypass diagnostic) —
both tiers agree the probe site is clear, matching the impostor; cruise
capture shows near puffs continuing into the far shelf without a pattern
seam; runway ground view matches the impostor's clear claim; planet disc
clean.

## Why it evaded diagnosis

- The corrupted field is statistically cloud-like; the near tier's output
  looked plausible in isolation for weeks.
- The far tier was a near-uniform veil until the 2026-07-23 regime producer
  gave the field real structure — only then did misalignment become visible.
- Shared-code reasoning kept "explaining" agreement: both paths used the
  same producer bytes, the same builder, byte-identical rotation quats
  (logged), matching Rust/WGSL uniform layouts — all true, and all
  irrelevant, because the divergence happened inside the GPU upload.

## Prevention / recurrence tells

- **Never mutate cube-texture asset data in place**; version by creating a
  new asset through `cloud_weather_image` and swapping handles (CLOUD-7's
  advection must follow this rule).
- The `far-only` tier diagnostic (`THALOS_SCREENSHOT_CLOUD_TIER`, ownership
  bypassed) plus the same-framing fill/overlap measurement in
  `artifacts/visual/runs/cloud_fill/` is the standing registration test.
- Tell: any consumer of a runtime-refreshed cubemap disagreeing spatially
  with a spawn-time consumer of "the same" data — suspect the upload, not
  the math. Rectangular face-block artifacts on a cube-sampled projection
  are the smoking gun.

## Falsified along the way (do not re-derive)

Quat convention/conjugate inversion (quats bit-identical); wind-advection
misregistration (~66 m in cold runs); layer-relative strata dead zone (a
real, separately fixed bug — thin decks between fixed strata heights); chord
overlap dilution (real, fixed via analytic per-segment clip); Rust/WGSL
uniform field-order mismatch; `surface_density_coupling` not applied.
