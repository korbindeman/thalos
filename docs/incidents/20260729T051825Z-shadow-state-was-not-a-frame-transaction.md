# INC-20260729T051825Z-shadow-state-was-not-a-frame-transaction: maps and receivers could use different frames

- **Date:** 2026-07-29
- **Surface:** moving cameras, floating-origin cell crossings, zoom/altitude changes, and accelerated solar time

## Symptom

Shadows looked correct in still captures but could shimmer, briefly shift, or
change softness while the camera or simulation moved. Some edges were also
softer than their caster contact implied.

## Root cause

Four independent discontinuities accumulated:

1. The floating-origin resource was written in `Update`, before camera drivers.
   On a cell crossing it could trail the frame by one 1 km cell.
2. Cascade cameras were placed in `Update`, while grass/tree/rock material
   drivers independently copied `SunShadowState` in the same schedule without
   ordering. A map could therefore be rendered with one matrix while a receiver
   sampled the previous one.
3. Footprint scale jumped between powers of two, the shadow-only sun advanced in
   0.1° steps, and inner cascades handed off at a hard boundary. Each mechanism
   moved many edges in one frame by design.
4. PCSS rotated its sparse tap disk from a per-fragment hash and treated the
   sun's full angular diameter as a filter radius. Without temporal AA the hash
   is visible noise, and the diameter/radius mismatch doubled penumbra width.

The contact tier had a related staging error: it was produced after opaque
shading from copied depth and consumed one frame later, and only the legacy
UDLOD material received it. That made it unsuitable as a stabilizer for the
default tile renderer.

## Fix

Shadow placement now runs in `PostUpdate`, after camera motion and immediately
before transform propagation. `RealSpaceOrigin`, cascade transforms, and the
craft mirror are a chain; one `Last` fan-out writes that exact block and map set
to all runtime-owned receivers before extraction.

Footprint changes are wall-clock smoothed, the shadow rig follows the continuous
lighting sun, its light basis is transported continuously, and adjacent
cascades overlap and cross-fade. PCSS uses a deterministic Vogel disk, the
physical solar angular radius, and a bilinear 2×2 contact kernel.

The contact pass now reads the current Bevy depth prepass (single-sample or
MSAA), runs before opaque shading, and is bound to the default tile material.

## Recurrence signal

`thalos::diagnostic::shadow` emits `stability_gauge` once per second.
`origin_frame_error_m` must remain below 0.01 m. `just diag` raises
`shadow_frame_desync` with the bad/total sample count and worst error if the
camera cell and cascade render frame ever diverge again.
