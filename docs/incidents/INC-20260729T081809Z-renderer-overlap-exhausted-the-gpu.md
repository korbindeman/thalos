# INC-20260729T081809Z — Renderer overlap exhausted the GPU

- **Date:** 2026-07-29
- **Status:** overlap mechanism fixed; not the complete adapter-loss diagnosis
- **Surface:** interactive game plus persistent/headless capture

## Symptom

The GPU became unusable until Windows rebooted. At 08:43 a capture host logged
`DeviceLost: Out of memory`; at 09:45 the interactive game aborted with
`0xc0000409`. Capture safety also stopped two hosts after each crossed 8 GiB RSS.

## Diagnosis

Both failures had two Thalos renderers active:

- game PID 14328 overlapped capture PID 29244 immediately before the explicit
  08:43 GPU OOM;
- game PID 12380 started while a persistent capture host was resident, then
  overlapped capture PID 22920 before the 09:45 abort.

The tile-share gauge saw the peers and divided its allowance, but that allowance
covers tile bytes—not each process's duplicated images, mesh slabs, shadows,
volumetrics, pipelines, or boot transients. A single-renderer session in the
same diagnostic window filled its working set, plateaued for about 18 minutes,
and did not crash. Windows recorded no WHEA hardware error, display-driver TDR,
or live-kernel hardware report around either incident.

The `memory_growth` finding was not evidence of a leak: it added tile live bytes
to the mesh slab allocation that contains them and measured peak minus the
pre-streaming first sample. The flagged session settled and stayed flat.

## Correction after the 11:20 single-renderer loss

The overlap evidence was real, and allowing two renderers remains unsafe, but
it was not sufficient to explain every GPU loss. At 11:20 the adapter vanished
again with exactly one canonical renderer (`instances=1`) and no capture host.
That recurrence refutes the stronger conclusion that overlap was the sole root
cause. See
[INC-20260729T092010Z](INC-20260729T092010Z-single-renderer-nvidia-adapter-loss.md).

## Fix

`thalos_diagnostics::renderer_lease` is now the shared process boundary. The
game and capture host acquire one machine-wide OS lease before Bevy/wgpu starts.
A competing launcher exits with code 4 and an owner record instead of creating
a second device. `just game` stops a resident capture host first; capture
classifies a refusal as `renderer busy`, with no retry or GPU quarantine.

## Recurrence tell

- `residency_gauge.instances` must remain 1 in canonical game/capture sessions.
- Starting a capture while the game is open must lead with
  `capture launcher exited: renderer busy`, create no runtime session, and
  leave the game healthy.
- `just game` with an idle persistent capture host must stop that host before
  the interactive renderer starts.
- Any overlapping canonical runtime sessions are a lease bypass, not a budget
  tuning problem.
- A loss with `instances=1` belongs to the single-renderer adapter-loss
  incident; do not weaken the lease or retune peer sharing to explain it.
