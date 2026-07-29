# INC-20260729T014529Z — Capture OOM retry stampede

- **Date:** 2026-07-29
- **Status:** fixed; client tests pass, runtime verification requires a healthy GPU
- **Surface:** persistent and cold headless capture

## Symptom

Parallel screenshot work made the workstation pause for seconds. A queued
`tex-look` capture printed that another client owned the lane, then started a
renderer, hit four wgpu `Out of Memory` errors, restarted once, and failed.
Another waiting `thalos-8-km` capture immediately started afterwards.

Windows/NVIDIA then reported `GPU is lost; reboot the system to recover`. The
last exact capture host reached about 12.6 GiB resident RAM and 70 threads before
it was stopped.

## Diagnosis

The client lock worked: renderer processes were not intentionally concurrent.
Four other facts combined:

1. Both saved viewpoints recorded a 3840×2160 desktop. Headless replay treated
   that metadata as an implicit request for a 4K render target and 2560×1440
   cloud targets.
2. Capture inherited the interactive renderer's 4 GiB tile-mesh allowance,
   despite sharing the workstation with the desktop and agent work.
3. Any dead capture host was classified as recoverable. The client did not
   distinguish an ordinary stale host from GPU OOM/device loss, so it immediately
   launched the renderer again.
4. The lock serialized only the current request. Once that failed and released
   ownership, the next already-waiting agent was free to repeat the fatal boot.

The `another capture client` line was therefore not evidence for multiple
renderers. It was evidence that the queue worked; the retry and lack of a shared
failure circuit breaker were the stampede.

## Fix

- Recorded viewport remains viewpoint metadata, but implicit headless replay
  fits it inside 1920×1080. `--size` is the explicit native/high-resolution path.
- Headless capture defaults to a 2 GiB tile budget and an 8 GiB host-RSS ceiling.
- OOM, device loss, submission timeout, and RSS runaway are terminal and never
  receive the generic one-shot restart.
- A fatal resource result writes a shared quarantine. Ordinary OOM/RSS pressure
  cools down for five minutes; device loss blocks the remainder of the current
  OS boot and clears automatically after reboot. Waiting agents fail without
  starting a renderer; `status` exposes the reason.
- Queue output now says what actually happens: compatible camera poses reuse one
  renderer sequentially.

## Recurrence tell

- A 4K saved viewpoint without `--size` must announce a 1920×1080 safety replay.
- Fatal GPU/resource logs must produce **zero** automatic renderer restarts.
- A second client during the cooldown must report quarantine without adding
  another `Running thalos_capture_host` line.
- A healthy host exceeding the RSS ceiling must be terminated before it can
  consume the workstation.

## Follow-up: OOM was accidentally promoted to reboot-long quarantine

The first circuit-breaker implementation checked generic `DeviceLost` text
before OOM. That ordering was wrong for WGPU's actual failure wording:

```text
Caught DeviceLost error: Unknown Out of memory
Quitting the application due to DeviceLost RenderError
```

The device object is torn down because allocation failed, but that is not the
same evidence as `Unknown Device is lost`. The classifier therefore assigned
ordinary OOM the whole-OS-boot policy instead of the intended five-minute
cooldown.

The classifier now gives explicit lost-device diagnoses highest priority, then
the causal OOM signature, and only uses generic `DeviceLost` as a fallback.
Exact historical OOM-induced and true-device-loss log pairs are regression
tests. An append-only
`artifacts/diagnostics/visual_capture_resource_events.jsonl` journal records
every quarantine transition, including fault age and system uptime, so the
remaining five-minute pressure window can be evaluated from real recovery
attempts without weakening the anti-stampede boundary blindly.
