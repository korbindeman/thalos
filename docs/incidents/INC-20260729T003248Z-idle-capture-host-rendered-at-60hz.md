# INC-20260729T003248Z — Idle capture host rendered at 60 Hz

- **Date:** 2026-07-29
- **Status:** fixed and runtime-verified
- **Surface:** persistent `just screenshot` / `just capture` / `just compare`

## Symptom

Capturing one still could leave the development machine saturated and its fans
spinning for minutes. Concurrent capture invocations also produced repeated
renderer launches, so work that should have reused one booted world instead paid
startup, terrain streaming, and pipeline setup several times.

## Diagnosis

Three mechanisms compounded:

1. `ScheduleRunnerPlugin::run_loop(1 / 60 s)` continued advancing the complete
   headless app after `PersistentCaptureServer::active_id` returned to `None`.
   The request driver stopped, but the real `ShipCamera` remained active, so
   Bevy still extracted and rendered the full scene at 60 Hz forever.
2. The Windows liveness probe shelled out to `tasklist`. A restricted process
   can receive `ERROR: Access denied` from `tasklist` even for a healthy capture
   host. The controller interpreted the missing PID text as “dead” and launched
   a replacement. The server log showed sub-second Cargo finishes followed by
   repeated host boots seconds apart, ruling out compilation as the repeated
   cost.
3. Capture clients had no ownership lock around the shared request, response,
   state, launcher, and log files. Two clients could each decide the other
   host/context was wrong and replace it.

The legitimate cold work is much smaller and bounded: the measured Thalos boot
reached surface settle in about one second and first full tile coverage about
ten seconds later. PNG encoding/readback was not the lingering load: successful
responses and saved files appeared while the host continued consuming resources.

## Fix

- A persistent host now sets the real capture camera inactive whenever no
  request owns it. That prevents render extraction and every downstream pass.
- The idle main loop sleeps 100 ms per poll instead of spinning the ECS at
  60 Hz. A request wakes within one poll and reactivates the camera before that
  frame's render extraction.
- Windows liveness uses `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` plus
  `GetExitCodeProcess`; an access-policy failure is treated conservatively as
  alive, never as permission to replace the process.
- The controller owns an atomic create-new client lock for the whole shot,
  batch, comparison, or reset. A second client waits without launching another
  renderer, reports the owning PID/command, clears dead owners, and times out
  rather than corrupting the control plane.

## Recurrence tell

- `artifacts/diagnostics/runtime.jsonl` should emit
  exactly one `persistent_render_activity` transition to `active:true` for a
  request and one to `active:false` after it.
- An idle host should retain its world/resources but stop producing GPU load.
- A concurrent invocation should print “another capture client is active” and
  wait; the server log must not gain another `Running
  thalos_capture_host.exe` entry.
- If a restricted shell cannot enumerate processes, it must not cause a healthy
  host restart.
