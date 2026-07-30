# INC-20260729T091959Z-capture-stop-abandoned-a-live-host: `stop_server` gave up after 5 s and left an untracked renderer running

- **Date:** 2026-07-29 · **Surface:** capture client restart path (`restart_stale_source` / `restart_startup_override` / `restart_incompatible_scene`)

## Symptom

During the 2026-07-29 renderer-overlap window (INC-20260729T081809Z), a capture
host that was being *replaced* kept rendering and streaming tiles for **5+
minutes after its replacement booted** — runtime.jsonl shows the old session's
residency gauges still updating 300 s past the new host's first record. The
replacement was then killed by the RSS watchdog at 8.1 GiB ("capture host
memory runaway") 30 s into its own boot, while its accounted GPU gauges summed
to ~2 GiB.

## Root cause

`stop_server` wrote a Shutdown request, waited **5 s** for the host to exit,
and on timeout **fell through silently**: it killed the launcher shim, deleted
`STATE_FILE`/`LAUNCHER_FILE`, and returned `Ok` — without ever force-killing
the host pid. A host mid-cold-stream services the request file from its own
busy main loop (~110 ms/frame during a massif fill) and can miss the graceful
window entirely; deleting the state files then made the survivor *untracked*,
so nothing would ever stop it. The wrong first hypothesis — a tile/terrain
memory leak in the diffusion backing — was ruled out by re-running the
"crashing" preset solo in three configs (warm canonical, warm diffusion,
cold-cache diffusion): all peak at 1.9 GiB RSS.

## Fix

`stop_server` now escalates: after the graceful window it force-terminates the
host's process tree and waits up to 10 s for **confirmed** exit; if the process
still survives, it returns an error and the shot **fails** instead of booting a
second renderer beside a live one (`start_server_once_inner` propagates). The
state files are only deleted once no tracked process is alive. This removes the
mechanism (an unowned renderer holding the GPU), not just the symptom — with
the renderer lease in place, the old behaviour would have stranded the lane in
"renderer busy" until someone found the zombie by hand.

## Tell

- tools.jsonl `host_stop_forced` counts a host that missed graceful shutdown.
- Any runtime.jsonl session whose gauges keep updating past a successor
  session's first record is this bug again (or a lease bypass — see
  INC-20260729T081809Z's tells).
- `frame_gauge.rss_mib` far above `tile_mib + slab_mib + mesh_cpu_mib +
  image_cpu_mib` means the growth is not owned by any accounted subsystem —
  look for a second renderer before suspecting a leak.
