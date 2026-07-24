# INC-20260724T185500Z — The persistent capture host inherits the caller's stdout pipe, so piped/background runs hang until timeout

**Status:** fixed (the host no longer inherits the client's std handles).

## Symptom

An agent runs a capture as a background task with its output captured — e.g.

```bash
just capture massif-aerial massif-ridge massif-valley 2>&1 | tail -6
```

— and the task never reports. It sits for the harness's full timeout (~10 min)
and is killed, with an **empty** output file, even though the captures
themselves succeeded minutes earlier. The agent has no result to act on, and
typically burns further turns re-running or "debugging" a capture that already
worked.

## Evidence

Observed live on 2026-07-24, ~15 minutes after a batch had completed:

- `artifacts/visual/latest/massif_{aerial,ridge,valley}.png` all written
  (timestamps 20:29), and `visual_capture_response.json` read
  `"ok": true, "message": "capture complete"`.
- Process tree at 20:44:

  ```
  ProcessId ParentProcessId Name                    Cmd
      11924           18840 tail.exe                tail.exe -25
      12256            2800 cargo.exe               cargo run -p thalos_capture_host --features dev-renderer
       4408           12256 cargo.exe               …cargo.exe run -p thalos_capture_host…
      15644            4408 thalos_capture_host.exe target\debug\thalos_capture_host.exe
  ```

  `thalos_capture.exe` (the client) is **absent** — it exited normally. `tail`
  is still alive, blocked on EOF.
- **Decisive:** killing `tail` (PID 11924) made the harness immediately report
  the background task as `completed (exit code 0)` — i.e. the pipeline had been
  blocked *only* on the pipe, and `just capture` itself had long since
  succeeded.

## Hypotheses considered

- **The client hangs / never exits** — ruled out: no `thalos_capture.exe` in
  the process list, and the task reported exit 0 the moment the pipe closed.
- **A capture actually failed or wedged** — ruled out: three valid PNGs plus an
  `ok: true` response predating the observation by 15 minutes.
- **`| tail` buffering alone** — insufficient: `tail` buffers *interim* output,
  but it still terminates at EOF. Something had to be holding the write end.
- **Inherited handle in a process designed to outlive the client (accepted)** —
  matches every observation, including the exact instant of release.
- **…but specifically via our std handles** — *falsified* by the fix attempt
  below: clearing their inherit flags changed nothing, so the leak also travels
  as inheritable duplicates invisible to `GetStdHandle`. This is why the fix
  had to be "inherit nothing" rather than "clear the handles we know about".

## Root cause

Windows `CreateProcess` with `bInheritHandles = TRUE` duplicates **every
inheritable handle** into the child, not only the ones named in `STARTUPINFO`.
The client's own stdout/stderr — the shell pipeline's write end, inherited from
the invoking shell and therefore itself inheritable — was passed down to the
detached `cargo run` launcher and its `thalos_capture_host` child, *despite*
their stdio being redirected to `artifacts/diagnostics/visual_capture_server.log`.

Because the persistent host is deliberately long-lived (that is the entire
point of the lane — ADR-20260721T192218Z), the inherited pipe handle is held
effectively forever. Any invocation whose output is captured through a pipe
therefore cannot terminate: the reader waits for an EOF that only arrives when
the host dies.

This makes the failure *worst* exactly where it hurts most — the **first**
capture of a session, which is the one that spawns the host.

## Fix

**First attempt, measured insufficient (recorded because it is the obvious
one):** clearing `HANDLE_FLAG_INHERIT` on the client's own
`STD_{INPUT,OUTPUT,ERROR}_HANDLE` before spawning, plus `DETACHED_PROCESS` in
the creation flags. An instrumented rerun still hung — the client exited at
~31 s with the capture already saved, while `tail` and the host stayed alive
past 152 s, and killing *only* the host released `tail` instantly. Inheritance
therefore also carries **inheritable duplicates of the pipe that never appear
as our std handles**, so plugging handles one at a time cannot be trusted.

**Shipped fix — spawn with an empty inherited handle table** (`spawn_detached_launcher`,
`tools/capture/src/main.rs`):

- Windows launches the host through PowerShell `Start-Process`, i.e.
  `ShellExecuteEx`, which creates the process with `bInheritHandles = FALSE`.
  That is a *guarantee* rather than an enumeration of leaks to plug.
- `Start-Process` only keeps `UseShellExecute = true` while no stream
  redirection is requested, so the log redirection moves into a generated
  `artifacts/diagnostics/visual_capture_launch.cmd` shim (`cargo … 1>>log 2>&1`)
  which opens the log itself. No handle is passed at all.
- `cargo run` still fronts the host inside the shim, preserving the bevy_dylib
  search-path contract (INC-0008). `-PassThru` returns the shim's pid, so
  `stop_server`'s tree-kill is unchanged.
- Liveness is now pid-based (`LauncherProcess::is_running`) on Windows, since
  the host is deliberately no longer our child; Unix keeps the real `Child`.

Unix needed no change: `std` sets `CLOEXEC` on every descriptor it does not
explicitly pass, and the host's stdio is redirected to the log.

**Verified** with the same instrument that exposed it — a `bash -c "just
screenshot … | tail -2"` job, polled every 5 s:

| path | before | after |
|---|---|---|
| cold spawn | job still `Running` past 152 s, `tail` alive | job `Completed` at 30 s, output delivered |
| host reuse | 5 min foreground timeout (exit 143) | job `Completed` at 5 s |

In both, the survivors afterwards are `cargo, cargo, thalos_capture_host` — the
persistent host stays up for reuse, which is the whole point of the lane.

## Prevention

- **A process spawned to outlive its parent must inherit nothing from the
  parent's console/pipes.** On Windows that requires explicitly dropping the
  inherit flag — redirecting the child's stdio is *not* sufficient. Applies to
  any future detached helper (a render farm worker, a watcher daemon), not just
  this host.
- Recorded as a fast-iteration invariant in CLAUDE.md.

## Recurrence tells

- A background/piped `just screenshot|capture|compare` that never reports while
  `artifacts/visual/latest/*.png` and `visual_capture_response.json` show the
  work finished.
- A live `tail`/reader process whose parent shell is gone, alongside a live
  `thalos_capture_host.exe`.
- The task completing the instant the host is stopped (`just capture-stop`) —
  the release, not a coincidence.
