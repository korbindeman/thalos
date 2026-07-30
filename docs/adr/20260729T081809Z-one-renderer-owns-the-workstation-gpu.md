# ADR-20260729T081809Z: One Thalos renderer owns the workstation GPU at a time

- **Status:** Accepted
- **Date:** 2026-07-29
- **Incident:** [INC-20260729T081809Z](../incidents/INC-20260729T081809Z-renderer-overlap-exhausted-the-gpu.md)

## Context

The tile renderer acquired a byte budget after its first GPU OOM, then divided
that budget between live renderer processes after two games exhausted the same
12 GB card. Both mitigations governed tile meshes only. A full renderer also
owns mesh slabs, images, cloud and ocean targets, shadows, pipelines, staging
buffers, and boot transients.

Two later failures made that distinction decisive:

- at 08:43, a game and capture host overlapped; the new host reported
  `DeviceLost: Out of memory`;
- at 09:40, a game booted beside the persistent capture host, another capture
  host crossed the 8 GiB RSS guard, and the game later aborted with the same
  `0xc0000409` signature as the earlier OOM incident.

The capture-client mutex correctly serializes capture *requests*. It cannot
protect the GPU from the interactive game because the game is not a capture
client. Dividing one budget cannot make the uncounted allocations safe, and
buying a larger card only changes how long an unbounded concurrency contract
takes to fail.

## Decision

Exactly one canonical game-shaped Thalos renderer may own a machine's GPU:
either the interactive game or the headless capture host.

- Both launchers acquire the same non-waiting OS lease before building Bevy or
  initializing wgpu.
- Windows uses a named kernel mutex; Unix uses `flock`. The OS releases either
  primitive after an abort, so an OOM cannot leave a stale ownership boundary.
- A JSON owner record in the user temp directory is diagnostic metadata only.
  It names the PID, role, command, workspace, and start time; it is never the
  ownership authority.
- `just game` stops the persistent capture host before launching. A direct game
  launch beside a capture host is refused with an actionable message.
- A capture requested while the game is open fails as `renderer busy` before
  Bevy creates a GPU device. It is not retried and does not quarantine the GPU.
- The tile-share heartbeat remains a secondary diagnostic/defense for
  noncanonical renderer examples, but canonical launch safety no longer
  depends on accounting one allocation class.

## Alternatives

- **Keep dividing the tile allowance.** Rejected: it worked, yet both later
  failures occurred because most renderer allocations sit outside that
  denominator.
- **Measure every GPU allocation and permit safe concurrency.** Deferred as
  observability, rejected as the safety boundary. Wgpu does not expose one
  portable whole-device budget, allocator fragmentation matters, and startup
  needs headroom before a feedback controller can react.
- **Let a larger GPU absorb concurrent renderers.** Rejected: it would make
  correctness depend on the developer's card and does nothing for the observed
  capture-host system-RAM runaway.
- **Make captures wait indefinitely behind a game.** Rejected: interactive
  sessions have no bounded duration. A prompt refusal tells an agent why no
  evidence was produced and avoids a hidden queue that may wake hours later.
- **Use a PID/mtime file as the lock.** Rejected: OOM terminates by abort and
  bypasses destructors. The OS primitive already has the required owner-death
  semantics.

## Consequences

Agents cannot capture while the user is playing, and two interactive instances
cannot run together. That is intentional: deterministic evidence and a stable
desktop outrank renderer concurrency on one workstation.

The capture resource quarantine remains necessary for faults inside the sole
host. The lease prevents overlap; it does not declare the 8 GiB host-RSS growth
fixed. A future multi-GPU implementation may reopen the decision only if the
lease is scoped to a selected physical adapter and every launcher identifies
that adapter before device creation.
