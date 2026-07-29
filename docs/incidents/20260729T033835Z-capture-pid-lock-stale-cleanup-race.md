# INC-20260729T033835Z — capture PID-lock stale cleanup admitted two owners

## Symptom

Two screenshot clients from the same checkout could both remain live while the
client-lock file was absent. They could then prepare or manipulate capture state
concurrently despite the advertised singleton contract.

## Mechanism

Atomic file creation serialized the happy path, but stale cleanup was a
non-atomic read/decide/delete sequence:

1. waiters A and B both read old owner X and decide X is dead;
2. A removes X's file and creates a new lock for A;
3. B executes its already-decided path deletion, which now deletes A's file;
4. B creates its own file, so both clients believe they own capture.

The token check in `Drop` could not repair ownership after that point. A second
scope bug made the lock and resource quarantine worktree-local even though all
worktrees share the physical GPU.

## Fix and recurrence tell

On Windows, ownership is now a named kernel mutex
`Local\ThalosMachineCaptureV1`. The kernel admits one owner and releases
abandoned ownership when a process dies, so there is no stale path to delete.
The owner JSON and resource-fault JSON live in the machine temp directory and
are metadata guarded by that mutex.

If two live `thalos_capture` clients show no “queued behind” relationship, or
if a worktree can avoid another worktree's quarantine, this incident has
recurred. A filesystem PID/token file must not be restored as the authority.
