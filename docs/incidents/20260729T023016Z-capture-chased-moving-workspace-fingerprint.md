# INC-20260729T023016Z — capture chased a moving workspace fingerprint

## Symptom

A capture started a renderer for source `A`. If any agent edited a capture
input during the build, the client stopped the healthy renderer, called source
`B` “the latest,” and rebuilt. It could repeat this three times while the
machine remained occupied and no screenshot was taken.

## Mechanism

One aggregate content-hash equality check was answering two different
questions:

1. Was the renderer launched from a state that includes the caller's starting
   work?
2. Is the shared checkout still byte-identical now?

Only the first is required. The second is useful provenance, but parallel work
makes inequality normal. Hashes have equality, not an ordering; treating every
different hash as stale turned a source-attribution guard into a moving-target
restart loop.

## Fix and recurrence tell

The invocation snapshot is now a **source floor**. The client prepares the
renderer once and accepts edits made later while that build is in flight.
Receipts retain exact `workspace_matches`, but also state
`source_floor_guaranteed` and `workspace_relation`. A false exact match never
triggers another build.

If capture output again says “refreshing [n/3]” merely because `requested !=
current`, this incident has recurred. Restart only when the resident host
predates the request floor or advertises a different launch build—not because
the checkout advanced after launch.
