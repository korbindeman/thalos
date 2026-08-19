# INC-20260818T200855Z: Headless capture waited on placeholder terrain

- **Date:** 2026-08-18 · **Surface:** cold `just screenshot` and every headless performance matrix

## Symptom

A cold `forest-stand` shot spent about 145 seconds in the render phase while a
warm repeat took about 14 seconds. Performance matrices repeatedly logged the
120-second loading hard timeout before beginning their actual target-view
stream. Eight matrices consumed about 38 minutes of one investigation.

The decisive tell was residency holding at 642 tiles throughout loading, then
jumping through 1,039 to 1,542 only after the timeout revealed `Running`.

## Root cause

The terrain loading step required `coverage_ready` for an immediate
`ShipOrbit` boot. The scripted capture camera could not pose the requested
scene until `AppState::Running`, so loading waited for complete coverage of the
placeholder orbit while the system that could select the real target waited on
loading. The timeout broke the cycle after 120 seconds.

The first handoff attempt relaxed only the tile-owned body inside the existing
per-body loop. Orbit prediction also nominated another terrain body as `Near`;
ordinary builds have no legacy renderer to make that second body resident, so
the loop still could not complete even with the main root at 642/642. Headless
ownership must therefore cover the complete wanted terrain set.

The performance harness then repeated a separate 120-frame stability wait for
every cell even though the camera and resident scene were unchanged. Worse,
it configured cell 1 before settlement, so tiles landing later could miss a
cell-specific mesh or bounds mutation.

## Fix

Headless capture now explicitly owns target-view terrain readiness for the
complete wanted set. The loading gate hands off once the tile renderer has
spawned a live root; after the scripted pose, the capture driver still requires
`coverage_ready() && settled()` before readback.

The benchmark state machine now settles that exact view once, before
configuring cell 1. Every cell then gets a 30-frame transition flush and the
existing 240-frame measurement window. Interactive loading behavior is
unchanged.

## Recurrence signal

Any cold headless run that logs the 120-second loading hard timeout, or any
matrix with more than one `headless_benchmark_ready` event, has regressed. The
shared ready event must precede the first config event, and capture receipts
must still record `terrain.settled: true`.
