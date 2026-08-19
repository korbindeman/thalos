# INC-20260818T084406Z: hot capture read partial tile residency

## Symptom

A deterministic `forest-stand` capture exited zero and wrote a plausible PNG,
but its receipt reported only 720 resident tiles for a 1,680-tile desired set.
The immediately preceding frame at the same camera and epoch was fully settled
at 1,680/1,680.

## Mechanism

A persistent-host request can move the camera or canonical time far enough to
replace the desired tile set. Ordinary frame warmup then expired while the new
selection was still landing. The existing readback gate watched only the tile
memory brake (`split_scale < 1`); this capture had `split_scale == 1`, so it
looked healthy to the client even though desired coverage was incomplete.

Hiding the bookkeeping `TileTerrainRoot` is not a valid diagnostic for this
path: resident meshes deliberately hang from the global big-space root for
precision. `TileBodyOrigin` identifies the actual rendered tile entities.

## Fix and recurrence tell

Every tile capture now holds after ordinary warmup until
`coverage_ready() && settled()`, bounded by the existing 180-second terrain
stream ceiling. The receipt records `terrain.settled` and `settle_wait_s`; a
timeout prints a warning and records `terrain_unsettled` in the tool lane.

The recurrence tell is a receipt with `resident_tiles != desired_tiles` or
`settled: false` after a warm-host camera/time jump. A PNG alone is not proof of
a complete capture.
