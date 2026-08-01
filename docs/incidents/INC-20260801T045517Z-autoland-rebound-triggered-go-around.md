# INC-20260801T045517Z: autoland treated suspension rebound as an aborted landing

- **Reported:** 2026-08-01, user, screenshots from the successful first LAND
  arrival
- **Recorder:** `runtime.jsonl` session `10360-1785559021717`
- **Fixed:** same change
- **Areas:** `route_autopilot`, approach aim-point tuning

## Symptom

LAND flew the approach correctly but touched near the beginning of the runway,
then immediately added full power and climbed away instead of braking to a
stop. The annunciator said `GO-AROUND: bounced on landing`.

## What the recorder showed

The approach itself was not the failure. On short final, cross-track remained
within centimetres and the achieved flight-path angle held about -2.9°.

| Time (Unix ms) | Evidence |
|---:|---|
| 1785559481411 | two main wheels loaded; normal force 635 kN |
| 1785559481710 | 0.25 s contact confirmation completed; `Flare → Rollout` |
| 1785559482006 | suspension rebound: wheels unloaded, only ~5.4 m above runway reference, still centred, brakes already selected |
| 1785559482081 | the 0.35 s airborne gate fired; throttle went from idle to full |
| 1785559483452 | wheels loaded again despite the go-around command |

The path aimed 300 m past the threshold. The aircraft crossed the threshold
below the planned 16 m height and first contact was roughly 150 m into the
strip, confirming the user's separate margin report.

## Mechanism

Touchdown confirmation and bounce detection had incompatible debounce scales:
0.25 s of contact committed LAND to rollout, but only 0.35 s without wheel load
revoked that commitment. The landing gear's intentionally soft compression and
firm rebound can unload its rays for that long during an ordinary first
touchdown. The rollout law also snapped from a 7° flare target to zero-pitch
rate control on the first airborne frame, helping a small rebound look like a
new flight phase.

This was not an aerodynamic bounce diagnosis and did not require retuning the
gear. The state machine misclassified a normal transient that the existing
suspension was designed to absorb.

## Fix

- A confirmed touchdown now remains in rollout through up to 2 s of continuous
  wheel unload. During that recovery it keeps idle throttle, full brakes/spoilers,
  centreline steering, wings level, and the flare pitch target. A genuinely
  sustained separation still goes around.
- The approach aim point moved from 300 m to 450 m past the threshold. At 3°
  that raises the planned threshold crossing from ~16 m to ~24 m and moves the
  expected contact into the touchdown zone without changing glideslope or flare.
- `appr_frame` and bounce `land_go_around` records now carry the contact and
  post-touchdown-airborne dwell values needed to distinguish the same failure.

## Recurrence tell

For a bounce go-around, read `post_touchdown_airborne_s` on
`event=land_go_around`. A value below `2.0` means the debounce contract regressed.
During ordinary rebound, `appr_frame` should keep `phase=Rollout`,
`throttle_cmd=0`, `brake_cmd=1`, and return to `weight_on_wheels=true` before
that dwell expires.
