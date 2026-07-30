# Expert review — 2026-07-29 — propagator (trial run)

- **Run:** 20260729T063800Z · commit `2fb6db6`
- **Slices:** `propagator` (`astrodynamicist`) — single-pair trial to calibrate the harness
- **Evidence:** full (`cargo check -p thalos_physics_canonical` ≈ 3 s; test evidence used)
- **Findings:** 5 confirmed · 0 plausible · 2 dropped

Every repro test was written into the real crate, run, and reverted;
`cargo test -p thalos_physics_canonical` is 118/118 green and
`git status --porcelain crates/simulation/` is empty.

**Evidence artifacts:** [`repro/20260729T063800Z-propagator/`](repro/20260729T063800Z-propagator/)
holds the paste-back test source for findings 1–3, plus a rebuild note for
finding 4 (whose replica source was lost — see that file). Read each header's
corrections section before acting: **every preserved test has at least one
correction from the refutation pass**, and finding 2's originally-proposed fix
would have made the code worse.

**Category verdict from the expert, worth keeping:** the conic math is clean.
`cartesian_to_elements` was checked component by component against
`keplerian_basis`'s non-standard (X, Z-swap, h∥−Y) convention — inclination pole,
node vector, ω-quadrant flip, circular-inclined argument-of-latitude flip, and the
retrograde equatorial branch are all correct. The coast is warp-rate independent
and free of energy drift, measured over 24 h of 1/6 s frames (518 k steps):
`Δa = −7.7e−7 m`, `Δe = 4.2e−13`, `|Δr|` vs one exact Kepler call `= 2.0e−3 m`.
**Every finding below is in the event-detection layer, not the propagation layer** —
so the fixes are localized, not an integrator rewrite.

---

## Confirmed

### `propagator-astro-1` — multi-revolution SOI encounters are invisible to prediction
**`major`** (filed `fundamental`) · `crates/simulation/physics_canonical/src/ship_propagator.rs:289-310`, `:487-505`

**Mechanism.** When `el.apoapsis_m < soi_radius` the coast caps `target_time` at
one period and sets `is_stable_orbit`. The sample loop is the only place
`detect_step_crossings` runs, so it covers one revolution; `extends_past_loop`
then appends a single `eval_at(original_target_time)` sample with **no crossing
check**. The ship's Kepler orbit does not precess — but the moon moves, so an
encounter that lines up on revolution 3 has no representation on revolution 1.

**Failure.** Reproduced by the refuter on the **shipped** system
(`assets/solar_system.ron`), not the expert's synthetic one — Thalos 1.378e24 kg /
3.186e6 m, Mira a = 1.91488e8 m, ship on a trans-Mira transfer rp 4,000 km /
ra 195,000 km phased for a third-apoapsis intercept:

```
one-shot stop_on_stable_orbit=false: terminator Horizon,    end_time 2601039.9, samples 144
one-shot stop_on_stable_orbit=true:  terminator StableOrbit, end_time  650260.0, samples 143
chunked walk (0.2 period):           SoiEnter Some((2, 1571096.13))   period = 650260.0 s
```

The same propagator finds the Mira encounter at 2.42 periods when stepped in
chunks; the one-shot call finds nothing under either flag. Player-visible: a
multi-revolution phasing intercept cannot be planned at all.

**Refuter corrections.** Two, both narrowing:
- The player-facing path terminates differently than filed. `flight_plan.rs:624`
  sets `stop_on_stable_orbit = leg_idx + 1 == leg_count && !burn_collided`, so the
  final leg returns `StableOrbit` at one period rather than `Horizon`. Same
  outcome, different branch than the expert's test exercised.
- The live-sim amplifier is much weaker than filed. `advance_vessel` re-queries
  the anchor every frame, so a skipped entry is re-detected at the next frame
  boundary — a one-frame error, not a permanent miss. The expert's
  `warp/60 > period` threshold needs `real_delta ≥ 0.065 s`, i.e. a ≤15 FPS hitch
  at 1e7×. Reachable, not normal.

Nothing chunks above this layer: `flight_plan.rs:541-558` passes `ephemeris_end`
(≈10 kyr) as the last leg's end, `propagation.rs:135`'s loop does not re-drive
after the straggler, `events.rs:193-196`'s `pair_limit` deliberately excludes the
straggler chord, and `scan_closest_approaches` reads the same one-period sample set.

**Fix.** Decouple event scanning from render sampling: keep one period of dense
render samples, continue the crossing scan across the horizon on a coarse
event-only walk (no `build_sample`, no Vec growth). The one-period cap itself is a
deliberate, load-bearing perf fix and should stay.

---

### `propagator-astro-2` — no terrain-scale resolution bound on the prediction collision scan
**`logic`** · `crates/simulation/physics_canonical/src/ship_propagator.rs:388-414`, `:1046-1068`

**Mechanism.** `needs_subdivide` compares swept path against
`MAX_PATH_RATIO × min_alt`, where `min_alt = (prev_state.position −
prev_body.position).length()` — a **radius**, not an altitude. `body_radius` and
`max_elevation_m` are never subtracted on this path; the only subtraction is in
`altitude_at_q`, which `needs_subdivide` does not call. So the cap cannot bound
terrain-scan resolution, and the only terrain probes are the two endpoints plus
`interior_min_altitude`'s fixed `s = 0.25/0.5/0.75`. The block comment above it
("both relative to the smaller endpoint **altitude**") contradicts the code.

The asymmetry is structural: the live path's step is bounded by
`warp_min_altitude_radii` (`simulation.rs:83-100`, explicitly sized so a frame's
advance "cannot punch through the body"), while prediction's is bounded only by
`coast_samples_per_segment: 128` — a render-density knob.

**Failure.** Independently reproduced by the refuter (own probe, not the expert's):
Earth-scale body, 5 km circular orbit, 10 km × 8 km ridge, 24 longitudes.

```
prediction (128/period): hit  8, missed 16 of 24
live-shaped (32/10 s)  : hit 24, missed  0 of 24
```

This contradicts `lib.rs:17-19`, which asserts *"`trajectory::propagate_flight_plan`
uses the same `ShipPropagator` … so 'where the ship is' and 'where it will be' can
never numerically diverge."*

**Refuter corrections.** Substantial — read these before acting:
- **The "keyed to radius is the bug" framing is wrong**, and so is the first half
  of the proposed fix. For the cap's actual purpose — bounding cubic-Hermite arc
  error — `path / |q|` *is* the right quantity (swept angle; 0.25 ≈ 14°/step).
  Re-keying it to altitude would subdivide to `MIN_STEP_S` for an entire grazing
  pass. The real defect is narrower: **there is no terrain-scale bound at all**,
  and the fix is the `sample_stride` clause alone.
- The CLAUDE.md *One propagator everywhere* citation is a misapplication — both
  paths do route through the same `ShipPropagator`. What is contradicted is the
  `lib.rs` module doc above.
- **Reachability is narrow.** Probe spacing is `2πr/512 ≈ 0.0123·r` — 39 km on
  Thalos, 10.7 km on Mira. Real `ProceduralSurface` relief is wider and shorter
  than the test ridge (`MASSIF_SITES` 44–48 km half-width, `MASSIF_PEAK_M ≈ 4.9 km`,
  `crates/domain/terrain/src/procedural.rs:534-590`), so a 10 km/8 km ridge does
  not exist in this world. The failure needs tall terrain narrower than ~1.2 % of
  the orbital radius — on Thalos, a few-km-altitude orbit inside a 1 bar
  atmosphere.

**Fix.** Add a `sample_stride` so the crossing scan runs at a physically-derived
step while `build_sample` emits at the caller's render density. Do **not** re-key
`needs_subdivide` to altitude.

---

### `propagator-astro-3` — a canonical `OnRails` surface impact is neither terminal nor reported
**`logic`** · `crates/simulation/physics_canonical/src/simulation.rs:741-767`, `:618-632`

**Mechanism.** `advance_vessel` returns `collision_time`; `step` replays the fleet
to it, sets `sim_time = event_time`, calls `warp.reset_immediate()`, and returns
`()` — the collision epoch, body, and speed are dropped. `AuthorityMode` is
unchanged and the vessel keeps its impact velocity. `mark_destroyed` has exactly
one caller (`local_physics/ground.rs:451`, the Avian contact path), so a canonical
impact never destroys anything. Next frame, `detect_step_crossings` gates on
`prev_alt > 0.0` (`ship_propagator.rs:1201`), and bisection has left the vessel
just below the surface — so no collision is flagged and the coast runs unbounded.

`docs/simulation/vessels.md` §4 states the intended behaviour outright: *"A
surface impact transitions it to the same destroyed or `BodyFixed` outcomes as an
active vessel."* This is the spec, unimplemented.

**Failure.** Reachable today: `crates/runtime/game/src/staging.rs:524` calls
`create_vessel` with `AuthorityMode::OnRails` — the landed stage-separation
slice — and a staged booster has periapsis inside the body by construction.
`apply_regime_authority` (`regime.rs:280+`) only ever transitions the *active*
craft off rails. Refuter's independent repro, Earth-sized body, 10,000 × 1,430 km:

```
first impact        = 1476.80 s (frame 89)      [expert measured 1476.562 s]
alt 10 s later      = -93 345 m
deepest             = -4 938 011 m               [expert measured -4 939 735 m]
warp resets         = 49 over 66 106 s          (~3 per revolution)
is_destroyed        = false
authority(OnRails)  = true
```

Player symptom: warp will not stay engaged, at ~3 stomps per revolution — worse
than the once-per-revolution the expert filed.

**Refuter struck one sub-claim as `wrong`.** The finding's "the warp reset
disables the game's only sub-surface safety net" is false. That guard
(`bridge.rs:110-113`) *is* `warp.reset_immediate()` — the same call just made, so
it is redundant at that instant, not disabled, and re-fires the moment the player
pushes warp above 1×. It also reads the *active* craft's state, so it never
applied to a detached vessel. **Strike that sentence.**

**Scope note.** The authority-transition half of the fix overlaps
`docs/backlog.md:121` (`BL-20260724T230226Z-shared-local-vessel-scene`, blocked).
What this evidence licenses is narrower and cheap: have `Simulation::step`
return/queue the collision `{craft, body, epoch, surface-relative speed}` it
already computed, and stop the record from coasting sub-surface.

---

### `propagator-astro-4` — golden-section refinement discards real collisions on non-unimodal terrain
**`robustness`** · `crates/simulation/physics_canonical/src/ship_propagator.rs:1111-1118`

**Mechanism.** `refine_collision` calls `golden_section_extremum` on
`f(t) = |q(t)| − body_radius − elev(dir_body(t))` across the whole step and
rejects the crossing if `f(t_extremum)` keeps `f_lo`'s sign.
`golden_section_extremum` (`:966-994`) is a textbook bracket-shrinker
(`let pick_left = if seek_min { fc < fd } else { fc > fd };`) with no restarts and
no sampling pass — it **requires** unimodality, which a procedural height field
does not provide. On rejection, control falls through to `samples.push(...)` and
the scan continues.

**Failure.** The expert declined to quantify this; **the refuter built the
demonstration**. Standalone replica of `hermite_cubic`, `altitude_at_q`,
`interior_min_altitude`, `golden_section_extremum`, `refine_collision` and the
coast loop, driven by the real `orbital_math::propagate_kepler`, against ridged
terrain at the established scale. 720 scenarios (60 ridge phases × 12 periapsis
altitudes):

```
detect_fired            = 192
  refine -> Some (kept) = 182
  refine -> None        =  10
    of which a REAL sub-surface root existed = 10
```

```
DIAG golden_s=0.7528 f_ext=+814.8m   true_min_s=0.2567 f_min=-1241.6m
DIAG golden_s=0.2831 f_ext=+885.4m   true_min_s=0.4902 f_min= -740.2m
```

The 0.75/0.25 mirroring is the signature of the first two probes at 0.382/0.618
steering the bracket to the wrong local minimum. Across four terrain
configurations, **100 % of `None` returns were real crossings** — the "Hermite
false positive" escape hatch this branch exists for never once fired legitimately.

**Refuter narrowed the consequence.** The filed "the ship flies through the
mountain" is stronger than the code supports: 9 of 10 rejections are re-caught by
a later step in the same revolution. The dominant defect is a **mislocated
impact** — up to one step late, ~40 s and ~300 km downrange at LEO speeds — with
total loss rare (1 of 720 scenarios). Also: the finding lists `:929-941`
(`refine_crossing`) as "same pattern", but unimodality genuinely holds there;
only the `refine_collision` site is defective.

**Fix.** Bisect on the bracket `[s_prev, s_probe]` that `interior_min_altitude`
already found. Caveat the finding does not carry: `interior_min_altitude` probes
the **Hermite** curve while `refine_collision` roots the **Kepler** curve, so that
bracket is a strong hint, not a guaranteed sign change — keep a fallback.

**Calibration.** The same sweep found **477** steps where a real sub-surface root
existed and `detect_step_crossings` never fired at all. Detection density
(finding 2) is the larger defect by an order of magnitude; this is the residual
behind it.

---

### `propagator-astro-5` — prediction tracks the burn frame, the autopilot flies it inertially fixed
**`minor`** (filed `design`) · `ship_propagator.rs:797-813` vs `crates/runtime/game/src/autopilot.rs:647-652`

**Mechanism.** `rk4_burn_step`'s `accel` closure calls `delta_v_to_world(...)` with
the *instantaneous* `vel`/`pos` inside every k1..k4 stage, so predicted thrust
rotates with velocity. Execution latches `direction: directive.direction` into
`AutopilotState::Burn` at ignition; the `Burn` arm destructures with `..` and
touches only throttle, and `attitude_target()` returns that latched vector as
`AttitudeDemand::PointNose` — bypassing the per-frame re-resolution in
`navigation.rs:222-249`. Termination is the direction-blind scalar
`delivered_dv` counter (`simulation.rs:998-1001`).

**Failure.** Refuter measured both halves. Burns are *centred* on `node.time`, and
the frozen direction comes from `pre_burn_state_at(node.time)` — the midpoint, not
ignition, which the expert's arithmetic missed:

| scenario | arc | Δv vector error | apoapsis error |
|---|---|---|---|
| 250 kN / 75 kg·s⁻¹, 2000 m/s, 200 km orbit | 60.2 s, 4.09° | 1.64 m/s = **0.082 %** | 5.7 km / 24,674 km = **0.023 %** |
| 25 kN / 7.5 kg·s⁻¹, same Δv | 601.6 s, 40.9° | 35.6 m/s = 1.78 % | 534 km = 2.26 % |

**Two of the finding's three load-bearing claims were killed:**
- **The spec citation is a misread.** `docs/simulation/simulation.md:8-29` marks
  `ManeuverFrame` — and `BurnFrameBehavior` in the same code block at `:872-924` —
  as *"target design, unbuilt"*. The spec does not require it; it sketches it.
  Strike "as the spec calls for".
- **The magnitude is below its own stated lower bound** on the headline case:
  0.082 %, not "0.1–1 %". The scaling claim holds and only the low-thrust row is
  material — and no Thalos stage with that TWR has been shown to exist.

**Fix.** If this earns a row, it should read *"prediction tracks the PRN frame,
the autopilot flies inertially fixed; pick one"* with the measured numbers — not
a spec-compliance framing.

---

## Dropped

| id | claim | verdict | reason |
|---|---|---|---|
| `propagator-astro-6` | Propagator re-derives `mu = mass_kg * G` instead of reading `BodyState.gm` — two independent sources | `wrong` | Not *independent*. `PhysicalParams` (`crates/domain/world/src/parsing.rs:78-85`) has no `gm` field and no serde default, so the premise "author a body from a measured GM" is unauthorable. One production `BodyDefinition` constructor (`:239`), no `.gm =` or body-level `.mass_kg =` mutation anywhere, one production ephemeris. And `physics_canonical` re-exports `thalos_world::body::G`, so `body_t0.mass_kg * G` and `G * b.physical.mass_kg` are the same IEEE-754 product with operands swapped — bit-identical. The claimed wandering orbit cannot be produced by any reachable state. |
| `propagator-astro-7` | 32 `TrajectorySample`s built per vessel per frame and discarded | `wrong` | Every fact checks out — refuter instrumented `BodyTrajectoryProvider::state` and measured 194 ephemeris calls/segment, 33 from `build_sample`, matching the filing. But the fix recovers ~2.3 µs per vessel per frame: **0.014 % of a 16.6 ms frame**. Zero while paused, skipped entirely for `BodyFixed`/`LocalRigidBody`, and 0.28 % even at a 20-vessel fleet. Below the perf lane's own noise floor and unobservable by any instrument in the tree. The finding also cites `BL-20260724T230226Z-persistent-stage-separation` for scaling; that row does not exist. If the sample budget is ever worth touching, the lever is step count (hint=2 → 1.15 µs vs 13.53 µs, 11×), which the finding does not propose. |

## Questions for a capture session

None. This slice is pure math and headless-verifiable end to end.
