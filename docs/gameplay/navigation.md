# Navigation, routes, and guidance

The system that answers three questions for a vehicle on or above a body:
**where am I going**, **what path gets me there**, and **how far off it am I**.

Today it covers exactly one case end to end — **fly an approach to a runway and
land** — but the model underneath is deliberately vehicle-agnostic, because the
same three questions are what a rover crossing a plain, a boat making a harbour,
and a submersible holding a depth all need. What is built and what is only
designed for is spelled out in [Scope, and what is deliberately deferred](#scope-and-what-is-deliberately-deferred).

Code:

| Where | What it owns |
|---|---|
| `crates/simulation/navigation` (`thalos_navigation`, **no Bevy**) | Waypoints, route frames, lateral path geometry, the bank-limited (Dubins) planner, the approach planner, the vertical profile, and the per-frame guidance function. Pure and unit-tested. |
| `crates/runtime/game/src/route.rs` (`crate::route`) | Selection (which runway end is armed), re-plan policy, craft-parameter derivation, and the one published `RouteState` every consumer reads. |
| `hud/mfd/widgets/nav_display.rs` + `assets/shaders/nav_display.wgsl` | The ND: a projection of `RouteState`. |
| `hud/pfd_panel.rs` (`update_approach_guidance`) | Localizer / glideslope deviation scales + flight director: another projection of the same `RouteState`. |

> **Naming trap.** `crate::navigation` is **not** this system — despite the name
> it is the attitude/SAS pointing modes (prograde, retrograde, target hold). It
> predates this module and is queued for a rename
> (`BL-20260730T005746Z-rename-attitude-modes`). Route navigation is
> `crate::route` + `thalos_navigation`.

## One authority, many projections

`RouteState` (sole writer `update_route_state`) is the single source of truth for
the active plan and the live deviations. **No display re-derives navigation.**
The ND draws the path it is given; the PFD deflects needles from deviations it is
given.

This is not tidiness. An ND that planned its own approach and a PFD that computed
its own localizer error would agree at first and drift apart later — different
update order, different rounding, a different idea of where the threshold is —
and the disagreement would be invisible right up until someone followed the
needle into the ground. There is one number, and everything shows that number.

## The waypoint model

A `Waypoint` is a body-fixed direction plus **optional** vertical and speed
constraints:

```rust
pub struct Waypoint {
    pub dir: DVec3,                            // body-fixed, so the planet can rotate under it
    pub vertical: Option<VerticalConstraint>,   // At / AtOrAbove / AtOrBelow / Window
    pub speed_m_s: Option<f64>,
    pub kind: WaypointKind,                     // Fix | FinalApproach | Threshold | Aim
}
```

**`vertical: None` is not missing data.** It is the correct state for a vehicle
with no vertical axis to command — a rover, a surface ship. Aircraft and
submersibles constrain it. Guidance must degrade to lateral-only rather than
inventing an altitude, which is why the vertical half of the guidance output is
separable from the lateral half.

Positions are **body-fixed**, never inertial: a route is nailed to the ground and
must survive the planet spinning under an inertial craft.

## Frames: lateral in a plane, vertical on the radius

Route geometry is planned in a `RouteFrame` — a local east/north tangent plane
anchored at a body-fixed origin (for an approach, the landing threshold).

- The lateral projection is **gnomonic** (project radially outward onto the
  tangent plane). Three properties earn it the job: it is **altitude-independent**
  (a craft at 3 km reads the same `(east, north)` as the ground beneath it), it is
  an **exact inverse** of the plane→sphere direction, and **straight lines in the
  plane are great circles** on the sphere. A naive chord projection fails the
  first one, drifting by `d · Δalt / R` — about 9 m over a 25 km final, which
  quietly corrupted route length, distance-to-go, and glideslope deviation
  together, in a way that reads as noise rather than a bug.
- **Altitude is never a plane coordinate.** The tangent plane runs `d²/2R` ≈ 75 m
  above the sphere at 30 km. Altitude is always height above the body reference
  radius, measured radially. (Thalos has no sea-level layer — sea level *is* 0 m.)

Angles use two conventions, converted only at the boundary: **compass heading**
(0 = north, clockwise) for anything a pilot or the HUD reads, and an internal
**math angle** (CCW from east) for all geometry. Because +y is north and θ is
CCW, a pilot's left turn is a mathematically positive rotation — which is why
`DubinsWord::L` really is "turn left".

## The lateral path

A path is a chain of `Leg`s — `Line`, `Arc` (constant radius, signed sweep), or
`Point` (a zero-length leg that still knows its heading) — with arclength
parameterisation and a closest-point query returning along-track distance and
**signed cross-track (positive = right of course)**.

`Leg::Point` exists because a zero-length `Line` has no direction, and a leg that
reports a fabricated direction puts a wrong *course* on the display. That was a
real defect: identical start/goal poses produced a zero-length line whose
direction read as "east".

### The bank-limited planner

Joining the craft's current position-and-heading to the start of a final approach
is exactly the classical **Dubins problem**: two arcs of the minimum turn radius
`v² / (g · tan φ_max)` joined by a straight tangent. `plan_dubins` implements the
four `CSC` words and deliberately omits the three-arc `CCC` words:

- A `CCC` solution is a tight S of back-to-back full-bank spirals — the wrong
  shape for an approach even when it is the shortest.
- **A `CSC` solution always exists**, so omitting `CCC` costs no coverage.
  Expanding LSL's discriminant gives
  `(d + sin α − sin β)² + (cos β − cos α)²` — a sum of squares, never negative.
  The planner therefore cannot fail on geometry, only on a bad radius.

Turn radius uses the **local** gravity the caller supplies: Thalos is not Earth,
and the same craft turns wider on a heavier world.

## The approach

`plan_approach(end, craft_position, craft_track, params)` produces one route:

```
   craft ───(bank-limited transition)───▶ join point ══════(straight final)═════▶ aim
                                                    threshold ─┘        └─ aim = threshold + 300 m
```

- **Every strip has two landable ends.** `RunwayEnd` is a first-class type
  (strip + which way you are landing) with its own designator, because which end
  you pick is the most consequential choice in an approach — not a boolean buried
  in a plan.
- The **final** is a straight segment aligned with the landing heading, 9 km by
  default, ending at an **aim point 300 m past the threshold**. Aiming the
  glideslope there sets the threshold crossing height at `300 · tan 3°` ≈ 16 m,
  matching real ILS practice. Note that `dtg = 0` is the *aim* point, so the
  threshold is crossed with `aim_inset` still to go.
- **Joining late is handled explicitly.** A craft already inside the final
  corridor is *not* sent back out to the nominal final approach point — a Dubins
  path to a fix 6 km behind you is a full turn-around, a correct answer to the
  wrong question when you are three kilometres out and lined up. The join point
  slides forward along the centreline, leaving a stabilised run
  (`min_capture_run_m`) onto the aim point. Past the aim point there is no final
  left, so the full pattern is planned instead (a go-around).

### The vertical profile (VNAV)

Parameterised by **distance-to-go**, not time and not waypoint index, so it is
one continuous function that the ND, the PFD's glideslope scale, and the
autopilot's vertical-speed command can all agree on with no state to keep in
sync:

```text
 alt
  │────────────────╮  cruise (level, whatever you were at when planned)
  │                 ╲  descent at cruise_descent_rad
  │                  ╰──────────╮  capture / platform altitude (level)
  │                              ╲  glideslope (3° default)
  └───────────────────────────────╲──▶  dtg → 0 at the aim point
```

Cruise altitude is floored at the capture altitude, so a craft already low gets a
level intercept instead of being told to climb in order to descend. Speed gates
(`FLAPS` / `GEAR` / `VAPP`) hang off the same distance-to-go axis; approach speed
is `1.3 × Vs` derived from the craft's own lift curve, mass, and the air density
**at the threshold** (not at cruise altitude, which would size it for the wrong
air).

## Guidance

`compute_guidance(plan, craft_state)` is pure — no integrators, no mode latches,
no memory — so every consumer reading it gets the same numbers, and anything
stateful (engagement, selection, re-plan policy) stays with the caller.

Outputs, with the pilot's sign sense throughout (positive cross-track and
positive localizer mean *you are right of where you should be*; positive
glideslope means *you are high*; positive bank command means *roll right*):

| Output | Notes |
|---|---|
| `cross_track_m`, `course_heading_rad`, `track_heading_rad`, `desired_heading_rad` | Lateral state and the track to fly to capture. |
| `dtg_m`, `along_m`, `threshold_range_m` | Route distance-to-go is **not** straight-line range to the threshold; both are published because they answer different questions. |
| `loc_deviation_rad`, `gs_deviation_rad` (+ `loc_deflection()` / `gs_deflection()`) | **Angular**, full scale ±2.5° / ±0.7° as per ILS. The same 40 m of error is full-scale on short final and nothing 20 km out — that sensitivity growth is the whole reason the instrument works. |
| `target_altitude_m`, `altitude_error_m`, `target_speed_m_s`, `next_gate` | Vertical + speed profile state. |
| `bank_command_rad`, `vertical_speed_command_m_s` | The flight director reads these today; the autoland will read the same ones. |
| `phase` (`Transition` / `Final` / `Touchdown`), `established` | `established` = inside both full-scale deflections. |

Lateral capture is an L1-style law: aim at a point one lookahead ahead on the
path, cap the correction at a 45° intercept, convert the resulting turn rate to
bank through `tan φ = ω v / g`.

## Re-plan policy — and the one rule that is about correctness

The plan is **not** rebuilt every frame; a path that jitters as airspeed wobbles
is unreadable and unflyable. It is rebuilt on selection change, and while still
maneuvering, when the craft has drifted more than 2 km from the planned path
(rate-limited to every 2 s).

**Once the craft is established on final, the plan freezes.** This is not an
optimisation. Re-planning from a position past the final approach point asks the
planner to fly back to a fix behind the craft, which it solves with a full
turn-around — the plan would loop the aircraft away from the runway it is three
kilometres from.

## Displays

### ND (the MFD's `ND` tab)

Heading-up top-down plot, craft fixed at the centre. Everything is projected
through **one** `RouteFrame` anchored at the craft, whose basis is built exactly
like the shared `hud::geo::local_enu_basis`, so ND and PFD headings agree by
construction.

- **Runways are drawn at true scale** from `StructureKind::Runway`'s real
  half-extents, with a threshold bar marking the end you land on. Only the
  *width* gets a minimum (a 90 m strip is a sub-pixel scratch at 20 km); length
  stays true, because length is the dimension a pilot is judging. A strip longer
  than the plot still draws — culling tests the nearest point of the strip, not
  its centre.
- **The route is drawn as a real polyline**, arcs included, with the final
  approach segment in its own brighter colour so "where the turn ends and the
  stabilised approach begins" reads at a glance. Waypoint symbols mark the join
  point, threshold, and aim point. A bearing caret rides the compass ring.
- Selection works **two ways**, and neither writes the selection directly (both
  send a `RouteRequest`, keeping `crate::route` the sole writer): click a runway
  on the plot (clicking the armed one again lands the other way), or use the
  `< > FLIP CLR` buttons for a strip that is off-plot.

### PFD

Localizer scale (horizontal, under the ladder) and glideslope scale (vertical,
inboard of the altitude tape), each with a deviation index, plus a two-axis
flight director and an `APPR RWY nn` annunciation. All hidden unless an approach
is armed.

**The index moves opposite the error** — the universal "fly toward the needle"
convention: being right of course puts the course to your left, so the index sits
left of centre. Inverting this yields a perfectly plausible instrument that flies
you off the side of the runway, so the mapping is pinned by a unit test.

## Verification

- **`just nd-preview`** (agent-runnable, seconds): renders the ND in eight
  approach situations — straight-in, offset intercept, overflown-and-turning-back,
  short final, reciprocal end, crosswind strip, 60 km out, idle — and writes
  `artifacts/visual/latest/nav_preview.png`. Every panel is a **real**
  `plan_approach` result, tessellated by the game's own `route::plan_display`,
  projected by the game's own `build_nav_scene`, drawn by the real shader. It is
  therefore genuine evidence about planner geometry, projection, scale, and
  symbology — and **not** evidence about ECS wiring, widget auto-selection, click
  handling, or the PFD scales, which still need an in-game check.
- Unit tests: `cargo test -p thalos_navigation` (geometry, planner, VNAV,
  guidance signs) and `cargo test -p thalos_runtime --lib -- nav_display`
  (scale, culling, decimation, range ladder).
- In-game: `just game runway-approach` → arm the runway, confirm the ND route and
  the PFD needles agree and both track as the aircraft maneuvers.

## Scope, and what is deliberately deferred

Built today: **runway approaches for aircraft**, selection, auto-generated
flyable path, VNAV, alignment indication.

Designed for but not built — the waypoint model, route frames, and guidance split
already accommodate these, so none of them needs a redesign:

| Deferred | What it needs | Notes |
|---|---|---|
| **Arbitrary waypoints** (`BL-20260730T005746Z-arbitrary-waypoints`) | A route as an ordered `Vec<Waypoint>` with leg-to-leg sequencing, plus a way to create one (map click / ND click / saved viewpoints). | The approach is already a special case of this: a two-leg route whose last leg is constrained to a runway centreline. |
| **Autoland / route autopilot** (`BL-20260730T005746Z-approach-autopilot`) | A `DemandSource` in `thalos_control` consuming `bank_command_rad` / `vertical_speed_command_m_s`, plus throttle to hold `target_speed_m_s`, plus flare and gear/flap scheduling from the speed gates. | The guidance already publishes every command it needs; the missing part is the control-bus seam and the flare law. The bank command is already clamped to the same limit the planner planned to. |
| **In-world 3D path** (`BL-20260730T005746Z-in-world-route-path`) | Render the route in the ship view as a ribbon or gates. | Purely a rendering job on `RouteState.display`; needs its own screenshot verification pass. |
| **Rovers and surface ships** | Lateral-only guidance: ignore the vertical half, replace the bank-limited planner with a curvature limit appropriate to the vehicle (or none), sequence waypoints on arrival radius. | `vertical: None` is already the correct, supported state; `Guidance`'s vertical fields are separable. |
| **Submersibles** | Vertical constraints become depth (negative altitude against the same radial axis) and the "glideslope" becomes a dive gradient. | The vertical profile is already a function of distance-to-go with no aircraft-specific assumption beyond the default angles. |
| **Terrain-aware routing** | Clearance checks along the planned path, and a minimum-safe-altitude term in the vertical profile. | Deliberately absent: the planner is pure geometry today and will happily route through a mountain. |
| **Multiple destinations / diversions** | Alternate selection and a fuel/range check. | The selector already enumerates every landable end on the body, nearest first. |
