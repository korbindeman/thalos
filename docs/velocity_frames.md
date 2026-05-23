# Velocity reference frames (navball speed mode)

Spec for the KSP-style navball **speed mode** — a per-craft *velocity
reference frame* (Orbit / Surface / Target) that sets which reference
velocity is subtracted before computing the speed readout, the navball
velocity markers, and the SAS attitude holds.

**Status: implemented 2026-05-21** (see §10 for what shipped + resolved decisions). Sections 2–3 describe today's code
(with file:line); sections 4–7 are the agreed target design; section 8
is the change set; section 9 lists what to verify.

> **Naming clash with `surface_gameplay.md`.** That doc uses "surface
> mode" for the *landed / body-fixed authority* treatment
> (warp-while-landed, map ground track, `AuthorityMode::BodyFixed`).
> This doc's **Surface** is the *navball velocity frame*. Both reference
> the rotating-surface frame but are independent features. Where they
> meet: the landed signal (`BodyFixed`) is one input to this doc's
> auto-switch (§5).

## 1. Goals

- Player can read and fly **orbital** velocity (relative to the body's
  center, inertial axes) or **surface** velocity (relative to the
  co-rotating surface), KSP-style.
- The active frame drives the **speed readout**, the **navball
  markers**, and the **SAS holds** consistently — one source of truth,
  no per-consumer divergence.
- The frame **auto-selects** from the craft's situation, with a
  **manual override** (click the readout). Auto improves on KSP for
  airless bodies (§5).
- **Target** frame (velocity relative to a selected target) is designed
  in from the start but inert until targeting exists for it.

## 2. Current state

### 2.1 The basis is computed three times, all hardcoded to orbital

Prograde / normal / radial are recomputed independently in three places,
each using the body-centered inertial frame (`rel_vel = craft.vel −
body.vel`):

- **SAS holds** — `compute_target_direction`
  ([navigation.rs:179](crates/game/src/navigation.rs:179)); Prograde arm
  at [:191](crates/game/src/navigation.rs:191), Normal via
  `orbital_frame` at [:205](crates/game/src/navigation.rs:205), Radial at
  [:223](crates/game/src/navigation.rs:223).
- **Navball markers** — `compute_marker_directions`
  ([markers.rs:274](crates/game/src/navball/markers.rs:274)); the
  `rel_pos`/`rel_vel` basis at
  [:302](crates/game/src/navball/markers.rs:302).
- **Speed readout** — `update`
  ([flight_panel.rs:153](crates/game/src/hud/flight_panel.rs:153)); label
  hardcoded `"ORBITAL VELOCITY"` at
  [:105](crates/game/src/hud/flight_panel.rs:105).

Adding Surface/Target by branching each of the three is exactly the
divergence the "one source of truth" invariant exists to prevent. §4
collapses them into one.

### 2.2 The frame math already exists (pure, in physics_canonical)

- Orbital reference velocity = `body.velocity` (body-centered inertial;
  see [body_centered.rs](crates/physics_canonical/src/body_centered.rs)).
- Surface reference velocity = `body.velocity + ω × r` —
  [`body_fixed_surface_velocity`](crates/physics_canonical/src/body_fixed.rs:18).
  `BodyState` carries `orientation` + `angular_velocity`.
- `orbital_frame(ship_pos, ship_vel, body_pos, body_vel)` returns the
  orbital prograde/normal/radial triad and already lives in
  `thalos_physics_canonical::maneuver` (used by navigation.rs).

### 2.3 SAS holds and markers already enumerate the directions

- `NavigationMode`
  ([navigation.rs:24](crates/game/src/navigation.rs:24)) — Stability,
  Prograde, Retrograde, Normal, AntiNormal, Radial{In,Out}, Target,
  AntiTarget, ManeuverNode.
- `MarkerKind` ([markers.rs:41](crates/game/src/navball/markers.rs:41)) —
  the same nine markers.
- **"Target" already exists, but as a *pointing* direction** ("aim nose
  at target"), not a *velocity frame*. KSP's Target speed mode reframes
  prograde/speed to be relative to the target's motion; both coexist
  (pink toward/away markers + yellow relative-velocity markers). The
  existing pink markers stay; this doc adds the velocity reframing.

### 2.4 The navball sphere is frame-independent

`NavballFrame`
([navball/attitude.rs](crates/game/src/navball/attitude.rs)) paints the
sphere in local East/North/Up. The speed mode does **not** rotate the
sphere — only the velocity markers + readout change. Contained change.

### 2.5 UI toggle pattern + situation signals

- The Ship/Map `ViewMode` toggle
  ([hud/view_mode_panel.rs](crates/game/src/hud/view_mode_panel.rs)) is
  the resource+button pattern to mirror.
- Landed signal: `AuthorityMode::BodyFixed`
  ([canonical.rs:144](crates/physics_canonical/src/canonical.rs:144)).
- Altitude: the HUD GND readout already computes AGL via the
  `HeightSource` registry (orbital_panel.rs); altitude above the
  reference sphere is `|r| − body.radius`.

## 3. Target abstraction today

`TargetBody { target: Option<usize> }` ([target.rs](crates/game/src/target.rs))
is a celestial-body world-id, forwarded into the `Simulation`. Good
enough for the Target frame's reference velocity (`= target_body.velocity`)
now; vessel-vs-vessel uses the same path later.

## 4. Design — one basis, one writer

### 4.1 `VelocityReferenceFrame` + `NavBasis` (physics_canonical)

```rust
pub enum VelocityReferenceFrame { Orbit, Surface, Target }

pub struct NavBasis {
    pub reference_vel: DVec3, // frame velocity at the craft
    pub prograde: DVec3,      // normalize(craft.vel − reference_vel)
    pub normal: DVec3,        // normalize(rel_pos × rel_vel)
    pub radial: DVec3,        // normalize(rel_pos)  (out from dominant body)
    pub speed: f64,           // |craft.vel − reference_vel|
}

pub fn nav_basis(
    frame: VelocityReferenceFrame,
    craft: StateVector,
    body: &BodyState,           // dominant body
    target: Option<&BodyState>, // for the Target frame
) -> Option<NavBasis>;
```

- `rel_pos = craft.position − body.position`.
- `reference_vel`: Orbit → `body.velocity`; Surface → `body.velocity +
  body.angular_velocity × rel_pos`; Target → `target.velocity` (None ⇒
  fn returns `None`).
- Pure, Bevy-free, unit-tested in physics_canonical (this crate is
  exempt from the no-gen-tests rule — that rule is terrain-only).

### 4.2 `VelocityFrameState` (game resource, sole writer)

One resource owns the active frame + computed basis; one system writes
it each frame (document **Sole writer:** in the doc comment, per the
single-writer invariant). Every consumer reads.

```rust
#[derive(Resource)]
pub struct VelocityFrameState {
    pub active: VelocityReferenceFrame,
    pub basis: Option<NavBasis>,
    manual_override: Option<VelocityReferenceFrame>,
    last_suggested: VelocityReferenceFrame, // override-clear edge detection
}
```

### 4.3 Consumers read the shared basis

- markers.rs `compute_marker_directions` → read `state.basis`
  (prograde/normal/radial). Target/ManeuverNode markers unchanged.
- navigation.rs `compute_target_direction` directional arms → read
  `state.basis`. Stability/Target/AntiTarget/ManeuverNode unchanged.
- flight_panel.rs → `basis.speed` + label from `active`.

This deletes the three duplicate basis computations (§2.1).

## 5. Auto-switch + manual override

```
suggested =
    Surface  if landed (AuthorityMode::BodyFixed) OR altitude < ceiling
    Orbit    otherwise
active = manual_override.unwrap_or(suggested)
```

- `altitude = |craft.position − body.position| − body.radius` (above the
  reference sphere; cheap and terrain-independent — on-ground is covered
  by the `BodyFixed` term and by `altitude → 0`).
- `ceiling` = the body's atmosphere top (Kármán) where it has one, else a
  **per-body authored** value (§6). **Target is never auto-selected.**
- **Override stickiness:** `manual_override` is set by a readout click
  and **cleared when `suggested` changes** from `last_suggested`
  (atmosphere/altitude boundary crossing, land, take-off, SOI change),
  so auto reclaims at the next real situation boundary. KSP-faithful.

**Improvement over KSP:** KSP's auto-switch is purely situation-based —
airless bodies have no in-atmosphere "Flying" band, so the navball stays
in Orbit through the whole descent and the player must manually switch to
Surface to land. The `altitude < ceiling` term makes airless descents
auto-select Surface. Deliberate, small divergence.

## 6. Per-body ceiling authoring

- Add an optional `surface_frame_ceiling_m` to the body definition;
  **default derived from radius** (~0.3–1% of radius, tuned by feel) so
  it scales across the system's size range rather than one global metres
  value that is huge on a moon and tiny on Thalos.
- For bodies with an atmosphere, the ceiling is the atmosphere top; the
  authored value is the airless fallback / explicit override.
- Exact home (the `solar_system.ron` physical block vs the per-body
  file) and the atmosphere-top query path are implementation-time
  decisions (§9).

## 7. Switching UI

- Make the speed readout
  ([flight_panel.rs:105](crates/game/src/hud/flight_panel.rs:105))
  interactive; a click cycles **Orbit → Surface → Target → Orbit**,
  **skipping Target when `TargetBody` is unset** (degrades to a 2-way
  toggle). Sets `manual_override`. Mirror the `view_mode_panel.rs` button
  pattern.
- Readout label follows `active`: `ORBITAL / SURFACE / TARGET VELOCITY`.
- No keybind (decision: click-only).

## 8. Change set (low-risk first)

1. **[physics_canonical]** `VelocityReferenceFrame` + `nav_basis(...)` +
   unit tests (per-frame reference velocity; degenerate zero-rel-vel ⇒
   `None`; the surface ω×r term). Bevy-free.
2. **[game]** `VelocityFrameState` resource + sole-writer system: compute
   `suggested` (needs altitude + ceiling), resolve `active`, fill
   `basis`. Ceiling source: radius-default first; atmosphere-top +
   authored override second.
3. **[game]** Route the three consumers (§4.3) through `state.basis`. No
   behavior change at the default (Orbit) — verify readout/markers/SAS
   are identical before exposing the toggle.
4. **[game]** Interactive readout: cycle + `manual_override` +
   boundary-clear; label follows mode.
5. **[game/assets]** Per-body ceiling field + default-from-radius; wire
   atmosphere top where available.
6. Target stays inert (out of the cycle) until a target can be set for
   it; no extra "enable" work — the path is built in step 1.

## 9. To verify / open

- **Atmosphere-top query** from the game side (`thalos_atmosphere` /
  Kármán authoring, [atmosphere.md](docs/atmosphere.md)) — confirm or use
  the radius-default until wired.
- **Per-body ceiling default fraction** — tune live.
- **Radial/normal in Target mode** — KSP de-emphasizes them; decide
  whether to hide. Minor UX, defer.
- **EVA landed → Surface:** EVA isn't in `BodyFixed` yet
  (surface_gameplay.md §2.2/§4.2); the `altitude < ceiling` term still
  selects Surface for low EVA, and landed-EVA Surface becomes exact once
  the EVA→BodyFixed migration lands.
- **Invariants at implementation:** the new sole-writer resource
  (`VelocityFrameState` ← its writer) belongs in CLAUDE.md's
  single-writer list, and the triplication removal strengthens "one
  source of truth." Update CLAUDE.md + ROADMAP when the work is
  scheduled/landed (per the "announce before you make it" rule) — not
  done here, since this is design only.

## 10. Implementation (landed 2026-05-21)

Shipped close to §4–§8. File map:

- **`thalos_physics_canonical::velocity_frame`** (new) —
  `VelocityReferenceFrame`, `NavBasis`, the pure `nav_basis` fn + unit
  tests. `NavBasis` carries `reference_vel` + `speed` (always defined) and
  `prograde`/`normal`/`radial` as `Option` (preserving the navball's
  per-marker hide behavior).
- **`BodyDefinition.surface_frame_ceiling_m: Option<f64>`** + parsing
  plumbing (`BodyFile`/`BodyDetailsFile`, merge + build). Authored in the
  per-body file (`assets/bodies/<name>.ron`); Mira carries an example value.
- **`thalos_game::velocity_frame`** (new) — `VelocityFrameState` resource +
  `update_velocity_frame` writer + `VelocityFramePlugin`, plus `next_frame`
  for the click cycle. Registered in `main.rs`; the writer runs in
  `SimStage::Physics` `.before(bridge::handle_attitude_controls)`.
- **SAS** — `navigation::compute_attitude_control` /
  `compute_target_direction` take the active frame; the velocity holds
  resolve through `active_nav_basis` (ephemeris-sourced, Physics stage).
- **Navball** — `navball::markers::compute_marker_directions` reads the
  active frame + `nav_basis` (solar-system-snapshot-sourced).
- **Readout** — `hud::flight_panel` is now a clickable `Button`; `update`
  drives the value + label from `nav_basis`, `handle_velocity_frame_click`
  cycles the frame.

Resolved open decisions:

- **Atmosphere top** = `terrestrial_atmosphere.karman_line_m` (f32 → f64).
  Gas giants have no surface ceiling and never enter Surface.
- **Ceiling home** = per-body file, plumbed like `terrestrial_atmosphere`.
  Resolution order: Kármán line → authored `surface_frame_ceiling_m` →
  `DEFAULT_CEILING_RADIUS_FRACTION` (0.5% of radius).
- **Resource shape** — `VelocityFrameState` holds only the *active frame*
  (+ sticky-override bookkeeping, modeled on `hud::orbital_panel`'s
  `AltitudeDisplay`); consumers evaluate `nav_basis` themselves at their
  stage-correct body-state source. This sidesteps the Physics-vs-`Sync`
  staleness trap, so no precomputed basis is stored. `active` has a single
  writer (`update_velocity_frame`); the click handler only sets the override
  (the `AltitudeDisplay` two-writer UI pattern), so this is intentionally
  *not* added to CLAUDE.md's strict single-writer list.

Still open (unchanged from §9):

- Radial/normal marker visibility in Target mode (KSP de-emphasizes them).
- Per-body ceiling tuning by feel.
- Landed-EVA Surface becomes exact once the EVA→BodyFixed migration
  (surface_gameplay.md §4.2) lands; until then the `altitude < ceiling`
  term already selects Surface for low EVA.
