//! Hull-ground interaction: floor backstop, surface friction, impact destruction.
//!
//! Split out of the former monolithic `local_physics.rs` (Phase B, `docs/simulation/regimes.md`).

#[allow(unused_imports)]
use super::*;

use bevy::math::{DQuat, DVec3};
use thalos_physics_canonical::types::VesselKind;
use thalos_physics_local::avian::{ContactGraph, LinearVelocity, Position, Rotation};
use thalos_physics_local::{
    ActiveLocalBubble, HeightSourceRegistry, LocalCraftBody, LocalCraftColliderPrimitives,
    LocalPrimitiveShape, craft_contacts_terrain,
};

use crate::rendering::SimulationState;
use crate::sim_clock::SimClock;

/// Coulomb friction tuning for a ship resting/sliding on its **hull** — gearless
/// craft (landers, rockets) or a craft on its belly. Wheeled craft get their
/// tangential ground reaction from the landing-gear model instead. Stick/slip
/// with a static and a kinetic coefficient, so a landed craft comes to a true
/// rest in finite time rather than the indefinite frictionless slide it had
/// before (the only ground force was the floor backstop, which removes the
/// into-surface velocity component only). Reflect-registered (for a future
/// debug UI); edit the defaults and rebuild to tune.
#[derive(Resource, Clone, Copy, Debug, Reflect)]
#[reflect(Resource)]
pub struct SurfaceFriction {
    /// Static coefficient: a craft whose per-frame tangential slip is below
    /// `μ_static · g · dt` sticks (its surface-parallel velocity is zeroed).
    pub mu_static: f64,
    /// Kinetic coefficient (≤ static): a faster-sliding craft decelerates at
    /// `μ_kinetic · g` along its slip direction until it drops to the stick band.
    pub mu_kinetic: f64,
    /// How close the deepest hull point must sit to the surface (metres) to
    /// count as in contact — a small band above the floor backstop's lift so
    /// a craft held exactly at the surface still reads as grounded.
    pub contact_margin_m: f64,
}

impl Default for SurfaceFriction {
    fn default() -> Self {
        // Metal/composite hull on rock/regolith: high grip, true stop. A small
        // static>kinetic gap gives the usual break-free-then-slide feel.
        Self {
            mu_static: 0.8,
            mu_kinetic: 0.6,
            contact_margin_m: 0.3,
        }
    }
}

/// Analytic ground backstop — a deterministic safety net that guarantees the
/// craft hull can never tunnel through the terrain, independent of the collision
/// mesh.
///
/// The terrain collider patch + `SweptCcd` are the *primary* contact layer, but
/// any mesh-based contact is probabilistic: a fast enough descent, an
/// edge-of-patch / not-yet-streamed tile, or a single missed sweep can let the
/// hull cross the surface — and the patch is a one-sided trimesh with no
/// "inside" to push back out of, so one missed frame becomes a permanent
/// fall-through. This system is the deterministic backstop. It samples terrain
/// height analytically (the same [`HeightSource`] the renderer and collider
/// read) directly under the craft and lifts any penetrating hull point back to
/// the surface, killing the into-surface velocity component. Because it is a
/// closed-form height query evaluated every frame — not a swept intersection —
/// it has no tunneling failure mode at any speed.
///
/// Ships only, and only while Avian owns translation ([`AvianRole::Full`]) —
/// the sole regime where Avian-integrated motion can drive the hull into the
/// ground. Under Kepler/OnRails coast the craft is far above the handoff band;
/// under warp / `BodyFixed` the pose is analytic and pinned. EVA is exempt (no
/// hull collider; the grounded controller owns its pose and clamps its own
/// terrain height). Runs after the force systems and just before
/// [`readback_local_craft`], so the corrected pose is what flows into canonical;
/// in `Full` the snap does not overwrite `Position`, so the correction persists
/// as the start state for Avian's next integration.
pub(crate) fn terrain_floor_backstop(
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    backstop: Res<TerrainFloorBackstop>,
    height_sources: Res<HeightSourceRegistry>,
    sim: Res<SimulationState>,
    mut craft_q: Query<
        (
            &mut Position,
            &Rotation,
            &mut LinearVelocity,
            &LocalCraftColliderPrimitives,
        ),
        With<LocalCraftBody>,
    >,
) {
    if !backstop.enabled
        || !authority.owns_translation()
        || sim.simulation.vessel_kind() != VesselKind::Ship
        || sim.simulation.is_destroyed()
    {
        return;
    }
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let Some(height_source) = height_sources.get(bubble.body_id) else {
        return;
    };
    let body = &sim.system.bodies[bubble.body_id];
    let Ok((mut position, rotation, mut linear_velocity, primitives)) =
        craft_q.get_mut(bubble.craft_entity)
    else {
        return;
    };
    // Ship Avian state lives in the surface-local frame: `Position` is
    // anchor-relative (small), so recover the body-center offset for the
    // radial direction, and convert to body-fixed axes for the height query.
    // `LinearVelocity` is surface-relative (a parked craft reads ~0).
    let r_center = bubble.frame.body_center_offset(position.0);
    let Some(dir) = r_center.try_normalize() else {
        return;
    };
    let dir_body = bubble.frame.rotation_body_to_frame.inverse() * dir;
    let Some(height) = height_source.sample_height_m(dir_body.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
    else {
        return;
    };
    let surface_radius = body.radius_m + height as f64;

    // Deepest hull point along the radial — the lowest the hull reaches toward
    // the surface, measured from the body centre. See [`deepest_hull_radial`].
    let deepest = deepest_hull_radial(r_center, rotation.0, primitives, dir);
    if !deepest.is_finite() {
        return;
    }

    // Only the depth *past the skin* is corrected — see
    // [`TerrainFloorBackstop::skin_m`]. Resting on the collider (penetration
    // shallower than the skin) leaves this ≤ 0 and the backstop stands down, so
    // it never becomes a second hard floor fighting the soft contact solver.
    let excess = (surface_radius - deepest) - backstop.skin_m;
    if excess <= 0.0 {
        return;
    }
    if excess > 1.0 {
        // A correction this deep means the primary contact layer let the hull
        // sink a metre+ past the skin — exactly the tunnelling the backstop
        // exists to catch. Surface it so the mesh layer's gap can be investigated.
        debug!(
            "terrain floor backstop caught a {:.2} m hull penetration (past {:.2} m skin) over {}",
            excess + backstop.skin_m,
            backstop.skin_m,
            body.name
        );
    }
    // Lift the hull out to skin depth and remove only the into-surface (negative
    // radial) velocity, so tangential motion (taxi / slide) is untouched.
    position.0 += dir * excess;
    let radial_speed = linear_velocity.0.dot(dir);
    if radial_speed < 0.0 {
        linear_velocity.0 -= dir * radial_speed;
    }
}

/// Tuning + kill-switch for [`terrain_floor_backstop`]. Reflect-registered
/// (for a future debug UI); edit the defaults and rebuild to toggle / tune it
/// while diagnosing ground contact.
#[derive(Resource, Debug, Clone, Copy, Reflect)]
#[reflect(Resource)]
pub struct TerrainFloorBackstop {
    /// Master enable. Off → the backstop never moves the craft (diagnostic
    /// isolation lever).
    pub enabled: bool,
    /// Allowed penetration skin, metres.
    ///
    /// The backstop is a *deep-penetration safety net*, **not** a zero-tolerance
    /// surface clamp. The primary contact layer — the terrain/runway collider,
    /// gear suspension, and hull friction — is a soft (XPBD) solver that tolerates
    /// small per-frame penetration and resolves it over substeps. A backstop that
    /// clamps at zero depth becomes a *second, stiffer* hard floor that disagrees
    /// with the collider by sub-metre amounts and fights it every frame →
    /// uncontrollable jitter. So the backstop ignores penetration shallower than
    /// this skin and corrects only the excess below it: normal resting contact is
    /// left entirely to the solver, while genuine tunnelling (metres deep) is
    /// still caught. The craft can never end up more than `skin_m` below the
    /// surface, and can never pass through.
    pub skin_m: f64,
}

impl Default for TerrainFloorBackstop {
    fn default() -> Self {
        Self {
            enabled: true,
            skin_m: 0.5,
        }
    }
}

/// Coulomb surface friction for a ship resting/sliding on its **hull** (no
/// weight on wheels). Velocity-level stick/slip on the tangential
/// (surface-parallel) component of the craft's body-fixed velocity, applied the
/// same frame [`terrain_floor_backstop`] removes the into-surface component and
/// just before [`readback_local_craft`] flows the corrected velocity into
/// canonical. Brings a landed gearless craft to a true rest in finite time
/// instead of the indefinite slide it had before — the only ground force was the
/// backstop, which touches the normal direction only.
///
/// Done at the velocity level (like the backstop), not as a force into the
/// acceleration accumulator: a velocity-level stick/slip cancels exactly within
/// the friction budget each frame, so it reaches a true stop regardless of step
/// size, whereas an `∝ v` force law only ever decays asymptotically (the bug
/// this replaces). Gravity here is central (radial), so it has no tangential
/// component — friction only has to remove residual slip, and the normal load
/// per unit mass is just `g = μ/r²`.
///
/// Wheeled craft are skipped: when any wheel bears load the landing-gear model
/// owns the tangential ground reaction (lateral grip + Coulomb rolling) and the
/// suspension holds the hull clear of the surface.
pub(crate) fn apply_surface_friction(
    clock: Res<SimClock>,
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    height_sources: Res<HeightSourceRegistry>,
    weight_on_wheels: Res<WeightOnWheels>,
    tuning: Res<SurfaceFriction>,
    sim: Res<SimulationState>,
    mut craft_q: Query<
        (
            &Position,
            &Rotation,
            &mut LinearVelocity,
            &LocalCraftColliderPrimitives,
        ),
        With<LocalCraftBody>,
    >,
) {
    if !authority.owns_translation()
        || sim.simulation.vessel_kind() != VesselKind::Ship
        || sim.simulation.is_destroyed()
        || weight_on_wheels.grounded
    {
        return;
    }
    let dt = clock.delta_secs_f64();
    if dt <= 0.0 {
        return;
    }
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let Some(height_source) = height_sources.get(bubble.body_id) else {
        return;
    };
    let body = &sim.system.bodies[bubble.body_id];
    let Ok((position, rotation, mut linear_velocity, primitives)) =
        craft_q.get_mut(bubble.craft_entity)
    else {
        return;
    };
    // Surface-local frame: recover the body-center offset for the radial
    // direction (local up), body-fixed axes for the height query.
    // `LinearVelocity` is surface-relative (the ground is static here, so
    // tangential velocity is the slip directly).
    let r_center = bubble.frame.body_center_offset(position.0);
    let Some(dir) = r_center.try_normalize() else {
        return;
    };
    let dir_body = bubble.frame.rotation_body_to_frame.inverse() * dir;
    let Some(height) = height_source.sample_height_m(dir_body.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
    else {
        return;
    };
    let surface_radius = body.radius_m + height as f64;

    // In hull contact? The backstop holds the deepest hull point ~at the surface;
    // a small margin keeps a resting craft reading grounded across sample noise.
    let deepest = deepest_hull_radial(r_center, rotation.0, primitives, dir);
    if !deepest.is_finite() || deepest > surface_radius + tuning.contact_margin_m {
        return;
    }

    let v = linear_velocity.0;
    let v_tan = v - dir * v.dot(dir);
    let speed = v_tan.length();
    if speed < 1.0e-9 {
        return;
    }
    // Normal load per unit mass = gravity into the (near-radial) surface. Mass
    // cancels in the velocity-level form, so it never enters.
    let g = body.gm / r_center.length_squared().max(1.0);
    let static_budget = tuning.mu_static * g * dt;
    if speed <= static_budget {
        // Stick: remove all tangential motion, keep the radial component.
        linear_velocity.0 = v - v_tan;
    } else {
        // Slip: kinetic friction opposes the slip at `μ_kinetic · g`.
        linear_velocity.0 = v - (v_tan / speed) * (tuning.mu_kinetic * g * dt);
    }
}

/// Deepest hull point along `dir` (local up): the minimum radial coordinate
/// (projection onto `dir`) over every collider primitive's true support point in
/// the `-dir` direction. Each primitive lives in the craft body frame, so its
/// world (body-fixed) centre is `position + R_craft · offset`. `dir` must be
/// unit-length. Shared by [`terrain_floor_backstop`] (penetration → lift) and
/// [`apply_surface_friction`] (contact test).
pub(crate) fn deepest_hull_radial(
    position: DVec3,
    rotation: DQuat,
    primitives: &LocalCraftColliderPrimitives,
    dir: DVec3,
) -> f64 {
    let mut deepest = f64::INFINITY;
    for prim in &primitives.0 {
        let center = position + rotation * prim.offset_m;
        // `a = (R_craft · R_prim)^T · dir` is unit-length (rotations preserve
        // norm, `dir` is unit); `shape_min_support` returns the shape's signed
        // support depth along `-a`, i.e. how far below its centre the hull
        // reaches along the radial.
        let a = (rotation * prim.rotation).inverse() * dir;
        deepest = deepest.min(center.dot(dir) + shape_min_support(prim.shape, a));
    }
    deepest
}

/// Minimum of `a · p` over the points `p` of a centred primitive shape — the
/// signed depth of the shape's support point in the `-a` direction. `a` must be
/// unit-length; the result is ≤ 0. Exact for cuboid/sphere/capsule/cylinder;
/// the cone is bounded conservatively by its enclosing cylinder (a backstop
/// erring toward catching the hull slightly early is safe).
pub(crate) fn shape_min_support(shape: LocalPrimitiveShape, a: DVec3) -> f64 {
    match shape {
        // `Collider::cuboid` takes full side lengths; support uses half-extents.
        LocalPrimitiveShape::Cuboid { x, y, z } => {
            -(a.x.abs() * x + a.y.abs() * y + a.z.abs() * z) * 0.5
        }
        LocalPrimitiveShape::Sphere { radius } => -radius,
        // Parry capsule/cylinder/cone principal axis is local Y. `length` is the
        // capsule's segment length (between hemisphere centres); `height` is the
        // full cylinder/cone height.
        LocalPrimitiveShape::Capsule { radius, length } => -(a.y.abs() * length * 0.5) - radius,
        LocalPrimitiveShape::Cylinder { radius, height }
        | LocalPrimitiveShape::Cone { radius, height } => {
            -(a.y.abs() * height * 0.5) - radius * (a.x * a.x + a.z * a.z).sqrt()
        }
    }
}

/// Short ring buffer of recent surface-relative approach speeds. The
/// impact detector reads its **peak** at contact onset rather than the
/// instantaneous speed because `SweptCcd` books the velocity arrest a frame
/// or two after the geometric sweep, and speculative collision can shave
/// the final approach frame — by contact-start the instantaneous speed is
/// already damped, but the peak across the last ~8 frames is still the true
/// approach speed (gravity changes it by only ~1 m/s over that window).
#[derive(Default)]
pub(crate) struct ImpactSpeedWindow {
    samples: [f64; Self::LEN],
    idx: usize,
}

impl ImpactSpeedWindow {
    const LEN: usize = 8;

    fn push(&mut self, speed_m_s: f64) {
        self.samples[self.idx] = speed_m_s;
        self.idx = (self.idx + 1) % Self::LEN;
    }

    fn peak(&self) -> f64 {
        self.samples.iter().copied().fold(0.0, f64::max)
    }

    fn clear(&mut self) {
        self.samples = [0.0; Self::LEN];
        self.idx = 0;
    }
}

/// Detect a destroying terrain impact and mark the craft destroyed.
///
/// Only meaningful while Avian owns translation (`AvianRole::Full`), which
/// is exactly when a terrain patch is attached and contacts are being
/// solved. Each frame we record the craft's surface-relative approach speed
/// (`v − ω × r`, the speed the co-rotating terrain collider actually sees);
/// on the **rising edge** of contact with the terrain patch we compare the
/// windowed peak approach speed against [`ShipParameters::impact_tolerance_m_s`]
/// and destroy the craft if it was coming in too hard.
///
/// EVA is exempt (the capsule has no collider, so no contacts). A craft that
/// is already destroyed short-circuits so debris settling on the ground
/// doesn't re-trigger. See `docs/simulation/surface.md`.
pub(crate) fn detect_terrain_impact(
    contact_graph: Res<ContactGraph>,
    active: Res<ActiveLocalBubble>,
    authority: Res<AvianAuthority>,
    weight_on_wheels: Res<WeightOnWheels>,
    mut sim: ResMut<SimulationState>,
    craft_q: Query<(&LinearVelocity, &Position), With<LocalCraftBody>>,
    mut speed_window: Local<ImpactSpeedWindow>,
    mut was_touching: Local<bool>,
) {
    // Only Full integrates contacts and owns the craft's velocity. Outside
    // it (coast, warp/Paused, BodyFixed) there is nothing to detect, and the
    // snapped canonical velocity would read as a false high-speed approach.
    let owns_translation = authority.owns_translation();
    let Some(bubble) = active.bubble.as_ref() else {
        *was_touching = false;
        speed_window.clear();
        return;
    };
    if !owns_translation
        || sim.simulation.vessel_kind() == VesselKind::Eva
        || sim.simulation.is_destroyed()
    {
        *was_touching = false;
        speed_window.clear();
        return;
    }
    let Ok((linear_velocity, _position)) = craft_q.get(bubble.craft_entity) else {
        return;
    };

    // Ships integrate in the surface-local frame and the ground collider is
    // static there, so the craft's SLF velocity is already the surface-relative
    // approach speed (a craft resting on the surface reads ~0). No `ω × r`
    // subtraction needed.
    let approach_speed = linear_velocity.0.length();
    speed_window.push(approach_speed);

    // Ground contact onset. A wheeled craft's hull is filtered out of solver
    // contact with the ground (gear is its sole interface), so use the gear's
    // weight-on-wheels signal for it; a gearless craft still contacts the
    // terrain heightfield directly, so fall back to the contact graph.
    let hull_touches = bubble
        .terrain_entity
        .is_some_and(|t| craft_contacts_terrain(&contact_graph, bubble.craft_entity, t));
    let touching = weight_on_wheels.grounded || hull_touches;
    let contact_started = touching && !*was_touching;
    *was_touching = touching;
    if !contact_started {
        return;
    }

    let impact_speed = speed_window.peak();
    let tolerance = sim.simulation.ship_params().impact_tolerance_m_s;
    if impact_speed > tolerance {
        warn!(
            "VESSEL DESTROYED: terrain impact at {:.1} m/s (tolerance {:.1} m/s)",
            impact_speed, tolerance
        );
        sim.simulation.mark_destroyed(impact_speed);
    }
}
