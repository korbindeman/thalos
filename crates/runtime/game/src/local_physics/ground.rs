//! Hull-ground interaction: floor backstop, surface friction, impact destruction.
//!
//! Split out of the former monolithic `local_physics.rs` (Phase B, `docs/simulation/regimes.md`).

#[allow(unused_imports)]
use super::*;

use bevy::math::{DQuat, DVec3};
use thalos_physics_canonical::types::VesselKind;
use thalos_physics_local::avian::{
    AngularVelocity, ContactGraph, LinearVelocity, Position, Rotation,
};
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
/// hull cross the surface — and the patch heightfield is a *surface*, not a
/// closed solid, so a deep enough crossing becomes a permanent fall-through.
/// For a **wheeled** craft the stakes are higher still: its hull is filtered
/// out of solver contact with the ground entirely
/// ([`thalos_physics_local::wheeled_craft_collision_layers`]), so whenever it
/// is not on its wheels this backstop *is* its ground contact. This system is
/// the deterministic layer. It samples terrain height analytically (the same
/// [`HeightSource`] the renderer and collider read) directly under the craft,
/// lifts any penetrating hull point back to the surface, and resolves the
/// residual approach as a **contact impulse at the deepest hull point** — with
/// the angular response a real contact has, so a craft that arrives wingtip-
/// or nose-first topples flat instead of freezing in whatever attitude it hit
/// with. Because it is a closed-form height query evaluated every frame — not
/// a swept intersection — it has no tunneling failure mode at any speed.
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
    friction: Res<SurfaceFriction>,
    gear_state: Res<GearState>,
    weight_on_wheels: Res<WeightOnWheels>,
    height_sources: Res<HeightSourceRegistry>,
    sim: Res<SimulationState>,
    mut craft_q: Query<
        (
            &mut Position,
            &Rotation,
            &mut LinearVelocity,
            &mut AngularVelocity,
            &LocalCraftColliderPrimitives,
        ),
        With<LocalCraftBody>,
    >,
    mut last_emit_sim_s: Local<f64>,
) {
    // Destroyed craft are NOT exempt: a wreck is inert debris that still needs
    // a floor. A destroyed wheeled craft has no gear forces (the gear system
    // stands down) and its hull is layer-filtered out of solver ground contact
    // — this backstop is its *only* ground interface, and skipping it dropped
    // the wreck straight through the planet the moment a finite impact
    // tolerance made destruction reachable.
    if !backstop.enabled
        || !authority.owns_translation()
        || sim.simulation.vessel_kind() != VesselKind::Ship
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
    let Ok((mut position, rotation, mut linear_velocity, mut angular_velocity, primitives)) =
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
    // the surface, measured from the body centre — plus the support point it
    // occurs at, which is where the contact impulse below acts. See
    // [`deepest_hull_support`].
    let (deepest, support_center) = deepest_hull_support(r_center, rotation.0, primitives, dir);
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
    // Structured record of every intervention window (1 Hz throttle so a craft
    // resting on the backstop doesn't dominate the lane). The load-bearing
    // combination is `gear_down = 1, weight_on_wheels = 0`: a wheeled craft
    // being carried by the backstop means its gear found no ground — the
    // buried-ray / belly-slide defect signature (INC-20260729T073116Z). Read by
    // `just diag` (`gear_carried_by_backstop`).
    let sim_time_s = sim.simulation.sim_time();
    if sim_time_s - *last_emit_sim_s >= 1.0 || sim_time_s < *last_emit_sim_s {
        *last_emit_sim_s = sim_time_s;
        info!(
            target: "thalos::diagnostic::local_physics",
            event = "backstop_intervention",
            penetration_m = excess + backstop.skin_m,
            excess_m = excess,
            gear_down = gear_state.down as u32,
            weight_on_wheels = weight_on_wheels.grounded as u32,
            destroyed = sim.simulation.is_destroyed() as u32,
            "terrain floor backstop carried the hull"
        );
    }
    // Lift the hull out to skin depth. Translation only — the attitude change
    // comes from the impulse below, integrated over successive frames.
    position.0 += dir * excess;

    // Resolve the residual approach as a **contact impulse at the support
    // point**, not a CoM velocity clamp. The old clamp removed the radial CoM
    // velocity with zero torque — and gravity enters the integrator as a pure
    // linear acceleration at the CoM, so *no* toppling moment existed anywhere
    // in the system: a craft pinned on a wingtip kept whatever attitude it
    // arrived with and stood on the wing indefinitely
    // (INC-20260729T073116Z-wingtip-stand-and-buried-nosewheel). An impulse at
    // the support point gives the angular response a real contact has: the
    // pinned point is pushed up, the CoM keeps falling under gravity, and the
    // craft topples flat like a real body, settling once the support point is
    // under the CoM.
    let params = sim.simulation.ship_params();
    let mass = sim.simulation.ship_mass_kg().max(1.0);
    let inertia_body = params.moment_of_inertia;
    let inv_mass = 1.0 / mass;
    let rot = rotation.0;
    // World-frame inverse-inertia application, guarding degenerate axes.
    let inv_inertia = |v: DVec3| -> DVec3 {
        let v_body = rot.inverse() * v;
        rot * DVec3::new(
            if inertia_body.x > 0.0 {
                v_body.x / inertia_body.x
            } else {
                0.0
            },
            if inertia_body.y > 0.0 {
                v_body.y / inertia_body.y
            } else {
                0.0
            },
            if inertia_body.z > 0.0 {
                v_body.z / inertia_body.z
            } else {
                0.0
            },
        )
    };
    // Arm from the CoM (the Avian rotation pivot, pinned to the craft's real
    // CoM at spawn) to the support point. Both are body-center-relative in SLF
    // axes; the difference is translation-invariant, so the post-lift shift is
    // immaterial.
    let com_center = r_center + rot * params.center_of_mass;
    let r_arm = support_center - com_center;
    let n = dir;

    // Normal impulse: kill the support point's into-surface velocity.
    let v_point = linear_velocity.0 + angular_velocity.0.cross(r_arm);
    let v_n = v_point.dot(n);
    if v_n >= 0.0 {
        return;
    }
    let k_n = inv_mass + n.dot(inv_inertia(r_arm.cross(n)).cross(r_arm));
    let j_n = -v_n / k_n.max(1.0e-12);
    linear_velocity.0 += n * (j_n * inv_mass);
    angular_velocity.0 += inv_inertia(r_arm.cross(n * j_n));

    // Coulomb friction impulse at the same point, capped by the normal impulse
    // (`μ_kinetic · j_n`), with the matching torque — a pinned wingtip drags
    // and digs in, it doesn't glide. This fires only on deep-penetration
    // frames (past the skin); shallow resting/sliding contact is still owned
    // by [`apply_surface_friction`] and the gear model.
    let v_point = linear_velocity.0 + angular_velocity.0.cross(r_arm);
    let v_tan = v_point - n * v_point.dot(n);
    let tan_speed = v_tan.length();
    if tan_speed <= 1.0e-9 {
        return;
    }
    let t = v_tan / tan_speed;
    let k_t = inv_mass + t.dot(inv_inertia(r_arm.cross(t)).cross(r_arm));
    let j_t = (tan_speed / k_t.max(1.0e-12)).min(friction.mu_kinetic * j_n);
    linear_velocity.0 -= t * (j_t * inv_mass);
    angular_velocity.0 -= inv_inertia(r_arm.cross(t * j_t));
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
    // Destroyed craft keep hull friction (like the backstop above): a wreck
    // must grind to a stop on its belly, not slide frictionless forever.
    if !authority.owns_translation()
        || sim.simulation.vessel_kind() != VesselKind::Ship
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
    let (deepest, _) = deepest_hull_support(r_center, rotation.0, primitives, dir);
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

/// Deepest hull point along `dir` (local up) and the support point it occurs
/// at: the minimum radial coordinate (projection onto `dir`) over every
/// collider primitive's true support point in the `-dir` direction, plus that
/// point's position in the same frame as `position`. Each primitive lives in
/// the craft body frame, so its world (body-fixed) centre is
/// `position + R_craft · offset`. `dir` must be unit-length. Shared by
/// [`terrain_floor_backstop`] (penetration → lift + contact impulse) and
/// [`apply_surface_friction`] (contact test). Returns `f64::INFINITY` depth
/// (and `position` as the point) for an empty primitive list.
pub(crate) fn deepest_hull_support(
    position: DVec3,
    rotation: DQuat,
    primitives: &LocalCraftColliderPrimitives,
    dir: DVec3,
) -> (f64, DVec3) {
    let mut deepest = f64::INFINITY;
    let mut support = position;
    for prim in &primitives.0 {
        let center = position + rotation * prim.offset_m;
        let prim_rot = rotation * prim.rotation;
        // `a = (R_craft · R_prim)^T · dir` is unit-length (rotations preserve
        // norm, `dir` is unit); `shape_support_point` returns the point of the
        // centred shape minimising `a · p` — the shape's support point in the
        // `-a` (toward-surface) direction.
        let a = prim_rot.inverse() * dir;
        let point = center + prim_rot * shape_support_point(prim.shape, a);
        let depth = point.dot(dir);
        if depth < deepest {
            deepest = depth;
            support = point;
        }
    }
    (deepest, support)
}

/// The point `p` of a centred primitive shape minimising `a · p` — the support
/// point in the `-a` direction, in the primitive's local frame. `a` must be
/// unit-length. Exact for cuboid/sphere/capsule/cylinder; the cone is bounded
/// conservatively by its enclosing cylinder (a backstop erring toward catching
/// the hull slightly early is safe). Face/edge-parallel directions pick the
/// face centre so a flat resting contact gets no spurious torque arm.
///
/// Shape conventions: `Collider::cuboid` takes full side lengths (support uses
/// half-extents). Parry capsule/cylinder/cone principal axis is local Y;
/// `length` is the capsule's segment length (between hemisphere centres),
/// `height` the full cylinder/cone height.
pub(crate) fn shape_support_point(shape: LocalPrimitiveShape, a: DVec3) -> DVec3 {
    // `signum` that treats a (near-)zero component as zero, so a face-parallel
    // direction supports at the face *centre* instead of an arbitrary corner —
    // a flat resting contact must not manufacture a torque arm.
    let sgn = |c: f64| if c.abs() < 1.0e-9 { 0.0 } else { c.signum() };
    match shape {
        LocalPrimitiveShape::Cuboid { x, y, z } => {
            DVec3::new(-sgn(a.x) * x, -sgn(a.y) * y, -sgn(a.z) * z) * 0.5
        }
        LocalPrimitiveShape::Sphere { radius } => -a * radius,
        LocalPrimitiveShape::Capsule { radius, length } => {
            DVec3::new(0.0, -sgn(a.y) * length * 0.5, 0.0) - a * radius
        }
        LocalPrimitiveShape::Cylinder { radius, height }
        | LocalPrimitiveShape::Cone { radius, height } => {
            let s = (a.x * a.x + a.z * a.z).sqrt();
            let rim = if s > 1.0e-12 {
                DVec3::new(-a.x / s, 0.0, -a.z / s) * radius
            } else {
                DVec3::ZERO
            };
            rim + DVec3::new(0.0, -sgn(a.y) * height * 0.5, 0.0)
        }
    }
}

/// Short ring buffer of recent surface-relative **sink rates** (into-surface
/// radial speed). The impact detector reads its **peak** at contact onset
/// rather than the instantaneous value because `SweptCcd` books the velocity
/// arrest a frame or two after the geometric sweep, and speculative collision
/// can shave the final approach frame — by contact-start the instantaneous
/// sink is already damped, but the peak across the last ~8 frames is still the
/// true approach (gravity changes it by only ~1 m/s over that window).
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
/// solved. Each frame we record the craft's **sink rate** — the into-surface
/// (radial) component of its surface-relative velocity, *not* the full speed:
/// a runway touchdown carries 60+ m/s of horizontal ground speed that a
/// wheels-first landing sheds harmlessly over the rollout, and counting it as
/// impact speed would destroy the craft on every normal landing. The radial
/// component is what the airframe actually absorbs at contact. On the
/// **rising edge** of ground contact we compare the windowed peak sink rate
/// against [`ShipParameters::impact_tolerance_m_s`] and destroy the craft if
/// it was coming down too hard.
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
    let Ok((linear_velocity, position)) = craft_q.get(bubble.craft_entity) else {
        return;
    };

    // Ships integrate in the surface-local frame and the ground collider is
    // static there, so the craft's SLF velocity is already surface-relative (a
    // craft resting on the surface reads ~0); no `ω × r` subtraction needed.
    // The recorded quantity is the *sink rate*: the into-surface (negative
    // radial) component only — see the system doc for why full speed is wrong.
    let up = bubble
        .frame
        .body_center_offset(position.0)
        .normalize_or_zero();
    let sink_rate = (-linear_velocity.0.dot(up)).max(0.0);
    speed_window.push(sink_rate);

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
            "VESSEL DESTROYED: terrain impact at {:.1} m/s sink rate (tolerance {:.1} m/s)",
            impact_speed, tolerance
        );
        sim.simulation.mark_destroyed(impact_speed);
    }
}
