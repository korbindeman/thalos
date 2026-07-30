// REPRO — `propagator-astro-3`
//
// CLAIM: A canonical `OnRails` surface impact is neither terminal nor reported.
// `Simulation::step` clamps `sim_time` and force-resets warp, then the next frame
// propagates the vessel straight through the body; it re-emerges and re-impacts
// once per revolution forever, stomping the player's warp each time.
//
// Report: docs/reviews/20260729T063800Z-propagator-trial.md
// Captured at commit 2fb6db6. NOT COMPILED — this file lives outside the crate
// on purpose. Paste into the `mod tests` block at the bottom of
// `crates/simulation/physics_canonical/src/simulation.rs`, run, then revert.
//
// ANCHORS: simulation.rs:741-767 (`step` replays to `collision_time`, sets
// `sim_time`, calls `warp.reset_immediate()`, returns `()` — the collision epoch,
// body and speed are discarded), :618-632 (`advance_vessel` returns
// `collision_time`), ship_propagator.rs:1201 (`detect_step_crossings` gates on
// `prev_alt > 0.0`, so a state that starts BELOW the surface is never flagged —
// this is why it coasts on through).
//
// DEPENDS ON (already present in that test module at 2fb6db6): `BodyDefinition`,
// `BodyKind`, `StateVector`, `GravityMode`, `Simulation`, `SimulationConfig`,
// `DVec3`, `HashMap`, `crate::types::G`.
//
// ── Reachability, and what the spec already says ────────────────────────────
//
// REACHABLE TODAY: `crates/runtime/game/src/staging.rs:524` calls `create_vessel`
// with `AuthorityMode::OnRails` — the landed stage-separation slice — and a staged
// booster has periapsis inside the body by construction. `apply_regime_authority`
// (`crates/runtime/game/src/regime.rs:280+`) early-returns on `active.bubble` and
// only ever transitions the ACTIVE craft off rails, so nothing else is taken off.
// There is no `remove_vessel`/retire API on `Simulation`, and `vessels.md` §4 says
// "Warp never deletes debris."
//
// THE FIX IS ALREADY THE SPEC. `docs/simulation/vessels.md` §4:
//   "Outside the local scene, an inactive vessel continues through the canonical
//    propagator and contact/event detection. A surface impact transitions it to
//    the same destroyed or `BodyFixed` outcomes as an active vessel."
//
// ── Corrections from the refutation pass ────────────────────────────────────
//
// 1. ONE SUB-CLAIM WAS STRUCK AS WRONG. The finding originally said the impact's
//    `warp.reset_immediate()` "disables the game's only sub-surface safety net"
//    at `crates/runtime/game/src/bridge.rs:110`. False: that guard's ENTIRE effect
//    IS `warp.reset_immediate()`, so it is redundant at that instant rather than
//    disabled, and it re-fires the moment the player pushes warp above 1x. Its
//    `> 1.0` gate is deliberate (it stops stomping warp for a player parked in a
//    sub-mean-radius valley). It also reads the ACTIVE craft's state, so it never
//    applied to a detached vessel anyway. The core finding stands without it.
//
// 2. THE RE-FIRE RATE IS WORSE THAN FILED. The refuter's independent repro
//    measured ~3 warp stomps per revolution, not one — the bisected contact state
//    lands marginally above the surface often enough to re-trigger on consecutive
//    frames.
//
// 3. SCOPE THE FIX NARROWLY. "Transition the vessel out of `OnRails`
//    (BodyFixed / destroyed)" overlaps the blocked `docs/backlog.md` row
//    `BL-20260724T230226Z-shared-local-vessel-scene`. What THIS evidence licenses
//    is cheaper: have `Simulation::step` return or queue the collision
//    `{craft, body, epoch, surface-relative speed}` it already computed, and stop
//    the record from coasting sub-surface.
//
// ── Observed output (verbatim) ──────────────────────────────────────────────
//
//   orbital period                     = 4298.9 s
//   first impact                       = sim_time 1476.562 s (frame 88)
//   altitude 10 s of sim time later    = -66920.0 m
//   deepest point reached              = -4939735.3 m (surface = 0)
//   sim time covered                   = 637635.6 s
//   warp force-resets (impact events)  = 400
//   `is_destroyed()` after all of this = false
//
//   thread '...onrails_surface_impact_neither_arrests_nor_reports' panicked at
//   crates\simulation\physics_canonical\src\simulation.rs:1687:9:
//   assertion `left == right` failed: a terminal impact fired 400 times
//     left: 400
//    right: 1
//
// Refuter's independent repro (own construction) reached first impact at
// 1476.80 s and a deepest point of -4,938,011 m — same numbers from a separate
// build of the scenario.
//
// THE ASSERTION: a terminal impact should fire exactly once. It fires 400 times
// over 637,636 s of sim time, the vessel reaches 4,939 km BELOW the surface
// (its true periapsis, straight through the body), and `is_destroyed()` is still
// false. Each of those 400 events force-resets the player's warp to 1x.

    /// REVIEW REPRO — a canonical (`OnRails`) surface impact neither arrests
    /// the vessel nor is reported to the caller; it only clamps `sim_time`
    /// and stomps warp. The vessel keeps Kepler-coasting *through* the body,
    /// re-emerges, and re-impacts once per revolution — forever.
    #[test]
    fn onrails_surface_impact_neither_arrests_nor_reports() {
        use crate::types::SolarSystemDefinition;

        let mu = crate::types::G * 5.972e24;
        let planet = BodyDefinition {
            id: 0, name: "Planet".to_string(), kind: BodyKind::Planet, parent: None,
            mass_kg: 5.972e24, radius_m: 6.371e6, color: [1.0; 3],
            rotation_period_s: 86_400.0, axial_tilt_rad: 0.0, gm: mu,
            soi_radius_m: f64::INFINITY, orbital_elements: None,
            terrain: thalos_world::TerrainConfig::None,
            ocean: None, tectonics: None, atmosphere: None,
            terrestrial_atmosphere: None, rings: None, surface_frame_ceiling_m: None,
        };
        let bodies = vec![planet];
        let system = SolarSystemDefinition {
            name: "T".to_string(), bodies: bodies.clone(),
            name_to_id: HashMap::from([("Planet".to_string(), 0)]), homeworld_id: 0,
        };

        // Apoapsis 10 000 km, periapsis 1 430 km — deep inside the 6 371 km
        // surface. Any staged booster or debris vessel on a decaying arc.
        let r = 1.0e7;
        let v = 0.5 * (mu / r).sqrt();
        let ship = StateVector {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, 0.0, v),
        };
        let impls = GravityMode::PatchedConics.build(&system, 1.0e6);
        let mut sim = Simulation::new(ship, impls, bodies, SimulationConfig::default());
        let period = std::f64::consts::TAU
            * ((mu / (2.0 * (0.5 * v * v - mu / r)).abs()).powi(3) / mu).sqrt();

        sim.warp.set_speed(1_000.0);
        let mut impacts = 0;
        let mut first_impact_time = None;
        let mut deepest_below_surface = 0.0_f64;
        let mut altitude_10s_after_impact = None;

        for frame in 0..40_000 {
            sim.step(1.0 / 60.0);
            // The *only* observable signal of a canonical collision.
            if sim.warp.speed() <= 1.0 {
                impacts += 1;
                if first_impact_time.is_none() {
                    first_impact_time = Some((sim.sim_time(), frame));
                }
                sim.warp.set_speed(1_000.0); // player re-engages warp
            }
            let alt = sim.ship_state().position.length() - 6.371e6;
            deepest_below_surface = deepest_below_surface.min(alt);
            if let Some((t0, _)) = first_impact_time
                && altitude_10s_after_impact.is_none()
                && sim.sim_time() > t0 + 10.0
            {
                altitude_10s_after_impact = Some(alt);
            }
        }

        let (t0, f0) = first_impact_time.expect("expected an impact");
        println!("orbital period                     = {period:.1} s");
        println!("first impact                       = sim_time {t0:.3} s (frame {f0})");
        println!("altitude 10 s of sim time later    = {:.1} m", altitude_10s_after_impact.unwrap());
        println!("deepest point reached              = {deepest_below_surface:.1} m (surface = 0)");
        println!("sim time covered                   = {:.1} s", sim.sim_time());
        println!("warp force-resets (impact events)  = {impacts}");
        println!("`is_destroyed()` after all of this = {}", sim.is_destroyed());

        assert_eq!(impacts, 1, "a terminal impact fired {impacts} times");
    }
