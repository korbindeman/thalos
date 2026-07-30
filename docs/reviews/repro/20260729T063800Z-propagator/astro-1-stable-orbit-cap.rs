// REPRO — `propagator-astro-1`
//
// CLAIM: A bound coast is truncated to exactly one orbital period and then jumps
// to the horizon via a single event-unchecked "straggler" sample, so every SOI
// entry on revolutions 2..N is invisible to prediction. A multi-revolution
// phasing orbit into a moon encounter cannot be planned.
//
// Report: docs/reviews/20260729T063800Z-propagator-trial.md
// Captured at commit 2fb6db6. NOT COMPILED — this file lives outside the crate
// on purpose. Paste the test body into the `mod tests` block at the bottom of
// `crates/simulation/physics_canonical/src/ship_propagator.rs`, run it, then
// revert. Re-check the anchors below against current source before trusting it.
//
// ANCHORS: ship_propagator.rs:289-310 (the one-period cap), :487-505 (the
// unchecked straggler), :267-288 (the in-code comment that makes the *cap*
// deliberate but scopes its "known limitation" to terrain only).
//
// DEPENDS ON (already present in that test module at 2fb6db6): `BodyDefinition`,
// `StateVector`, `CoastRequest`, `SegmentTerminator`, `KeplerianPropagator`,
// `PatchedConics`, `G`, `SUN_GM`, `AU`, `DVec3`, `HashMap`.
//
// ── Two corrections from the refutation pass — fold these into any fix ───────
//
// 1. THIS TEST EXERCISES THE INTERMEDIATE-LEG BRANCH, not the player-facing one.
//    `trajectory/flight_plan.rs:624` sets
//        stop_on_stable_orbit = leg_idx + 1 == leg_count && !burn_collided
//    so the FINAL leg — the one the player's orbit line is drawn from — passes
//    `true` and returns `StableOrbit` at exactly one period, rather than
//    `Horizon` at the requested horizon. Same outcome (the encounter is
//    invisible), different branch. The test below prints both.
//
// 2. A STRONGER VARIANT EXISTS. The refuter reproduced this on the *shipped*
//    system (`assets/solar_system.ron`: Thalos 1.378e24 kg / 3.186e6 m, Mira
//    a = 1.91488e8 m) with a trans-Mira transfer rp 4,000 km / ra 195,000 km
//    phased for a third-apoapsis intercept, and measured:
//
//        one-shot stop_on_stable_orbit=false: terminator Horizon,     end_time 2601039.9, samples 144
//        one-shot stop_on_stable_orbit=true:  terminator StableOrbit,  end_time  650260.0, samples 143
//        chunked walk (0.2 period):           SoiEnter Some((2, 1571096.13))   period = 650260.0 s
//
//    i.e. the same propagator finds the Mira encounter at 2.42 periods when
//    stepped in chunks, and nothing when asked in one call. That variant was not
//    preserved in source; rebuild it against `assets/solar_system.ron` if a
//    shipped-system demonstration is needed.
//
// ── Observed output (synthetic Earth/Moon version below, verbatim) ───────────
//
//   ship period 861595 s (9.97 d), apoapsis 3.844e8 m, planet SOI 9.246e8 m, moon SOI 6.620e7 m
//   one-shot coast to 3 periods : terminator Horizon, end_time 2584786 s, samples 144
//   chunked coast (0.4 period)  : SoiEnter Some((2, 2073203.4704969772))
//
//   thread '...stable_orbit_cap_hides_a_later_revolution_moon_encounter' panicked at
//   crates\simulation\physics_canonical\src\ship_propagator.rs:2237:9:
//   one-shot coast never reported the moon encounter that the chunked propagation of the
//   SAME trajectory finds at Some((2, 2073203.4704969772))
//
// THE ASSERTION: the one-shot coast to 3 periods must report `SoiEnter { body: 2 }`.
// It reports `Horizon`. The chunked walk over the identical trajectory, using the
// identical propagator, finds the encounter at t = 2,073,203 s — inside the
// requested window. The failure is that the event scan was capped along with the
// sampling, not that the cap exists.

    /// REVIEW REPRO — a coast is capped at ONE orbital period whenever the
    /// orbit is bound inside its SOI, then jumps analytically to the horizon
    /// with a single unchecked "straggler" sample. Every SOI crossing on
    /// revolutions 2..N is therefore invisible — including the standard
    /// multi-revolution phasing orbit into a moon encounter.
    #[test]
    fn stable_orbit_cap_hides_a_later_revolution_moon_encounter() {
        use crate::types::{BodyKind, OrbitalElements, SolarSystemDefinition};

        let sun_mass = SUN_GM / G;
        let planet_mass = 5.972e24;
        let moon_mass = 7.35e22;
        let moon_a = 3.844e8_f64;
        let planet_gm = G * planet_mass;

        let mk = |id: usize, name: &str, kind: BodyKind, parent: Option<usize>,
                  mass: f64, radius: f64, soi: f64, el: Option<OrbitalElements>| BodyDefinition {
            id, name: name.into(), kind, parent,
            mass_kg: mass, radius_m: radius, color: [1.0; 3],
            rotation_period_s: 86_400.0, axial_tilt_rad: 0.0,
            gm: G * mass, soi_radius_m: soi, orbital_elements: el,
            terrain: thalos_world::TerrainConfig::None,
            ocean: None, tectonics: None, atmosphere: None,
            terrestrial_atmosphere: None, rings: None, surface_frame_ceiling_m: None,
        };

        // Moon phase chosen so the ship's apoapsis and the moon coincide on
        // the THIRD apoapsis passage (t = 5P/2), not the first.
        let moon_period = std::f64::consts::TAU * (moon_a.powi(3) / planet_gm).sqrt();
        let rp = 7.0e6_f64;
        let ra = moon_a;
        let a = 0.5 * (rp + ra);
        let e = (ra - rp) / (ra + rp);
        let ship_period = std::f64::consts::TAU * (a.powi(3) / planet_gm).sqrt();
        let encounter_t = 2.5 * ship_period;
        let moon_nu0 = (std::f64::consts::PI
            - std::f64::consts::TAU * encounter_t / moon_period)
            .rem_euclid(std::f64::consts::TAU);

        let bodies = vec![
            mk(0, "Sun", BodyKind::Star, None, sun_mass, 6.957e8, f64::INFINITY, None),
            mk(1, "Planet", BodyKind::Planet, Some(0), planet_mass, 6.371e6,
               AU * (planet_mass / sun_mass).powf(0.4),
               Some(OrbitalElements { semi_major_axis_m: AU, eccentricity: 0.0,
                   inclination_rad: 0.0, lon_ascending_node_rad: 0.0,
                   arg_periapsis_rad: 0.0, true_anomaly_rad: 0.0 })),
            mk(2, "Moon", BodyKind::Moon, Some(1), moon_mass, 1.737e6,
               moon_a * (moon_mass / planet_mass).powf(0.4),
               Some(OrbitalElements { semi_major_axis_m: moon_a, eccentricity: 0.0,
                   inclination_rad: 0.0, lon_ascending_node_rad: 0.0,
                   arg_periapsis_rad: 0.0, true_anomaly_rad: moon_nu0 })),
        ];
        let mut name_to_id = HashMap::new();
        for b in &bodies { name_to_id.insert(b.name.clone(), b.id); }
        let system = SolarSystemDefinition {
            name: "EM".into(), bodies, name_to_id, homeworld_id: 1,
        };
        let pc = PatchedConics::new(&system, 3.156e9);
        let propagator = KeplerianPropagator::default();

        let planet = pc.state(1, crate::canonical::Epoch(0.0));
        let vp = (planet_gm * (1.0 + e) / rp).sqrt();
        let state = StateVector {
            position: planet.position + DVec3::new(rp, 0.0, 0.0),
            velocity: planet.velocity + DVec3::new(0.0, 0.0, vp),
        };
        let horizon = 3.0 * ship_period;

        println!("ship period {:.0} s ({:.2} d), apoapsis {:.3e} m, planet SOI {:.3e} m, moon SOI {:.3e} m",
            ship_period, ship_period / 86_400.0, ra,
            system.bodies[1].soi_radius_m, system.bodies[2].soi_radius_m);

        // --- what prediction actually asks for: one call to the horizon ---
        let one_shot = propagator.coast_segment(CoastRequest {
            state, time: 0.0, soi_body: 1, target_time: horizon,
            stop_on_stable_orbit: false, sample_count_hint: 128,
            ephemeris: &pc, bodies: &system.bodies,
        });
        println!("one-shot coast to 3 periods : terminator {:?}, end_time {:.0} s, samples {}",
            one_shot.terminator, one_shot.end_time, one_shot.samples.len());

        // --- same propagator, stepped in sub-period chunks ---
        let mut t = 0.0_f64;
        let mut st = state;
        let mut chunked = None;
        while t < horizon {
            let res = propagator.coast_segment(CoastRequest {
                state: st, time: t, soi_body: 1,
                target_time: (t + 0.4 * ship_period).min(horizon),
                stop_on_stable_orbit: false, sample_count_hint: 128,
                ephemeris: &pc, bodies: &system.bodies,
            });
            if let SegmentTerminator::SoiEnter { body, time } = res.terminator {
                chunked = Some((body, time));
                break;
            }
            st = res.end_state;
            t = res.end_time;
        }
        println!("chunked coast (0.4 period)  : SoiEnter {chunked:?}");

        assert!(
            matches!(one_shot.terminator, SegmentTerminator::SoiEnter { body: 2, .. }),
            "one-shot coast never reported the moon encounter that the chunked \
             propagation of the SAME trajectory finds at {chunked:?}"
        );
    }
