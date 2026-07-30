// REPRO — `propagator-astro-2`
//
// CLAIM: There is no terrain-scale resolution bound on the prediction collision
// scan. Its spatial resolution is set by the caller's *render* sample-count hint,
// and the step-subdivision cap that looks like it should bound it is keyed to
// distance-from-body-centre, so it never fires near the surface. Prediction
// (128 samples/period) probes the ground track every ~78 km while the live path
// (32 samples/frame) probes every few metres.
//
// Report: docs/reviews/20260729T063800Z-propagator-trial.md
// Captured at commit 2fb6db6. NOT COMPILED — this file lives outside the crate
// on purpose. Paste into the `mod tests` block at the bottom of
// `crates/simulation/physics_canonical/src/ship_propagator.rs`, run, then revert.
//
// ANCHORS: ship_propagator.rs:388-414 (`needs_subdivide`; `min_alt` is
// `(prev_state.position - prev_body.position).length()`, a RADIUS — `body_radius`
// and `max_elevation_m` are never subtracted on this path), :1046-1068
// (`interior_min_altitude`'s three fixed probes at s = 0.25/0.5/0.75).
// Contradicts `lib.rs:17-19`, which asserts the two paths "can never numerically
// diverge".
//
// DEPENDS ON (already present in that test module at 2fb6db6): `sun_earth_system()`,
// `EARTH_GM`, `TerrainProvider`, `BodyId`, `StateVector`, `CoastRequest`,
// `SegmentTerminator`, `KeplerianPropagator`, `DVec3`, `Arc`. The `RidgeTerrain`
// provider below is defined by this repro and must be pasted with the test.
//
// ── Corrections from the refutation pass — READ BEFORE FIXING ────────────────
//
// 1. THE FILED FIX IS WRONG AND WOULD MAKE THINGS WORSE. The finding proposed
//    re-keying `needs_subdivide` to altitude-above-terrain. Do not. For that
//    cap's actual purpose — bounding cubic-Hermite arc-tracking error —
//    `path / |q|` IS the correct quantity (it is the swept angle; 0.25 ≈ 14° per
//    step). Re-keying it to altitude demands ~1.25 km of path at 5 km altitude,
//    i.e. ~0.16 s steps at orbital speed, subdividing to `MIN_STEP_S = 1e-3` for
//    an entire grazing pass. The real defect is narrower: there is no
//    terrain-scale bound AT ALL. The fix is the `sample_stride` clause alone —
//    let the crossing scan run at a physically-derived step while `build_sample`
//    emits only at the caller's render density.
//
// 2. THE CLAUDE.md "One propagator everywhere" CITATION IS A MISAPPLICATION.
//    Both paths do route through the same `ShipPropagator`. What is actually
//    contradicted is the `lib.rs` module doc quoted above.
//
// 3. THIS TEST'S RIDGE IS A STRAWMAN AT THE STATED NUMBERS; THE CLASS IS NOT.
//    Probe spacing is 2*pi*r/512 ≈ 0.0123*r, independent of mass: ~39 km on
//    Thalos (radius_m 3186000), ~10.7 km on Mira (869 km). Real `ProceduralSurface`
//    relief is WIDER AND SHORTER than this test's ridge — `MASSIF_SITES` are
//    44–48 km half-width with `MASSIF_PEAK_M` ≈ 4.9 km
//    (crates/domain/terrain/src/procedural.rs:534-590) — so a 10 km-wide,
//    8 km-tall ridge does not exist in this world. Reaching the failure needs a
//    stable orbit with periapsis inside the terrain band on a body whose *tall*
//    terrain is narrower than ~1.2 % of the orbital radius: on Thalos that means
//    a few-km-altitude orbit inside a 1 bar atmosphere; on the airless bodies the
//    probe spacing is already fine relative to their relief. Keep the test — it
//    isolates the mechanism cleanly — but do not quote its 13/24 as a
//    field-reachable rate.
//
// ── Observed output (verbatim) ──────────────────────────────────────────────
//
//   ridge 10000 m wide / 8000 m tall, orbit at 5000 m altitude, period 5067 s
//   prediction (128 samples / period): missed the impact in 13/24 ridge placements
//   live       ( 32 samples / frame ): missed the impact in 0/24 ridge placements
//
//   thread '...prediction_and_live_disagree_on_terrain_impact' panicked at
//   crates\simulation\physics_canonical\src\ship_propagator.rs:2065:9:
//   assertion `left == right` failed: prediction and live propagation disagree about hitting terrain
//     left: 13
//    right: 0
//
// An independent replica by the refuter (own construction, not this source) gave
// 16/24 vs 0/24 — same direction and magnitude.
//
// THE ASSERTION: prediction and live propagation must agree on whether a given
// trajectory hits terrain. Over 24 evenly spaced ridge longitudes they disagree
// 13 times, always in the same direction: prediction misses what live catches.

    /// One tall, narrow ridge at a fixed body-fixed longitude.
    struct RidgeTerrain { center_lon_rad: f64, half_width_rad: f64, height_m: f64 }

    impl TerrainProvider for RidgeTerrain {
        fn surface_elevation_m(&self, _body: BodyId, dir_body: DVec3) -> f64 {
            let lon = dir_body.z.atan2(dir_body.x);
            let mut d = (lon - self.center_lon_rad).abs();
            if d > std::f64::consts::PI { d = std::f64::consts::TAU - d; }
            if d < self.half_width_rad { self.height_m } else { 0.0 }
        }
        fn max_elevation_m(&self, _body: BodyId) -> f64 { self.height_m }
    }

    #[test]
    fn prediction_and_live_disagree_on_terrain_impact() {
        let (system, pc) = sun_earth_system();
        let body_r = 6.371e6_f64;
        let alt = 5_000.0_f64;
        let r = body_r + alt;
        let v = (EARTH_GM / r).sqrt();
        let period = std::f64::consts::TAU * (r.powi(3) / EARTH_GM).sqrt();

        // 10 km wide, 8 km tall ridge: the orbit passes 3 km *below* its top.
        let ridge_width_m = 10_000.0;
        let half_width = 0.5 * ridge_width_m / body_r;

        let mut coarse_misses = 0;
        let mut fine_misses = 0;
        let samples = 24;
        for i in 0..samples {
            let center = std::f64::consts::TAU * (i as f64) / (samples as f64);
            let terrain = Arc::new(RidgeTerrain {
                center_lon_rad: center, half_width_rad: half_width, height_m: 8_000.0,
            });
            let propagator = KeplerianPropagator::default().with_terrain(terrain);
            let earth = pc.state(1, crate::canonical::Epoch(0.0));
            let state = StateVector {
                position: earth.position + DVec3::new(r, 0.0, 0.0),
                velocity: earth.velocity + DVec3::new(0.0, 0.0, v),
            };

            // --- prediction shape: one call, 128 samples over the period ---
            let pred = propagator.coast_segment(CoastRequest {
                state, time: 0.0, soi_body: 1, target_time: period * 2.0,
                stop_on_stable_orbit: true, sample_count_hint: 128,
                ephemeris: &pc, bodies: &system.bodies,
            });
            let pred_hit = matches!(pred.terminator, SegmentTerminator::Collision { .. });

            // --- live shape: `advance_vessel`'s per-frame coast, 1000x warp
            //     (16.67 s of sim time per 1/60 s frame), 32 samples each ---
            let frame_dt = 1000.0 / 60.0;
            let mut live_state = state;
            let mut t = 0.0_f64;
            let mut live_hit = false;
            while t < period {
                let res = propagator.coast_segment(CoastRequest {
                    state: live_state, time: t, soi_body: 1,
                    target_time: t + frame_dt, stop_on_stable_orbit: false,
                    sample_count_hint: 32, ephemeris: &pc, bodies: &system.bodies,
                });
                if matches!(res.terminator, SegmentTerminator::Collision { .. }) {
                    live_hit = true;
                    break;
                }
                live_state = res.end_state;
                t = res.end_time;
            }

            if !pred_hit { coarse_misses += 1; }
            if !live_hit { fine_misses += 1; }
        }
        println!("ridge {ridge_width_m:.0} m wide / 8000 m tall, orbit at {alt:.0} m altitude, period {period:.0} s");
        println!("prediction (128 samples / period): missed the impact in {coarse_misses}/{samples} ridge placements");
        println!("live       ( 32 samples / frame ): missed the impact in {fine_misses}/{samples} ridge placements");
        assert_eq!(coarse_misses, fine_misses,
            "prediction and live propagation disagree about hitting terrain");
    }
