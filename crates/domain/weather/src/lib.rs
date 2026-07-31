//! Planetary weather simulation: a reduced-gravity shallow-water layer on the
//! sphere, plus an advected moisture tracer, producing the SYNOPTIC-scale
//! organization of a cloud field — jets, Rossby waves, cyclones with wound
//! frontal bands, an ITCZ convergence line, clear subsidence belts.
//!
//! ## Why a simulation at all
//!
//! The weather-cube producer previously painted this structure with warped
//! noise (curl warps, analytic vortex rotations, latitude gaussians). Every
//! round of tuning converged on the same verdict: domain-warped noise reads as
//! marbled fluid, uniformly swirly, unmistakably synthetic (2026-07-31). The
//! shapes that read as *weather* — filament fronts, comma clouds, dry slots —
//! are histories of material advected by a rotating flow, and the cheapest
//! honest source of such a history is the flow itself. One coarse 2-D layer is
//! enough; this is the "barotropic vorticity / shallow water" tier of the
//! technique survey, the only rung that is not a fake.
//!
//! ## Model
//!
//! Single-layer shallow water on a lat-lon grid, equatorial beta included via
//! the full Coriolis parameter; semi-Lagrangian advection (unconditionally
//! stable), explicit pressure gradient + divergence with the gravity-wave
//! speed tuned to an equivalent-barotropic ~40 m/s (Rossby radius ≈ 400 km on
//! Thalos — cyclones come out at the right size by construction). Zonal-mean
//! forcing maintains the general circulation: relaxation to a trades /
//! westerlies / polar-easterlies wind profile and to an ITCZ-convergent
//! meridional flow; seeded stochastic vortex stirring along the storm tracks
//! maintains the eddy field the (missing) baroclinic instability would supply.
//! Moisture is a passive tracer with evaporation toward a latitude-dependent
//! saturation and a precipitation sink; cloud is diagnosed from moisture and
//! convergence.
//!
//! Determinism: fixed step, seeded stirring — the same params give the same
//! field bit-for-bit, which the weather-cube contract requires.
//!
//! ## Numerical dodges (deliberate, visual-grade)
//!
//! - **Zonal metric cap**: 1/cos(lat) factors are capped at cos(60°) so the
//!   gravity-wave CFL limit is set by mid-latitudes, not by the pole rows.
//!   High-latitude zonal wave propagation is slowed; nothing the disc shows.
//! - **Polar sponge**: above ~78° the state relaxes to the zonal base state.
//!   The weather cube's polar caps are climatology anyway.
//! - **Plain Laplacian smoothing** stands in for turbulent dissipation.

use rayon::prelude::*;

const TAU: f32 = std::f32::consts::TAU;
const PI: f32 = std::f32::consts::PI;

#[derive(Clone, Debug)]
pub struct WeatherSimParams {
    pub seed: u64,
    /// Grid columns (longitude). 512 → ~39 km cells on Thalos.
    pub nx: usize,
    /// Grid rows (latitude).
    pub ny: usize,
    pub radius_m: f32,
    /// Planetary rotation rate. Earth-like default; Thalos day assumed similar.
    pub omega_rad_s: f32,
    /// Mean ITCZ latitude (signed, radians). Earth's annual mean is ~+6°.
    pub itcz_lat_rad: f32,
    /// Equivalent gravity-wave speed, m/s. Sets the Rossby radius c/f and
    /// therefore the emergent eddy size.
    pub c_wave_m_s: f32,
    pub dt_s: f32,
    /// Relaxation time toward the zonal base circulation, seconds.
    pub relax_s: f32,
    /// Rayleigh drag on velocity anomalies, seconds.
    pub drag_s: f32,
    /// Nondimensional smoothing strength (fraction of the diffusive CFL
    /// limit). ~0.05.
    pub smooth_frac: f32,
    /// Seconds between stochastic storm-track stirring events.
    pub stir_interval_s: f32,
    /// Peak tangential velocity of one injected stirring vortex, m/s.
    pub stir_speed_m_s: f32,
    /// Injected vortex core radius, meters.
    pub stir_radius_m: f32,
    /// Vortices injected per stirring event (split across hemispheres).
    pub stir_count: usize,
    /// Storm-track center latitude (unsigned, radians).
    pub storm_lat_rad: f32,
    /// Moisture evaporation timescale, seconds.
    pub evap_s: f32,
    /// Precipitation timescale, seconds (fast; keeps q pinned near the
    /// saturation knee so fronts print as sharp moisture edges).
    pub precip_s: f32,
    /// Fraction of saturation where precipitation starts.
    pub precip_knee: f32,
}

impl Default for WeatherSimParams {
    fn default() -> Self {
        Self {
            seed: 0,
            nx: 512,
            ny: 256,
            radius_m: 3.186e6,
            omega_rad_s: 7.27e-5,
            itcz_lat_rad: 0.10,
            c_wave_m_s: 40.0,
            dt_s: 360.0,
            relax_s: 6.0 * 86_400.0,
            drag_s: 10.0 * 86_400.0,
            smooth_frac: 0.02,
            stir_interval_s: 10_800.0,
            stir_speed_m_s: 8.0,
            stir_radius_m: 450_000.0,
            stir_count: 3,
            storm_lat_rad: 0.92,
            evap_s: 4.0 * 86_400.0,
            precip_s: 0.5 * 86_400.0,
            precip_knee: 0.82,
        }
    }
}

/// The zonal metric factor 1/cos(lat) is capped here — see the module notes.
const ZONAL_METRIC_CAP_COS: f32 = 0.5;
/// Polar sponge start (radians of |lat|). The sim's honest domain ends here;
/// poleward of it the state relaxes to the zonal base and the weather-cube
/// producer's climatology owns the look. Chosen just above the storm tracks —
/// the capped-metric rows above it generate grid noise if left free.
const SPONGE_LAT: f32 = 1.15;
/// Velocity sanity clamp, m/s.
const V_CLAMP: f32 = 60.0;
/// Reference layer thickness, meters. Only c² = g'·H is dynamically
/// meaningful; H_REF fixes the split so h reads in meters.
const H_REF: f32 = 8000.0;

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(f32::EPSILON)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// splitmix64 — deterministic, seedable, dependency-free.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub struct WeatherSim {
    pub params: WeatherSimParams,
    /// Zonal velocity, m/s, row-major `ny × nx`, row 0 = south pole edge.
    u: Vec<f32>,
    v: Vec<f32>,
    /// Layer-thickness anomaly, meters.
    h: Vec<f32>,
    /// Moisture, 0..~1 (fraction of tropical saturation).
    q: Vec<f32>,
    // Double buffers.
    u1: Vec<f32>,
    v1: Vec<f32>,
    h1: Vec<f32>,
    q1: Vec<f32>,
    /// Instantaneous precipitation rate (q units per second) from the last
    /// step. Precipitating regions are the deep-convective bright cloud —
    /// the ITCZ line and frontal bands — which pure moisture cannot show
    /// because the precipitation knee clips q at the same relative level
    /// everywhere.
    precip: Vec<f32>,
    /// Per-row latitude (radians).
    lat: Vec<f32>,
    /// Per-row base zonal wind (the maintained general circulation).
    u_base: Vec<f32>,
    /// Per-row base meridional wind (trade convergence into the ITCZ).
    v_base: Vec<f32>,
    /// Per-row saturation moisture.
    qsat: Vec<f32>,
    /// Per-row Coriolis parameter.
    f_cor: Vec<f32>,
    /// Per-row thickness in geostrophic balance with `u_base`. The maintained
    /// jet MUST have its balancing pressure field: without it the Coriolis
    /// rotation of the mean wind pumps inertial oscillations every step and
    /// the run blows up within days (first probe run).
    h_base: Vec<f32>,
    /// Per-row polar-sponge strength (0 = free, 1 = pinned).
    sponge: Vec<f32>,
    rng: Rng,
    time_s: f64,
    next_stir_s: f64,
}

impl WeatherSim {
    pub fn new(params: WeatherSimParams) -> Self {
        let (nx, ny) = (params.nx, params.ny);
        let n = nx * ny;
        let mut lat = vec![0.0f32; ny];
        let mut u_base = vec![0.0f32; ny];
        let mut v_base = vec![0.0f32; ny];
        let mut qsat = vec![0.0f32; ny];
        let mut f_cor = vec![0.0f32; ny];
        let mut sponge = vec![0.0f32; ny];
        let gauss = |x: f32, c: f32, w: f32| (-((x - c) / w) * ((x - c) / w)).exp();
        for j in 0..ny {
            let phi = -PI / 2.0 + (j as f32 + 0.5) * PI / ny as f32;
            lat[j] = phi;
            let a = phi.abs();
            // Trades / westerlies / polar easterlies. The westerly jet peak
            // sits at the storm-track latitude by construction.
            u_base[j] = -7.0 * gauss(phi, params.itcz_lat_rad, 0.24)
                + 14.0 * gauss(a, params.storm_lat_rad, 0.24)
                - 2.0 * gauss(a, 1.45, 0.15);
            // Trade-wind convergence into the ITCZ: the band's cloud comes
            // from this line of meeting air, not from painted coverage.
            let toward = if phi > params.itcz_lat_rad { -1.0 } else { 1.0 };
            v_base[j] = toward * 2.4 * gauss(phi, params.itcz_lat_rad, 0.22);
            qsat[j] = 0.25 + 0.75 * phi.cos().powf(1.5);
            f_cor[j] = 2.0 * params.omega_rad_s * phi.sin();
            sponge[j] = smoothstep(SPONGE_LAT, 1.35, a);
        }
        // Geostrophic balance: f * u = -g' * dh/dy, integrated over rows and
        // centered to zero area-weighted mean.
        let g_eff = params.c_wave_m_s * params.c_wave_m_s / H_REF;
        let dy = params.radius_m * PI / ny as f32;
        let mut h_base = vec![0.0f32; ny];
        for j in 1..ny {
            let f_mid = 0.5 * (f_cor[j] + f_cor[j - 1]);
            let u_mid = 0.5 * (u_base[j] + u_base[j - 1]);
            h_base[j] = h_base[j - 1] - f_mid * u_mid / g_eff * dy;
        }
        let mut wsum = 0.0f32;
        let mut hsum = 0.0f32;
        for j in 0..ny {
            let w = lat[j].cos();
            wsum += w;
            hsum += w * h_base[j];
        }
        let h_mean = hsum / wsum.max(1.0e-6);
        for value in &mut h_base {
            *value -= h_mean;
        }

        let mut u = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        let mut h = vec![0.0f32; n];
        let mut q = vec![0.0f32; n];
        for j in 0..ny {
            for i in 0..nx {
                u[j * nx + i] = u_base[j];
                v[j * nx + i] = v_base[j];
                h[j * nx + i] = h_base[j];
                q[j * nx + i] = 0.7 * qsat[j];
            }
        }
        let rng = Rng(params.seed ^ 0x5EA5_0EA7_1E55_C0DE);
        Self {
            u1: u.clone(),
            v1: v.clone(),
            h1: h.clone(),
            q1: q.clone(),
            precip: vec![0.0; n],
            u,
            v,
            h,
            q,
            lat,
            u_base,
            v_base,
            qsat,
            f_cor,
            h_base,
            sponge,
            rng,
            time_s: 0.0,
            next_stir_s: 0.0,
            params,
        }
    }

    pub fn nx(&self) -> usize {
        self.params.nx
    }

    pub fn ny(&self) -> usize {
        self.params.ny
    }

    pub fn time_days(&self) -> f64 {
        self.time_s / 86_400.0
    }

    pub fn u_field(&self) -> &[f32] {
        &self.u
    }

    pub fn v_field(&self) -> &[f32] {
        &self.v
    }

    pub fn h_field(&self) -> &[f32] {
        &self.h
    }

    pub fn q_field(&self) -> &[f32] {
        &self.q
    }

    pub fn precip_field(&self) -> &[f32] {
        &self.precip
    }

    pub fn lat_of_row(&self, j: usize) -> f32 {
        self.lat[j]
    }

    pub fn qsat_of_row(&self, j: usize) -> f32 {
        self.qsat[j]
    }

    /// Advance the sim by `days` of model time.
    pub fn run_days(&mut self, days: f32) {
        let steps = (days * 86_400.0 / self.params.dt_s).ceil() as usize;
        for _ in 0..steps {
            self.step();
        }
    }

    /// One `dt` step. Semi-Lagrangian advection of (u, v, h, q), then the
    /// local physics, all cell-parallel over the previous state.
    pub fn step(&mut self) {
        if self.time_s >= self.next_stir_s {
            self.stir();
            self.next_stir_s = self.time_s + f64::from(self.params.stir_interval_s);
        }

        let p = &self.params;
        let (nx, ny) = (p.nx, p.ny);
        let dt = p.dt_s;
        let r = p.radius_m;
        let dlam = TAU / nx as f32;
        let dphi = PI / ny as f32;
        let dy = r * dphi;
        // g' * H folded into one constant: PGF uses c² * grad(h) / H_REF and
        // continuity uses H_REF * div, so only c² matters.
        let h_ref = H_REF;
        let g_eff = p.c_wave_m_s * p.c_wave_m_s / h_ref;
        // Diffusion at a fraction of the 5-point stability limit at the
        // mid-latitude cell size.
        let dx_mid = r * dlam * 0.71;
        let kappa_dt = p.smooth_frac * 0.25 * dx_mid * dx_mid;

        let u0 = &self.u;
        let v0 = &self.v;
        let h0 = &self.h;
        let q0 = &self.q;
        let lat = &self.lat;
        let u_base = &self.u_base;
        let v_base = &self.v_base;
        let qsat = &self.qsat;
        let f_cor = &self.f_cor;
        let sponge = &self.sponge;

        // Bilinear sample with zonal wrap and meridional clamp.
        let sample = |field: &[f32], xf: f32, yf: f32| -> f32 {
            let yf = yf.clamp(0.0, (ny - 1) as f32);
            let y0 = yf.floor() as usize;
            let y1 = (y0 + 1).min(ny - 1);
            let ty = yf - y0 as f32;
            let xw = xf.rem_euclid(nx as f32);
            let x0 = xw.floor() as usize % nx;
            let x1 = (x0 + 1) % nx;
            let tx = xw - xw.floor();
            let a = field[y0 * nx + x0] + (field[y0 * nx + x1] - field[y0 * nx + x0]) * tx;
            let b = field[y1 * nx + x0] + (field[y1 * nx + x1] - field[y1 * nx + x0]) * tx;
            a + (b - a) * ty
        };

        let h_base = &self.h_base;

        // ── Pass 1: momentum. Forward-backward scheme — velocities update
        // first from the OLD thickness; thickness then updates from the NEW
        // velocities. Plain forward Euler on the wave terms amplifies every
        // step and blew up the first probe run within days.
        self.u1
            .par_chunks_mut(nx)
            .zip(self.v1.par_chunks_mut(nx))
            .enumerate()
            .for_each(|(j, (row_u, row_v))| {
                let phi = lat[j];
                let cos_metric = phi.cos().max(ZONAL_METRIC_CAP_COS);
                let dx = r * dlam * cos_metric;
                let tan_phi = phi.tan().clamp(-4.0, 4.0);
                let kappa_row = kappa_dt * (1.0 + 4.0 * smoothstep(0.95, 1.25, phi.abs()));
                let jm = j.saturating_sub(1);
                let jp = (j + 1).min(ny - 1);
                for i in 0..nx {
                    let idx = j * nx + i;
                    let im = (i + nx - 1) % nx;
                    let ip = (i + 1) % nx;
                    let u_c = u0[idx];
                    let v_c = v0[idx];

                    let x_dep = i as f32 - u_c * dt / (r * cos_metric * dlam);
                    let y_dep = j as f32 - v_c * dt / (r * dphi);
                    let mut u_n = sample(u0, x_dep, y_dep);
                    let mut v_n = sample(v0, x_dep, y_dep);

                    // Pressure gradient from the old thickness.
                    let dhdx = (h0[j * nx + ip] - h0[j * nx + im]) / (2.0 * dx);
                    let dhdy = (h0[jp * nx + i] - h0[jm * nx + i]) / (2.0 * dy);
                    u_n -= dt * g_eff * dhdx;
                    v_n -= dt * g_eff * dhdy;

                    // Curvature terms.
                    u_n += dt * u_c * v_c * tan_phi / r;
                    v_n -= dt * u_c * u_c * tan_phi / r;

                    // Coriolis as an exact rotation (stable at any f*dt).
                    let ang = f_cor[j] * dt;
                    let (s, c) = ang.sin_cos();
                    let (u_r, v_r) = (u_n * c + v_n * s, -u_n * s + v_n * c);
                    u_n = u_r;
                    v_n = v_r;

                    // Maintained circulation + drag.
                    u_n += dt * (u_base[j] - u_n) / p.relax_s;
                    v_n += dt * (v_base[j] - v_n) / p.relax_s;
                    u_n -= dt * (u_n - u_base[j]) / p.drag_s;
                    v_n -= dt * (v_n - v_base[j]) / p.drag_s;

                    // Smoothing (Laplacian of the old field).
                    let lap = |f: &[f32]| {
                        (f[j * nx + ip] + f[j * nx + im] - 2.0 * f[idx]) / (dx * dx)
                            + (f[jp * nx + i] + f[jm * nx + i] - 2.0 * f[idx]) / (dy * dy)
                    };
                    u_n += kappa_row * lap(u0);
                    v_n += kappa_row * lap(v0);

                    // Polar sponge.
                    let sp = sponge[j] * 0.20;
                    u_n += sp * (u_base[j] - u_n);
                    v_n += sp * (0.0 - v_n);

                    row_u[i] = u_n.clamp(-V_CLAMP, V_CLAMP);
                    row_v[i] = v_n.clamp(-V_CLAMP, V_CLAMP);
                }
            });

        // ── Pass 2: thickness + moisture, divergence from the NEW velocities.
        let u_new = &self.u1;
        let v_new = &self.v1;
        self.h1
            .par_chunks_mut(nx)
            .zip(self.q1.par_chunks_mut(nx))
            .zip(self.precip.par_chunks_mut(nx))
            .enumerate()
            .for_each(|(j, ((row_h, row_q), row_p))| {
                let phi = lat[j];
                let cos_metric = phi.cos().max(ZONAL_METRIC_CAP_COS);
                let dx = r * dlam * cos_metric;
                // Extra dissipation where the metric cap distorts the
                // dynamics; free rows keep the sharp setting.
                let kappa_row = kappa_dt * (1.0 + 4.0 * smoothstep(0.95, 1.25, phi.abs()));
                // SST-style ITCZ moisture source: the convergence line's deep
                // convection feeds on a warm-surface evaporation maximum the
                // single layer cannot produce on its own.
                let itcz_d = (phi - p.itcz_lat_rad) / 0.08;
                let evap_boost = 1.0 + 3.5 * (-itcz_d * itcz_d).exp();
                let jm = j.saturating_sub(1);
                let jp = (j + 1).min(ny - 1);
                for i in 0..nx {
                    let idx = j * nx + i;
                    let im = (i + nx - 1) % nx;
                    let ip = (i + 1) % nx;

                    let x_dep = i as f32 - u0[idx] * dt / (r * cos_metric * dlam);
                    let y_dep = j as f32 - v0[idx] * dt / (r * dphi);
                    let mut h_n = sample(h0, x_dep, y_dep);
                    let mut q_n = sample(q0, x_dep, y_dep);

                    // The denominator uses the SAME capped metric as the
                    // momentum operators. Mixing the real cos(lat) here with
                    // the capped one there manufactures spurious divergence
                    // at high latitude — grid-scale hatching that the
                    // moisture coupling then integrates (probe round 3).
                    let dudx = (u_new[j * nx + ip] - u_new[j * nx + im]) / (2.0 * dx);
                    let dvdy = ((v_new[jp * nx + i] * lat[jp].cos())
                        - (v_new[jm * nx + i] * lat[jm].cos()))
                        / (2.0 * dy * cos_metric);
                    let div = dudx + dvdy;
                    h_n -= dt * h_ref * div;
                    h_n += dt * (h_base[j] - h_n) / (16.0 * 86_400.0);

                    // Moisture: evaporation toward saturation, precipitation
                    // past the knee. Fast precipitation keeps q pinned near
                    // the knee so advection prints fronts as sharp edges.
                    let qs = qsat[j];
                    q_n += dt * evap_boost * (qs - q_n) / p.evap_s;
                    let excess = q_n - p.precip_knee * qs;
                    let rain = if excess > 0.0 { excess / p.precip_s } else { 0.0 };
                    q_n -= dt * rain;
                    row_p[i] = rain;
                    // Subsidence drying / convergence moistening — the term
                    // that gives the moisture field its CONTRAST: column
                    // stretching, q̇ = −k·div·q, with k of O(1) (k = 1 is the
                    // exact passive-column value; a little more stands in for
                    // the missing vertical structure). The first cut used
                    // k ~ 10³ — three orders too strong, since the comparison
                    // scale is the evaporation rate, not unity — and q slammed
                    // binary within a day. Divergent flow dries (subtropical
                    // belts carve themselves); convergent flow moistens (ITCZ
                    // and frontal bands print bright).
                    q_n -= dt * 2.0 * div * q_n;

                    let lap = |f: &[f32]| {
                        (f[j * nx + ip] + f[j * nx + im] - 2.0 * f[idx]) / (dx * dx)
                            + (f[jp * nx + i] + f[jm * nx + i] - 2.0 * f[idx]) / (dy * dy)
                    };
                    h_n += kappa_row * lap(h0);
                    // Moisture is the field that must hold FRONTS — filament
                    // sharpness is the whole visual point — so it diffuses at
                    // a fraction of the dynamical smoothing.
                    q_n += 0.3 * kappa_row * lap(q0);

                    let sp = sponge[j] * 0.20;
                    h_n += sp * (h_base[j] - h_n);
                    q_n += sp * (0.6 * qs - q_n);

                    row_h[i] = h_n.clamp(-4.0 * h_ref, 4.0 * h_ref);
                    row_q[i] = q_n.clamp(0.0, 1.6);
                }
            });

        std::mem::swap(&mut self.u, &mut self.u1);
        std::mem::swap(&mut self.v, &mut self.v1);
        std::mem::swap(&mut self.h, &mut self.h1);
        std::mem::swap(&mut self.q, &mut self.q1);
        self.time_s += f64::from(dt);
    }

    /// Inject a batch of storm-track stirring vortices — the stand-in for
    /// baroclinic instability. Cyclonic-biased, hemisphere-signed.
    fn stir(&mut self) {
        let p = &self.params;
        let (nx, ny) = (p.nx, p.ny);
        for _ in 0..p.stir_count {
            let hemi = if self.rng.f32() < 0.5 { 1.0 } else { -1.0 };
            let lat_c = hemi * (p.storm_lat_rad + 0.35 * (self.rng.f32() - 0.5));
            let lon_c = TAU * self.rng.f32();
            // 70 % cyclonic: counterclockwise in the north.
            let cyclonic = if self.rng.f32() < 0.7 { 1.0 } else { -1.0 };
            let sign = cyclonic * hemi.signum();
            let speed = p.stir_speed_m_s * (0.5 + self.rng.f32());
            let r0 = p.stir_radius_m * (0.7 + 0.6 * self.rng.f32());
            let reach = 3.0 * r0;

            let r_planet = p.radius_m;
            let g_eff = p.c_wave_m_s * p.c_wave_m_s / H_REF;
            // Balance the injected swirl with its height field: for
            // v_θ = s·V·x·exp((1−x²)/2), geostrophy g'·dh/dr = f·v_θ
            // integrates to h(x) = −(f·s·V·r0/g')·exp((1−x²)/2). Injecting
            // bare momentum leaves the vortex unbalanced, and the resulting
            // gravity waves flooded the divergence field until the moisture
            // coupling was integrating wave noise instead of weather
            // (probe round 2).
            let f_here = 2.0 * p.omega_rad_s * lat_c.sin();
            let h_amp = -f_here * sign * speed * r0 / g_eff;
            for j in 0..ny {
                let phi = self.lat[j];
                let dy_m = (phi - lat_c) * r_planet;
                if dy_m.abs() > reach {
                    continue;
                }
                let cos_phi = phi.cos().max(0.05);
                for i in 0..nx {
                    let lon = (i as f32 + 0.5) / nx as f32 * TAU;
                    let mut dlon = lon - lon_c;
                    if dlon > PI {
                        dlon -= TAU;
                    }
                    if dlon < -PI {
                        dlon += TAU;
                    }
                    let dx_m = dlon * r_planet * cos_phi;
                    if dx_m.abs() > reach {
                        continue;
                    }
                    let dist = (dx_m * dx_m + dy_m * dy_m).sqrt().max(1.0);
                    let x = dist / r0;
                    // Rankine-like: solid-body core, smooth exponential tail.
                    let shape = (0.5 * (1.0 - x * x)).exp();
                    let vt = sign * speed * x * shape;
                    let idx = j * nx + i;
                    self.u[idx] += vt * (-dy_m / dist);
                    self.v[idx] += vt * (dx_m / dist);
                    self.h[idx] += h_amp * shape;
                }
            }
        }
    }

    /// Relative vorticity, s⁻¹ (diagnostic).
    pub fn vorticity(&self) -> Vec<f32> {
        self.spatial_derivative(true)
    }

    /// Horizontal divergence, s⁻¹ (diagnostic; convergence = negative).
    pub fn divergence(&self) -> Vec<f32> {
        self.spatial_derivative(false)
    }

    fn spatial_derivative(&self, curl: bool) -> Vec<f32> {
        let p = &self.params;
        let (nx, ny) = (p.nx, p.ny);
        let r = p.radius_m;
        let dlam = TAU / nx as f32;
        let dphi = PI / ny as f32;
        let dy = r * dphi;
        let mut out = vec![0.0f32; nx * ny];
        for j in 0..ny {
            // Same capped metric as the prognostic operators — see the
            // divergence note in `step`.
            let cos_metric = self.lat[j].cos().max(ZONAL_METRIC_CAP_COS);
            let dx = r * dlam * cos_metric;
            let jm = j.saturating_sub(1);
            let jp = (j + 1).min(ny - 1);
            for i in 0..nx {
                let im = (i + nx - 1) % nx;
                let ip = (i + 1) % nx;
                let val = if curl {
                    (self.v[j * nx + ip] - self.v[j * nx + im]) / (2.0 * dx)
                        - ((self.u[jp * nx + i] * self.lat[jp].cos())
                            - (self.u[jm * nx + i] * self.lat[jm].cos()))
                            / (2.0 * dy * cos_metric)
                } else {
                    (self.u[j * nx + ip] - self.u[j * nx + im]) / (2.0 * dx)
                        + ((self.v[jp * nx + i] * self.lat[jp].cos())
                            - (self.v[jm * nx + i] * self.lat[jm].cos()))
                            / (2.0 * dy * cos_metric)
                };
                out[j * nx + i] = val;
            }
        }
        out
    }

    /// Diagnosed cloud coverage in [0, 1]: moisture relative to saturation,
    /// enhanced where the flow converges (uplift). This is the field the
    /// weather-cube producer consumes as its synoptic occupancy.
    pub fn cloud(&self) -> Vec<f32> {
        let div = self.divergence();
        let (nx, ny) = (self.params.nx, self.params.ny);
        let mut out = vec![0.0f32; nx * ny];
        // Precipitation scale: the equilibrium rain rate under a strong
        // moisture source (see the ITCZ evaporation boost).
        let rain_ref = 2.8e-6;
        for j in 0..ny {
            let qs = self.qsat[j];
            for i in 0..nx {
                let idx = j * nx + i;
                let moist = self.q[idx] / qs;
                let conv = (-div[idx] * 5.0e4).clamp(0.0, 2.0);
                let base = smoothstep(0.55, 1.05, moist);
                let lifted = smoothstep(0.25, 1.3, conv);
                let raining = smoothstep(0.45, 1.2, self.precip[idx] / (rain_ref * qs));
                out[idx] = (0.80 * base
                    + 0.40 * lifted * (0.35 + 0.65 * base)
                    + 0.55 * raining)
                    .clamp(0.0, 1.0);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_params(seed: u64) -> WeatherSimParams {
        WeatherSimParams {
            seed,
            nx: 128,
            ny: 64,
            ..WeatherSimParams::default()
        }
    }

    /// The weather-cube contract requires the sim to be a pure function of
    /// its params: same seed, same field, bit for bit. Anything time-, rng-,
    /// or thread-order-dependent breaks cube regeneration.
    #[test]
    fn same_seed_is_bit_identical() {
        let run = || {
            let mut sim = WeatherSim::new(small_params(0xBEEF));
            sim.run_days(3.0);
            sim.cloud()
        };
        assert_eq!(run(), run());
    }

    /// Long-run stability: no NaN, velocities off the sanity clamp, moisture
    /// in range. The first implementation blew up within days (forward-Euler
    /// gravity waves + unbalanced forcing) — this pins the fix.
    #[test]
    fn thirty_days_stays_bounded() {
        let mut sim = WeatherSim::new(small_params(0x7A10));
        sim.run_days(30.0);
        let mut clamped = 0usize;
        for (&u, &v) in sim.u_field().iter().zip(sim.v_field()) {
            assert!(u.is_finite() && v.is_finite(), "non-finite velocity");
            if u.abs() >= V_CLAMP - 0.5 || v.abs() >= V_CLAMP - 0.5 {
                clamped += 1;
            }
        }
        let frac = clamped as f32 / sim.u_field().len() as f32;
        assert!(
            frac < 0.001,
            "{:.3}% of cells ride the velocity clamp - the sim is blowing up",
            frac * 100.0
        );
        for &q in sim.q_field() {
            assert!(q.is_finite() && (0.0..=1.6).contains(&q), "moisture out of range: {q}");
        }
    }
}
