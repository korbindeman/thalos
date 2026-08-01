//! The finding model and the thresholds that produce one.
//!
//! The whole point of this tool is signal-to-noise: a triage pass that reports
//! everything is a triage pass nobody reads, and one that reports nothing is a
//! lie. So a check earns its place only if it can state a **falsifiable claim
//! with a denominator** ("3 of 11 shots failed", not "captures look slow"), and
//! every threshold below is a named constant with the reason it sits where it
//! does.
//!
//! Two severities, deliberately. `Attention` means *a human or agent should
//! look today*; `Watch` means *this is drifting, and if it repeats it becomes
//! Attention*. There is no third level, because a level nobody acts on is the
//! definition of noise.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Look at this today.
    Attention,
    /// Drifting; act if it repeats.
    Watch,
}

impl fmt::Display for Severity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Severity::Attention => "ATTENTION",
            Severity::Watch => "WATCH",
        })
    }
}

/// One thing worth a closer look.
#[derive(Clone, Debug)]
pub struct Finding {
    pub severity: Severity,
    /// Stable key. The triage routine dedupes filed work against this, so
    /// renaming one re-files everything it covers — treat it as an API.
    pub id: &'static str,
    /// The claim, with its numbers and denominator, in one line.
    pub headline: String,
    /// At most a few evidence lines. A finding that needs more than this is
    /// really an investigation, and the investigation belongs in the record it
    /// points at.
    pub detail: Vec<String>,
    /// Where to go next: a file, an event filter, or a command.
    pub next: String,
}

/// A vehicle flow effect (plume, reentry shock layer) sizes itself from the
/// craft's measured visual bounds. Below this many measured descendant meshes the
/// sweep has plainly not found the vehicle, and every attached effect is running
/// on a default size — which renders as an effect fitted to the wrong body rather
/// than as anything missing, so nothing else reports it.
///
/// Two is the floor rather than one because the smallest authored craft still
/// carries a hull plus at least one appendage; a single measured mesh has always
/// meant the sweep stopped early.
pub const FLOW_BOUNDS_MESH_MIN: u64 = 2;

/// Aspect ratio above which a measured craft box is suspiciously cubic for a
/// vehicle. Real craft are elongated; a box whose longest and shortest axes agree
/// this closely usually means the sweep measured one placeholder mesh instead of
/// the hull.
pub const FLOW_BOUNDS_CUBIC_RATIO: f64 = 1.05;

impl Finding {
    pub fn new(
        severity: Severity,
        id: &'static str,
        headline: impl Into<String>,
        next: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            id,
            headline: headline.into(),
            detail: Vec::new(),
            next: next.into(),
        }
    }

    pub fn with_detail(mut self, line: impl Into<String>) -> Self {
        // Cap here rather than at every call site: an unbounded evidence dump
        // is the failure mode this tool exists to avoid.
        if self.detail.len() < MAX_DETAIL_LINES {
            self.detail.push(line.into());
        }
        self
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "severity": self.severity.to_string(),
            "id": self.id,
            "headline": self.headline,
            "detail": self.detail,
            "next": self.next,
        })
    }
}

/// Evidence lines kept per finding.
pub const MAX_DETAIL_LINES: usize = 4;

/// LAND completion is only valid at a near-zero wheel speed. The controller's
/// own threshold is 0.5 m/s; 0.75 leaves room for the diagnostic sample and
/// physics step to straddle the boundary without hiding a rolling disengage.
pub const LAND_COMPLETION_SPEED_M_S_ATTENTION: f64 = 0.75;

/// Go-arounds per session that mean LAND is not converging on an approach.
///
/// One go-around is the mode working: an approach went out of tolerance and it
/// repositioned. Two in a session is bad luck or a marginal aircraft. Three is
/// the controller's own retry limit, so reaching it means the session ended in
/// `UNABLE` and the player never landed — the outcome this check exists to
/// surface.
pub const LAND_GO_AROUNDS_ATTENTION: usize = 3;

/// Approach re-plans per session before the route is churning rather than
/// converging.
///
/// The recorded failure re-planned every ~47 s for the whole approach, because
/// the drift trigger fired on cross-track that a *planned* course reversal
/// creates by construction. Each rebuild teleported distance-to-go and the
/// entire vertical profile with it. A healthy approach re-plans when the pilot
/// changes the selection and otherwise not at all, so a handful in one session
/// is already the signature of that loop returning.
pub const ROUTE_REPLANS_ATTENTION: usize = 4;

/// Committed rejoins per session before the follower is not holding its path.
///
/// One or two per approach is the mechanism working: the craft was blown off,
/// a way back was committed, it flew it. Committing repeatedly means it keeps
/// falling off a route it is supposed to be tracking — which is the lateral
/// tracking failure, showing up one level above where it happens. The commit is
/// already rate-limited to one per 20 s, so reaching this count takes minutes of
/// sustained failure to track.
pub const ROUTE_REJOINS_ATTENTION: usize = 4;

/// ORBIT promises these live achieved-element tolerances before publishing
/// completion. The reader repeats the contract so a future executor regression
/// cannot emit a plausible-looking false success.
pub const ORBIT_APSIS_ERROR_M_ATTENTION: f64 = 2_000.0;
pub const ORBIT_INCLINATION_ERROR_RAD_ATTENTION: f64 = 0.01;
/// The ascent controller begins throttling at 35 kPa. Three one-second samples
/// above 120% of that limit distinguish a transient crossing from a sustained
/// max-Q guidance failure.
pub const ORBIT_DYNAMIC_PRESSURE_PA_ATTENTION: f64 = 42_000.0;
pub const ORBIT_MAX_Q_SAMPLES_ATTENTION: usize = 3;

/// Staging is *commanded* on a predicted burnout; thrust collapse survives only
/// as a backup trigger. Every `stage_unpredicted` is therefore a prediction the
/// sequencer got wrong, and one is already worth reading — the vehicle staged
/// late, with a thrust dropout the guidance did not plan for. Set at 1 because
/// the healthy count is exactly zero: any nonzero value falsifies the
/// prediction, which is the whole reason the backup path still logs instead of
/// silently covering for it.
pub const STAGE_UNPREDICTED_ATTENTION: usize = 1;

/// A refused staging request means the sequencer passed its interlocks and the
/// canonical staging op still declined — a real contradiction between the two,
/// not a tuning question. One is worth a look.
pub const STAGE_REFUSED_ATTENTION: usize = 1;

// ── Capture-lane thresholds ─────────────────────────────────────────────────
//
// The capture lane is the throughput floor for agent work, so its bar is the
// strictest in the file: one failed shot is already worth a look, because it
// cost an agent a full cycle and may have cost a wrong conclusion.

/// Any failed shot is Attention. Failures are not routine here.
pub const CAPTURE_FAILURES_ATTENTION: usize = 1;
/// Abandoned runs are usually an interrupted invocation (Ctrl-C), so they only
/// become interesting in bulk.
pub const CAPTURE_ABANDONED_WATCH: usize = 3;
/// Shots needing a host retry, as a fraction of shots. A retry is a full boot
/// paid twice.
pub const CAPTURE_RETRY_FRACTION_ATTENTION: f64 = 0.20;
/// Fraction of shots that had to boot or reboot the host. Some rebuilding is
/// normal while editing Rust; a lane that reboots for most shots means the
/// reuse path is broken and every agent pays 1.5–2.5 min per image.
pub const CAPTURE_BOOT_FRACTION_WATCH: f64 = 0.50;
pub const CAPTURE_BOOT_FRACTION_ATTENTION: f64 = 0.80;
/// Below this many shots, rates are noise rather than signal.
pub const CAPTURE_RATE_MIN_SHOTS: usize = 5;
/// p95 wall time for a successful shot. A warm reuse is seconds; a rebuild is
/// ~1.5–2.5 min, so 120 s is "most shots are rebuilding" and 300 s is "something
/// is wrong with the lane, not with Cargo".
pub const CAPTURE_P95_MS_WATCH: f64 = 120_000.0;
pub const CAPTURE_P95_MS_ATTENTION: f64 = 300_000.0;
/// Total time lost waiting for the machine-wide capture lock.
pub const CAPTURE_LOCK_WAIT_MS_WATCH: f64 = 600_000.0;

// ── Runtime thresholds ──────────────────────────────────────────────────────

/// Frame spikes recorded in one session. The collector already cools down 5 s
/// between dumps, so twenty of them is a session that hitched throughout.
pub const SPIKES_WATCH: usize = 5;
pub const SPIKES_ATTENTION: usize = 20;
/// A 2 s gauge whose p95 CPU frame time is above 30 fps.
pub const SLOW_FRAME_MS: f64 = 33.3;
/// Consecutive-ish gauges over [`SLOW_FRAME_MS`] before it is more than a load
/// hitch: 3 gauges ≈ 6 s, 10 ≈ 20 s of sustained sub-30 fps.
pub const SLOW_GAUGES_WATCH: usize = 3;
pub const SLOW_GAUGES_ATTENTION: usize = 10;
/// Memory the render-mesh slabs plus tile residency grew across one session.
/// Growth on this scale is the open accumulation question behind the tile OOM
/// (INC-20260725T012104Z), so it is checked explicitly rather than left to a
/// crash to reveal.
pub const MEMORY_GROWTH_MIB_ATTENTION: f64 = 1024.0;
/// The cascade cameras live outside big_space and must use exactly the cell
/// origin the current frame renders against. Anything above one centimetre is
/// not floating-point noise; it is a scheduling/frame mismatch and can move
/// every shadow edge at once.
pub const SHADOW_ORIGIN_ERROR_M_ATTENTION: f64 = 0.01;
/// Reversals of the cascade-mode switch (`active_cascades`, 4 near-surface vs 2
/// craft-local) within one session. Entering craft-local mode parks cascades 2–3
/// and turns every ground shadow off in one frame, so the switch is only ever
/// correct as a one-way consequence of a real climb or descent: a genuine ascent
/// reads 4 → 2 and STAYS, contributing zero reversals. A reversal (4 → 2 → 4)
/// means the altitude sat on the threshold and the mode strobed, which the
/// player sees as the whole world's shadows blinking. Two tolerates one honest
/// round trip — a climb and a descent inside the same session; beyond that the
/// hysteresis band (`SHADOW_CRAFT_LOCAL_EXIT_M`) is not doing its job.
pub const SHADOW_MODE_REVERSALS_ATTENTION: usize = 2;
/// Craft-local shadow mode is legitimate only when the VIEW is high. The sky
/// lane's `environment_paint.altitude_m` is an independent, view-anchored
/// record of where the view actually was, so a craft-local gauge bracketed by
/// paint samples below this altitude means the gate resolved the view from the
/// wrong thing — the INC-20260731T004704Z class (the gate read the orbiting
/// craft at a 600 m god view), which renders a plausible frame with every
/// non-craft shadow missing and no error anywhere. Set below the mode's own
/// 40 km exit threshold so a genuine descent that is one frame from unlatching
/// does not fire, and far above any god view or surface flight.
pub const SHADOW_CRAFT_LOCAL_SURFACE_ALT_M: f64 = 35_000.0;
/// Sustained samples (~1 Hz gauge) of craft-local-at-surface before it is a
/// finding. Three tolerates the seam frames of an honest teleport from orbit
/// down to a pad, where one or two gauges can straddle the transition.
pub const SHADOW_CRAFT_LOCAL_SURFACE_SAMPLES_ATTENTION: usize = 3;
/// Pairing window between a shadow gauge and the nearest sky paint sample.
/// Paint publishes at roughly 0.5 Hz, so 3 s always brackets one sample while
/// staying too short to pair across a real climb through the threshold.
pub const SHADOW_VIEW_ALT_PAIR_WINDOW_MS: u128 = 3_000;
/// Below this, the planet painted into the reflection cubemap is effectively a
/// flat tint — the impostor bake is bound but is not varying, which is how a
/// broken body-fixed rotation or a blank bake presents. Set well under the
/// 0.046 measured on Thalos from a 200 km orbit (≈ 51 % of mean luminance) and
/// far above the 0.0 a genuine constant produces, so it separates the two
/// without firing on a legitimately bland body.
pub const REFLECTION_ALBEDO_SPREAD_FLAT: f64 = 0.005;
/// Half-angle of the planet surface actually visible from the paint point,
/// below which `albedo_spread` says nothing about the bake. The event publishes
/// the disc's angular *radius* (`planet_ang_deg` = asin(R/(R+h))), so the
/// visible spherical cap is `90° - planet_ang_deg` — a footprint of radius
/// `R · cap` on the ground.
///
/// This exists because the spread test cannot be read without it. Triage
/// 2026-07-31 measured two clean populations over 24 h: the 200 km orbital view
/// (`planet_ang_deg` 70.2 → cap 19.8°, footprint ≈ 2,000 km on Thalos) reads
/// spread 0.046 — continents against ocean; a ~7 km view (`planet_ang_deg` 87.4
/// → cap 2.6°, footprint ≈ 270 km) reads 0.0018 across **168 of 373 paints**,
/// with a different, brighter `albedo_mean`. The low reading there is the
/// correct answer — from 7 km up the reflected ground is one landscape — so
/// judging it flagged every low-altitude session and buried the defect the
/// check exists for.
///
/// 10° (footprint radius ≈ 0.17 R, ≈ 1,050 km on Thalos) is set to span several
/// biomes and coastline on any authored body, so a bound bake that reads
/// constant across it really is blank or wrongly rotated. It leaves ~2× margin
/// under the orbital framing this check was built for (`orbit-hull`, cap 19.8°)
/// and ~4× over the near-surface population it must ignore. On a 6,000 km body
/// it starts judging from roughly 90 km up.
pub const REFLECTION_VISIBLE_CAP_MIN_DEG: f64 = 10.0;
/// Share of the reflection cubemap the planet must cover before a missing bake
/// is worth reporting. Below this the planet is a minor feature of the
/// reflected sky; at and above it, reflecting a flat disc is a visible defect
/// on a polished hull.
pub const REFLECTION_DISC_FRAC_MIN: f64 = 0.05;
/// Whole-card dedicated-memory occupancy. Above 90% the driver has little
/// headroom for transient allocations, compaction, or the desktop compositor.
pub const GPU_MEMORY_USED_FRAC_ATTENTION: f64 = 0.90;
/// Ada normally controls around its low-80s target. A sustained sample at or
/// above 88 °C is outside the expected envelope and separates thermal pressure
/// from an allocation/backend failure.
pub const GPU_TEMPERATURE_C_ATTENTION: f64 = 88.0;
/// NVML's software-thermal-slowdown reason. NVIDIA defines this bit as the GPU
/// or memory exceeding its maximum operating temperature. The memory sensor is
/// not exposed on every GeForce card, so this is stronger evidence than the
/// visible core temperature alone.
pub const GPU_SW_THERMAL_SLOWDOWN_MASK: u64 = 0x20;
/// NVML's hardware-thermal-slowdown reason. Unlike ordinary boost management,
/// this means the hardware protection path reduced core clocks sharply.
pub const GPU_HW_THERMAL_SLOWDOWN_MASK: u64 = 0x40;
/// NVML's broader hardware-slowdown bit. It accompanied hardware thermal
/// slowdown in the incident that motivated this check and is kept as evidence.
pub const GPU_HW_SLOWDOWN_MASK: u64 = 0x08;
/// Ignore a one-sample clock-transition blip, but report thermal limiting that
/// appears for roughly three seconds within one runtime session.
pub const GPU_THERMAL_THROTTLE_SAMPLES_ATTENTION: usize = 3;
// ── lane hygiene ────────────────────────────────────────────────────────────

/// Below this volume, a dominant event is not yet a cost worth acting on.
pub const NOISE_MIN_RECORDS: usize = 5_000;
/// Share of all records one event may hold before it is worth questioning.
pub const NOISE_SHARE_WATCH: f64 = 0.40;

/// The tile budget brake coarsens terrain to stay inside VRAM. Any engagement
/// means a capture-verified framing may have rendered coarser than authored.
pub const TILE_BRAKE_SCALE: f64 = 0.999;

/// A wheeled craft carried by the terrain floor backstop instead of its gear
/// (`backstop_intervention` with `gear_down = 1, weight_on_wheels = 0`) is the
/// buried-suspension-ray signature (INC-20260729T073116Z): the craft slides
/// belly-down with no braking authority. The events are 1 Hz-throttled, so
/// each one is ~one second spent in the state; 2+ means sustained carrying
/// rather than a single touchdown transient.
pub const GEAR_BACKSTOP_CARRY_EVENTS_ATTENTION: usize = 2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detail_lines_are_capped() {
        let mut finding = Finding::new(Severity::Watch, "test", "headline", "next");
        for index in 0..20 {
            finding = finding.with_detail(format!("line {index}"));
        }
        assert_eq!(finding.detail.len(), MAX_DETAIL_LINES);
    }
}
