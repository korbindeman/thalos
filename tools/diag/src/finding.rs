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
/// Below this, the planet painted into the reflection cubemap is effectively a
/// flat tint — the impostor bake is bound but is not varying, which is how a
/// broken body-fixed rotation or a blank bake presents. Set well under the
/// 0.046 measured on Thalos from a 200 km orbit (≈ 51 % of mean luminance) and
/// far above the 0.0 a genuine constant produces, so it separates the two
/// without firing on a legitimately bland body.
pub const REFLECTION_ALBEDO_SPREAD_FLAT: f64 = 0.005;
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
