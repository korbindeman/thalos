//! Reflect-friendly mirrors of canonical craft state.

use bevy::prelude::*;

/// Reflect-registered mirror of the canonical `CraftState`, refreshed
/// once per frame by the runtime's `refresh_craft_state_mirror` (its **sole
/// writer**). The canonical state lives in `thalos_physics_canonical` (no
/// Bevy dependency), so it cannot derive `Reflect` directly; this resource is
/// a read-only Reflect-registered projection (for the HUD / a future debug
/// overlay).
#[derive(Resource, Reflect, Default, Clone, Debug)]
#[reflect(Resource)]
pub struct CraftStateMirror {
    pub sim_time_s: f64,
    pub warp_speed: f64,
    pub position_m: [f64; 3],
    pub velocity_m_s: [f64; 3],
    /// World-frame angular velocity of the craft (rad/s). In the local
    /// bubble this is the live Avian body rate (written back to canonical by
    /// the runtime's `readback_local_craft`); a diagnostic for attitude /
    /// control-stability work — a steady non-decaying oscillation here is
    /// SAS chatter. See `docs/simulation/control.md`.
    pub angular_velocity_rad_s: [f64; 3],
    pub mass_kg: f64,
    pub dominant_body_id: u32,
    /// Discriminant name of `AuthorityMode` (variant fields elided).
    pub authority: String,
    /// Whole-craft structural failure from a terrain impact. See
    /// `docs/simulation/surface.md`.
    pub destroyed: bool,
    /// Surface-relative approach speed (m/s) of the destroying impact;
    /// `0.0` unless `destroyed`.
    pub last_impact_speed_m_s: f64,
    /// Aggregate thrust currently pushed into the propagator (N). Zero
    /// means no engine is producing thrust this frame (e.g. air-breathing
    /// jets with no intake air). Diagnostic mirror of
    /// `ship_params().thrust_n`.
    pub thrust_n: f64,
    /// Altitude above the dominant body's reference radius (m).
    pub altitude_m: f64,
    /// Whether the dominant body has a `terrestrial_atmosphere` block.
    pub has_atmosphere: bool,
    /// Kármán line of the dominant body's atmosphere (m); 0 if none.
    pub karman_line_m: f64,
    /// Whether the ship is currently inside the breathable column (the
    /// gate air-breathing jets check). Diagnostic mirror of the runtime's
    /// `fuel::ship_in_atmosphere`.
    pub in_atmosphere: bool,
    /// Number of engines that passed every propulsion gate this frame
    /// (enabled, positive thrust/isp, atmosphere ok, intake satisfied,
    /// reactants present). Zero with `in_atmosphere == true` means the
    /// gate that killed thrust is *not* the atmosphere.
    pub propulsion_engine_count: u32,
}
