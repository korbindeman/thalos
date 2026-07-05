use std::sync::Arc;

use bevy::prelude::*;
use thalos_body_render::CLOUD_BAND_COUNT;
use thalos_physics_canonical::{
    body_trajectory_provider::BodyTrajectoryProvider, canonical::Epoch, simulation::Simulation,
    types::BodyStates,
};
use thalos_terrain::DynamicSurfaceState;
use thalos_world::{BodyId, SolarSystemDefinition};

use crate::SimStage;

/// Central simulation state: the long-lived authority that advances time,
/// craft state, flight plans, and the active body trajectory provider.
#[derive(Resource)]
pub struct SimulationState {
    pub simulation: Simulation,
    pub system: SolarSystemDefinition,
    pub ephemeris: Arc<dyn BodyTrajectoryProvider>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudBandEnvironmentState {
    pub phases: [f64; CLOUD_BAND_COUNT],
    pub scroll_rate_rad_s: f64,
    pub differential_rotation: f64,
}

impl CloudBandEnvironmentState {
    // Constructed once banded-cloud bodies are wired at spawn (see
    // `install_cloud_band_state`); kept as the clamping constructor.
    #[allow(dead_code)]
    pub fn new(scroll_rate_rad_s: f64, differential_rotation: f64) -> Self {
        Self {
            phases: [0.0; CLOUD_BAND_COUNT],
            scroll_rate_rad_s,
            differential_rotation: differential_rotation.clamp(0.0, 1.0),
        }
    }

    pub fn advance(&mut self, dt: f64) {
        if dt == 0.0 || self.scroll_rate_rad_s.abs() < 1.0e-12 {
            return;
        }

        for i in 0..CLOUD_BAND_COUNT {
            let sin2 = i as f64 / (CLOUD_BAND_COUNT - 1) as f64;
            let lat_factor = 1.0 - self.differential_rotation * sin2;
            let omega = self.scroll_rate_rad_s * lat_factor;
            self.phases[i] = (self.phases[i] + omega * dt).rem_euclid(std::f64::consts::TAU);
        }
    }
}

/// Per-body large-scale cloud-coverage weather parameters: the source the
/// volumetric-cloud renderer projects into its planet-fixed equirect coverage
/// map (latitude bands + low-frequency variation). This is the future weather
/// system's write target — evolve the fields (or, later, a full coverage grid)
/// and bump `version`; the renderer re-uploads on version change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudWeatherState {
    /// Noise seed for the large-scale coverage variation.
    pub seed: u64,
    /// Mean overcast fraction in [0, 1] (0.38 ≈ broken/scattered cloud).
    pub coverage_mean: f32,
    /// Amplitude of the latitude-band modulation (ITCZ / subtropical dry
    /// belts / mid-latitude storm tracks), added to the mean.
    pub band_strength: f32,
    /// Amplitude of the low-frequency noise variation (clear patches vs.
    /// overcast regions), centred on the mean.
    pub variation: f32,
    /// Re-upload trigger: the renderer regenerates its coverage texture when
    /// this changes.
    pub version: u32,
}

impl Default for CloudWeatherState {
    fn default() -> Self {
        Self {
            seed: 0x7A105_C10D5,
            coverage_mean: 0.38,
            band_strength: 0.18,
            variation: 0.45,
            version: 0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BodyEnvironmentState {
    /// Mutable runtime state for terrain-owned dynamic layers: seasonal ice,
    /// active dunes, and later weather/tide-driven surface overlays.
    pub dynamic_surface: Option<DynamicSurfaceState>,
    /// Atmospheric cloud-band motion and phases. Kept here, not on render
    /// components, so map impostors, ship impostors, terrain skies, and future
    /// weather systems all see the same cloud state.
    pub cloud_bands: Option<CloudBandEnvironmentState>,
    /// Large-scale volumetric-cloud coverage weather (terrestrial-atmosphere
    /// bodies). Same ownership rationale as `cloud_bands`.
    pub cloud_weather: Option<CloudWeatherState>,
}

/// Canonical evaluated solar-system state for the current game frame.
///
/// This is the source that projections consume. Bevy entities, impostor
/// materials, terrain tile providers, map snapshots, and atmosphere passes may
/// cache derived data, but they should not independently evaluate or own body
/// state. Future wind, storms, tides, and dune migration belong in
/// [`BodyEnvironmentState`] so every projection reads the same runtime
/// environment for a body.
///
/// **Sole writer:** [`sync_solar_system_state`] (in [`SimStage::Sync`]). All
/// other systems read it; environment mutators go through `environment_mut`.
#[derive(Resource, Debug, Default)]
pub struct SolarSystemState {
    pub states: Option<BodyStates>,
    pub time: f64,
    pub environment: Vec<BodyEnvironmentState>,
}

impl SolarSystemState {
    pub fn environment_mut(&mut self, body_id: BodyId) -> Option<&mut BodyEnvironmentState> {
        self.environment.get_mut(body_id)
    }

    fn ensure_body_capacity(&mut self, body_count: usize) {
        if self.environment.len() < body_count {
            self.environment
                .resize_with(body_count, BodyEnvironmentState::default);
        }
    }

    // Forward environment-install API, ready for spawn-time wiring:
    // `install_cloud_band_state` lights up the `update_cloud_bands` drift
    // loop the moment a body is given cloud bands. Kept symmetric with the live
    // `install_cloud_weather`.
    #[allow(dead_code)]
    pub fn install_cloud_band_state(&mut self, body_id: BodyId, state: CloudBandEnvironmentState) {
        self.ensure_body_capacity(body_id + 1);
        self.environment[body_id].cloud_bands = Some(state);
    }

    pub fn install_cloud_weather(&mut self, body_id: BodyId, state: CloudWeatherState) {
        self.ensure_body_capacity(body_id + 1);
        self.environment[body_id].cloud_weather = Some(state);
    }

}

pub fn sync_solar_system_state(
    sim: Res<SimulationState>,
    mut solar_system: ResMut<SolarSystemState>,
) {
    let epoch = Epoch(sim.simulation.sim_time());
    if solar_system.states.is_some() && (solar_system.time - epoch.0).abs() < f64::EPSILON {
        return;
    }

    if let Some(states) = solar_system.states.as_mut() {
        sim.ephemeris.states_into(epoch, states);
    } else {
        let mut states = Vec::with_capacity(sim.ephemeris.body_count());
        sim.ephemeris.states_into(epoch, &mut states);
        solar_system.states = Some(states);
    }
    solar_system.time = epoch.0;
    solar_system.ensure_body_capacity(sim.ephemeris.body_count());
}

pub struct SolarSystemStatePlugin;

impl Plugin for SolarSystemStatePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SolarSystemState>()
            .add_systems(Update, sync_solar_system_state.in_set(SimStage::Sync));
    }
}
