//! **Experimental A/B toggle**: Bevy's stock atmosphere in place of the
//! custom `BodySky` fullscreen pass.
//!
//! When [`GraphicsSettings::stock_atmosphere`] is on, every body with an
//! authored scattering layer gets a stock [`Atmosphere`] component on its
//! real-space grid entity (whose `GlobalTransform` *is* the planet center in
//! render space — exactly the contract the 0.19 planet-centered atmosphere
//! wants), the ship camera gets [`AtmosphereSettings`] with
//! [`AtmosphereMode::Raymarched`] (the mode built for planets seen from
//! orbit), and the custom `BodySky` pass is force-hidden.
//!
//! This is a **look-comparison tool, not a replacement**: the stock pass
//! deliberately bypasses the `thalos::lighting` spine, so while it is on we
//! lose everything the `BodySky` pass composites — the analytic ray-traced
//! ocean, the volumetric-cloud composite, the star-crush coupling, and the
//! spine's aerial perspective on terrain. The scattering coefficients are
//! derived from the same authored `AtmosphericScattering` block the custom
//! pass reads (β = τ_v / H per term), so the two skies are fed identical
//! physics inputs; brightness differs because the stock pass is photometric
//! (scales with the sun `DirectionalLight` illuminance) while ours uses the
//! spine's arbitrary flux units. The authored artistic knobs
//! (`strength`, `multi_scatter_gain`) are intentionally **not** applied —
//! they compensate for our unit system, not physical density.
//!
//! Toggle lives in Settings → Graphics ("Stock Bevy atmosphere"), so the two
//! skies can be A/B'd live in one session.

use bevy::light::Atmosphere;
use bevy::light::atmosphere::{Falloff, PhaseFunction, ScatteringMedium, ScatteringTerm};
use bevy::pbr::{AtmosphereMode, AtmosphereSettings};
use bevy::prelude::*;
use thalos_body_render::{AtmosphereBlock, SolidPlanetMaterial};
use thalos_world::{BodyId, atmosphere::AtmosphericScattering};

use super::SimulationState;
use super::ground_terrain::BodySky;
use super::types::{CelestialBody, RealSpaceBody, SolidPlanetMaterials};
use crate::camera::ShipCamera;
use crate::coords::SHIP_SCALE;
use crate::graphics_settings::GraphicsSettings;

/// Earth-like Mie absorption as a fraction of Mie scattering (Hillaire 2020
/// uses β_abs ≈ 0.11 · β_sca). Our authored model folds absorption into a
/// single Mie optical depth, so the split is reconstructed here.
const MIE_ABSORPTION_RATIO: f32 = 0.11;

/// Build a stock [`ScatteringMedium`] from the authored scattering block.
///
/// The authored values are vertical optical depths (τ_v = β · H), so the
/// per-meter coefficients the stock model wants are β = τ_v / H. Stock
/// `Falloff::Exponential`'s `scale` is the scale height as a fraction of the
/// total atmosphere height (cf. `ScatteringMedium::earth`: `8.0 / 60.0`).
fn build_medium(scattering: &AtmosphericScattering, atmosphere_height_m: f32) -> ScatteringMedium {
    let rayleigh_beta =
        Vec3::from(scattering.vertical_optical_depth) / scattering.rayleigh_scale_height_m;
    let mie_beta = scattering.mie_optical_depth / scattering.mie_scale_height_m;
    ScatteringMedium::new(
        256,
        256,
        [
            ScatteringTerm {
                absorption: Vec3::ZERO,
                scattering: rayleigh_beta,
                falloff: Falloff::Exponential {
                    scale: scattering.rayleigh_scale_height_m / atmosphere_height_m,
                },
                phase: PhaseFunction::Rayleigh,
            },
            ScatteringTerm {
                absorption: Vec3::splat(mie_beta * MIE_ABSORPTION_RATIO),
                scattering: Vec3::splat(mie_beta),
                falloff: Falloff::Exponential {
                    scale: scattering.mie_scale_height_m / atmosphere_height_m,
                },
                phase: PhaseFunction::Mie {
                    asymmetry: scattering.mie_asymmetry,
                },
            },
        ],
    )
    .with_label("stock_atmosphere_authored")
}

/// Keep the stock-atmosphere components in sync with the toggle.
///
/// Enabled: insert [`Atmosphere`] on each atmospheric body's real-space grid
/// entity and [`AtmosphereSettings`] (raymarched) on the ship camera.
/// Disabled: remove the per-body [`Atmosphere`] components — but **keep**
/// `AtmosphereSettings` on the camera. The extract system only clears the
/// render world's `ExtractedAtmosphere` for cameras that still carry
/// `AtmosphereSettings`; removing both in the same frame would strand a stale
/// extracted copy and the stock sky would keep rendering forever. With
/// settings present and zero `Atmosphere` entities the pass is fully idle.
pub(super) fn sync_stock_atmosphere(
    mut commands: Commands,
    settings: Res<GraphicsSettings>,
    sim: Res<SimulationState>,
    mut media: ResMut<Assets<ScatteringMedium>>,
    mut medium_cache: Local<bevy::platform::collections::HashMap<BodyId, Handle<ScatteringMedium>>>,
    bodies: Query<(Entity, &RealSpaceBody, Has<Atmosphere>)>,
    camera: Query<(Entity, Has<AtmosphereSettings>), With<ShipCamera>>,
) {
    if !settings.stock_atmosphere {
        for (entity, _, has_atmosphere) in &bodies {
            if has_atmosphere {
                commands.entity(entity).remove::<Atmosphere>();
            }
        }
        return;
    }

    if let Ok((camera_entity, has_settings)) = camera.single()
        && !has_settings
    {
        commands.entity(camera_entity).insert(AtmosphereSettings {
            rendering_method: AtmosphereMode::Raymarched,
            // Default is 32 km — tuned for ground-level scenes. Stretch the
            // aerial-view LUT across the whole in-atmosphere sightline so
            // distant terrain still picks up haze.
            aerial_view_lut_max_distance: 3.2e5,
            ..default()
        });
    }

    for (entity, real_body, has_atmosphere) in &bodies {
        if has_atmosphere {
            continue;
        }
        let Some(body) = sim.system.bodies.get(real_body.body_id) else {
            continue;
        };
        let Some(atmo) = body.terrestrial_atmosphere.as_ref() else {
            continue;
        };
        let Some(scattering) = atmo.scattering.as_ref() else {
            continue;
        };
        if atmo.karman_line_m <= 0.0 {
            continue;
        }
        let medium = medium_cache
            .entry(real_body.body_id)
            .or_insert_with(|| media.add(build_medium(scattering, atmo.karman_line_m)))
            .clone();
        let radius = body.radius_m as f32;
        commands.entity(entity).insert(Atmosphere {
            inner_radius: radius,
            outer_radius: radius + atmo.karman_line_m,
            ground_albedo: Vec3::from(body.color),
            medium,
        });
    }
}

/// Keep the **ship-view impostor's** inline atmosphere in step with the toggle.
///
/// Beyond the terrain LOD swap the body renders through `SolidPlanetMaterial`,
/// whose `params.atmosphere` block draws its own rim halo + on-disc aerial
/// perspective. With the stock atmosphere on, that stacks a second limb ring
/// on top of the stock raymarch (the "double atmosphere" seen from orbit) —
/// so zero the block while the toggle is on (the shader early-outs on the
/// `strength == 0` gate, keeping the baked continents/ocean disc) and restore
/// the authored block when it goes off. Ship material only: the map camera
/// never carries the stock atmosphere, so the map disc + halo keep the custom
/// look. `update_solid_planet_params` never writes `params.atmosphere`, so
/// there is no writer conflict.
pub(super) fn sync_impostor_atmosphere_with_stock(
    settings: Res<GraphicsSettings>,
    sim: Res<SimulationState>,
    bodies: Query<(&CelestialBody, &SolidPlanetMaterials)>,
    mut materials: ResMut<Assets<SolidPlanetMaterial>>,
) {
    let body_defs = sim.simulation.bodies();
    for (body, mats) in &bodies {
        let Some(def) = body_defs.get(body.body_id) else {
            continue;
        };
        // Only bodies whose authored atmosphere actually scatters need the
        // swap; airless bodies sit at strength 0 in both states.
        let Some(atmo) = def.terrestrial_atmosphere.as_ref() else {
            continue;
        };
        let authored = AtmosphereBlock::from_terrestrial(atmo, (1.0 / SHIP_SCALE) as f32);
        if authored.atmos_geom.z <= 0.0 {
            continue;
        }
        // Peek before mutating; `get_mut` re-uploads the uniform even on
        // identical writes (same pattern as `update_solid_planet_params`).
        let Some(mat) = materials.get(&mats.ship) else {
            continue;
        };
        let currently_zero = mat.params.atmosphere.atmos_geom.z == 0.0;
        if settings.stock_atmosphere == currently_zero {
            continue;
        }
        if let Some(mut mat) = materials.get_mut(&mats.ship) {
            mat.params.atmosphere = if settings.stock_atmosphere {
                AtmosphereBlock::default()
            } else {
                authored
            };
        }
    }
}

/// While the stock atmosphere is on, force the custom `BodySky` pass hidden.
///
/// Runs after `sync_body_render_lod` (the sky's normal visibility owner,
/// which re-evaluates every frame), so switching the toggle off needs no
/// cleanup here — the LOD pass restores the custom sky on the next frame.
pub(super) fn suppress_body_sky_for_stock_atmosphere(
    settings: Res<GraphicsSettings>,
    mut skies: Query<&mut Visibility, With<BodySky>>,
) {
    if !settings.stock_atmosphere {
        return;
    }
    for mut vis in &mut skies {
        if *vis != Visibility::Hidden {
            *vis = Visibility::Hidden;
        }
    }
}
