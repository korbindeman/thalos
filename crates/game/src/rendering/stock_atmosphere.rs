//! Canonical rocky-body sky: Bevy's raymarched atmosphere, with the superseded
//! custom `BodySky` atmosphere retained only as a debug A/B fallback.
//!
//! Unless [`GraphicsSettings::legacy_body_sky`] is explicitly on, the
//! atmospheric body selected by the canonical render-camera [`ViewAnchor`] gets a stock
//! [`Atmosphere`] component on a camera-local proxy entity. The proxy position
//! is projected in f64 from [`ViewAnchor`] each frame; this avoids handing
//! Bevy a planet-grid `GlobalTransform` that can be transiently overwritten by
//! the atmosphere component's default-placement hook under BigSpace. The ship
//! camera gets
//! [`AtmosphereSettings`] with
//! [`AtmosphereMode::Raymarched`] (the mode built for planets seen from
//! orbit), and the custom `BodySky` pass is force-hidden.
//!
//! The debug fallback remains useful for matched captures while the retained
//! `BodySky` composites are migrated independently. Rayleigh coefficients are
//! derived from the same authored `AtmosphericScattering` block the custom pass reads
//! (`β = τ_v / H`); the legacy aerosol value is projected as a relative load
//! around Bevy's Earth Mie baseline, and Bevy's Earth ozone term is added. This
//! is intentionally an Earth-reference convergence path, not a numerically
//! identical replay of the old shader. Brightness also differs because the
//! stock pass is photometric (scales with the sun `DirectionalLight`
//! illuminance) while ours uses the spine's arbitrary flux units. The authored
//! artistic knobs (`strength`, `multi_scatter_gain`) are intentionally **not** applied —
//! they compensate for our unit system, not physical density.
//!
//! Settings → Graphics exposes "Legacy custom atmosphere (debug)" solely for
//! matched A/B verification.

use bevy::light::Atmosphere;
use bevy::light::atmosphere::{Falloff, PhaseFunction, ScatteringMedium, ScatteringTerm};
use bevy::pbr::{AtmosphereMode, AtmosphereSettings};
use bevy::prelude::*;
use thalos_body_render::{AtmosphereBlock, SolidPlanetMaterial};
use thalos_world::{BodyId, atmosphere::AtmosphericScattering};

use super::ground_terrain::BodySky;
use super::types::{CelestialBody, RealSpaceBody, SolidPlanetMaterials};
use super::view_anchor::ViewAnchor;
use super::{SimulationState, SolarSystemState};
use crate::camera::ShipCamera;
use crate::coords::SHIP_SCALE;
use crate::graphics_settings::GraphicsSettings;
use crate::screenshot::ScreenshotConfig;

/// Resolve the interactive setting plus the optional deterministic headless
/// override. The screenshot resource is absent in normal gameplay, and unlike
/// mutating `GraphicsSettings` this cannot leak an A/B selection into
/// `user/settings.ron` through the settings autosave.
fn enabled(settings: &GraphicsSettings, screenshot: Option<&ScreenshotConfig>) -> bool {
    screenshot
        .and_then(|cfg| cfg.atmosphere.stock_override())
        .unwrap_or(!settings.legacy_body_sky)
}

fn active_body(
    settings: &GraphicsSettings,
    screenshot: Option<&ScreenshotConfig>,
    view_anchor: &ViewAnchor,
) -> Option<BodyId> {
    enabled(settings, screenshot)
        .then_some(view_anchor.resolved)?
        .map(|anchor| anchor.body)
}

/// The one Bevy-atmosphere projection for the active ship view. Thalos remains
/// N-body; Bevy's renderer receives one camera-local atmosphere at a time, as
/// its extraction contract requires.
#[derive(Component)]
pub(super) struct BevyAtmosphereProxy {
    body_id: BodyId,
}

/// The legacy model authored `0.025` as Thalos's clean-aerosol reference. Bevy's
/// Earth medium uses a much lower aerosol coefficient and a strongly absorbing
/// split; preserve the authored value as a *relative loading* around that
/// physical baseline instead of feeding it to Bevy as a literal scattering
/// optical depth (which produces the opaque milk-blue result in the matched
/// orbital capture).
const AUTHORED_EARTH_MIE_REFERENCE: f32 = 0.025;
const BEVY_EARTH_MIE_SCATTERING: f32 = 0.444e-6;
const BEVY_EARTH_MIE_ABSORPTION: f32 = 3.996e-6;
const OZONE_CENTER_M: f32 = 45_000.0;
const OZONE_WIDTH_M: f32 = 18_000.0;
/// Thalos terrain still emits in the shading spine's arbitrary scene-flux
/// units, while Bevy's atmosphere is driven by photometric directional-light
/// luminance. Until those surfaces share one photometric bind group (F7), an
/// unscaled Earth column overwhelms the darker terrain even though its spectral
/// shape is correct. This is the one adapter calibration knob, applied to both
/// scattering and extinction so the raymarch stays energy-consistent.
const BEVY_ATMOSPHERE_DENSITY_SCALE: f32 = 0.1;

/// Build a stock [`ScatteringMedium`] from the authored scattering block.
///
/// Rayleigh remains a direct physical projection: authored vertical optical
/// depth is `τ_v = β · H`, hence `β = τ_v / H`. Aerosols retain the authored
/// loading *relative* to Thalos's clean reference, but use Bevy's Earth Mie
/// scattering/absorption split. The final term is Bevy's Earth ozone profile;
/// the reference's blue limb and neutral surface color depend on that missing
/// wavelength-selective absorption.
///
/// Stock `Falloff` coordinates are fractions of the configured atmosphere
/// shell, so all physical altitudes are divided by `atmosphere_height_m`.
fn build_medium(scattering: &AtmosphericScattering, atmosphere_height_m: f32) -> ScatteringMedium {
    let rayleigh_beta =
        Vec3::from(scattering.vertical_optical_depth) / scattering.rayleigh_scale_height_m;
    let mie_load = (scattering.mie_optical_depth / AUTHORED_EARTH_MIE_REFERENCE).max(0.0);
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
                absorption: Vec3::splat(BEVY_EARTH_MIE_ABSORPTION * mie_load),
                scattering: Vec3::splat(BEVY_EARTH_MIE_SCATTERING * mie_load),
                falloff: Falloff::Exponential {
                    scale: scattering.mie_scale_height_m / atmosphere_height_m,
                },
                phase: PhaseFunction::Mie {
                    asymmetry: scattering.mie_asymmetry,
                },
            },
            ScatteringTerm {
                absorption: Vec3::new(0.650e-6, 1.881e-6, 0.085e-6),
                scattering: Vec3::ZERO,
                falloff: Falloff::Tent {
                    center: OZONE_CENTER_M / atmosphere_height_m,
                    width: OZONE_WIDTH_M / atmosphere_height_m,
                },
                phase: PhaseFunction::Isotropic,
            },
        ],
    )
    .with_density_multiplier(BEVY_ATMOSPHERE_DENSITY_SCALE)
    .with_label("thalos_bevy_earth_atmosphere")
}

/// Keep the stock-atmosphere components in sync with the toggle.
///
/// Enabled: project the active atmospheric body onto one camera-local proxy and
/// insert [`AtmosphereSettings`] (raymarched) on the ship camera. Disabled:
/// despawn the proxy — but **keep**
/// `AtmosphereSettings` on the camera. The extract system only clears the
/// render world's `ExtractedAtmosphere` for cameras that still carry
/// `AtmosphereSettings`; removing both in the same frame would strand a stale
/// extracted copy and the stock sky would keep rendering forever. With
/// settings present and zero `Atmosphere` entities the pass is fully idle.
pub(super) fn sync_stock_atmosphere(
    mut commands: Commands,
    settings: Res<GraphicsSettings>,
    screenshot: Option<Res<ScreenshotConfig>>,
    sim: Res<SimulationState>,
    cache: Res<SolarSystemState>,
    view_anchor: Res<ViewAnchor>,
    mut media: ResMut<Assets<ScatteringMedium>>,
    mut medium_cache: Local<bevy::platform::collections::HashMap<BodyId, Handle<ScatteringMedium>>>,
    legacy_body_atmospheres: Query<Entity, (With<RealSpaceBody>, With<Atmosphere>)>,
    mut proxies: Query<(
        Entity,
        &mut BevyAtmosphereProxy,
        &mut Atmosphere,
        &mut Transform,
    )>,
    camera: Query<(Entity, &GlobalTransform, Has<AtmosphereSettings>), With<ShipCamera>>,
) {
    // Cleanup from the old experiment, which attached Atmosphere directly to
    // every real-space body grid. The canonical path owns one proxy instead.
    for entity in &legacy_body_atmospheres {
        commands.entity(entity).remove::<Atmosphere>();
    }

    let active = enabled(&settings, screenshot.as_deref())
        .then_some(view_anchor.resolved)
        .flatten();
    let Some(active) = active else {
        for (entity, ..) in &mut proxies {
            commands.entity(entity).despawn();
        }
        return;
    };

    let Ok((camera_entity, camera_global, has_settings)) = camera.single() else {
        return;
    };
    if !has_settings {
        commands.entity(camera_entity).insert(AtmosphereSettings {
            rendering_method: AtmosphereMode::Raymarched,
            // Default is 32 km — tuned for ground-level scenes. Stretch the
            // aerial-view LUT across the whole in-atmosphere sightline so
            // distant terrain still picks up haze.
            aerial_view_lut_max_distance: 3.2e5,
            ..default()
        });
    }

    let Some(body) = sim.system.bodies.get(active.body) else {
        return;
    };
    let Some(atmo) = body.terrestrial_atmosphere.as_ref() else {
        return;
    };
    let Some(scattering) = atmo.scattering.as_ref() else {
        return;
    };
    if atmo.karman_line_m <= 0.0 {
        return;
    }
    let Some(state) = cache
        .states
        .as_deref()
        .and_then(|states| states.get(active.body))
    else {
        return;
    };
    // ViewAnchor stores the f64 body-fixed camera vector. Reproject it with the
    // current body orientation: body center relative to the floating-origin
    // camera is the negative of that vector in world axes.
    let camera_to_planet = -(state.orientation * active.cam_body).as_vec3();
    // Bevy extracts both the camera and atmosphere `GlobalTransform`s. BigSpace
    // keeps them small, but the camera is not necessarily *at* the render
    // origin (the screenshot rig leaves that origin near the parked craft).
    // Express the proxy in that same render-world frame; treating the camera
    // as zero subtracts its boom offset twice and moves the atmosphere off the
    // visible planet.
    let render_center = camera_global.translation() + camera_to_planet;
    let medium = medium_cache
        .entry(active.body)
        .or_insert_with(|| media.add(build_medium(scattering, atmo.karman_line_m)))
        .clone();
    let radius = body.radius_m as f32;
    let projected = Atmosphere {
        inner_radius: radius,
        outer_radius: radius + atmo.karman_line_m,
        ground_albedo: Vec3::from(body.color),
        medium,
    };

    let mut found = false;
    for (entity, mut proxy, mut atmosphere, mut transform) in &mut proxies {
        if found {
            commands.entity(entity).despawn();
            continue;
        }
        found = true;
        transform.translation = render_center;
        if proxy.body_id != active.body {
            proxy.body_id = active.body;
            *atmosphere = projected.clone();
        }
    }
    if !found {
        commands.spawn((
            BevyAtmosphereProxy {
                body_id: active.body,
            },
            projected,
            Transform::from_translation(render_center),
            Name::new(format!("{} Bevy Atmosphere", body.name)),
        ));
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
    screenshot: Option<Res<ScreenshotConfig>>,
    sim: Res<SimulationState>,
    view_anchor: Res<ViewAnchor>,
    bodies: Query<(&CelestialBody, &SolidPlanetMaterials)>,
    mut materials: ResMut<Assets<SolidPlanetMaterial>>,
) {
    let stock_body = active_body(&settings, screenshot.as_deref(), &view_anchor);
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
        let should_zero = stock_body == Some(body.body_id);
        if should_zero == currently_zero {
            continue;
        }
        if let Some(mut mat) = materials.get_mut(&mats.ship) {
            mat.params.atmosphere = if should_zero {
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
    screenshot: Option<Res<ScreenshotConfig>>,
    view_anchor: Res<ViewAnchor>,
    mut skies: Query<(&BodySky, &mut Visibility)>,
) {
    let Some(active_body) = active_body(&settings, screenshot.as_deref(), &view_anchor) else {
        return;
    };
    for (sky, mut vis) in &mut skies {
        if sky.body_id != active_body {
            continue;
        }
        if *vis != Visibility::Hidden {
            *vis = Visibility::Hidden;
        }
    }
}
