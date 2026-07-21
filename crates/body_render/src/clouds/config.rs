use bevy::prelude::*;

use super::images::{RENDER_HEIGHT, RENDER_WIDTH};

#[derive(Resource, Clone, Copy, Reflect)]
#[reflect(Resource)]
/// The configuration that gets passed to the compute shader that renders the clouds.
///
/// The resource gets added automatically by `CloudsPlugin`. However, you can overwrite it
/// by inserting a new instance of it.
///
/// # Example
///
/// ```rust ignore
/// App::new()
///     .add_plugins((DefaultPlugins, CloudsPlugin))
///     .insert_resource(CloudsConfig {clouds_coverage: 0.6, ..default()})
///     .run();
/// ```
pub struct CloudsConfig {
    /// Hard cap on view-ray steps. The marcher still exits at the shell edge or
    /// once transmittance is exhausted; raising this mainly extends grazing
    /// horizon rays. CLOUD-0 exposes it as a capture-quality control.
    pub clouds_raymarch_steps_count: u32,
    /// Number of raymarching steps for shadowing.
    /// More steps reduces noise but requires more computational power
    pub clouds_shadow_raymarch_steps_count: u32,
    /// Radius of the planet the clouds encompass. Determines the curvature of the cloud layer near
    /// the horizon.
    pub planet_radius: f32,
    /// Height of the `clouds_bottom_height` of the cloud layer.
    pub clouds_bottom_height: f32,
    /// Height of the `clouds_top_height` of the cloud layer.
    pub clouds_top_height: f32,
    /// Global coverage scale on the planet-fixed weather map: the local
    /// overcast fraction is `coverage_map(dir) * clouds_coverage`. With the
    /// default all-1 map this behaves like the original scalar knob
    /// (0.0 = no clouds, 1.0 = full overcast).
    pub clouds_coverage: f32,
    /// Determines how much the base cloud structure is eroded by higher-frequency,
    /// lower-amplitude detail noise.
    pub clouds_detail_strength: f32,
    /// Softness of the clouds
    pub clouds_base_edge_softness: f32,
    /// Softness of the `clouds_bottom_height` of the clouds
    pub clouds_bottom_softness: f32,
    /// `clouds_density` of the clouds between 0.0 and 1.0
    pub clouds_density: f32,
    /// Step size of raymarching steps for calculating the shadow inside clouds
    pub clouds_shadow_raymarch_step_size: f32,
    /// Step size exponential multiplication factor of raymarching steps for calculating the
    /// shadow inside clouds
    pub clouds_shadow_raymarch_step_multiply: f32,
    /// Scattering factor for forward scattering lobe. See Frostbite paper in README for details.
    pub forward_scattering_g: f32,
    /// Scattering factor for backward scattering lobe. See Frostbite paper in README for details.
    pub backward_scattering_g: f32,
    /// Factor between 0.0 and 1.0 for mixing forward and backward scattering.
    pub scattering_lerp: f32,
    /// The color of ambient lighting at the `clouds_top_height` of the clouds.
    pub clouds_ambient_color_top: Vec4,
    /// The color of ambient lighting at the `clouds_bottom_height` of the clouds.
    pub clouds_ambient_color_bottom: Vec4,
    /// Minimal transmittance in a ray, if transmittance is too low the ray is discarded.
    pub clouds_min_transmittance: f32,
    /// Characteristic world-space period of the base cloud shape, metres.
    pub clouds_base_shape_scale_m: f32,
    /// Characteristic world-space period of edge erosion detail, metres.
    pub clouds_detail_scale_m: f32,
    /// Direction towards the sun.
    pub sun_dir: Vec4,
    /// Color of the sun (HDR, RGBA).
    pub sun_color: Vec4,
    /// Linear cloud single-scatter albedo. Kept separate from sun radiance so
    /// the canonical atmosphere sky-view LUT can illuminate the volume without
    /// baking a second artist-authored ambient colour into the light source.
    pub cloud_albedo: Vec4,
    /// Strength of reprojection. 0.0 means we don't mix the current frame with the last frame.
    /// 0.95 means we take 5% of the current frame and 95% of last frame and combine those two to
    /// reduce noise.
    /// Automatically updates each frame.
    pub reprojection_strength: f32,
    /// Determines whether the egui UI is visible or not. Requires the `debug` feature.
    pub ui_visible: bool,
    /// Resolution of the image we're writing to.
    pub render_resolution: Vec2,
    /// Fraction of the ship camera's physical viewport used by the cloud
    /// targets. The resulting extent is aligned to the compute workgroup.
    pub resolution_scale: f32,
    /// Enables the rotating 3x3 sparse update. Temporal-disabled/reference
    /// captures turn this off and raymarch every target pixel.
    pub sparse_march: bool,
    /// Invalidates all temporal samples when target size, active body,
    /// weather, or simulation continuity changes.
    pub history_epoch: u32,
    /// Velocity of the wind, metres/second in the body-fixed frame: `x` is
    /// zonal drift (eastward surface speed at the equator — applied as a slow
    /// rotation of the whole cloud field about the body's spin axis, so the
    /// drift stays glued to the sphere), `y`/`z` drift the detail-erosion
    /// noise for slow "boiling". Later this becomes a weather-system output.
    pub wind_velocity: Vec3,
}

impl Default for CloudsConfig {
    fn default() -> Self {
        let sun_dir = Vec3::new(-0.7, 0.5, 0.75).normalize();
        Self {
            // 80 samples let an inside-layer tangent ray integrate 40 km at
            // the 500 m coarse step before the near-volume distance fade.
            clouds_raymarch_steps_count: 80,
            clouds_shadow_raymarch_steps_count: 6,
            planet_radius: 6_371_000.0,
            clouds_bottom_height: 1250.0,
            clouds_top_height: 2400.0,
            clouds_coverage: 0.5,
            clouds_detail_strength: 0.27,
            clouds_base_edge_softness: 0.1,
            clouds_bottom_softness: 0.25,
            clouds_density: 0.03,
            clouds_shadow_raymarch_step_size: 10.0,
            clouds_shadow_raymarch_step_multiply: 1.3,
            forward_scattering_g: 0.8,
            backward_scattering_g: -0.2,
            scattering_lerp: 0.5,
            clouds_ambient_color_top: Vec4::new(149.0, 167.0, 200.0, 0.0) * (1.5 / 225.0),
            clouds_ambient_color_bottom: Vec4::new(39.0, 67.0, 87.0, 0.0) * (1.5 / 225.0),
            clouds_min_transmittance: 0.1,
            clouds_base_shape_scale_m: 8_000.0,
            clouds_detail_scale_m: 450.0,
            sun_dir: Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, 0.0),
            sun_color: Vec4::new(1.0, 0.9, 0.85, 1.0) * 1.4,
            cloud_albedo: Vec4::ONE,
            reprojection_strength: 0.95,
            ui_visible: true,
            render_resolution: Vec2::new(RENDER_WIDTH as f32, RENDER_HEIGHT as f32),
            resolution_scale: 2.0 / 3.0,
            sparse_march: true,
            history_epoch: 1,
            wind_velocity: Vec3::new(-1.1, 0.0, 2.3),
        }
    }
}

impl CloudsConfig {
    /// Project a physical viewport into a stable, workgroup-aligned cloud
    /// target. Keeping this policy here makes interactive resize and headless
    /// quality captures use the same path.
    pub fn set_viewport_resolution(&mut self, viewport: UVec2) {
        if viewport.x == 0 || viewport.y == 0 {
            return;
        }
        let scale = self.resolution_scale.clamp(0.25, 1.0);
        let align = |value: u32| ((value.max(8) + 7) / 8) * 8;
        self.render_resolution = Vec2::new(
            align((viewport.x as f32 * scale).round() as u32) as f32,
            align((viewport.y as f32 * scale).round() as u32) as f32,
        );
    }
}
