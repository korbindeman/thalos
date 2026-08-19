use bevy::{
    asset::{AssetEventSystems, AssetId},
    camera::visibility::NoFrustumCulling,
    light::NotShadowCaster,
    mesh::MeshVertexBufferLayoutRef,
    pbr::{Material, MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    reflect::TypePath,
    render::render_resource::{
        AsBindGroup, CompareFunction, RenderPipelineDescriptor, SpecializedMeshPipelineError,
    },
    shader::ShaderRef,
    transform::TransformSystems,
};
use thalos_atmosphere::BEVY_EARTH_RADIUS_M;
use thalos_body_shading::{LIGHT_AT_1AU, PlanetLightingPlugin};
use thalos_clouds::{
    CameraMatrices, CloudDistanceTexture, CloudRenderTexture, CloudShadowFrame, CloudShadowMap,
    CloudSurfaceDensityMap, CloudWeatherMap, CloudsConfig, CloudsImage,
    CloudsPlugin as CloudsMechanismPlugin, WEATHER_FACE_SIZE, cloud_weather_image,
};
use thalos_render_foundation::{SceneDepthImage, SceneDepthPlugin};
use thalos_runtime::preferences::{GraphicsPreferences, QualityOverrides, effective_graphics};
use thalos_weather::cloud_cube::{COVERAGE_SCALE, CloudWeatherField};
use thalos_world::CloudClimate;

use crate::{
    camera::{TerrainCamera, TerrainCameraSet},
    world::SolarClock,
};

pub(crate) const CLOUD_SHADER: &str = "korsou://shaders/cloud_composite.wgsl";

const DENSITY: f32 = 0.0026;
const BOTTOM_SOFTNESS: f32 = 0.16;
const DETAIL_STRENGTH: f32 = 0.16;
const EDGE_SOFTNESS: f32 = 0.055;
const SUN_FLUX_SCALE: f32 = 0.36;
const AMBIENT_TOP_SCALE: f32 = 0.085;
const AMBIENT_BOTTOM_SCALE: f32 = 0.042;

pub struct CloudsPlugin;

impl Plugin for CloudsPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<PlanetLightingPlugin>() {
            app.add_plugins(PlanetLightingPlugin);
        }
        if !app.is_plugin_added::<SceneDepthPlugin>() {
            app.add_plugins(SceneDepthPlugin);
        }
        if !app.is_plugin_added::<CloudsMechanismPlugin>() {
            app.add_plugins(CloudsMechanismPlugin);
        }
        app.add_plugins(MaterialPlugin::<CloudLayerMaterial>::default())
            .add_systems(Startup, init_cloud_appearance)
            .add_systems(
                PostStartup,
                (setup_cloud_weather, setup_cloud_composite).chain(),
            )
            .add_systems(
                PostUpdate,
                (
                    drive_clouds
                        .after(TransformSystems::Propagate)
                        .after(TerrainCameraSet::Projection),
                    refresh_cloud_composite_bindings.before(AssetEventSystems),
                ),
            );
    }
}

#[derive(Asset, AsBindGroup, TypePath, Clone)]
struct CloudLayerMaterial {
    #[texture(0, sample_type = "depth")]
    scene_depth: Handle<Image>,
    #[texture(1, sample_type = "float", filterable = false)]
    cloud_layer: Handle<Image>,
    #[texture(2, sample_type = "float", filterable = false)]
    cloud_distance: Handle<Image>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CloudBindingSignature {
    scene_depth: (AssetId<Image>, UVec2),
    cloud_layer: (AssetId<Image>, UVec2),
    cloud_distance: (AssetId<Image>, UVec2),
}

#[derive(Resource)]
struct CloudCompositeBindings {
    material: Handle<CloudLayerMaterial>,
    signature: Option<CloudBindingSignature>,
}

impl Material for CloudLayerMaterial {
    fn vertex_shader() -> ShaderRef {
        CLOUD_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        CLOUD_SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Premultiplied
    }

    fn depth_bias(&self) -> f32 {
        -2.0e9
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = Some(false);
            depth.depth_compare = Some(CompareFunction::Always);
        }
        Ok(())
    }
}

fn caribbean_climate() -> CloudClimate {
    CloudClimate {
        seed: 0xC0_A5_70,
        coverage: 0.38,
        band_strength: 0.18,
        variation: 0.40,
        type_mix: [0.12, 0.72, 0.16],
        albedo: [0.96, 0.97, 1.0],
        scroll_rate: 4.7e-6,
        differential_rotation: 0.35,
        wind_m_s: [8.0, 1.0],
        base_altitude_m: 800.0,
        thickness_m: 4_500.0,
        density: 1.0,
        precipitation_threshold: 0.72,
        storm_threshold: 0.86,
        weather_scale_km: 900.0,
        base_shape_scale_m: 8_000.0,
        detail_scale_m: 450.0,
    }
}

fn init_cloud_appearance(mut config: ResMut<CloudsConfig>) {
    config.clouds_coverage = COVERAGE_SCALE;
    config.clouds_density = DENSITY;
    config.clouds_detail_strength = DETAIL_STRENGTH;
    config.clouds_base_edge_softness = EDGE_SOFTNESS;
    config.clouds_bottom_softness = BOTTOM_SOFTNESS;
    config.clouds_shadow_raymarch_steps_count = 3;
    config.clouds_shadow_raymarch_step_size = 700.0;
    config.clouds_shadow_raymarch_step_multiply = 2.0;
    config.planet_radius = BEVY_EARTH_RADIUS_M;
}

fn setup_cloud_weather(
    mut images: ResMut<Assets<Image>>,
    mut clouds_image: ResMut<CloudsImage>,
    mut weather: ResMut<CloudWeatherMap>,
    mut surface_density: ResMut<CloudSurfaceDensityMap>,
) {
    let climate = caribbean_climate();
    info!("deriving Caribbean cloud weather field");
    let field = CloudWeatherField::from_climate(&climate);
    assert_eq!(
        field.face_size, WEATHER_FACE_SIZE,
        "weather field face size must match the cloud renderer"
    );
    let weather_handle = images.add(cloud_weather_image(
        field.rgba8_mip_chain(),
        field.face_size,
        CloudWeatherField::MIP_LEVELS,
    ));
    let strata_handle = images.add(cloud_weather_image(
        field.surface_density_rgba8_mip_chain(),
        field.face_size,
        CloudWeatherField::MIP_LEVELS,
    ));
    clouds_image.weather_image = weather_handle.clone();
    clouds_image.surface_density_image = strata_handle.clone();
    weather.handle = weather_handle;
    surface_density.handle = strata_handle;
}

fn setup_cloud_composite(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CloudLayerMaterial>>,
    scene_depth: Res<SceneDepthImage>,
    cloud_layer: Res<CloudRenderTexture>,
    cloud_distance: Res<CloudDistanceTexture>,
) {
    let material = materials.add(CloudLayerMaterial {
        scene_depth: scene_depth.handle.clone(),
        cloud_layer: cloud_layer.handle.clone(),
        cloud_distance: cloud_distance.handle.clone(),
    });
    commands.insert_resource(CloudCompositeBindings {
        material: material.clone(),
        signature: None,
    });
    commands.spawn((
        Mesh3d(meshes.add(Rectangle::new(2.0, 2.0))),
        MeshMaterial3d(material),
        NoFrustumCulling,
        NotShadowCaster,
        Name::new("Caribbean clouds"),
    ));
}

/// Re-prepare the material bind group after any sampled image is resized or
/// replaced. Bevy recreates the GPU image view, but an unchanged material
/// keeps its old view; the cloud compute pass then writes one texture while
/// this composite samples the zero-filled predecessor.
fn refresh_cloud_composite_bindings(
    scene_depth: Res<SceneDepthImage>,
    cloud_layer: Res<CloudRenderTexture>,
    cloud_distance: Res<CloudDistanceTexture>,
    images: Res<Assets<Image>>,
    mut bindings: ResMut<CloudCompositeBindings>,
    mut materials: ResMut<Assets<CloudLayerMaterial>>,
) {
    let Some(scene_depth_image) = images.get(&scene_depth.handle) else {
        return;
    };
    let Some(cloud_layer_image) = images.get(&cloud_layer.handle) else {
        return;
    };
    let Some(cloud_distance_image) = images.get(&cloud_distance.handle) else {
        return;
    };
    let signature = CloudBindingSignature {
        scene_depth: (scene_depth.handle.id(), scene_depth_image.size()),
        cloud_layer: (cloud_layer.handle.id(), cloud_layer_image.size()),
        cloud_distance: (cloud_distance.handle.id(), cloud_distance_image.size()),
    };
    if bindings.signature == Some(signature) {
        return;
    }

    let Some(mut material) = materials.get_mut(&bindings.material) else {
        return;
    };
    material.scene_depth = scene_depth.handle.clone();
    material.cloud_layer = cloud_layer.handle.clone();
    material.cloud_distance = cloud_distance.handle.clone();
    bindings.signature = Some(signature);
}

#[allow(clippy::too_many_arguments)]
fn drive_clouds(
    camera: Query<(&GlobalTransform, &Camera), With<TerrainCamera>>,
    clock: Res<SolarClock>,
    graphics: Res<GraphicsPreferences>,
    overrides: Res<QualityOverrides>,
    mut cam_mat: ResMut<CameraMatrices>,
    mut config: ResMut<CloudsConfig>,
    mut cloud_shadow: ResMut<CloudShadowMap>,
    time: Res<Time>,
    mut wind_angle: Local<f32>,
) {
    config.shadow_frame = CloudShadowFrame::default();
    cloud_shadow.frame = CloudShadowFrame::default();

    let climate = caribbean_climate();
    let enabled = effective_graphics(&graphics, &overrides).clouds;
    if !enabled {
        cam_mat.translation = Vec3::new(0.0, config.planet_radius * 1.0e3 + 1.0e9, 0.0);
        return;
    }

    let Ok((cam_gt, camera)) = camera.single() else {
        return;
    };
    if let Some(viewport) = camera.physical_viewport_size() {
        config.set_viewport_resolution(viewport);
    }

    let earth_center = Vec3::new(0.0, -BEVY_EARTH_RADIUS_M, 0.0);
    let to_cam = cam_gt.translation() - earth_center;
    if to_cam.length() < 1.0 {
        return;
    }

    *wind_angle = (*wind_angle + climate.wind_m_s[0] * time.delta_secs() / BEVY_EARTH_RADIUS_M)
        .rem_euclid(std::f32::consts::TAU);
    let q_bw = Quat::from_rotation_y(*wind_angle);
    let cam_body = q_bw * to_cam;
    cam_mat.translation = cam_body;
    let mut view_mat = Mat4::from_quat(q_bw) * cam_gt.to_matrix();
    view_mat.w_axis = (cam_body * 1.0e-4).extend(1.0);
    cam_mat.inverse_camera_view = view_mat;
    cam_mat.inverse_camera_projection = camera.computed.clip_from_view.inverse();

    let sun_world = clock.sun_direction();
    let sun_body = q_bw * sun_world;
    let scene_flux = LIGHT_AT_1AU;

    config.planet_radius = BEVY_EARTH_RADIUS_M;
    config.clouds_bottom_height = climate.base_altitude_m.max(0.0);
    config.clouds_top_height =
        (climate.base_altitude_m + climate.thickness_m).max(config.clouds_bottom_height + 1.0);
    config.clouds_density = DENSITY * climate.density.max(0.0);
    config.clouds_base_shape_scale_m = climate.base_shape_scale_m.max(500.0);
    config.clouds_detail_scale_m = climate.detail_scale_m.max(50.0);
    config.wind_velocity = Vec3::new(climate.wind_m_s[0], climate.wind_m_s[1], 0.0);
    config.sun_dir = Vec4::new(sun_body.x, sun_body.y, sun_body.z, 0.0);
    config.cell_evolution_s =
        (f64::from(clock.day_of_year) * 86_400.0 + clock.local_seconds) as f32;

    let shadow_frame = CloudShadowFrame::resolve(cam_body, sun_body, BEVY_EARTH_RADIUS_M);
    config.shadow_frame = shadow_frame;
    cloud_shadow.frame = shadow_frame;
    cloud_shadow.world_to_body = q_bw;
    cloud_shadow.body_center_ws = earth_center;
    cloud_shadow.sun_body = sun_body;
    cloud_shadow.strength = 1.0;

    let cloud_albedo = Vec3::from_array(climate.albedo).max(Vec3::ZERO);
    config.cloud_albedo = cloud_albedo.extend(1.0);
    let sun_mu = cam_body.normalize_or_zero().dot(sun_body);
    let day_t = ((sun_mu + 0.04) / 0.28).clamp(0.0, 1.0);
    let day_blend = day_t * day_t * (3.0 - 2.0 * day_t);
    let sun_chromaticity = Vec3::new(1.0, 0.84, 0.72).lerp(Vec3::new(1.0, 0.97, 0.93), day_blend);
    let horizon_transmittance = 0.85 + 0.15 * day_blend;
    let sun_rgb =
        sun_chromaticity * cloud_albedo * scene_flux * SUN_FLUX_SCALE * horizon_transmittance;
    config.sun_color = Vec4::new(sun_rgb.x, sun_rgb.y, sun_rgb.z, 1.0);
    let horizon_ambient = 0.28 + 0.72 * day_blend;
    config.clouds_ambient_color_top =
        (Vec3::new(0.42, 0.58, 0.88) * scene_flux * AMBIENT_TOP_SCALE * horizon_ambient)
            .extend(0.0);
    config.clouds_ambient_color_bottom =
        (Vec3::new(0.30, 0.36, 0.48) * scene_flux * AMBIENT_BOTTOM_SCALE * horizon_ambient)
            .extend(0.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{
        asset::{AssetApp, AssetEvent, AssetPlugin, RenderAssetUsages},
        render::render_resource::{Extent3d, TextureDimension, TextureFormat},
    };

    fn test_image(width: u32, height: u32) -> Image {
        Image::new_uninit(
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::all(),
        )
    }

    #[test]
    fn virtual_earth_puts_the_origin_on_the_surface() {
        let earth_center = Vec3::new(0.0, -BEVY_EARTH_RADIUS_M, 0.0);
        let cam_body = Vec3::ZERO - earth_center;
        assert!((cam_body.length() - BEVY_EARTH_RADIUS_M).abs() < 1.0);
        assert!((cam_body.normalize().dot(Vec3::Y) - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn caribbean_climate_is_trade_wind_cumulus() {
        let climate = caribbean_climate();
        assert!(climate.coverage < 0.5);
        assert!(climate.type_mix[1] > climate.type_mix[0]);
        assert!(climate.base_altitude_m < 2_000.0);
    }

    #[test]
    fn resized_cloud_target_invalidates_the_composite_material() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Image>()
            .init_asset::<CloudLayerMaterial>()
            .add_systems(
                PostUpdate,
                refresh_cloud_composite_bindings.before(AssetEventSystems),
            );

        let (scene_depth, cloud_layer, cloud_distance) = {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();
            (
                images.add(test_image(1600, 900)),
                images.add(test_image(1280, 720)),
                images.add(test_image(1280, 720)),
            )
        };
        let material = app
            .world_mut()
            .resource_mut::<Assets<CloudLayerMaterial>>()
            .add(CloudLayerMaterial {
                scene_depth: scene_depth.clone(),
                cloud_layer: cloud_layer.clone(),
                cloud_distance: cloud_distance.clone(),
            });
        app.insert_resource(SceneDepthImage {
            handle: scene_depth,
        })
        .insert_resource(CloudRenderTexture {
            handle: cloud_layer.clone(),
        })
        .insert_resource(CloudDistanceTexture {
            handle: cloud_distance,
        })
        .insert_resource(CloudCompositeBindings {
            material: material.clone(),
            signature: None,
        });

        app.update();
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<CloudLayerMaterial>>>()
            .drain()
            .for_each(drop);

        app.world_mut()
            .resource_mut::<Assets<Image>>()
            .get_mut(&cloud_layer)
            .expect("test cloud target must exist")
            .resize(Extent3d {
                width: 1072,
                height: 600,
                depth_or_array_layers: 1,
            });
        app.update();

        let events = app
            .world_mut()
            .resource_mut::<Messages<AssetEvent<CloudLayerMaterial>>>()
            .drain()
            .collect::<Vec<_>>();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AssetEvent::Modified { id } if *id == material.id()))
        );
        assert_eq!(
            app.world()
                .resource::<CloudCompositeBindings>()
                .signature
                .expect("binding signature must be recorded")
                .cloud_layer
                .1,
            UVec2::new(1072, 600),
        );
    }
}
