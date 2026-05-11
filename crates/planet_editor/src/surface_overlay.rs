//! Editor surface overlays for compiled terrain metadata.
//!
//! Plate and biome overlays share the same rendering path: build a
//! vertex-colored shell, sample one compiled metadata layer per triangle,
//! and place the shell in the preview body's display orientation.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;
use thalos_terrain_gen::cubemap::dir_to_face_uv;
use thalos_terrain_gen::{BiomeMixTexel, Cubemap, PlateKind, TectonicActivity, TectonicSystem};

/// Edit-time component carrying compiled metadata for editor overlays.
/// Inserted by `finalize_terrain_bake` when a terrain bake succeeds.
#[derive(Component, Clone)]
pub struct PreviewSurfaceOverlays {
    pub tectonics: Option<TectonicSystem>,
    pub biome_weights: Cubemap<BiomeMixTexel>,
}

/// Toggleable overlay layers. Disabled by default so the normal terrain
/// preview remains unobstructed until explicitly requested.
#[derive(Resource, Default)]
pub struct SurfaceOverlayState {
    pub show_plates: bool,
    pub show_biomes: bool,
}

/// Plugin: registers overlay state and the shared mesh-sync system.
pub struct SurfaceOverlayPlugin;

impl Plugin for SurfaceOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SurfaceOverlayState>()
            .add_systems(Update, sync_surface_overlays);
    }
}

#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum SurfaceOverlayLayer {
    Plates,
    Biomes,
}

#[derive(Component)]
struct SurfaceOverlayEntity {
    layer: SurfaceOverlayLayer,
}

/// Inflate shells slightly past the impostor surface so overlays win the
/// depth test without visibly floating away from the planet. Biomes sit a
/// hair above plates so both can be toggled for quick comparison.
const PLATE_OVERLAY_LIFT: f32 = 1.012;
const BIOME_OVERLAY_LIFT: f32 = 1.018;
const OVERLAY_ALPHA: f32 = 1.0;
const OVERLAY_LAT_SEGMENTS: usize = 48;
const OVERLAY_LON_SEGMENTS: usize = 96;

fn sync_surface_overlays(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: Res<SurfaceOverlayState>,
    render_radius: Res<SurfaceOverlayRenderRadius>,
    orientation: Res<SurfaceOverlayOrientation>,
    preview_q: Query<(Entity, Ref<PreviewSurfaceOverlays>)>,
    mut overlay_q: Query<(Entity, &SurfaceOverlayEntity, &mut Transform)>,
) {
    let Ok((preview_entity, preview)) = preview_q.single() else {
        for (entity, _, _) in &mut overlay_q {
            commands.entity(entity).despawn();
        }
        return;
    };

    let wants_plates = state.show_plates && preview.tectonics.is_some();
    let wants_biomes = state.show_biomes;
    let rebuild = preview.is_changed() || render_radius.is_changed();
    let body_to_world = orientation.0.inverse();
    let orientation_changed = orientation.is_changed();
    let mut has_plates = false;
    let mut has_biomes = false;

    for (entity, marker, mut transform) in &mut overlay_q {
        let wanted = match marker.layer {
            SurfaceOverlayLayer::Plates => wants_plates,
            SurfaceOverlayLayer::Biomes => wants_biomes,
        };

        if !wanted || rebuild {
            commands.entity(entity).despawn();
            continue;
        }

        if orientation_changed && transform.rotation != body_to_world {
            transform.rotation = body_to_world;
        }
        match marker.layer {
            SurfaceOverlayLayer::Plates => has_plates = true,
            SurfaceOverlayLayer::Biomes => has_biomes = true,
        }
    }

    if wants_plates && !has_plates {
        if let Some(sys) = preview.tectonics.as_ref() {
            spawn_overlay(
                &mut commands,
                &mut meshes,
                &mut materials,
                preview_entity,
                SurfaceOverlayLayer::Plates,
                Transform::from_rotation(body_to_world),
                build_spherical_overlay_mesh(render_radius.0 * PLATE_OVERLAY_LIFT, |dir| {
                    plate_overlay_color(sys, dir)
                }),
            );
        }
    }

    if wants_biomes && !has_biomes {
        spawn_overlay(
            &mut commands,
            &mut meshes,
            &mut materials,
            preview_entity,
            SurfaceOverlayLayer::Biomes,
            Transform::from_rotation(body_to_world),
            build_spherical_overlay_mesh(render_radius.0 * BIOME_OVERLAY_LIFT, |dir| {
                biome_overlay_color(&preview.biome_weights, dir)
            }),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_overlay(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    parent: Entity,
    layer: SurfaceOverlayLayer,
    transform: Transform,
    mesh: Mesh,
) {
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        alpha_mode: AlphaMode::Opaque,
        unlit: true,
        perceptual_roughness: 1.0,
        metallic: 0.0,
        ..default()
    });

    let name = match layer {
        SurfaceOverlayLayer::Plates => "Tectonic Plate Overlay",
        SurfaceOverlayLayer::Biomes => "Biome Overlay",
    };

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(material),
        transform,
        ChildOf(parent),
        SurfaceOverlayEntity { layer },
        NoFrustumCulling,
        NotShadowCaster,
        NotShadowReceiver,
        Name::new(name),
    ));
}

/// Resource carrying the planet's display radius. Owned by the editor
/// (set once at startup); broken out as a resource so overlays do not have
/// to import the editor's private constants.
#[derive(Resource)]
pub struct SurfaceOverlayRenderRadius(pub f32);

/// Resource carrying the same world-to-body orientation quaternion sent to
/// the planet shader. Overlay meshes are authored in body-local space, so
/// their transform uses this value's inverse.
#[derive(Resource, Default)]
pub struct SurfaceOverlayOrientation(pub Quat);

fn build_spherical_overlay_mesh(radius: f32, mut color_at: impl FnMut(Vec3) -> [f32; 4]) -> Mesh {
    let vertex_count = OVERLAY_LAT_SEGMENTS * OVERLAY_LON_SEGMENTS * 6;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(vertex_count);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(vertex_count);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(vertex_count);

    for lat_i in 0..OVERLAY_LAT_SEGMENTS {
        let lat0 = -std::f32::consts::FRAC_PI_2
            + std::f32::consts::PI * lat_i as f32 / OVERLAY_LAT_SEGMENTS as f32;
        let lat1 = -std::f32::consts::FRAC_PI_2
            + std::f32::consts::PI * (lat_i + 1) as f32 / OVERLAY_LAT_SEGMENTS as f32;

        for lon_i in 0..OVERLAY_LON_SEGMENTS {
            let lon0 = std::f32::consts::TAU * lon_i as f32 / OVERLAY_LON_SEGMENTS as f32;
            let lon1 = std::f32::consts::TAU * (lon_i + 1) as f32 / OVERLAY_LON_SEGMENTS as f32;

            let p00 = direction_from_lat_lon(lat0, lon0);
            let p10 = direction_from_lat_lon(lat1, lon0);
            let p11 = direction_from_lat_lon(lat1, lon1);
            let p01 = direction_from_lat_lon(lat0, lon1);

            push_overlay_triangle(
                radius,
                [p00, p10, p11],
                &mut color_at,
                &mut positions,
                &mut normals,
                &mut colors,
            );
            push_overlay_triangle(
                radius,
                [p00, p11, p01],
                &mut color_at,
                &mut positions,
                &mut normals,
                &mut colors,
            );
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh
}

fn direction_from_lat_lon(lat: f32, lon: f32) -> Vec3 {
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    Vec3::new(cos_lat * cos_lon, sin_lat, cos_lat * sin_lon)
}

fn push_overlay_triangle(
    radius: f32,
    dirs: [Vec3; 3],
    color_at: &mut impl FnMut(Vec3) -> [f32; 4],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    let sample_dir = (dirs[0] + dirs[1] + dirs[2]).normalize_or_zero();
    let color = color_at(sample_dir);

    for dir in dirs {
        positions.push((dir * radius).to_array());
        normals.push(dir.to_array());
        colors.push(color);
    }
}

fn plate_overlay_color(sys: &TectonicSystem, dir: Vec3) -> [f32; 4] {
    let sample = sys.sample(dir);
    let c = plate_color(sample.plate_kind, sample.plate_id.0).to_linear();
    [c.red, c.green, c.blue, OVERLAY_ALPHA]
}

fn biome_overlay_color(weights: &Cubemap<BiomeMixTexel>, dir: Vec3) -> [f32; 4] {
    let biome_id = sample_dominant_biome(weights, dir);
    let c = biome_color(biome_id).to_linear();
    [c.red, c.green, c.blue, OVERLAY_ALPHA]
}

fn sample_dominant_biome(weights: &Cubemap<BiomeMixTexel>, dir: Vec3) -> u8 {
    let (face, u, v) = dir_to_face_uv(dir);
    let res = weights.resolution();
    let x = ((u * res as f32) as u32).min(res - 1);
    let y = ((v * res as f32) as u32).min(res - 1);
    weights.get(face, x, y).biome_ids[0]
}

fn plate_color(kind: PlateKind, plate_id: u32) -> Color {
    let h = thalos_terrain_gen::seeding::splitmix64(plate_id as u64 ^ 0xB1ADE0FF);
    let hue_unit = ((h & 0xFFFF) as f32) / 65535.0;
    let (hue_deg, sat, val) = match kind {
        PlateKind::Continental => {
            let hue = if hue_unit < 0.5 {
                hue_unit * 120.0
            } else {
                300.0 + (hue_unit - 0.5) * 120.0
            };
            (hue, 0.65, 0.85)
        }
        PlateKind::Oceanic => (180.0 + hue_unit * 60.0, 0.70, 0.55),
    };
    Color::hsv(hue_deg, sat, val)
}

fn biome_color(biome_id: u8) -> Color {
    const COLORS: [[u8; 3]; 16] = [
        [214, 69, 80],
        [45, 137, 239],
        [62, 183, 98],
        [244, 188, 66],
        [158, 96, 214],
        [33, 184, 169],
        [239, 111, 47],
        [218, 218, 230],
        [132, 184, 64],
        [215, 91, 167],
        [86, 114, 214],
        [180, 136, 72],
        [89, 194, 219],
        [185, 82, 72],
        [124, 206, 144],
        [166, 166, 166],
    ];

    if let Some(rgb) = COLORS.get(biome_id as usize) {
        return color_from_srgb8(*rgb);
    }

    let h = thalos_terrain_gen::seeding::splitmix64(biome_id as u64 ^ 0xB10B_1A5E);
    let hue = ((h & 0xFFFF) as f32) / 65535.0 * 360.0;
    Color::hsv(hue, 0.72, 0.88)
}

fn color_from_srgb8(rgb: [u8; 3]) -> Color {
    Color::srgb(
        rgb[0] as f32 / 255.0,
        rgb[1] as f32 / 255.0,
        rgb[2] as f32 / 255.0,
    )
}

/// Activity-mode label for the editor sidebar. Doesn't strictly belong
/// here, but it's the only place that needs to map an enum to a string.
pub fn activity_label(activity: TectonicActivity) -> &'static str {
    match activity {
        TectonicActivity::Active => "Active",
        TectonicActivity::StagnantLid => "Stagnant lid",
        TectonicActivity::Frozen { .. } => "Frozen",
    }
}
