//! Editor surface overlays for compiled terrain metadata.
//!
//! Plate and biome overlays render through a single vertex-colored shell.
//! Each overlay quad samples both compiled layers (subject to which are
//! enabled) and emits an averaged color. A single mesh avoids the
//! sort-order flicker that two alpha-blended shells produced when both
//! were enabled.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;
use thalos_input::planet_editor::PlanetEditorInputIntent;
use thalos_terrain::cubemap::face_uv_to_dir;
use thalos_terrain::{
    BiomeMixTexel, Cubemap, CubemapFace, PlateKind, TectonicActivity, TectonicSystem,
};

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

#[derive(Component)]
struct SurfaceOverlayEntity;

/// Inflate the shell slightly past the impostor surface so the overlay
/// wins the depth test without visibly floating away from the planet.
const OVERLAY_LIFT: f32 = 1.015;
/// Overlay alpha. Tuned to keep the underlying terrain visible while the
/// overlay color still reads dominantly. When the planet is forced
/// fullbright (which the editor does whenever any overlay is on), this
/// blend rate lands somewhere around 60/40 overlay/terrain.
const OVERLAY_ALPHA: f32 = 0.6;
/// Overlay quads per cubemap face edge. Six faces × this², two triangles
/// per quad. Picked to match the topology of the underlying biome /
/// tectonic data (cubemap-aligned, no pole singularity) while keeping the
/// mesh inside ~200k triangles for cheap rebuilds. When the source
/// cubemap is finer than this, each overlay quad represents a
/// `(cm_res / OVERLAY_FACE_RES)²` block and the dominant biome wins.
const OVERLAY_FACE_RES: u32 = 128;

fn sync_surface_overlays(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: Res<SurfaceOverlayState>,
    render_radius: Res<SurfaceOverlayRenderRadius>,
    orientation: Res<SurfaceOverlayOrientation>,
    input: Res<PlanetEditorInputIntent>,
    preview_q: Query<(Entity, Ref<PreviewSurfaceOverlays>)>,
    mut overlay_q: Query<(
        Entity,
        &SurfaceOverlayEntity,
        &mut Transform,
        &mut Visibility,
    )>,
    mut last_state: Local<(bool, bool)>,
) {
    let Ok((preview_entity, preview)) = preview_q.single() else {
        for (entity, _, _, _) in &mut overlay_q {
            commands.entity(entity).despawn();
        }
        return;
    };

    let wants_plates = state.show_plates && preview.tectonics.is_some();
    let wants_biomes = state.show_biomes;
    let wants_overlay = wants_plates || wants_biomes;
    // Space-held suppresses overlays without despawning them — a rebuild
    // on each press/release would be ~50–100ms, but flipping Visibility
    // is free.
    let suppress = input.overlay_suppress;
    let target_visibility = if suppress {
        Visibility::Hidden
    } else {
        Visibility::Visible
    };
    // Bevy's resource change detection fires whenever a system declares
    // `ResMut<SurfaceOverlayState>` and mutably derefs it — which the egui
    // panel does every frame even when the user isn't toggling anything.
    // Track the last-observed values locally to detect real transitions.
    let current_state = (wants_plates, wants_biomes);
    let state_changed = *last_state != current_state;
    if state_changed {
        *last_state = current_state;
    }
    let rebuild = preview.is_changed() || render_radius.is_changed() || state_changed;
    let body_to_world = orientation.0.inverse();
    let orientation_changed = orientation.is_changed();
    let mut has_overlay = false;

    for (entity, _, mut transform, mut visibility) in &mut overlay_q {
        if !wants_overlay || rebuild {
            commands.entity(entity).despawn();
            continue;
        }

        if orientation_changed && transform.rotation != body_to_world {
            transform.rotation = body_to_world;
        }
        if *visibility != target_visibility {
            *visibility = target_visibility;
        }
        has_overlay = true;
    }

    if wants_overlay && !has_overlay {
        let tectonics = preview.tectonics.as_ref();
        let biome_weights = &preview.biome_weights;
        let mesh = build_cube_overlay_mesh(render_radius.0 * OVERLAY_LIFT, |face, x, y, res| {
            combined_overlay_color(
                wants_plates,
                wants_biomes,
                tectonics,
                biome_weights,
                face,
                x,
                y,
                res,
            )
        });

        spawn_overlay(
            &mut commands,
            &mut meshes,
            &mut materials,
            preview_entity,
            Transform::from_rotation(body_to_world),
            mesh,
            target_visibility,
        );
    }
}

fn spawn_overlay(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    parent: Entity,
    transform: Transform,
    mesh: Mesh,
    visibility: Visibility,
) {
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        perceptual_roughness: 1.0,
        metallic: 0.0,
        ..default()
    });

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(material),
        transform,
        visibility,
        ChildOf(parent),
        SurfaceOverlayEntity,
        NoFrustumCulling,
        NotShadowCaster,
        NotShadowReceiver,
        Name::new("Surface Overlay"),
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

/// Build a cubemap-topology overlay shell. Six face grids of
/// `OVERLAY_FACE_RES × OVERLAY_FACE_RES` quads are spherified via the
/// same `face_uv_to_dir` mapping the cubemap data uses, so each overlay
/// quad sits over the cubemap block it samples and the sphere has no
/// pole singularity.
fn build_cube_overlay_mesh(
    radius: f32,
    mut color_at: impl FnMut(CubemapFace, u32, u32, u32) -> [f32; 4],
) -> Mesh {
    let res = OVERLAY_FACE_RES;
    let quads_per_face = (res * res) as usize;
    let vertex_count = 6 * quads_per_face * 6;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(vertex_count);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(vertex_count);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(vertex_count);

    let inv_res = 1.0 / res as f32;
    for face in CubemapFace::ALL {
        for y in 0..res {
            for x in 0..res {
                let u0 = x as f32 * inv_res;
                let u1 = (x + 1) as f32 * inv_res;
                let v0 = y as f32 * inv_res;
                let v1 = (y + 1) as f32 * inv_res;

                let p00 = face_uv_to_dir(face, u0, v0);
                let p10 = face_uv_to_dir(face, u1, v0);
                let p11 = face_uv_to_dir(face, u1, v1);
                let p01 = face_uv_to_dir(face, u0, v1);

                let color = color_at(face, x, y, res);

                push_overlay_quad(
                    radius,
                    [p00, p10, p11, p01],
                    color,
                    &mut positions,
                    &mut normals,
                    &mut colors,
                );
            }
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

fn push_overlay_quad(
    radius: f32,
    corners: [Vec3; 4],
    color: [f32; 4],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    // CCW-from-outside winding: face_uv_to_dir's u/v axes flip handedness
    // on each face relative to the outward normal, so the naive
    // (p00, p10, p11) order is back-facing. Reversing each triangle gives
    // consistent front-facing geometry across all six faces.
    let [p00, p10, p11, p01] = corners;
    for dir in [p00, p11, p10, p00, p01, p11] {
        positions.push((dir * radius).to_array());
        normals.push(dir.to_array());
        colors.push(color);
    }
}

/// Mixes the active overlay layers into a single linear-RGB + alpha
/// triple per quad. When only one layer is on its color passes through
/// unchanged; when both are on the colors are averaged. Caller has
/// already filtered out the plate layer if the body has no tectonics.
fn combined_overlay_color(
    want_plate: bool,
    want_biome: bool,
    tectonics: Option<&TectonicSystem>,
    biome_weights: &Cubemap<BiomeMixTexel>,
    face: CubemapFace,
    x: u32,
    y: u32,
    res: u32,
) -> [f32; 4] {
    let mut acc = [0.0f32; 3];
    let mut count = 0u32;
    if want_plate && let Some(sys) = tectonics {
        let c = plate_overlay_rgb(sys, face, x, y, res);
        acc[0] += c[0];
        acc[1] += c[1];
        acc[2] += c[2];
        count += 1;
    }
    if want_biome {
        let c = biome_overlay_rgb(biome_weights, face, x, y, res);
        acc[0] += c[0];
        acc[1] += c[1];
        acc[2] += c[2];
        count += 1;
    }
    if count == 0 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let inv = 1.0 / count as f32;
    [acc[0] * inv, acc[1] * inv, acc[2] * inv, OVERLAY_ALPHA]
}

fn plate_overlay_rgb(
    sys: &TectonicSystem,
    face: CubemapFace,
    x: u32,
    y: u32,
    res: u32,
) -> [f32; 3] {
    let u = (x as f32 + 0.5) / res as f32;
    let v = (y as f32 + 0.5) / res as f32;
    let sample = sys.sample(face_uv_to_dir(face, u, v));
    let c = plate_color(sample.plate_kind, sample.plate_id.0).to_linear();
    [c.red, c.green, c.blue]
}

fn biome_overlay_rgb(
    weights: &Cubemap<BiomeMixTexel>,
    face: CubemapFace,
    x: u32,
    y: u32,
    res: u32,
) -> [f32; 3] {
    let biome_id = dominant_biome_in_block(weights, face, x, y, res);
    let c = biome_color(biome_id).to_linear();
    [c.red, c.green, c.blue]
}

/// Histogram the dominant biome id across the cubemap texels that fall
/// under the given overlay quad. When the cubemap is coarser than the
/// overlay (cm_res ≤ res) the block degenerates to a single texel.
fn dominant_biome_in_block(
    weights: &Cubemap<BiomeMixTexel>,
    face: CubemapFace,
    x: u32,
    y: u32,
    res: u32,
) -> u8 {
    let cm_res = weights.resolution();
    let x_min = (x as u64 * cm_res as u64 / res as u64) as u32;
    let x_max = (((x + 1) as u64 * cm_res as u64).div_ceil(res as u64) as u32).min(cm_res);
    let y_min = (y as u64 * cm_res as u64 / res as u64) as u32;
    let y_max = (((y + 1) as u64 * cm_res as u64).div_ceil(res as u64) as u32).min(cm_res);
    let x_max = x_max.max(x_min + 1);
    let y_max = y_max.max(y_min + 1);

    let mut counts = [0u32; 256];
    for cy in y_min..y_max {
        for cx in x_min..x_max {
            let texel = weights.get(face, cx, cy);
            if texel.is_empty() {
                continue;
            }
            counts[texel.biome_ids[0] as usize] += 1;
        }
    }

    counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, c)| *c)
        .map(|(i, _)| i as u8)
        .unwrap_or(0)
}

fn plate_color(kind: PlateKind, plate_id: u32) -> Color {
    let h = thalos_terrain::seeding::splitmix64(plate_id as u64 ^ 0xB1ADE0FF);
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

    let h = thalos_terrain::seeding::splitmix64(biome_id as u64 ^ 0xB10B_1A5E);
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
