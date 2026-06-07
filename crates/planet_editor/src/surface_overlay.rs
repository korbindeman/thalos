//! On-planet overlay shell for the editor's selected field.
//!
//! The overlay projects whatever field the field viewer has selected (albedo,
//! height, material id, tectonic plates, biomes…) onto a single vertex-colored
//! shell wrapping the impostor. A single mesh avoids the sort-order flicker
//! that alpha-blended shells produced, and routing every field through the same
//! [`crate::ui::field_overlay_rgb_linear`] sampler the equirect viewer uses
//! keeps the on-planet overlay identical to the flattened preview.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;
use thalos_input::planet_editor::PlanetEditorInputIntent;
use thalos_terrain::cubemap::face_uv_to_dir;
use thalos_terrain::{CubemapFace, TectonicActivity};

use crate::state::{ActivePreviewSurface, EquirectFieldKind, EquirectViewerState, PreviewPlanet};
use crate::ui::{equirect_lod, field_overlay_rgb_linear};

/// Plugin: registers the overlay-sync system. Overlay enable/field selection
/// live on [`EquirectViewerState`]; the data comes from [`ActivePreviewSurface`].
pub struct SurfaceOverlayPlugin;

impl Plugin for SurfaceOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, sync_surface_overlays);
    }
}

#[derive(Component)]
struct SurfaceOverlayEntity;

/// Inflate the shell slightly past the impostor surface so the overlay
/// wins the depth test without visibly floating away from the planet.
const OVERLAY_LIFT: f32 = 1.015;
/// Overlay alpha. Tuned to keep the underlying terrain visible while the
/// overlay color still reads dominantly. When the planet is forced
/// fullbright (which the editor does whenever the overlay is on), this
/// blend rate lands somewhere around 60/40 overlay/terrain.
const OVERLAY_ALPHA: f32 = 0.6;
/// Overlay quads per cubemap face edge. Six faces × this², two triangles
/// per quad. Picked to match the topology of the underlying cubemap data
/// (cubemap-aligned, no pole singularity) while keeping the mesh inside
/// ~200k triangles for cheap rebuilds.
const OVERLAY_FACE_RES: u32 = 128;

#[allow(clippy::too_many_arguments)]
fn sync_surface_overlays(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    equirect: Res<EquirectViewerState>,
    active_surface: Res<ActivePreviewSurface>,
    render_radius: Res<SurfaceOverlayRenderRadius>,
    orientation: Res<SurfaceOverlayOrientation>,
    input: Res<PlanetEditorInputIntent>,
    preview_q: Query<Entity, With<PreviewPlanet>>,
    mut overlay_q: Query<(
        Entity,
        &SurfaceOverlayEntity,
        &mut Transform,
        &mut Visibility,
    )>,
    mut last_key: Local<Option<(bool, EquirectFieldKind, String)>>,
) {
    let Ok(preview_entity) = preview_q.single() else {
        for (entity, _, _, _) in &mut overlay_q {
            commands.entity(entity).despawn();
        }
        return;
    };

    let surface = active_surface.surface.as_ref();
    let dynamic_state = active_surface.dynamic_state.as_ref();
    let wants_overlay = equirect.overlay_on_planet && surface.is_some() && dynamic_state.is_some();

    // Space-held suppresses overlays without despawning them — a rebuild on
    // each press/release would be ~50–100ms, but flipping Visibility is free.
    let suppress = input.overlay_suppress;
    let target_visibility = if suppress {
        Visibility::Hidden
    } else {
        Visibility::Visible
    };

    // Bevy resource change detection fires whenever the egui panel mutably
    // derefs `EquirectViewerState`, which it does every frame. Track the
    // overlay-relevant inputs locally to detect real transitions; the field
    // selection and the active body both require a mesh rebuild.
    let key = (
        wants_overlay,
        equirect.selected,
        active_surface.body_name.clone(),
    );
    let key_changed = last_key.as_ref() != Some(&key);
    if key_changed {
        *last_key = Some(key);
    }
    // A fresh bake of the same body changes the surface data without changing
    // the key, so also rebuild when `ActivePreviewSurface` is written.
    let rebuild = render_radius.is_changed() || key_changed || active_surface.is_changed();

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
        // Unwraps guarded by `wants_overlay`.
        let surface = surface.unwrap();
        let dynamic_state = dynamic_state.unwrap();
        let field = equirect.selected;
        // LOD matched to the overlay mesh density (≈ four faces around the
        // equator) so field sampling reads at a comparable scale.
        let lod = equirect_lod(surface, OVERLAY_FACE_RES * 4);

        let mesh = build_cube_overlay_mesh(render_radius.0 * OVERLAY_LIFT, |face, x, y, res| {
            let u = (x as f32 + 0.5) / res as f32;
            let v = (y as f32 + 0.5) / res as f32;
            let dir = face_uv_to_dir(face, u, v);
            let [r, g, b] = field_overlay_rgb_linear(surface, dynamic_state, field, dir, lod);
            [r, g, b, OVERLAY_ALPHA]
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

/// Activity-mode label for the editor sidebar. Doesn't strictly belong
/// here, but it's the only place that needs to map an enum to a string.
pub fn activity_label(activity: TectonicActivity) -> &'static str {
    match activity {
        TectonicActivity::Active => "Active",
        TectonicActivity::StagnantLid => "Stagnant lid",
        TectonicActivity::Frozen { .. } => "Frozen",
    }
}
