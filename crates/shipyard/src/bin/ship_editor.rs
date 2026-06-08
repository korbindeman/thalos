//! Interactive 3D ship editor for the shipyard crate. Temporary home — once
//! the workflow settles, this should probably move out to its own crate so
//! `thalos_shipyard` can stay a headless library.
//!
//! Workflow:
//! - Left panel: parts palette + file I/O. Clicking a part arms it as
//!   "pending" — a popup then lists free attach nodes on the existing ship
//!   to place the pending part at.
//! - Right panel: inspector for the selected part (editable params,
//!   resource pools, delete).
//! - Viewport: orbit camera (right-drag + scroll), gizmo spheres at each
//!   attach node, parts rendered as cylinders/frustums sized from their
//!   attach-node diameters.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::gestures::PinchGesture;
use bevy::mesh::{Indices, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::ecs::system::SystemParam;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::picking::Pickable;
use bevy::picking::events::{Click, DragEnd, DragStart, Pointer};
use bevy::picking::hover::HoverMap;
use bevy::picking::mesh_picking::ray_cast::RayCastVisibility;
use bevy::picking::mesh_picking::{MeshPickingPlugin, MeshPickingSettings};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;
use bevy::window::PrimaryWindow;
use bevy_egui::{EguiContextSettings, EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;

use thalos_celestial::Universe;
use thalos_celestial::generate::{DefaultGenParams, generate_default};
use thalos_input::enhanced::{ActionSources, EnhancedInputSystems};
use thalos_input::settings::InputSettings;
use thalos_input::shipyard::{ShipyardInputIntent, ShipyardInputPlugin};
use thalos_shipyard::blueprint::default_params_for;
use thalos_shipyard::sizing::propagate_node_sizes;
use thalos_shipyard::*;

const SHIPS_DIR: &str = "ships";
const CATALOG_PATH: &str = "assets/parts.ron";

/// Radial segment count for cylindrical/frustum part meshes. Bevy's
/// default is 32, which leaves a visibly faceted silhouette at editor
/// zoom levels. Cost is negligible at the part counts we render.
const PART_RESOLUTION: u32 = 128;

/// Cursor-distance threshold (in pixels) separating a click from a
/// drag. Used by deselect-on-empty-click and by orbit-while-pending so
/// the click/drag boundary is consistent — strict-less is "click",
/// `>=` is "drag".
const CLICK_THRESHOLD_PX: f32 = 4.0;

#[derive(Clone, Debug)]
struct SavedShip {
    slug: String,
    name: String,
}

#[derive(Deserialize)]
struct ShipFileHeader {
    name: String,
}

fn schema_ship_name(name: &str) -> String {
    let name = name.trim();
    if name.is_empty() {
        "Unnamed".into()
    } else {
        name.into()
    }
}

fn slugify_ship_name(name: &str) -> String {
    let mut slug = String::new();
    let mut pending_separator = false;

    for c in name.trim().chars() {
        if c.is_ascii_alphanumeric() {
            if pending_separator && !slug.is_empty() {
                slug.push('-');
            }
            slug.push(c.to_ascii_lowercase());
            pending_separator = false;
        } else {
            pending_separator = !slug.is_empty();
        }
    }

    if slug.is_empty() {
        "unnamed".into()
    } else {
        slug
    }
}

fn ship_path_for_name(name: &str) -> PathBuf {
    ship_path_for_slug(&slugify_ship_name(name))
}

fn ship_path_for_slug(slug: &str) -> PathBuf {
    PathBuf::from(SHIPS_DIR).join(format!("{slug}.ron"))
}

/// Stable ordering inside each palette category. Within each kind we sort
/// by display name in the caller.
fn kind_order(entry: &CatalogEntry) -> u8 {
    match entry {
        CatalogEntry::Pod(_) => 0,
        CatalogEntry::Engine(_) => 1,
        CatalogEntry::Intake(_) => 2,
        CatalogEntry::Decoupler(_) => 3,
        CatalogEntry::Adapter(_) => 4,
        CatalogEntry::Tank(_) => 5,
        CatalogEntry::Fuselage(_) => 6,
        CatalogEntry::Wing(_) => 7,
        CatalogEntry::Gear(_) => 8,
    }
}

fn palette_category_order(entry: &CatalogEntry) -> u8 {
    match entry {
        CatalogEntry::Pod(_) => 0,
        CatalogEntry::Engine(_) => 1,
        CatalogEntry::Intake(_) => 2,
        CatalogEntry::Tank(_) => 3,
        CatalogEntry::Adapter(_) | CatalogEntry::Decoupler(_) => 4,
        CatalogEntry::Fuselage(_) => 4,
        CatalogEntry::Wing(_) => 4,
        CatalogEntry::Gear(_) => 5,
    }
}

fn palette_category_label(entry: &CatalogEntry) -> &'static str {
    match entry {
        CatalogEntry::Pod(_) => "Command Pods",
        CatalogEntry::Engine(_) => "Engines",
        CatalogEntry::Intake(_) => "Intakes",
        CatalogEntry::Tank(_) => "Propellant Tanks",
        CatalogEntry::Adapter(_) | CatalogEntry::Decoupler(_) | CatalogEntry::Fuselage(_) => {
            "Structure"
        }
        CatalogEntry::Wing(_) => "Aerodynamics",
        CatalogEntry::Gear(_) => "Landing Gear",
    }
}

fn meters_label(value: f32) -> String {
    format!("{value:.1} m")
}

fn palette_part_summary(entry: &CatalogEntry) -> String {
    match entry {
        CatalogEntry::Pod(p) => {
            format!(
                "{} · Diameter {} · {:.1} t dry",
                p.geometry.label(),
                meters_label(p.diameter),
                p.dry_mass / 1000.0
            )
        }
        CatalogEntry::Engine(e) => {
            format!(
                "{} · {} · Diameter {} · {:.0} kN · {:.0} s",
                e.optimized_for.label(),
                e.geometry.label(),
                meters_label(e.diameter),
                e.thrust / 1000.0,
                e.isp
            )
        }
        CatalogEntry::Intake(i) => format!(
            "Diameter {} · area {:.2} m² · {}",
            meters_label(i.diameter),
            i.capture.area_m2 * i.capture.efficiency,
            i.capture.kind.label()
        ),
        CatalogEntry::Decoupler(_) => match default_params_for(entry) {
            PartParams::Decoupler { diameter } => {
                format!("Default diameter {} · staging", meters_label(diameter))
            }
            _ => "Parametric diameter".into(),
        },
        CatalogEntry::Adapter(_) => match default_params_for(entry) {
            PartParams::Adapter {
                diameter,
                target_diameter,
            } => format!(
                "Default {} to {} diameter",
                meters_label(diameter),
                meters_label(target_diameter)
            ),
            _ => "Parametric diameter".into(),
        },
        CatalogEntry::Tank(_) => match default_params_for(entry) {
            PartParams::Tank { diameter, length } => format!(
                "Default diameter {} · length {}",
                meters_label(diameter),
                meters_label(length)
            ),
            _ => "Parametric diameter".into(),
        },
        CatalogEntry::Fuselage(_) => match default_params_for(entry) {
            PartParams::Fuselage {
                length, max_width, ..
            } => format!(
                "Loft body · default Ø{} · length {} · upswept tail",
                meters_label(max_width),
                meters_label(length)
            ),
            _ => "Stationed-loft fuselage".into(),
        },
        CatalogEntry::Wing(_) => match default_params_for(entry) {
            PartParams::Wing {
                span,
                root_chord,
                tip_chord,
                ..
            } => format!(
                "Span {} · chord {}→{} · click a hull to mount",
                meters_label(span),
                meters_label(root_chord),
                meters_label(tip_chord)
            ),
            _ => "Parametric wing".into(),
        },
        CatalogEntry::Gear(g) => match default_params_for(entry) {
            PartParams::Gear {
                strut_length,
                wheel_radius,
            } => format!(
                "{} · strut {} · wheel Ø{} · click a belly to mount",
                if g.track_fraction > 0.0 {
                    "Main (L/R)"
                } else {
                    "Nose"
                },
                meters_label(strut_length),
                meters_label(wheel_radius * 2.0)
            ),
            _ => "Parametric gear".into(),
        },
    }
}

fn palette_part_button(ui: &mut egui::Ui, entry: &CatalogEntry) -> bool {
    let label = format!("{}\n{}", entry.display_name(), palette_part_summary(entry));
    ui.add(
        egui::Button::new(label)
            .wrap()
            .min_size(egui::vec2(ui.available_width(), 38.0)),
    )
    .on_hover_text(entry.kind_name())
    .clicked()
}

fn ship_name_from_ron(text: &str) -> Option<String> {
    ron::from_str::<ShipFileHeader>(text)
        .ok()
        .map(|header| schema_ship_name(&header.name))
}

fn list_ships() -> Vec<SavedShip> {
    let dir = PathBuf::from(SHIPS_DIR);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<SavedShip> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("ron") {
                return None;
            }
            let slug = p
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())?;
            let name = std::fs::read_to_string(&p)
                .ok()
                .and_then(|text| ship_name_from_ron(&text))
                .unwrap_or_else(|| slug.clone());
            Some(SavedShip { slug, name })
        })
        .collect();
    out.sort_by_key(|ship| (ship.name.to_ascii_lowercase(), ship.slug.clone()));
    out
}

fn main() {
    let catalog = match PartCatalog::load_from_path(CATALOG_PATH) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load parts catalog from {CATALOG_PATH}: {e}");
            std::process::exit(1);
        }
    };

    App::new()
        .insert_resource(
            InputSettings::load_from_path("assets/input.ron")
                .expect("Failed to load input bindings from assets/input.ron"),
        )
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Thalos Shipyard".into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(bevy::asset::AssetPlugin {
                    // Resolve shaders from the workspace-root `assets/` dir,
                    // matching `thalos_game` and `thalos_body_editor`.
                    file_path: "../../assets".to_string(),
                    ..default()
                }),
        )
        .insert_resource(catalog)
        .add_plugins(EguiPlugin::default())
        .add_plugins(ShipyardInputPlugin)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(MeshPickingPlugin)
        .insert_resource(MeshPickingSettings {
            require_markers: false,
            // `VisibleInView` so hidden handles (resize arrow, non-pending
            // pins) don't absorb clicks from the body behind them.
            ray_cast_visibility: RayCastVisibility::VisibleInView,
        })
        .add_plugins(ShipyardPlugin)
        .add_plugins(SkyBackdropPlugin)
        .init_resource::<EditorState>()
        .init_resource::<TankResizeDrag>()
        .init_resource::<DeselectTracker>()
        .init_resource::<BuildOrientation>()
        .init_resource::<SymmetryMode>()
        .init_resource::<NextSymmetryId>()
        .init_resource::<PlacementSnap>()
        .init_resource::<PlacementPreview>()
        .init_resource::<SkyBackdropEnabled>()
        .add_systems(Startup, setup)
        .add_systems(
            PreUpdate,
            gate_shipyard_input_sources.before(EnhancedInputSystems::Update),
        )
        .add_systems(
            Update,
            (
                orbit_camera,
                recenter_camera_on_orientation_change,
                process_commands,
                rebuild_visuals,
                sync_symmetry_groups
                    .before(rebuild_wing_visuals)
                    .before(rebuild_nacelle_visuals),
                rebuild_wing_visuals,
                rebuild_nacelle_visuals,
                rebuild_gear_visuals,
                update_part_transforms.after(propagate_node_sizes),
                update_placement_preview.after(update_part_transforms),
                update_node_pin_style,
                disable_egui_pointer_capture,
                sync_self_nodes,
            ),
        )
        .add_systems(
            Update,
            (
                spawn_tank_resize_arrow,
                update_tank_resize_arrow.after(update_part_transforms),
                update_tank_resize_drag,
                update_selection_highlight
                    .after(rebuild_visuals)
                    .after(rebuild_wing_visuals)
                    .after(rebuild_nacelle_visuals)
                    .after(rebuild_gear_visuals),
                update_part_shader_params.after(rebuild_visuals),
                update_part_shader_highlight.after(rebuild_visuals),
                deselect_on_empty_click,
                propagate_coupled_material.after(rebuild_visuals),
                sync_shrouds.after(update_part_transforms),
                update_shroud_transparency.after(sync_shrouds),
            ),
        )
        .add_systems(EguiPrimaryContextPass, editor_ui)
        .run();
}

// ---------------------------------------------------------------------------
// Resources / components
// ---------------------------------------------------------------------------

/// A part the user has armed in the palette but not yet placed. Held by
/// [`EditorState::pending`] until the user clicks an attach node or
/// drops it on an empty canvas as the new root.
#[derive(Clone, Debug)]
struct PendingPart {
    catalog_id: CatalogId,
    params: PartParams,
}

#[derive(Resource, Default)]
struct EditorState {
    ship_root: Option<Entity>,
    ship_entity: Option<Entity>,
    ship_name: String,
    selected: Option<Entity>,
    pending: Option<PendingPart>,
    place_at: Option<(Entity, String)>,
    /// A pending surface-mount placement: `(host part, world hit point,
    /// mount kind)`. Consumed by `process_commands` which derives the
    /// mount-kind-specific `(station, angle)` pair.
    place_surface_at: Option<(Entity, Vec3, SurfaceMountKind)>,
    delete_selected: bool,
    set_as_root: bool,
    save_requested: bool,
    load_target: Option<String>,
    delete_file: Option<String>,
    refresh_list: bool,
    ship_list: Vec<SavedShip>,
    status: String,
}

/// KSP-style editor symmetry mode. When `mirror` is on, placing a footprint
/// part stamps a linked mirror counterpart across the host X = 0 plane.
#[derive(Resource, Default)]
struct SymmetryMode {
    mirror: bool,
}

/// Magnetic angle snap for body-skin (cylinder) mounts. On by default — the
/// mount azimuth rounds to [`BODY_SKIN_SNAP_STEP`] increments so gear/wings
/// land dead-on the belly / sides as the cursor sweeps around the fuselage.
#[derive(Resource)]
struct PlacementSnap {
    enabled: bool,
}

impl Default for PlacementSnap {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Placement-mode toggles bundled into one [`SystemParam`] so `process_commands`
/// stays under Bevy's 16-argument system limit.
#[derive(SystemParam)]
struct PlacementModes<'w> {
    symmetry: Res<'w, SymmetryMode>,
    snap: Res<'w, PlacementSnap>,
}

/// Snap increment for body-skin mount azimuth — 15° (24 positions around the
/// cylinder). The belly (π), top (0), and sides (±π/2) are all exact steps.
const BODY_SKIN_SNAP_STEP: f32 = std::f32::consts::TAU / 24.0;

fn snap_body_skin_angle(angle: f32) -> f32 {
    (angle / BODY_SKIN_SNAP_STEP).round() * BODY_SKIN_SNAP_STEP
}

/// Live placement-preview state. Holds the one reused ghost entity plus the
/// signature of the mesh currently on it, so the (small) ghost mesh is rebuilt
/// only when the host / snapped angle / part params actually change, not every
/// frame the cursor moves.
#[derive(Resource, Default)]
struct PlacementPreview {
    entity: Option<Entity>,
    sig: Option<PreviewSig>,
}

/// What the preview ghost mesh depends on. Station is excluded — it only moves
/// the ghost along the body axis (the transform), it doesn't reshape the mesh.
#[derive(Clone, PartialEq)]
struct PreviewSig {
    host: Entity,
    angle: f32,
    parent_radius: f32,
    params: PartParams,
}

/// Monotonic source of [`SymmetryGroup`] ids for newly stamped groups.
#[derive(Resource, Default)]
struct NextSymmetryId(u32);

impl NextSymmetryId {
    fn next(&mut self) -> u32 {
        let id = self.0;
        self.0 += 1;
        id
    }
}

#[derive(Resource)]
struct EditorAssets {
    part_material: Handle<StandardMaterial>,
    /// Matte dark finish for landing gear bodies — distinct from the stainless
    /// hull. The selection-highlight system falls back to this (not
    /// `part_material`) for gear visuals so wheels never read as steel.
    gear_material: Handle<StandardMaterial>,
    hover_material: Handle<StandardMaterial>,
    selected_material: Handle<StandardMaterial>,
    pending_node_material: Handle<StandardMaterial>,
    node_mesh: Handle<Mesh>,
    resize_arrow_mesh: Handle<Mesh>,
    resize_arrow_material: Handle<StandardMaterial>,
    /// Translucent green ghost for the live placement preview.
    preview_material: Handle<StandardMaterial>,
    /// Translucent cyan x-ray ghost for the gear stow-bay box. High depth bias
    /// so the reserved volume reads *through* the opaque fuselage skin.
    gear_bay_material: Handle<StandardMaterial>,
}

#[derive(Component)]
struct PartVisual;

/// Marker on a wing's mesh child. Distinct from [`PartVisual`] so the
/// body-of-revolution rebuild (`rebuild_visuals`) never despawns wing
/// geometry — `rebuild_wing_visuals` owns it.
#[derive(Component)]
struct WingVisual;

#[derive(Component)]
struct NacelleVisual;

/// Marker on a gearbox's mesh child. Distinct from [`PartVisual`] so
/// `rebuild_visuals` (the body-of-revolution rebuild) never touches gear
/// geometry — `rebuild_gear_visuals` owns it, like wings/nacelles.
#[derive(Component)]
struct GearVisual;

/// Marker on a gearbox's **stow-bay** ghost child — the x-ray box showing the
/// volume inside the fuselage that will house the gear when retracted. Rendered
/// translucent and non-pickable; `rebuild_gear_visuals` owns it alongside
/// [`GearVisual`].
#[derive(Component)]
struct GearBayVisual;

/// Marker on the live placement-preview ghost — the translucent silhouette of
/// the pending footprint part following the cursor across a host surface. One
/// reused entity, tracked by [`PlacementPreview`].
#[derive(Component)]
struct PreviewGhost;

#[derive(Component)]
struct PartBody(Entity);

/// Per-part `ShipPartMaterial` asset handle, cached on the part entity
/// so it survives child rebuilds (e.g. resizing a tank despawns and
/// respawns the body, but the material asset — and its tint state — is
/// stable). Used by any part that carries [`PartMaterial`] — tanks and
/// decouplers today.
#[derive(Component, Clone)]
struct PartShaderHandle(Handle<ShipPartMaterial>);

#[derive(Component)]
struct AttachNodePin {
    part: Entity,
    node_id: NodeId,
}

#[derive(Component)]
struct TankResizeArrow {
    tank: Entity,
}

#[derive(Resource, Default)]
struct TankResizeDrag {
    active: Option<TankDragState>,
}

struct TankDragState {
    tank: Entity,
    start_length: f32,
    start_cursor: Vec2,
    screen_axis: Vec2,
    world_per_pixel: f32,
}

/// Tracks the cursor at mouse-down when the press landed on empty space,
/// so a release at near-the-same position clears the selection but a
/// press→drag→release (camera orbit) does not.
#[derive(Resource, Default)]
struct DeselectTracker {
    press_cursor: Option<Vec2>,
}

/// Vertical (rocket / VAB) vs horizontal (aircraft / SPH) build layout.
/// Purely a display + interaction frame: parts are always authored along
/// the body +Y axis; horizontal lays the whole assembly down so the body
/// axis runs fore/aft and the dorsal (+Z) side faces up, like KSP's
/// Spaceplane Hangar. The rotation is applied rigidly to every part in
/// `update_part_transforms`; placement / resize convert pointer hits back
/// through its inverse so building stays correct in either layout.
#[derive(Resource, Default)]
struct BuildOrientation {
    horizontal: bool,
}

impl BuildOrientation {
    fn rotation(&self) -> Quat {
        if self.horizontal {
            // Nose (+Y) → −Z (forward), dorsal (+Z) → +Y (up), span (X) stays
            // level — the craft lies down facing away from the camera.
            Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)
        } else {
            Quat::IDENTITY
        }
    }
}

#[derive(Component)]
struct OrbitCamera {
    focus: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

fn setup(
    mut commands: Commands,
    mut mats: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut state: ResMut<EditorState>,
) {
    commands.insert_resource(EditorAssets {
        // Double-sided so a wing's thin slab (and its reflected panel) reads
        // from both faces; closed bodies-of-revolution are unaffected.
        part_material: mats.add(StandardMaterial {
            // Same stainless-steel base as the procedural `ShipPartMaterial`
            // (fuel tank) so wings, nacelles, gear, and pod/engine bodies read
            // as one material. Double-sided so a wing's thin slab reads from
            // both faces; closed bodies-of-revolution are unaffected.
            double_sided: true,
            cull_mode: None,
            ..stainless_steel_base()
        }),
        gear_material: mats.add(StandardMaterial {
            double_sided: true,
            cull_mode: None,
            ..landing_gear_base()
        }),
        hover_material: mats.add(StandardMaterial {
            base_color: Color::srgb(0.82, 0.85, 0.88),
            perceptual_roughness: 0.4,
            metallic: 0.6,
            emissive: LinearRgba::rgb(0.08, 0.08, 0.08),
            double_sided: true,
            cull_mode: None,
            ..default()
        }),
        selected_material: mats.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.9, 1.0),
            perceptual_roughness: 0.4,
            metallic: 0.6,
            emissive: LinearRgba::rgb(0.15, 0.35, 0.7),
            double_sided: true,
            cull_mode: None,
            ..default()
        }),
        pending_node_material: mats.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.9, 1.0),
            emissive: LinearRgba::rgb(0.1, 0.6, 0.9),
            ..default()
        }),
        node_mesh: meshes.add(Sphere::new(0.5).mesh()),
        resize_arrow_mesh: meshes.add(Cone::new(0.3, 0.8).mesh()),
        resize_arrow_material: mats.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.75, 0.2),
            emissive: LinearRgba::rgb(0.9, 0.5, 0.05),
            perceptual_roughness: 0.5,
            unlit: false,
            ..default()
        }),
        preview_material: mats.add(StandardMaterial {
            base_color: Color::srgba(0.4, 1.0, 0.5, 0.45),
            emissive: LinearRgba::rgb(0.05, 0.3, 0.1),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            double_sided: true,
            cull_mode: None,
            ..default()
        }),
        gear_bay_material: mats.add(StandardMaterial {
            base_color: Color::srgba(0.2, 0.85, 1.0, 0.22),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            double_sided: true,
            cull_mode: None,
            // Large bias forces the ghost in front of the opaque hull so the
            // reserved bay volume is visible through the fuselage (x-ray).
            depth_bias: 1.0e9,
            ..default()
        }),
    });

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(8.0, 4.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitCamera {
            focus: Vec3::new(0.0, -2.0, 0.0),
            distance: 12.0,
            yaw: 0.8,
            pitch: 0.4,
        },
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        PointLight {
            intensity: 400_000.0,
            ..default()
        },
        Transform::from_xyz(-6.0, 4.0, -4.0),
    ));

    state.ship_name = "New Ship".into();
    state.ship_list = list_ships();
    state.status = "Click a part to begin".into();
}

// ---------------------------------------------------------------------------
// Visuals
// ---------------------------------------------------------------------------

struct VisualSpec {
    mesh: Mesh,
    height: f32,
}

/// `top` node diameter of a host part, or a sensible default. Single source
/// for the surface-mount radius lookups so they stay consistent.
fn host_top_diameter(nodes: &Query<&AttachNodes>, host: Entity) -> f32 {
    nodes
        .get(host)
        .ok()
        .and_then(|n| n.get("top").map(|nd| nd.diameter))
        .unwrap_or(2.0)
}

fn visual_spec(
    nodes: &AttachNodes,
    pod: Option<&CommandPod>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    tank: Option<&FuelTank>,
    fuselage: Option<&Fuselage>,
    engine: Option<&Engine>,
    intake: Option<&AirIntake>,
) -> Option<VisualSpec> {
    if let Some(p) = pod {
        // Inline cockpit: no body mesh (the fuselage nose is the nose).
        if matches!(p.geometry, PodGeometry::Inline) {
            return None;
        }
        let (radius_top, radius_bottom, h) = pod_visual_profile(p.diameter, p.geometry);
        let mesh = match p.geometry {
            // Rounded ogive nose (airliner radome) vs the plain capsule cone.
            PodGeometry::AircraftCockpit => build_cockpit_mesh(p.diameter, h),
            PodGeometry::Inline => unreachable!("handled above"),
            PodGeometry::Capsule => ConicalFrustum {
                radius_top,
                radius_bottom,
                height: h,
            }
            .mesh()
            .resolution(PART_RESOLUTION)
            .into(),
        };
        Some(VisualSpec { mesh, height: h })
    } else if dec.is_some() {
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let h = 0.2;
        Some(VisualSpec {
            mesh: Cylinder::new(d * 0.5, h)
                .mesh()
                .resolution(PART_RESOLUTION)
                .into(),
            height: h,
        })
    } else if let Some(a) = adapter {
        let top_d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let bot_d = a.target_diameter;
        let h = ((top_d + bot_d) * 0.5).max(0.4);
        Some(VisualSpec {
            mesh: ConicalFrustum {
                radius_top: top_d * 0.5,
                radius_bottom: bot_d * 0.5,
                height: h,
            }
            .mesh()
            .resolution(PART_RESOLUTION)
            .into(),
            height: h,
        })
    } else if let Some(t) = tank {
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let h = t.length;
        Some(VisualSpec {
            mesh: Cylinder::new(d * 0.5, h)
                .mesh()
                .resolution(PART_RESOLUTION)
                .into(),
            height: h,
        })
    } else if let Some(f) = fuselage {
        // Barrel diameter inherits from the `top` node (parent-driven), like
        // a tank; the loft generator scales the rest to it.
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(f.max_width);
        Some(VisualSpec {
            mesh: build_fuselage_mesh(f, d),
            height: f.length,
        })
    } else if let Some(e) = engine {
        match e.geometry {
            EngineGeometry::RocketBell => {
                let (r_top, r_bot, h) = engine_visual_profile(e.diameter);
                Some(VisualSpec {
                    mesh: ConicalFrustum {
                        radius_top: r_top,
                        radius_bottom: r_bot,
                        height: h,
                    }
                    .mesh()
                    .resolution(PART_RESOLUTION)
                    .into(),
                    height: h,
                })
            }
            EngineGeometry::JetNacelle => Some(VisualSpec {
                mesh: build_jet_nacelle_body_mesh(e),
                height: jet_nacelle_length(e),
            }),
        }
    } else if let Some(i) = intake {
        Some(VisualSpec {
            mesh: Cylinder::new(i.diameter * 0.5, i.length)
                .mesh()
                .resolution(PART_RESOLUTION)
                .into(),
            height: i.length,
        })
    } else {
        None
    }
}

/// Engine body silhouette: `(radius_top, radius_bottom, height)` for a
/// given engine diameter. Single source for both the engine mesh and the
/// matching shroud geometry — drift between the two would leave the
/// shroud edge either floating off the engine or clipping into it.
fn engine_visual_profile(diameter: f32) -> (f32, f32, f32) {
    (diameter * 0.35, diameter * 0.5, diameter * 0.9)
}

/// Pick `ShipPartMaterial` uniforms for a given part. Length / radius
/// drive the procedural panel + rivet layout; each part picks its own
/// dimensions so the pattern reads consistently across tank–decoupler
/// boundaries without sharing an asset handle.
fn ship_part_params(
    nodes: &AttachNodes,
    tank: Option<&FuelTank>,
    fuselage: Option<&Fuselage>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    seed: u32,
) -> ShipPartParams {
    let top_r = nodes.get("top").map(|n| n.diameter * 0.5).unwrap_or(0.5);
    // Tanks and decouplers are cylinders; adapters are conical frustums
    // from `top_r` at the mesh's +Y end to `target_diameter / 2` at -Y.
    let (radius_top, radius_bottom, length) = if let Some(t) = tank {
        (top_r, top_r, t.length)
    } else if let Some(f) = fuselage {
        // Near-cylindrical barrel: the panel shader treats it like a tank.
        (top_r, top_r, f.length)
    } else if dec.is_some() {
        (top_r, top_r, 0.2)
    } else if let Some(a) = adapter {
        let bot_r = a.target_diameter * 0.5;
        let h = (top_r + bot_r).max(0.4); // same formula as `visual_spec`
        let dr = top_r - bot_r;
        let slant = (h * h + dr * dr).sqrt();
        (top_r, bot_r, slant)
    } else {
        (top_r, top_r, 1.0)
    };
    ShipPartParams {
        length,
        radius_top,
        radius_bottom,
        seed,
        ..default()
    }
}

type VisualQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AttachNodes,
        Option<&'static CommandPod>,
        Option<&'static Decoupler>,
        Option<&'static Adapter>,
        Option<&'static FuelTank>,
        Option<&'static Fuselage>,
        Option<&'static Engine>,
        Option<&'static AirIntake>,
        Option<&'static SurfaceMount>,
        Option<&'static Children>,
        Option<&'static PartShaderHandle>,
        Has<PartMaterial>,
    ),
    Or<(Added<Part>, Changed<AttachNodes>)>,
>;

fn rebuild_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    assets: Res<EditorAssets>,
    state: Res<EditorState>,
    parts: VisualQuery,
    stale: Query<(), Or<(With<PartVisual>, With<AttachNodePin>)>>,
) {
    for (
        e,
        nodes,
        pod,
        dec,
        adapter,
        tank,
        fuselage,
        engine,
        intake,
        surface,
        children,
        part_shader,
        has_part_mat,
    ) in parts.iter()
    {
        if let Some(ch) = children {
            for c in ch.into_iter() {
                if stale.get(*c).is_ok() {
                    commands.entity(*c).despawn();
                }
            }
        }

        if engine.is_some_and(|e| e.geometry == EngineGeometry::JetNacelle)
            && surface.is_some_and(|m| m.kind == SurfaceMountKind::WingPylon)
        {
            continue;
        }

        // ---- Body visual --------------------------------------------------
        if let Some(spec) = visual_spec(nodes, pod, dec, adapter, tank, fuselage, engine, intake) {
            let mesh = meshes.add(spec.mesh);

            // Parts carrying `PartMaterial` render with `ShipPartMaterial`
            // (procedural stainless); others use the shared
            // `StandardMaterial`. The ship-material asset is created lazily
            // on first rebuild and cached on the part entity so resizing
            // doesn't churn assets or drop per-part state (seed/tint).
            let body_id = if has_part_mat {
                let params = ship_part_params(nodes, tank, fuselage, dec, adapter, e.index_u32());
                let handle = match part_shader {
                    Some(h) => h.0.clone(),
                    None => {
                        let h = ship_materials.add(ShipPartMaterial {
                            base: stainless_steel_base(),
                            extension: ShipPartExtension { params },
                        });
                        commands.entity(e).insert(PartShaderHandle(h.clone()));
                        h
                    }
                };
                commands
                    .spawn((
                        Mesh3d(mesh),
                        MeshMaterial3d(handle),
                        Transform::from_xyz(0.0, -spec.height * 0.5, 0.0),
                        Visibility::default(),
                        PartVisual,
                        PartBody(e),
                        Pickable::default(),
                    ))
                    .observe(on_body_click)
                    .id()
            } else {
                let initial_material = if Some(e) == state.selected {
                    assets.selected_material.clone()
                } else {
                    assets.part_material.clone()
                };
                commands
                    .spawn((
                        Mesh3d(mesh),
                        MeshMaterial3d(initial_material),
                        Transform::from_xyz(0.0, -spec.height * 0.5, 0.0),
                        Visibility::default(),
                        PartVisual,
                        PartBody(e),
                        Pickable::default(),
                    ))
                    .observe(on_body_click)
                    .id()
            };
            commands.entity(e).add_child(body_id);
        }

        // ---- Attach node pins --------------------------------------------
        for (id, node) in &nodes.nodes {
            let pin = commands
                .spawn((
                    Mesh3d(assets.node_mesh.clone()),
                    MeshMaterial3d(assets.pending_node_material.clone()),
                    Transform::from_translation(node.offset),
                    Visibility::Hidden,
                    AttachNodePin {
                        part: e,
                        node_id: id.clone(),
                    },
                    Pickable::default(),
                ))
                .observe(on_pin_click)
                .id();
            commands.entity(e).add_child(pin);
        }
    }
}

/// Build (or rebuild) the mesh child for each wing whose shape, mount, or
/// host diameter just changed. Wings are surface-mounted lifting surfaces,
/// not bodies of revolution, so they live outside `rebuild_visuals`: the
/// mesh is generated in the host-local frame by [`build_wing_mesh`] and the
/// wing entity's transform (set in `update_part_transforms`) places it on
/// the hull. Uses the shared part materials so selection / hover highlight
/// flows through `update_selection_highlight` like any other part.
fn rebuild_wing_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Res<EditorAssets>,
    state: Res<EditorState>,
    wings: Query<
        (Entity, &Wing, &SurfaceMount, Option<&Children>),
        Or<(Added<Wing>, Changed<Wing>, Changed<SurfaceMount>)>,
    >,
    host_nodes: Query<&AttachNodes>,
    hosts: Query<&Fuselage>,
    stale: Query<(), With<WingVisual>>,
) {
    for (e, wing, mount, children) in wings.iter() {
        if let Some(ch) = children {
            for c in ch.into_iter() {
                if stale.get(*c).is_ok() {
                    commands.entity(*c).despawn();
                }
            }
        }
        let top_d = host_top_diameter(&host_nodes, mount.parent);
        let (parent_radius, _) =
            host_mount_geometry(hosts.get(mount.parent).ok(), top_d, mount.station, mount.angle);
        let mesh = meshes.add(build_wing_mesh(wing, mount.angle, parent_radius));
        let material = if Some(e) == state.selected {
            assets.selected_material.clone()
        } else {
            assets.part_material.clone()
        };
        let body = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                Transform::IDENTITY,
                Visibility::default(),
                WingVisual,
                PartBody(e),
                Pickable::default(),
            ))
            .observe(on_body_click)
            .id();
        let parent = e;
        commands.queue(move |world: &mut World| {
            if world.get_entity(parent).is_ok() {
                world.entity_mut(body).insert(ChildOf(parent));
            } else {
                world.entity_mut(body).despawn();
            }
        });
    }
}

fn rebuild_nacelle_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Res<EditorAssets>,
    state: Res<EditorState>,
    engines: Query<
        (Entity, &Engine, &SurfaceMount, Option<&Children>),
        Or<(Added<Engine>, Changed<SurfaceMount>)>,
    >,
    wings: Query<&Wing>,
    surface_mounts: Query<&SurfaceMount>,
    host_nodes: Query<&AttachNodes>,
    hosts: Query<&Fuselage>,
    stale: Query<(), With<NacelleVisual>>,
) {
    for (e, engine, mount, children) in engines.iter() {
        if engine.geometry != EngineGeometry::JetNacelle
            || mount.kind != SurfaceMountKind::WingPylon
        {
            continue;
        }
        if let Some(ch) = children {
            for c in ch.into_iter() {
                if stale.get(*c).is_ok() {
                    commands.entity(*c).despawn();
                }
            }
        }

        let Ok(wing) = wings.get(mount.parent) else {
            continue;
        };
        let Ok(wing_mount) = surface_mounts.get(mount.parent) else {
            continue;
        };
        let top_d = host_top_diameter(&host_nodes, wing_mount.parent);
        let (parent_radius, _) = host_mount_geometry(
            hosts.get(wing_mount.parent).ok(),
            top_d,
            wing_mount.station,
            wing_mount.angle,
        );
        let mesh = meshes.add(build_jet_nacelle_pylon_mesh(
            engine,
            JetNacelleMount {
                wing,
                wing_mount_angle: wing_mount.angle,
                parent_radius,
                span_fraction: mount.station,
                chord_fraction: mount.angle,
            },
        ));
        let material = if Some(e) == state.selected {
            assets.selected_material.clone()
        } else {
            assets.part_material.clone()
        };
        let body = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                Transform::IDENTITY,
                Visibility::default(),
                NacelleVisual,
                PartBody(e),
                Pickable::default(),
            ))
            .observe(on_body_click)
            .id();
        let parent = e;
        commands.queue(move |world: &mut World| {
            if world.get_entity(parent).is_ok() {
                world.entity_mut(body).insert(ChildOf(parent));
            } else {
                world.entity_mut(body).despawn();
            }
        });
    }
}

/// Build (or rebuild) the mesh child for each gearbox whose dimensions, mount,
/// or host diameter just changed. A gearbox is a single footprint part that
/// draws all of its legs ([`build_gear_mesh`]); the mesh is in the host-local
/// frame and the gear entity's transform (set in `update_part_transforms`)
/// places it on the belly. Mirrors `rebuild_wing_visuals` — no symmetry.
fn rebuild_gear_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Res<EditorAssets>,
    state: Res<EditorState>,
    gears: Query<
        (Entity, &Gear, &SurfaceMount, Option<&Children>),
        Or<(Added<Gear>, Changed<Gear>, Changed<SurfaceMount>)>,
    >,
    host_nodes: Query<&AttachNodes>,
    hosts: Query<&Fuselage>,
    stale: Query<(), Or<(With<GearVisual>, With<GearBayVisual>)>>,
) {
    for (e, gear, mount, children) in gears.iter() {
        if let Some(ch) = children {
            for c in ch.into_iter() {
                if stale.get(*c).is_ok() {
                    commands.entity(*c).despawn();
                }
            }
        }
        let top_d = host_top_diameter(&host_nodes, mount.parent);
        let (parent_radius, _) =
            host_mount_geometry(hosts.get(mount.parent).ok(), top_d, mount.station, mount.angle);
        let mesh = meshes.add(build_gear_mesh(gear, mount.angle, parent_radius));
        let material = if Some(e) == state.selected {
            assets.selected_material.clone()
        } else {
            assets.gear_material.clone()
        };
        let body = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                Transform::IDENTITY,
                Visibility::default(),
                GearVisual,
                PartBody(e),
                Pickable::default(),
            ))
            .observe(on_body_click)
            .id();
        // Stow-bay ghost: an x-ray translucent box inside the fuselage. Not
        // pickable, no `PartBody` (so the selection-highlight system leaves its
        // ghost material alone), no click observer.
        let bay = commands
            .spawn((
                Mesh3d(meshes.add(build_gear_bay_mesh(gear, mount.angle, parent_radius))),
                MeshMaterial3d(assets.gear_bay_material.clone()),
                Transform::IDENTITY,
                Visibility::default(),
                GearBayVisual,
                Pickable::IGNORE,
            ))
            .id();
        let parent = e;
        commands.queue(move |world: &mut World| {
            if world.get_entity(parent).is_ok() {
                world.entity_mut(body).insert(ChildOf(parent));
                world.entity_mut(bay).insert(ChildOf(parent));
            } else {
                world.entity_mut(body).despawn();
                world.entity_mut(bay).despawn();
            }
        });
    }
}

/// KSP linked symmetry: keep every mirror counterpart in lockstep with its
/// group's primary — params copied (handed fields negated) and mount
/// reflected across the host X = 0 plane — so editing or moving the primary
/// updates the whole group. Writes are change-guarded so they don't
/// re-trigger the rebuild systems every frame.
fn sync_symmetry_groups(
    groups: Query<(Entity, &SymmetryGroup)>,
    mut mounts: Query<&mut SurfaceMount>,
    mut wings: Query<&mut Wing>,
) {
    let mut primary_of_group: HashMap<u32, Entity> = HashMap::new();
    let mut members: HashMap<u32, Vec<Entity>> = HashMap::new();
    let mut role_of: HashMap<Entity, (u32, SymmetryRole)> = HashMap::new();
    for (e, g) in groups.iter() {
        members.entry(g.id).or_default().push(e);
        role_of.insert(e, (g.id, g.role));
        if g.role == SymmetryRole::Primary {
            primary_of_group.insert(g.id, e);
        }
    }
    // A host's mirror counterpart (for nested WingPylon parents): the member
    // of the host's own group with the opposite role.
    let host_mirror = |host: Entity| -> Option<Entity> {
        let (gid, role) = role_of.get(&host)?;
        let want = match role {
            SymmetryRole::Primary => SymmetryRole::Mirror,
            SymmetryRole::Mirror => SymmetryRole::Primary,
        };
        members
            .get(gid)?
            .iter()
            .copied()
            .find(|m| role_of.get(m).map(|(_, r)| *r) == Some(want))
    };

    for (gid, mems) in &members {
        let Some(&primary) = primary_of_group.get(gid) else {
            continue;
        };
        let Some(p_mount) = mounts.get(primary).ok().copied() else {
            continue;
        };
        let p_wing = wings.get(primary).ok().cloned();
        for &m in mems {
            if m == primary {
                continue;
            }
            let (parent, angle) = match p_mount.kind {
                // Same host, reflected azimuth.
                SurfaceMountKind::BodySkin => (p_mount.parent, -p_mount.angle),
                // The host wing is itself mirrored; mount on its counterpart
                // at the same local coords — the host reflection does the work.
                SurfaceMountKind::WingPylon => {
                    (host_mirror(p_mount.parent).unwrap_or(p_mount.parent), p_mount.angle)
                }
            };
            let target = SurfaceMount {
                parent,
                kind: p_mount.kind,
                station: p_mount.station,
                angle,
            };
            if let Ok(mut mm) = mounts.get_mut(m)
                && *mm != target
            {
                *mm = target;
            }
            if let Some(w) = &p_wing
                && let Ok(mut mw) = wings.get_mut(m)
            {
                let mut tw = w.clone();
                tw.incidence = -w.incidence; // incidence is the one handed param
                if *mw != tw {
                    *mw = tw;
                }
            }
        }
    }
}

fn update_part_transforms(
    ships: Query<&Ship>,
    attachments: Query<(Entity, &Attachment)>,
    surface_mounts: Query<(Entity, &SurfaceMount)>,
    nodes: Query<&AttachNodes>,
    hosts: Query<&Fuselage>,
    orientation: Res<BuildOrientation>,
    mut transforms: Query<&mut Transform, With<Part>>,
) {
    let mut children_map: HashMap<Entity, Vec<(Entity, Attachment)>> = HashMap::new();
    for (e, att) in attachments.iter() {
        children_map
            .entry(att.parent)
            .or_default()
            .push((e, att.clone()));
    }

    for ship in ships.iter() {
        if let Ok(mut t) = transforms.get_mut(ship.root) {
            t.translation = Vec3::ZERO;
            t.rotation = Quat::IDENTITY;
        }
        let mut queue: VecDeque<Entity> = VecDeque::from([ship.root]);
        while let Some(parent) = queue.pop_front() {
            let parent_pos = transforms
                .get(parent)
                .map(|t| t.translation)
                .unwrap_or(Vec3::ZERO);
            let Ok(parent_nodes) = nodes.get(parent) else {
                continue;
            };
            let parent_pos_and_nodes: Vec<(Entity, Vec3)> = children_map
                .get(&parent)
                .map(|kids| {
                    kids.iter()
                        .filter_map(|(c, att)| {
                            let pn = parent_nodes.get(&att.parent_node)?;
                            let child_offset = nodes
                                .get(*c)
                                .ok()
                                .and_then(|cn| cn.get(&att.my_node))
                                .map(|n| n.offset)
                                .unwrap_or(Vec3::ZERO);
                            Some((*c, parent_pos + pn.offset - child_offset))
                        })
                        .collect()
                })
                .unwrap_or_default();
            for (child, pos) in parent_pos_and_nodes {
                if let Ok(mut ct) = transforms.get_mut(child) {
                    ct.translation = pos;
                    ct.rotation = Quat::IDENTITY;
                }
                queue.push_back(child);
            }
        }
    }

    // Surface-mounted parts sit in their host-local frame. Body-skin mounts
    // (wings) move down the host body axis; wing-pylon mounts (nacelles)
    // inherit the wing origin because the pylon mesh carries the offsets.
    //
    // Process in dependency order — BodySkin first, then WingPylon — because a
    // nacelle's parent is a wing that is itself a surface mount. Reading a
    // parent before it has been positioned this frame would pull its stale
    // (already rigid-rotated) translation from the previous frame, which the
    // rigid rotation below would then rotate a second time. Two passes keep
    // every parent upright and freshly positioned before its child reads it.
    let position_mount = |transforms: &mut Query<&mut Transform, With<Part>>,
                          part: Entity,
                          mount: &SurfaceMount| {
        let Ok(parent_t) = transforms.get(mount.parent).map(|t| t.translation) else {
            return;
        };
        let local_offset = match mount.kind {
            SurfaceMountKind::BodySkin => {
                let host_height = nodes
                    .get(mount.parent)
                    .ok()
                    .and_then(|n| n.get("bottom").map(|nd| -nd.offset.y))
                    .unwrap_or(0.0);
                // On a loft host the centerline rises toward the tail
                // (upsweep) / drops at the nose (droop); the mount must follow
                // it along +Z. Flat (zero) for a plain cylinder host.
                let top_d = host_top_diameter(&nodes, mount.parent);
                let (_, v_offset) =
                    host_mount_geometry(hosts.get(mount.parent).ok(), top_d, mount.station, 0.0);
                Vec3::new(0.0, -mount.station * host_height, v_offset)
            }
            SurfaceMountKind::WingPylon => Vec3::ZERO,
        };
        if let Ok(mut pt) = transforms.get_mut(part) {
            pt.translation = parent_t + local_offset;
            pt.rotation = Quat::IDENTITY;
        }
    };
    for (part, mount) in surface_mounts.iter() {
        if mount.kind == SurfaceMountKind::BodySkin {
            position_mount(&mut transforms, part, mount);
        }
    }
    for (part, mount) in surface_mounts.iter() {
        if mount.kind == SurfaceMountKind::WingPylon {
            position_mount(&mut transforms, part, mount);
        }
    }

    // Build-layout: everything above is computed in the upright build frame;
    // lay the whole assembly down rigidly for the horizontal (aircraft)
    // layout. Identity in vertical mode, so this is a no-op for rockets.
    let r = orientation.rotation();
    if r != Quat::IDENTITY {
        for mut t in transforms.iter_mut() {
            t.translation = r * t.translation;
            t.rotation = r;
        }
    }
}

/// Re-centre the orbit camera when the build layout flips, so the craft
/// stays framed (it moves from a tall upright stack to a level fuselage).
fn recenter_camera_on_orientation_change(
    orientation: Res<BuildOrientation>,
    mut cam: Query<&mut OrbitCamera>,
) {
    if !orientation.is_changed() {
        return;
    }
    for mut c in cam.iter_mut() {
        c.focus = if orientation.horizontal {
            Vec3::ZERO
        } else {
            Vec3::new(0.0, -2.0, 0.0)
        };
    }
}

fn update_node_pin_style(
    state: Res<EditorState>,
    catalog: Res<PartCatalog>,
    assets: Res<EditorAssets>,
    attachments: Query<&Attachment>,
    mut pins: Query<(
        &AttachNodePin,
        &mut MeshMaterial3d<StandardMaterial>,
        &mut Visibility,
    )>,
) {
    let occupied: HashSet<(Entity, String)> = attachments
        .iter()
        .map(|a| (a.parent, a.parent_node.clone()))
        .collect();
    let pending_uses_nodes = state.pending.as_ref().is_some_and(|p| {
        !matches!(p.params, PartParams::Wing { .. })
            && !catalog.resolve(&p.catalog_id).is_ok_and(|entry| {
                matches!(
                    entry,
                    CatalogEntry::Engine(e) if e.geometry == EngineGeometry::JetNacelle
                )
            })
    });

    for (pin, mut mat, mut vis) in pins.iter_mut() {
        let is_occupied = occupied.contains(&(pin.part, pin.node_id.clone()));

        // Pins only appear while a part is pending, and only on unoccupied
        // nodes.
        *vis = if pending_uses_nodes && !is_occupied {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };

        mat.0 = assets.pending_node_material.clone();
    }
}

/// bevy_egui's default `capture_pointer_input` writes a fake top-priority
/// PointerHits for the egui context entity whenever egui wants pointer
/// input, which redirects every click away from our meshes. Disable it and
/// filter picks manually via `is_pointer_over_area` below.
fn disable_egui_pointer_capture(mut q: Query<&mut EguiContextSettings>) {
    for mut s in q.iter_mut() {
        if s.capture_pointer_input {
            s.capture_pointer_input = false;
        }
    }
}

/// Propagate a part's own kind-component values (e.g. `CommandPod.diameter`,
/// `FuelTank.length`, `Engine.diameter`) into its own AttachNodes, only
/// touching the component when a value actually differs. This way editor
/// sliders drive AttachNodes → rebuild_visuals deterministically, without
/// the kind component's spurious Changed signals causing per-frame respawns.
///
/// For parametric radius parts (Decoupler/Adapter/FuelTank) the sync is
/// bidirectional by root state:
/// - **Root**: `self.diameter → nodes.top` so the Diameter slider drives
///   the part's visual size.
/// - **Child**: `nodes.top → self.diameter` so the diameter inherited via
///   `sizing::propagate_node_sizes` is mirrored onto the component. This
///   way a later re-root starts from the displayed size instead of
///   snapping back to the palette's placeholder.
fn sync_self_nodes(
    mut q: Query<(
        &mut AttachNodes,
        Option<&Attachment>,
        Option<&CommandPod>,
        Option<&mut Decoupler>,
        Option<&mut Adapter>,
        Option<&mut FuelTank>,
        Option<&Engine>,
    )>,
) {
    for (mut nodes, attachment, pod, mut dec, mut adapter, mut tank, engine) in q.iter_mut() {
        let is_root = attachment.is_none();
        let mut targets: Vec<(String, f32, Vec3)> = Vec::new();
        if let Some(p) = pod {
            let d = p.diameter;
            targets.push((
                "bottom".into(),
                d,
                Vec3::new(0.0, -d * p.geometry.length_factor(), 0.0),
            ));
        }
        // Read kind-component fields through `as_ref()` so the borrow only
        // goes through Bevy's `Mut::deref` (no Changed trigger). The write
        // path below reaches for `as_mut()` only when the value actually
        // needs to change.
        if let Some(d) = dec.as_ref() {
            let self_d = d.diameter;
            let top_d = if is_root {
                targets.push(("top".into(), self_d, Vec3::ZERO));
                self_d
            } else {
                let inherited = nodes.get("top").map(|n| n.diameter).unwrap_or(self_d);
                if (self_d - inherited).abs() > f32::EPSILON
                    && let Some(m) = dec.as_mut()
                {
                    m.diameter = inherited;
                }
                inherited
            };
            targets.push(("bottom".into(), top_d, Vec3::new(0.0, -0.2, 0.0)));
        }
        if let Some(a) = adapter.as_ref() {
            let self_d = a.diameter;
            let target_d = a.target_diameter;
            let top_d = if is_root {
                targets.push(("top".into(), self_d, Vec3::ZERO));
                self_d
            } else {
                let inherited = nodes.get("top").map(|n| n.diameter).unwrap_or(self_d);
                if (self_d - inherited).abs() > f32::EPSILON
                    && let Some(m) = adapter.as_mut()
                {
                    m.diameter = inherited;
                }
                inherited
            };
            let h = ((top_d + target_d) * 0.5).max(0.4);
            targets.push(("bottom".into(), target_d, Vec3::new(0.0, -h, 0.0)));
        }
        if let Some(t) = tank.as_ref() {
            let self_d = t.diameter;
            let length = t.length;
            let top_d = if is_root {
                targets.push(("top".into(), self_d, Vec3::ZERO));
                self_d
            } else {
                let inherited = nodes.get("top").map(|n| n.diameter).unwrap_or(self_d);
                if (self_d - inherited).abs() > f32::EPSILON
                    && let Some(m) = tank.as_mut()
                {
                    m.diameter = inherited;
                }
                inherited
            };
            targets.push(("bottom".into(), top_d, Vec3::new(0.0, -length, 0.0)));
        }
        if let Some(e) = engine {
            targets.push(("top".into(), e.diameter, Vec3::ZERO));
        }
        let needs_update = targets.iter().any(|(id, d, off)| {
            nodes
                .get(id)
                .map(|n| {
                    (n.diameter - *d).abs() > f32::EPSILON
                        || n.offset.distance_squared(*off) > f32::EPSILON
                })
                .unwrap_or(false)
        });
        if !needs_update {
            continue;
        }
        for (id, d, off) in &targets {
            if let Some(n) = nodes.nodes.get_mut(id) {
                n.diameter = *d;
                n.offset = *off;
            }
        }
    }
}

fn pointer_over_egui(contexts: &mut EguiContexts) -> bool {
    contexts
        .ctx_mut()
        .map(|c| c.is_pointer_over_area())
        .unwrap_or(false)
}

fn gate_shipyard_input_sources(
    mut action_sources: ResMut<ActionSources>,
    mut contexts: EguiContexts,
) {
    let (pointer_busy, keyboard_busy) = contexts
        .ctx_mut()
        .map(|ctx| {
            (
                ctx.is_pointer_over_area() || ctx.wants_pointer_input(),
                ctx.wants_keyboard_input(),
            )
        })
        .unwrap_or((false, false));
    thalos_input::gating::set_mouse_sources(&mut action_sources, !pointer_busy);
    thalos_input::gating::set_keyboard_source(&mut action_sources, !keyboard_busy);
}

// ---------------------------------------------------------------------------
// Tank resize arrow (parametric handle)
// ---------------------------------------------------------------------------

/// Spawn a single resize arrow per fuel tank on creation. The arrow is a
/// child of the tank entity, hidden until the tank becomes the current
/// selection, and positioned each frame by `update_tank_resize_arrow`.
fn spawn_tank_resize_arrow(
    mut commands: Commands,
    assets: Res<EditorAssets>,
    new_tanks: Query<Entity, Added<FuelTank>>,
) {
    for tank in new_tanks.iter() {
        let arrow = commands
            .spawn((
                Mesh3d(assets.resize_arrow_mesh.clone()),
                MeshMaterial3d(assets.resize_arrow_material.clone()),
                Transform::default(),
                Visibility::Hidden,
                TankResizeArrow { tank },
                Pickable::default(),
            ))
            .observe(on_arrow_drag_start)
            .observe(on_arrow_drag_end)
            .id();
        commands.entity(tank).add_child(arrow);
    }
}

/// Show the arrow only while the owning tank is selected; each frame, place
/// it on the camera-facing side of the tank at mid-height with the tip
/// pointing down along the tank's growth axis.
fn update_tank_resize_arrow(
    state: Res<EditorState>,
    orientation: Res<BuildOrientation>,
    tanks: Query<(&FuelTank, &AttachNodes), Without<TankResizeArrow>>,
    cameras: Query<
        &Transform,
        (
            With<OrbitCamera>,
            Without<TankResizeArrow>,
            Without<FuelTank>,
        ),
    >,
    mut arrows: Query<(&TankResizeArrow, &mut Transform, &mut Visibility)>,
) {
    let Ok(cam_transform) = cameras.single() else {
        return;
    };

    for (arrow, mut transform, mut vis) in arrows.iter_mut() {
        let is_selected = state.selected == Some(arrow.tank);
        let Ok((tank, nodes)) = tanks.get(arrow.tank) else {
            if *vis != Visibility::Hidden {
                *vis = Visibility::Hidden;
            }
            continue;
        };

        if !is_selected {
            if *vis != Visibility::Hidden {
                *vis = Visibility::Hidden;
            }
            continue;
        }
        if *vis != Visibility::Inherited {
            *vis = Visibility::Inherited;
        }

        // Place the arrow on the camera's right so it doesn't occlude the
        // tank body. The arrow is a child of the (possibly laid-down) tank,
        // so convert the world camera-right into the tank's build frame
        // before using it as a local offset; the length axis stays local −Y.
        let cam_right = orientation.rotation().inverse() * cam_transform.right().as_vec3();
        let right_xz = Vec2::new(cam_right.x, cam_right.z)
            .try_normalize()
            .unwrap_or(Vec2::X);
        let radius = nodes.get("top").map(|n| n.diameter * 0.5).unwrap_or(0.5);
        let offset_r = radius + 0.55;
        transform.translation = Vec3::new(
            right_xz.x * offset_r,
            -tank.length * 0.5,
            right_xz.y * offset_r,
        );
        // Bevy's Cone has its tip at +Y and base at -Y; rotate PI around X
        // to point the tip down (i.e., the direction the tank grows). Local,
        // so it composes with the tank's build-layout rotation.
        transform.rotation = Quat::from_rotation_x(std::f32::consts::PI);
    }
}

/// On drag start: snapshot the tank's current length, the cursor origin,
/// and project the world growth axis (-Y) into screen space. Subsequent
/// cursor motion is decomposed along that axis and rescaled to world units.
fn on_arrow_drag_start(
    trigger: On<Pointer<DragStart>>,
    arrows: Query<&TankResizeArrow>,
    tanks: Query<(&FuelTank, &Transform)>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    orientation: Res<BuildOrientation>,
    mut drag: ResMut<TankResizeDrag>,
) {
    let event = trigger.event();
    let Ok(arrow) = arrows.get(event.entity) else {
        return;
    };
    let Ok((tank, tank_transform)) = tanks.get(arrow.tank) else {
        return;
    };
    let Ok((camera, cam_transform)) = camera_q.single() else {
        return;
    };

    let origin_world = tank_transform.translation;
    // Tanks grow along the body −Y axis, which the build layout may have
    // laid down — use the rotated axis so the drag tracks the visible length.
    let grow_world = origin_world + orientation.rotation() * Vec3::NEG_Y;
    let Ok(origin_screen) = camera.world_to_viewport(cam_transform, origin_world) else {
        return;
    };
    let Ok(grow_screen) = camera.world_to_viewport(cam_transform, grow_world) else {
        return;
    };

    let axis = grow_screen - origin_screen;
    let axis_len = axis.length();
    if axis_len < 1e-3 {
        return;
    }

    drag.active = Some(TankDragState {
        tank: arrow.tank,
        start_length: tank.length,
        start_cursor: event.pointer_location.position,
        screen_axis: axis / axis_len,
        world_per_pixel: 1.0 / axis_len,
    });
}

fn on_arrow_drag_end(_trigger: On<Pointer<DragEnd>>, mut drag: ResMut<TankResizeDrag>) {
    drag.active = None;
}

/// Apply the active drag to the tank's length each frame. Bails (and
/// clears) if the button was released without a DragEnd — can happen when
/// the pointer leaves the window mid-drag.
fn update_tank_resize_drag(
    mut drag: ResMut<TankResizeDrag>,
    windows: Query<&Window, With<PrimaryWindow>>,
    input: Res<ShipyardInputIntent>,
    mut tanks: Query<(&mut FuelTank, &AttachNodes)>,
) {
    let Some(state) = drag.active.as_ref() else {
        return;
    };
    if !input.primary_pressed {
        drag.active = None;
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Some(cursor) = window.cursor_position() else {
        return;
    };

    let cursor_delta = cursor - state.start_cursor;
    let pixels_along = cursor_delta.dot(state.screen_axis);
    let world_growth = pixels_along * state.world_per_pixel;
    let raw_length = state.start_length + world_growth;
    // Magnetic snap: smooth drag in-between, stick to nearest 0.5 within
    // a small neighborhood so users can dial in round values without
    // losing fine control.
    const SNAP_GRID: f32 = 0.5;
    const SNAP_THRESHOLD: f32 = 0.06;
    const MAX_LENGTH_OVER_DIAMETER: f32 = 8.0;
    let nearest = (raw_length / SNAP_GRID).round() * SNAP_GRID;
    let length = if (raw_length - nearest).abs() < SNAP_THRESHOLD {
        nearest
    } else {
        raw_length
    };

    if let Ok((mut tank, nodes)) = tanks.get_mut(state.tank) {
        let diameter = nodes
            .get("top")
            .map(|n| n.diameter)
            .unwrap_or(tank.diameter);
        let new_length = length.clamp(0.5, MAX_LENGTH_OVER_DIAMETER * diameter);
        if (tank.length - new_length).abs() > f32::EPSILON {
            tank.length = new_length;
        }
    }
}

/// Clear selection when the user clicks on empty space. Tracks the
/// press cursor so a camera orbit (press → drag → release) doesn't
/// deselect at release.
fn deselect_on_empty_click(
    mut tracker: ResMut<DeselectTracker>,
    input: Res<ShipyardInputIntent>,
    windows: Query<&Window, With<PrimaryWindow>>,
    hover_map: Res<HoverMap>,
    pickables: Query<(), Or<(With<PartBody>, With<AttachNodePin>, With<TankResizeArrow>)>>,
    mut state: ResMut<EditorState>,
    mut contexts: EguiContexts,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let cursor = window.cursor_position();

    if input.primary_started {
        if pointer_over_egui(&mut contexts) {
            tracker.press_cursor = None;
        } else {
            let on_pickable = hover_map
                .0
                .values()
                .any(|hovers| hovers.keys().any(|e| pickables.get(*e).is_ok()));
            tracker.press_cursor = if on_pickable { None } else { cursor };
        }
    }

    if input.primary_released
        && let (Some(press), Some(current)) = (tracker.press_cursor.take(), cursor)
        && (current - press).length() < CLICK_THRESHOLD_PX
    {
        state.selected = None;
    }
}

/// Keep `ShipPartMaterial` uniforms in sync with the part's dimensions
/// (tank length, decoupler/tank radius). Triggered whenever the
/// kind-component or attach nodes change, so slider and resize-drag
/// updates flow through to the panel / rivet layout live.
fn update_part_shader_params(
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    parts: Query<
        (
            &AttachNodes,
            &PartShaderHandle,
            Option<&FuelTank>,
            Option<&Fuselage>,
            Option<&Decoupler>,
            Option<&Adapter>,
        ),
        Or<(
            Changed<FuelTank>,
            Changed<Fuselage>,
            Changed<Decoupler>,
            Changed<Adapter>,
            Changed<AttachNodes>,
        )>,
    >,
) {
    for (nodes, handle, tank, fuselage, dec, adapter) in parts.iter() {
        let Some(mat) = ship_materials.get_mut(&handle.0) else {
            continue;
        };
        let params = ship_part_params(nodes, tank, fuselage, dec, adapter, mat.extension.params.seed);
        mat.extension.params.length = params.length;
        mat.extension.params.radius_top = params.radius_top;
        mat.extension.params.radius_bottom = params.radius_bottom;
    }
}

/// Selection / hover tint for parts rendering through `ShipPartMaterial`
/// (tanks, decouplers). Writes into the material's tint uniform rather
/// than swapping handles so each part keeps its procedural detail.
/// Shrouds are excluded — they manage their own hover feedback via
/// `update_shroud_transparency`.
fn update_part_shader_highlight(
    state: Res<EditorState>,
    hover_map: Res<HoverMap>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    bodies: Query<(Entity, &PartBody, &MeshMaterial3d<ShipPartMaterial>), Without<ShroudBody>>,
) {
    let hovered: HashSet<Entity> = hover_map
        .0
        .values()
        .flat_map(|hovers| hovers.keys().copied())
        .collect();

    for (body_entity, body, mesh_mat) in bodies.iter() {
        let target = if Some(body.0) == state.selected {
            Vec3::new(0.88, 1.0, 1.35)
        } else if hovered.contains(&body_entity) {
            Vec3::new(1.08, 1.08, 1.12)
        } else {
            Vec3::ONE
        };
        if let Some(mat) = ship_materials.get_mut(&mesh_mat.0)
            && (mat.extension.params.tint - target).length_squared() > 1.0e-6
        {
            mat.extension.params.tint = target;
        }
    }
}

/// Swap each part body's material based on selection and hover state.
/// Priority: selected > hovered > default.
fn update_selection_highlight(
    state: Res<EditorState>,
    assets: Res<EditorAssets>,
    hover_map: Res<HoverMap>,
    mut bodies: Query<(
        Entity,
        &PartBody,
        Has<GearVisual>,
        &mut MeshMaterial3d<StandardMaterial>,
    )>,
) {
    let hovered: HashSet<Entity> = hover_map
        .0
        .values()
        .flat_map(|hovers| hovers.keys().copied())
        .collect();

    for (body_entity, body, is_gear, mut mat) in bodies.iter_mut() {
        let target = if Some(body.0) == state.selected {
            &assets.selected_material
        } else if hovered.contains(&body_entity) {
            &assets.hover_material
        } else if is_gear {
            &assets.gear_material
        } else {
            &assets.part_material
        };
        if mat.0.id() != target.id() {
            mat.0 = target.clone();
        }
    }
}

fn on_body_click(
    click: On<Pointer<Click>>,
    bodies: Query<&PartBody>,
    wings: Query<(), With<Wing>>,
    catalog: Res<PartCatalog>,
    mut state: ResMut<EditorState>,
    mut contexts: EguiContexts,
) {
    if pointer_over_egui(&mut contexts) {
        return;
    }
    let Ok(body) = bodies.get(click.entity) else {
        return;
    };
    let pending_surface_kind = state.pending.as_ref().and_then(|pending| {
        // Wings and landing gear both body-skin-mount on a hull (not a wing).
        if matches!(pending.params, PartParams::Wing { .. } | PartParams::Gear { .. })
            && wings.get(body.0).is_err()
        {
            return Some(SurfaceMountKind::BodySkin);
        }
        let entry = catalog.resolve(&pending.catalog_id).ok()?;
        match entry {
            CatalogEntry::Engine(e)
                if e.geometry == EngineGeometry::JetNacelle && wings.get(body.0).is_ok() =>
            {
                Some(SurfaceMountKind::WingPylon)
            }
            _ => None,
        }
    });
    if let Some(kind) = pending_surface_kind {
        if let Some(pos) = click.hit.position {
            state.place_surface_at = Some((body.0, pos, kind));
        }
        return;
    }
    if state.pending.is_some() {
        state.status = "Pick a compatible surface or attach node for the pending part".into();
        return;
    }
    state.selected = Some(body.0);
}

fn on_pin_click(
    click: On<Pointer<Click>>,
    pins: Query<&AttachNodePin>,
    mut state: ResMut<EditorState>,
    mut contexts: EguiContexts,
) {
    if pointer_over_egui(&mut contexts) {
        return;
    }
    if let Ok(pin) = pins.get(click.entity) {
        if state.pending.is_some() {
            state.place_at = Some((pin.part, pin.node_id.clone()));
        } else {
            state.selected = Some(pin.part);
        }
    }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

fn orbit_camera(
    mut cam: Query<(&mut Transform, &mut OrbitCamera)>,
    input: Res<ShipyardInputIntent>,
    mut pinch: MessageReader<PinchGesture>,
    mut contexts: EguiContexts,
    state: Res<EditorState>,
    resize_drag: Res<TankResizeDrag>,
    hover_map: Res<HoverMap>,
    orientation: Res<BuildOrientation>,
    arrows: Query<(), With<TankResizeArrow>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut press_cursor: Local<Option<Vec2>>,
    mut orbit_active: Local<bool>,
) {
    let pointer_over_egui = contexts
        .ctx_mut()
        .map(|c| c.is_pointer_over_area() || c.wants_pointer_input())
        .unwrap_or(false);

    let pointer_on_arrow = hover_map
        .0
        .values()
        .any(|hovers| hovers.keys().any(|e| arrows.get(*e).is_ok()));

    let delta = input.camera_motion;
    let wheel = input.camera_wheel;
    let mut pinch_d: f32 = 0.0;
    for p in pinch.read() {
        pinch_d += p.0;
    }

    let shift = input.precision_slow;

    // Click/drag arbitration for LMB: we want a press→release on a pin to
    // fire `Pointer<Click>`, which Bevy's picking only emits when the same
    // entity is hovered at press and at release. Rotating the camera mid-
    // press moves the world under the cursor and breaks that. So while a
    // part is pending we hold orbit until the cursor has moved past
    // CLICK_THRESHOLD_PX from the press location; once over, we stay in
    // orbit mode for the remainder of the press. With no pending part
    // there's no click target to protect, so orbit is unconditional.
    let cursor = windows.single().ok().and_then(|w| w.cursor_position());
    if input.primary_started {
        *press_cursor = cursor;
        *orbit_active = false;
    }
    if input.primary_released {
        *press_cursor = None;
        *orbit_active = false;
    }
    if !*orbit_active
        && let (Some(press), Some(current)) = (*press_cursor, cursor)
        && (current - press).length() >= CLICK_THRESHOLD_PX
    {
        *orbit_active = true;
    }

    // Also suppress while the pointer is over a resize arrow (or actively
    // dragging one) so the camera doesn't twitch between mouse-down and
    // DragStart firing.
    let orbit_allowed = !pointer_over_egui
        && resize_drag.active.is_none()
        && !pointer_on_arrow
        && (state.pending.is_none() || *orbit_active);

    for (mut t, mut orbit) in cam.iter_mut() {
        if orbit_allowed && input.primary_pressed {
            orbit.yaw -= delta.x * 0.005;
            orbit.pitch = (orbit.pitch - delta.y * 0.005).clamp(-1.5, 1.5);
        }

        if !pointer_over_egui && (wheel.x.abs() > 0.0 || wheel.y.abs() > 0.0) {
            if shift {
                orbit.distance = (orbit.distance * (1.0 - wheel.y * 0.05)).clamp(2.0, 200.0);
            } else {
                // Vertical scroll: pan along the build's long axis (body +Y),
                // which the horizontal layout lays down to −Z — so scrolling
                // tracks the fuselage in either layout instead of always world-up.
                if wheel.y.abs() > 0.0 {
                    let pan = wheel.y * orbit.distance * 0.015;
                    orbit.focus += orientation.rotation() * Vec3::Y * pan;
                }
                // Horizontal scroll (trackpad two-finger): pan perpendicular to
                // the build axis using the camera's current azimuth so left/right
                // always matches what's on screen regardless of orbit angle.
                if wheel.x.abs() > 0.0 {
                    let cam_right = Quat::from_rotation_y(orbit.yaw) * Vec3::X;
                    let pan = wheel.x * orbit.distance * 0.015;
                    orbit.focus += cam_right * pan;
                }
            }
        }

        // Trackpad pinch zooms regardless of shift.
        if !pointer_over_egui && pinch_d.abs() > 0.0 {
            orbit.distance = (orbit.distance * (1.0 - pinch_d * 8.0)).clamp(2.0, 200.0);
        }

        let rot = Quat::from_euler(EulerRot::YXZ, orbit.yaw, -orbit.pitch, 0.0);
        let offset = rot * Vec3::new(0.0, 0.0, orbit.distance);
        t.translation = orbit.focus + offset;
        t.look_at(orbit.focus, Vec3::Y);
    }
}

// ---------------------------------------------------------------------------
// Blueprint <-> ECS
// ---------------------------------------------------------------------------

type CollectQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static CatalogRef,
        &'static PartResources,
        Option<&'static Decoupler>,
        Option<&'static Adapter>,
        Option<&'static FuelTank>,
        Option<&'static Fuselage>,
        Option<&'static Wing>,
        Option<&'static Gear>,
    ),
>;

fn collect_blueprint(
    ship: &Ship,
    parts: &CollectQuery,
    attachments: &Query<(Entity, &Attachment)>,
    surface_mounts: &Query<(Entity, &SurfaceMount)>,
    groups: &Query<(Entity, &SymmetryGroup)>,
) -> Option<ShipBlueprint> {
    // Child graph unions both placement kinds so wings are ordered and
    // indexed alongside the node-stacked hull.
    let mut child_map: HashMap<Entity, Vec<Entity>> = HashMap::new();
    for (e, att) in attachments.iter() {
        child_map.entry(att.parent).or_default().push(e);
    }
    for (e, sm) in surface_mounts.iter() {
        child_map.entry(sm.parent).or_default().push(e);
    }

    let mut ordered: Vec<Entity> = Vec::new();
    let mut queue: VecDeque<Entity> = VecDeque::from([ship.root]);
    while let Some(e) = queue.pop_front() {
        ordered.push(e);
        if let Some(kids) = child_map.get(&e) {
            for c in kids {
                queue.push_back(*c);
            }
        }
    }

    let idx: HashMap<Entity, usize> = ordered.iter().enumerate().map(|(i, e)| (*e, i)).collect();

    let mut part_blueprints = Vec::with_capacity(ordered.len());
    for e in &ordered {
        let (_, cat_ref, res, dec, adapter, tank, fuselage, wing, gear) = parts.get(*e).ok()?;
        let params = if let Some(d) = dec {
            PartParams::Decoupler {
                diameter: d.diameter,
            }
        } else if let Some(a) = adapter {
            PartParams::Adapter {
                diameter: a.diameter,
                target_diameter: a.target_diameter,
            }
        } else if let Some(t) = tank {
            PartParams::Tank {
                diameter: t.diameter,
                length: t.length,
            }
        } else if let Some(f) = fuselage {
            PartParams::Fuselage {
                length: f.length,
                max_width: f.max_width,
                max_height: f.max_height,
                roundness: f.roundness,
                nose_fraction: f.nose_fraction,
                nose_bluntness: f.nose_bluntness,
                tail_fraction: f.tail_fraction,
                nose_droop: f.nose_droop,
                tail_upsweep: f.tail_upsweep,
                tail_tip_diameter: f.tail_tip_diameter,
                tail_bluntness: f.tail_bluntness,
            }
        } else if let Some(w) = wing {
            PartParams::Wing {
                span: w.span,
                root_chord: w.root_chord,
                tip_chord: w.tip_chord,
                sweep: w.sweep,
                dihedral: w.dihedral,
                thickness: w.thickness,
                incidence: w.incidence,
            }
        } else if let Some(g) = gear {
            PartParams::Gear {
                strut_length: g.strut_length,
                wheel_radius: g.wheel_radius,
            }
        } else {
            // Pods and engines carry no per-instance params.
            PartParams::None
        };
        // Persist amounts only — capacities are recomputed from the
        // catalog at load time.
        let resources: HashMap<thalos_shipyard::Resource, f32> =
            res.pools.iter().map(|(r, p)| (*r, p.amount)).collect();
        part_blueprints.push(PartBlueprint {
            catalog_id: cat_ref.id.clone(),
            params,
            resources: Some(resources),
        });
    }

    let mut connections = Vec::new();
    for (e, att) in attachments.iter() {
        if let (Some(&ci), Some(&pi)) = (idx.get(&e), idx.get(&att.parent)) {
            connections.push(Connection {
                parent: pi,
                parent_node: att.parent_node.clone(),
                child: ci,
                child_node: att.my_node.clone(),
            });
        }
    }

    let mut surface = Vec::new();
    for (e, sm) in surface_mounts.iter() {
        if let (Some(&ci), Some(&pi)) = (idx.get(&e), idx.get(&sm.parent)) {
            surface.push(SurfaceConnection {
                parent: pi,
                child: ci,
                kind: sm.kind,
                station: sm.station,
                angle: sm.angle,
                symmetry_group: groups.get(e).ok().map(|(_, g)| g.id),
            });
        }
    }

    Some(ShipBlueprint {
        name: schema_ship_name(&ship.name),
        root: 0,
        parts: part_blueprints,
        connections,
        surface_mounts: surface,
    })
}

/// The symmetry-group members of `host`, primary first, or `[host]` if the
/// host isn't part of a group. Used to stamp a footprint part onto every
/// counterpart of a symmetric host (KSP nested symmetry — a nacelle on a
/// mirrored wing lands on both wings).
fn host_group_members(host: Entity, groups: &Query<(Entity, &SymmetryGroup)>) -> Vec<Entity> {
    let Ok((_, hg)) = groups.get(host) else {
        return vec![host];
    };
    let gid = hg.id;
    let mut primary = None;
    let mut mirrors = Vec::new();
    for (e, g) in groups.iter() {
        if g.id == gid {
            match g.role {
                SymmetryRole::Primary => primary = Some(e),
                SymmetryRole::Mirror => mirrors.push(e),
            }
        }
    }
    let mut out: Vec<Entity> = primary.into_iter().collect();
    out.extend(mirrors);
    if out.is_empty() { vec![host] } else { out }
}

/// The entity whose params the inspector should edit for a given selection:
/// the selection's symmetry-group **primary** if it belongs to a group, else
/// the selection itself. `sync_symmetry_groups` copies the primary onto its
/// mirror counterparts every frame, so editing a counterpart directly would be
/// reverted next frame — its inspector sliders would look dead. KSP-style, an
/// edit on any member is applied to the controlling (primary) part and the
/// mirrors follow.
fn symmetry_edit_target(sel: Entity, groups: &Query<(Entity, &SymmetryGroup)>) -> Entity {
    let Ok((_, sg)) = groups.get(sel) else {
        return sel;
    };
    match sg.role {
        SymmetryRole::Primary => sel,
        SymmetryRole::Mirror => groups
            .iter()
            .find(|(_, g)| g.id == sg.id && g.role == SymmetryRole::Primary)
            .map(|(e, _)| e)
            .unwrap_or(sel),
    }
}

/// Resolve a body-skin (cylinder) hit into a `(station, angle)` pair, with
/// optional magnetic angle snapping. Shared by the commit path
/// ([`surface_mount_from_hit`]) and the live placement preview so the ghost and
/// the placed part land at exactly the same spot.
fn body_skin_mount(
    parent: Entity,
    world_pos: Vec3,
    part_transforms: &Query<&Transform, With<Part>>,
    host_nodes: &Query<&AttachNodes>,
    orientation: &BuildOrientation,
    snap: bool,
) -> (f32, f32) {
    let parent_t = part_transforms
        .get(parent)
        .map(|t| t.translation)
        .unwrap_or(Vec3::ZERO);
    // Undo the build-layout rotation so the hit lands in the upright build
    // frame, where all persisted surface coordinates are defined.
    let local = orientation.rotation().inverse() * (world_pos - parent_t);
    let host = host_nodes.get(parent).ok();
    let radius = host
        .and_then(|n| n.get("top").map(|nd| nd.diameter * 0.5))
        .unwrap_or(1.0);
    let height = host
        .and_then(|n| n.get("bottom").map(|nd| -nd.offset.y))
        .unwrap_or(radius * 2.0);
    let station = if height > 0.0 {
        (-local.y / height).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let mut angle = local.x.atan2(local.z);
    if snap {
        angle = snap_body_skin_angle(angle);
    }
    (station, angle)
}

/// Resolve a hull/wing surface hit into the persisted `(station, angle)`
/// pair for `kind`. Symmetry is no longer decided here — the global
/// [`SymmetryMode`] + the host's own symmetry drive group stamping at the
/// call site. `snap` magnetically rounds the azimuth of body-skin mounts.
fn surface_mount_from_hit(
    kind: SurfaceMountKind,
    parent: Entity,
    world_pos: Vec3,
    part_transforms: &Query<&Transform, With<Part>>,
    host_nodes: &Query<&AttachNodes>,
    surface_mounts: &Query<(Entity, &SurfaceMount)>,
    wings: &Query<&Wing>,
    orientation: &BuildOrientation,
    snap: bool,
) -> Option<(f32, f32, String)> {
    let parent_t = part_transforms
        .get(parent)
        .map(|t| t.translation)
        .unwrap_or(Vec3::ZERO);
    // Undo the build-layout rotation so the hit lands in the upright build
    // frame, where all persisted surface coordinates are defined.
    let local = orientation.rotation().inverse() * (world_pos - parent_t);

    match kind {
        SurfaceMountKind::BodySkin => {
            let (station, angle) =
                body_skin_mount(parent, world_pos, part_transforms, host_nodes, orientation, snap);
            Some((station, angle, "Mounted wing".into()))
        }
        SurfaceMountKind::WingPylon => {
            let wing = wings.get(parent).ok()?;
            let (_, wing_mount) = surface_mounts.iter().find(|(e, _)| *e == parent)?;
            let parent_radius = host_nodes
                .get(wing_mount.parent)
                .ok()
                .and_then(|n| n.get("top").map(|nd| nd.diameter * 0.5))
                .unwrap_or(1.0);
            // The click is on a specific wing entity; project it onto that
            // wing's own panel frame.
            let frame = wing_panel_frame(wing, wing_mount.angle, parent_radius);
            let span_axis = frame.tip_center - frame.root_center;
            let span_len2 = span_axis.length_squared();
            let station = if span_len2 > f32::EPSILON {
                ((local - frame.root_center).dot(span_axis) / span_len2).clamp(0.08, 0.92)
            } else {
                0.5
            };
            let chord = frame.chord_at(wing, station).max(0.1);
            let chord_center = frame.center_at(station);
            let chord_fraction =
                ((local - chord_center).dot(frame.fore_dir) / chord).clamp(-0.4, 0.4);
            Some((station, chord_fraction, "Mounted jet nacelle with pylon".into()))
        }
    }
}

/// Live placement preview. While a body-skin footprint part (gear / wing) is
/// pending and the cursor hovers a compatible hull, draw a translucent ghost of
/// it at the snapped `(station, angle)` it would land on — so you aim instead of
/// clicking and praying. The ghost is one reused entity; its (small) mesh is
/// rebuilt only when the snapped angle / host / params change. WingPylon
/// (nacelle-on-wing) mounts aren't cylinder mounts, so they get no preview yet.
fn update_placement_preview(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut contexts: EguiContexts,
    state: Res<EditorState>,
    catalog: Res<PartCatalog>,
    snap: Res<PlacementSnap>,
    orientation: Res<BuildOrientation>,
    assets: Res<EditorAssets>,
    hover_map: Res<HoverMap>,
    mut preview: ResMut<PlacementPreview>,
    bodies: Query<&PartBody>,
    wing_marker: Query<(), With<Wing>>,
    part_transforms: Query<&Transform, With<Part>>,
    host_nodes: Query<&AttachNodes>,
    hosts: Query<&Fuselage>,
    mut ghosts: Query<
        (&mut Transform, &mut Visibility, &mut Mesh3d),
        (With<PreviewGhost>, Without<Part>),
    >,
) {
    // Compute the desired ghost placement (None ⇒ hide it). An IIFE so the many
    // early-outs read cleanly; its borrows end before we mutate `preview`.
    let placement: Option<(Vec3, Quat, PreviewSig, Option<Handle<Mesh>>)> = (|| {
        let pending = state.pending.as_ref()?;
        // Only body-skin footprint parts get a preview (gear, wings).
        if !matches!(pending.params, PartParams::Wing { .. } | PartParams::Gear { .. }) {
            return None;
        }
        if pointer_over_egui(&mut contexts) {
            return None;
        }

        // Hovered host = a non-wing body under the cursor, with its hit point.
        let mut found: Option<(Entity, Vec3)> = None;
        'outer: for hits in hover_map.0.values() {
            for (hovered, data) in hits.iter() {
                let Ok(pb) = bodies.get(*hovered) else {
                    continue;
                };
                if wing_marker.get(pb.0).is_ok() {
                    continue; // gear/wing mount on a hull, not on a wing
                }
                let Some(pos) = data.position else {
                    continue;
                };
                found = Some((pb.0, pos));
                break 'outer;
            }
        }
        let (host, hit) = found?;

        let (station, angle) = body_skin_mount(
            host,
            hit,
            &part_transforms,
            &host_nodes,
            &orientation,
            snap.enabled,
        );
        let host_n = host_nodes.get(host).ok();
        let top_d = host_top_diameter(&host_nodes, host);
        let (parent_radius, v_offset) =
            host_mount_geometry(hosts.get(host).ok(), top_d, station, angle);
        let host_height = host_n
            .and_then(|n| n.get("bottom").map(|nd| -nd.offset.y))
            .unwrap_or(0.0);
        let host_t = part_transforms
            .get(host)
            .map(|t| t.translation)
            .unwrap_or(Vec3::ZERO);

        // Match where `update_part_transforms` puts a body-skin part: the host
        // (already post-layout-rotation) plus the rotated local station offset
        // (including the loft's centerline upsweep along +Z).
        let r = orientation.rotation();
        let translation = host_t + r * Vec3::new(0.0, -station * host_height, v_offset);

        let sig = PreviewSig {
            host,
            angle,
            parent_radius,
            params: pending.params.clone(),
        };
        // Rebuild the mesh only when the silhouette actually changed (or there's
        // no ghost entity to carry it yet).
        let mesh = if preview.sig.as_ref() != Some(&sig) || preview.entity.is_none() {
            let m = match &pending.params {
                PartParams::Gear {
                    strut_length,
                    wheel_radius,
                } => {
                    let (track_fraction, wheels_per_leg) = catalog
                        .resolve(&pending.catalog_id)
                        .ok()
                        .and_then(|e| match e {
                            CatalogEntry::Gear(g) => Some((g.track_fraction, g.wheels_per_leg)),
                            _ => None,
                        })
                        .unwrap_or((0.0, 1));
                    let g = Gear {
                        strut_length: *strut_length,
                        wheel_radius: *wheel_radius,
                        track_fraction,
                        wheels_per_leg,
                        dry_mass: 0.0,
                    };
                    build_gear_mesh(&g, angle, parent_radius)
                }
                PartParams::Wing {
                    span,
                    root_chord,
                    tip_chord,
                    sweep,
                    dihedral,
                    thickness,
                    incidence,
                } => {
                    let w = Wing {
                        span: *span,
                        root_chord: *root_chord,
                        tip_chord: *tip_chord,
                        sweep: *sweep,
                        dihedral: *dihedral,
                        thickness: *thickness,
                        incidence: *incidence,
                        dry_mass: 0.0,
                    };
                    build_wing_mesh(&w, angle, parent_radius)
                }
                _ => return None,
            };
            Some(meshes.add(m))
        } else {
            None
        };

        Some((translation, r, sig, mesh))
    })();

    match placement {
        None => {
            if let Some(prev) = preview.entity
                && let Ok((_, mut vis, _)) = ghosts.get_mut(prev)
            {
                *vis = Visibility::Hidden;
            }
            preview.sig = None;
        }
        Some((translation, rotation, sig, mesh)) => {
            let mut updated = false;
            if let Some(prev) = preview.entity
                && let Ok((mut t, mut vis, mut mesh3d)) = ghosts.get_mut(prev)
            {
                t.translation = translation;
                t.rotation = rotation;
                *vis = Visibility::Visible;
                if let Some(h) = &mesh {
                    mesh3d.0 = h.clone();
                }
                updated = true;
            }
            if !updated && let Some(h) = mesh {
                let id = commands
                    .spawn((
                        Mesh3d(h),
                        MeshMaterial3d(assets.preview_material.clone()),
                        Transform::from_translation(translation).with_rotation(rotation),
                        Visibility::Visible,
                        PreviewGhost,
                        Pickable::IGNORE,
                    ))
                    .id();
                preview.entity = Some(id);
            }
            preview.sig = Some(sig);
        }
    }
}

// ---------------------------------------------------------------------------
// Command processing (save / load / delete / place)
// ---------------------------------------------------------------------------

fn process_commands(
    mut commands: Commands,
    mut state: ResMut<EditorState>,
    mut ships: Query<&mut Ship>,
    parts_q: CollectQuery,
    attachments: Query<(Entity, &Attachment)>,
    surface_mounts: Query<(Entity, &SurfaceMount)>,
    part_transforms: Query<&Transform, With<Part>>,
    host_nodes: Query<&AttachNodes>,
    wings: Query<&Wing>,
    groups: Query<(Entity, &SymmetryGroup)>,
    orientation: Res<BuildOrientation>,
    modes: PlacementModes,
    mut next_sym_id: ResMut<NextSymmetryId>,
    all_parts: Query<Entity, With<Part>>,
    all_ships: Query<Entity, With<Ship>>,
    catalog: Res<PartCatalog>,
) {
    // ---- Save ---------------------------------------------------------
    if state.save_requested {
        state.save_requested = false;
        if let Some(ship_entity) = state.ship_entity
            && let Ok(ship) = ships.get(ship_entity)
        {
            match collect_blueprint(ship, &parts_q, &attachments, &surface_mounts, &groups) {
                Some(bp) => match bp.to_ron() {
                    Ok(text) => {
                        let path = ship_path_for_name(&bp.name);
                        if let Err(e) = std::fs::create_dir_all(SHIPS_DIR) {
                            state.status = format!("mkdir failed: {e}");
                        } else {
                            match std::fs::write(&path, text) {
                                Ok(()) => {
                                    state.status = format!("Saved {}", path.display());
                                    state.refresh_list = true;
                                }
                                Err(e) => state.status = format!("Save failed: {e}"),
                            }
                        }
                    }
                    Err(e) => state.status = format!("Serialize failed: {e}"),
                },
                None => state.status = "Failed to collect blueprint".into(),
            }
        }
    }

    // ---- Load ---------------------------------------------------------
    if let Some(slug) = state.load_target.take() {
        let path = ship_path_for_slug(&slug);
        match std::fs::read_to_string(&path) {
            Ok(text) => match ShipBlueprint::from_ron(&text) {
                Ok(bp) => {
                    for e in all_parts.iter() {
                        commands.entity(e).despawn();
                    }
                    for e in all_ships.iter() {
                        commands.entity(e).despawn();
                    }
                    state.ship_root = None;
                    state.ship_entity = None;
                    state.selected = None;

                    let path_disp = path.display().to_string();
                    commands.queue(move |world: &mut World| {
                        let catalog = world.resource::<PartCatalog>().clone();
                        let mut cmds = world.commands();
                        let ship_entity = match bp.spawn(&mut cmds, &catalog) {
                            Ok(e) => e,
                            Err(err) => {
                                let mut st = world.resource_mut::<EditorState>();
                                st.status = format!("Spawn failed: {err}");
                                return;
                            }
                        };
                        world.flush();
                        let (root, name) = world
                            .get::<Ship>(ship_entity)
                            .map(|s| (Some(s.root), s.name.clone()))
                            .unwrap_or((None, String::new()));
                        let mut st = world.resource_mut::<EditorState>();
                        st.ship_entity = Some(ship_entity);
                        st.ship_root = root;
                        st.selected = root;
                        st.ship_name = name;
                        st.status = format!("Loaded {path_disp}");
                    });
                }
                Err(e) => state.status = format!("Parse failed: {e}"),
            },
            Err(e) => state.status = format!("Read failed: {e}"),
        }
    }

    // ---- Delete file --------------------------------------------------
    if let Some(slug) = state.delete_file.take() {
        let path = ship_path_for_slug(&slug);
        match std::fs::remove_file(&path) {
            Ok(()) => {
                state.status = format!("Deleted {}", path.display());
                state.refresh_list = true;
            }
            Err(e) => state.status = format!("Delete failed: {e}"),
        }
    }

    // ---- Refresh list -------------------------------------------------
    if state.refresh_list {
        state.refresh_list = false;
        state.ship_list = list_ships();
    }

    // ---- Delete selected ---------------------------------------------
    // Deleting the root clears the whole canvas (despawns ship + all
    // parts). Deleting a non-root part despawns its subtree.
    if state.delete_selected {
        state.delete_selected = false;
        if let Some(sel) = state.selected {
            if Some(sel) == state.ship_root {
                if let Some(se) = state.ship_entity
                    && let Ok(ship) = ships.get(se)
                {
                    state.ship_name = ship.name.clone();
                }
                for e in all_parts.iter() {
                    commands.entity(e).despawn();
                }
                for e in all_ships.iter() {
                    commands.entity(e).despawn();
                }
                state.ship_root = None;
                state.ship_entity = None;
                state.selected = None;
                state.status = "Cleared canvas".into();
            } else {
                let mut child_map: HashMap<Entity, Vec<Entity>> = HashMap::new();
                for (e, att) in attachments.iter() {
                    child_map.entry(att.parent).or_default().push(e);
                }
                // Surface-mounted wings ride with their host: deleting a
                // fuselage must take its wings too.
                for (e, sm) in surface_mounts.iter() {
                    child_map.entry(sm.parent).or_default().push(e);
                }
                let mut to_remove: Vec<Entity> = Vec::new();
                // Deleting any symmetry-group member deletes the whole group
                // (KSP) — seed the walk with all counterparts of the selection.
                let mut stack = host_group_members(sel, &groups);
                while let Some(e) = stack.pop() {
                    to_remove.push(e);
                    if let Some(kids) = child_map.get(&e) {
                        stack.extend(kids.iter().copied());
                    }
                }
                for e in to_remove {
                    commands.entity(e).despawn();
                }
                state.selected = state.ship_root;
                state.status = "Deleted selection".into();
            }
        }
    }

    // ---- Set selection as root ---------------------------------------
    // Walk from the selection up through Attachment components to the
    // current root; reverse each link by inserting an Attachment on the
    // former parent pointing at the former child, with parent_node /
    // my_node swapped. Parts off the chain keep their attachments, so
    // branches follow their original subtree.
    if state.set_as_root {
        state.set_as_root = false;
        if let Some(sel) = state.selected
            && Some(sel) != state.ship_root
        {
            let att_map: HashMap<Entity, Attachment> =
                attachments.iter().map(|(e, a)| (e, a.clone())).collect();
            let mut chain: Vec<(Entity, Attachment)> = Vec::new();
            let mut current = sel;
            while let Some(att) = att_map.get(&current) {
                chain.push((current, att.clone()));
                current = att.parent;
            }
            commands.entity(sel).remove::<Attachment>();
            for (entity, att) in chain {
                commands.entity(att.parent).insert(Attachment {
                    parent: entity,
                    parent_node: att.my_node,
                    my_node: att.parent_node,
                });
            }
            if let Some(ship_entity) = state.ship_entity
                && let Ok(mut ship) = ships.get_mut(ship_entity)
            {
                ship.root = sel;
            }
            state.ship_root = Some(sel);
            state.status = "Re-rooted ship".into();
        }
    }

    // ---- Place pending part at a given (parent, node) -----------------
    if let Some((parent, node)) = state.place_at.take() {
        let Some(pending) = state.pending.take() else {
            return;
        };
        match ShipBlueprint::spawn_part(
            &mut commands,
            &catalog,
            &pending.catalog_id,
            pending.params,
            None,
        ) {
            Ok(child) => {
                commands.entity(child).insert(Attachment {
                    parent,
                    parent_node: node,
                    my_node: "top".into(),
                });
                state.selected = Some(child);
                state.status = "Placed part".into();
            }
            Err(e) => state.status = format!("Spawn failed: {e}"),
        }
    }

    // ---- Mount pending footprint part on a surface -------------------
    // Body-skin mounts (wings) derive station/azimuth from a hull hit.
    // Wing-pylon mounts (jet nacelles) derive span/chord position from
    // the host wing hit.
    if let Some((parent, world_pos, kind)) = state.place_surface_at.take() {
        if let Some(pending) = state.pending.take() {
            let Some((station, angle, status)) = surface_mount_from_hit(
                kind,
                parent,
                world_pos,
                &part_transforms,
                &host_nodes,
                &surface_mounts,
                &wings,
                &orientation,
                modes.snap.enabled,
            ) else {
                state.status = "Surface placement failed".into();
                state.pending = Some(pending);
                return;
            };

            // Landing gear is a self-contained gearbox — it draws its own legs,
            // so it is *always* a single mount regardless of the Mirror toggle
            // or a (hypothetically) symmetric host. Special-cased before the
            // wing/nacelle symmetry path below.
            let is_gear = matches!(pending.params, PartParams::Gear { .. });

            // KSP symmetry stamping. If the clicked host is itself a mirrored
            // pair (e.g. a nacelle onto a wing), stamp one part per host
            // member — nested symmetry. Otherwise, if mirror mode is on and
            // this is an off-centreline body-skin mount, stamp the reflected
            // pair on the same host. Else a single part.
            let host_members = host_group_members(parent, &groups);
            let stamps: Vec<(Entity, f32, f32)> = if is_gear {
                vec![(parent, station, angle)]
            } else if host_members.len() > 1 {
                host_members.iter().map(|&h| (h, station, angle)).collect()
            } else if modes.symmetry.mirror
                && kind == SurfaceMountKind::BodySkin
                && angle.sin().abs() > 0.3
            {
                vec![(parent, station, angle), (parent, station, -angle)]
            } else {
                vec![(parent, station, angle)]
            };

            let group_id = (stamps.len() > 1).then(|| next_sym_id.next());
            let mut first: Option<Entity> = None;
            for (i, (host, st, ang)) in stamps.iter().enumerate() {
                match ShipBlueprint::spawn_part(
                    &mut commands,
                    &catalog,
                    &pending.catalog_id,
                    pending.params.clone(),
                    None,
                ) {
                    Ok(child) => {
                        let mut ec = commands.entity(child);
                        ec.insert(SurfaceMount {
                            parent: *host,
                            kind,
                            station: *st,
                            angle: *ang,
                        });
                        if let Some(gid) = group_id {
                            let role = if i == 0 {
                                SymmetryRole::Primary
                            } else {
                                SymmetryRole::Mirror
                            };
                            ec.insert(SymmetryGroup { id: gid, role });
                        }
                        first.get_or_insert(child);
                    }
                    Err(e) => state.status = format!("Spawn failed: {e}"),
                }
            }
            if let Some(sel) = first {
                state.selected = Some(sel);
                // `surface_mount_from_hit` labels every body-skin hit "Mounted
                // wing"; gear shares that path but isn't a wing.
                state.status = if is_gear {
                    "Mounted landing gear".into()
                } else {
                    status
                };
            }
        }
    }

    // ---- Auto-place pending as root on empty canvas ------------------
    if state.ship_root.is_none() && state.pending.is_some() {
        // Footprint parts need a host — they can't be roots. Keep them
        // pending and nudge the user to add structure first.
        let needs_surface_host = state.pending.as_ref().is_some_and(|p| {
            matches!(p.params, PartParams::Wing { .. } | PartParams::Gear { .. })
                || catalog.resolve(&p.catalog_id).is_ok_and(|entry| {
                    matches!(
                        entry,
                        CatalogEntry::Engine(e) if e.geometry == EngineGeometry::JetNacelle
                    )
                })
        });
        if needs_surface_host {
            state.status = "Add a hull first, then click a compatible surface".into();
            return;
        }
        let pending = state.pending.take().unwrap();
        match ShipBlueprint::spawn_part(
            &mut commands,
            &catalog,
            &pending.catalog_id,
            pending.params,
            None,
        ) {
            Ok(part) => {
                let ship = commands
                    .spawn(Ship {
                        name: state.ship_name.clone(),
                        root: part,
                    })
                    .id();
                state.ship_root = Some(part);
                state.ship_entity = Some(ship);
                state.selected = Some(part);
                state.status = "Placed root".into();
            }
            Err(e) => state.status = format!("Spawn failed: {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

type InspectorQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static CatalogRef,
        &'static AttachNodes,
        Option<&'static mut CommandPod>,
        Option<&'static mut Decoupler>,
        Option<&'static mut Adapter>,
        Option<&'static mut FuelTank>,
        Option<&'static mut Fuselage>,
        Option<&'static mut Engine>,
        Option<&'static mut AirIntake>,
        Option<&'static mut Wing>,
        Option<&'static mut Gear>,
        Option<&'static mut PartResources>,
    ),
>;

fn draw_ship_stats(ui: &mut egui::Ui, stats: &ShipStats) {
    // Δv is reported per-stage in the Staging panel (the whole-ship rocket
    // equation is misleading for a multi-stage vessel), so it is not repeated
    // here. `vacuum` is still used for the whole-ship burn-time line.
    let vacuum = stats.vacuum_delta_v();
    ui.label(format!("Wet mass: {}", format_mass_kg(stats.wet_mass_kg())));
    ui.label(format!("Dry mass: {}", format_mass_kg(stats.dry_mass_kg)));
    ui.label(format!(
        "Propellant: {}",
        format_mass_kg(stats.propellant_mass_kg)
    ));
    ui.label(format!("Thrust: {}", format_thrust(stats.total_thrust_n)));
    if stats.wet_mass_kg() > 0.0 && stats.total_thrust_n > 0.0 {
        ui.label(format!("TWR: {:.2}", stats.current_acceleration() / G0));
    }
    if let Some(burn_s) = vacuum.burn_time_s {
        ui.label(format!("Full burn: {}", format_duration_s(burn_s)));
    }
    // Geometry-derived "will it fly" feedback. There is no flight model
    // yet (M6; Thalos has no atmosphere) — these are design references.
    if stats.wing_area_m2 > 0.0 {
        ui.label(format!("Wing area: {:.1} m²", stats.wing_area_m2));
        ui.label(format!("MAC: {:.2} m", stats.mean_aerodynamic_chord_m));
    }
}

/// Per-stage Δv / fuel breakdown, one card per stage in firing order. Stages
/// are derived from decoupler position (there is no authored stage list), so
/// this is a readout — you reorder staging by moving decouplers in the part
/// tree, not by dragging here. Tanks are previewed full.
fn draw_staging(ui: &mut egui::Ui, summaries: &[StageSummary]) {
    if summaries.is_empty() {
        ui.label("(no stages)");
        return;
    }

    let total_dv: f64 = summaries.iter().map(|s| s.delta_v_m_s).sum();
    ui.label(format!("Total Δv: {}", format_delta_v(total_dv)));
    ui.add_space(4.0);

    for s in summaries {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong(format!("Stage {}", s.number));
                ui.separator();
                if s.has_engine {
                    ui.label(format_delta_v(s.delta_v_m_s));
                } else {
                    ui.weak("drop only");
                }
            });
            if s.fuel_kg > 0.0 {
                ui.label(format!("Fuel: {}", format_mass_kg(s.fuel_kg)));
            }
            for res in thalos_shipyard::Resource::MASS_BEARING {
                let Some(totals) = s.resources.get(&res) else {
                    continue;
                };
                if totals.capacity <= 0.0 && totals.amount <= 0.0 {
                    continue;
                }
                let frac = if totals.capacity > 0.0 {
                    (totals.amount / totals.capacity).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };
                ui.add(
                    egui::ProgressBar::new(frac)
                        .desired_height(8.0)
                        .text(format!(
                            "{} {}",
                            res.display_name(),
                            format_mass_kg(totals.mass_kg)
                        )),
                );
            }
        });
    }
}

fn format_delta_v(meters_per_second: f64) -> String {
    if meters_per_second.abs() >= 9_999.5 {
        format!("{:.2} km/s", meters_per_second / 1_000.0)
    } else {
        format!("{:.0} m/s", meters_per_second)
    }
}

fn format_mass_kg(kg: f64) -> String {
    if kg.abs() >= 999_500.0 {
        format!("{:.2} kt", kg / 1_000_000.0)
    } else if kg.abs() >= 9_999.5 {
        format!("{:.1} t", kg / 1_000.0)
    } else {
        format!("{:.0} kg", kg)
    }
}

fn format_thrust(newtons: f64) -> String {
    if newtons.abs() >= 999_500.0 {
        format!("{:.2} MN", newtons / 1_000_000.0)
    } else if newtons.abs() >= 999.5 {
        format!("{:.1} kN", newtons / 1_000.0)
    } else {
        format!("{:.0} N", newtons)
    }
}

fn format_duration_s(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.0}s", seconds)
    } else if seconds < 3600.0 {
        let minutes = (seconds / 60.0).floor();
        let secs = seconds - minutes * 60.0;
        format!("{minutes:.0}m {secs:.0}s")
    } else {
        let hours = (seconds / 3600.0).floor();
        let minutes = ((seconds - hours * 3600.0) / 60.0).floor();
        format!("{hours:.0}h {minutes:.0}m")
    }
}

fn inspector_params(
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    tank: Option<&FuelTank>,
    fuselage: Option<&Fuselage>,
    wing: Option<&Wing>,
    gear: Option<&Gear>,
) -> PartParams {
    if let Some(d) = dec {
        PartParams::Decoupler {
            diameter: d.diameter,
        }
    } else if let Some(a) = adapter {
        PartParams::Adapter {
            diameter: a.diameter,
            target_diameter: a.target_diameter,
        }
    } else if let Some(t) = tank {
        PartParams::Tank {
            diameter: t.diameter,
            length: t.length,
        }
    } else if let Some(f) = fuselage {
        PartParams::Fuselage {
            length: f.length,
            max_width: f.max_width,
            max_height: f.max_height,
            roundness: f.roundness,
            nose_fraction: f.nose_fraction,
            nose_bluntness: f.nose_bluntness,
            tail_fraction: f.tail_fraction,
            nose_droop: f.nose_droop,
            tail_upsweep: f.tail_upsweep,
            tail_tip_diameter: f.tail_tip_diameter,
            tail_bluntness: f.tail_bluntness,
        }
    } else if let Some(w) = wing {
        PartParams::Wing {
            span: w.span,
            root_chord: w.root_chord,
            tip_chord: w.tip_chord,
            sweep: w.sweep,
            dihedral: w.dihedral,
            thickness: w.thickness,
            incidence: w.incidence,
        }
    } else if let Some(g) = gear {
        PartParams::Gear {
            strut_length: g.strut_length,
            wheel_radius: g.wheel_radius,
        }
    } else {
        PartParams::None
    }
}

fn editor_ui(
    mut contexts: EguiContexts,
    mut state: ResMut<EditorState>,
    mut part_queries: ParamSet<(InspectorQuery, CollectQuery)>,
    mut ships: Query<&mut Ship>,
    attachments: Query<(Entity, &Attachment)>,
    surface_mounts: Query<(Entity, &SurfaceMount)>,
    groups: Query<(Entity, &SymmetryGroup)>,
    catalog: Res<PartCatalog>,
    mut sky: ResMut<SkyBackdropEnabled>,
    mut clear_color: ResMut<ClearColor>,
    mut orientation: ResMut<BuildOrientation>,
    mut symmetry_mode: ResMut<SymmetryMode>,
    mut placement_snap: ResMut<PlacementSnap>,
    diagnostics: Res<DiagnosticsStore>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let ctx = ctx.clone();

    // Collect the blueprint once; both the aggregate stats and the per-stage
    // staging preview are projections of it.
    let (ship_stats, stage_summaries) = {
        let collect_parts = part_queries.p1();
        let blueprint = state.ship_root.and_then(|root| {
            let ship = Ship {
                name: String::new(),
                root,
            };
            collect_blueprint(&ship, &collect_parts, &attachments, &surface_mounts, &groups)
        });
        let stats = blueprint.as_ref().map(|bp| bp.stats(&catalog));
        let staging = blueprint.as_ref().map(|bp| bp.stage_summaries(&catalog));
        (stats, staging)
    };

    // -------- Left palette --------
    egui::SidePanel::left("palette")
        .default_width(180.0)
        .show(&ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
            let fps = diagnostics
                .get(&FrameTimeDiagnosticsPlugin::FPS)
                .and_then(|d| d.smoothed())
                .unwrap_or(0.0);
            ui.label(format!("FPS: {:.0}", fps));
            ui.separator();
            ui.heading("Parts");
            // Sort by category/kind/display name so palette ordering is
            // stable across runs (HashMap iteration is not).
            let mut entries: Vec<(&CatalogId, &CatalogEntry)> = catalog.parts.iter().collect();
            entries.sort_by_key(|(_, e)| {
                (
                    palette_category_order(e),
                    kind_order(e),
                    e.display_name().to_string(),
                )
            });
            let mut current_category = None;
            for (id, entry) in entries {
                let category = palette_category_label(entry);
                if current_category != Some(category) {
                    if current_category.is_some() {
                        ui.add_space(6.0);
                    }
                    ui.label(egui::RichText::new(category).strong());
                    current_category = Some(category);
                }

                if palette_part_button(ui, entry) {
                    state.pending = Some(PendingPart {
                        catalog_id: id.clone(),
                        params: default_params_for(entry),
                    });
                }
            }

            ui.separator();
            ui.heading("Ship");
            ui.horizontal(|ui| {
                ui.label("Name:");
                if let Some(se) = state.ship_entity {
                    if let Ok(mut ship) = ships.get_mut(se) {
                        ui.text_edit_singleline(&mut ship.name);
                    }
                } else {
                    ui.text_edit_singleline(&mut state.ship_name);
                }
            });
            ui.add_enabled_ui(state.ship_entity.is_some(), |ui| {
                if ui.button("Save").clicked() {
                    state.save_requested = true;
                }
            });
            if ui.button("Refresh list").clicked() {
                state.refresh_list = true;
            }

            ui.separator();
            ui.heading("Ship stats");
            match &ship_stats {
                Some(Ok(stats)) => draw_ship_stats(ui, stats),
                Some(Err(e)) => {
                    ui.colored_label(egui::Color32::from_rgb(220, 110, 60), format!("{e}"));
                }
                None => {
                    ui.label("(no ship)");
                }
            }

            ui.separator();
            ui.heading("Saved ships");
            let ship_list = state.ship_list.clone();
            if ship_list.is_empty() {
                ui.label("(none)");
            }
            for saved in ship_list {
                ui.horizontal(|ui| {
                    if ui.button("Load").clicked() {
                        state.load_target = Some(saved.slug.clone());
                    }
                    if ui.button("X").clicked() {
                        state.delete_file = Some(saved.slug.clone());
                    }
                    ui.label(&saved.name)
                        .on_hover_text(format!("{}.ron", saved.slug));
                });
            }

            ui.separator();
            ui.heading("Symmetry");
            ui.checkbox(&mut symmetry_mode.mirror, "Mirror (2×)")
                .on_hover_text(
                    "KSP-style: placing a wing/gear off-centre stamps a linked left/right pair. \
                     A part placed on a mirrored wing (e.g. a nacelle) auto-mirrors onto both.",
                );
            ui.checkbox(&mut placement_snap.enabled, "Angle snap (15°)")
                .on_hover_text(
                    "Magnetic snapping around the fuselage: a body-skin mount's azimuth rounds to \
                     15° steps as the cursor sweeps the hull, so gear/wings land dead-on the \
                     belly / sides. Off = free placement.",
                );

            ui.separator();
            ui.heading("View");
            {
                // Read through bypass_change_detection so the mere act of
                // rendering the checkbox doesn't mark BuildOrientation as
                // changed every frame (which would fire recenter_camera_on_
                // orientation_change and reset orbit.focus on every frame).
                let mut horiz = orientation.bypass_change_detection().horizontal;
                if ui
                    .checkbox(&mut horiz, "Horizontal layout (aircraft)")
                    .on_hover_text(
                        "Lay the build down like KSP's SPH — fuselage fore/aft, wings level, fin up.",
                    )
                    .changed()
                {
                    orientation.horizontal = horiz;
                }
            }
            if ui.checkbox(&mut sky.0, "Celestial backdrop").changed() {
                // Black clears behind the additively-blended stars so
                // they read as points of light; the default grey washes
                // them out.
                clear_color.0 = if sky.0 {
                    Color::BLACK
                } else {
                    ClearColor::default().0
                };
            }

            ui.separator();
            ui.label(format!("Status: {}", state.status));
            if state.pending.is_some() {
                let pending = state.pending.as_ref().unwrap();
                let surface_hint = matches!(
                    pending.params,
                    PartParams::Wing { .. } | PartParams::Gear { .. }
                ) || catalog.resolve(&pending.catalog_id).is_ok_and(|entry| {
                    matches!(
                        entry,
                        CatalogEntry::Engine(e) if e.geometry == EngineGeometry::JetNacelle
                    )
                });
                ui.colored_label(
                    egui::Color32::YELLOW,
                    if surface_hint {
                        "Pending part — click a compatible surface to place."
                    } else {
                        "Pending part — pick an attach node to place."
                    },
                );
                if ui.button("Cancel pending").clicked() {
                    state.pending = None;
                }
            }
            }); // scroll area
        });

    // -------- Right inspector --------
    egui::SidePanel::right("inspector")
        .default_width(260.0)
        .show(&ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Inspector");
            let Some(sel) = state.selected else {
                ui.label("(no selection)");
                return;
            };
            // KSP symmetry: edit the group's primary regardless of which member
            // was clicked. `sync_symmetry_groups` propagates the change to the
            // counterpart(s); editing a mirror counterpart directly would be
            // reverted next frame, leaving its sliders looking dead.
            let sel = symmetry_edit_target(sel, &groups);
            let mut parts = part_queries.p0();
            let Ok((
                entity,
                catalog_ref,
                nodes,
                mut pod,
                mut dec,
                mut adapter,
                mut tank,
                mut fuselage,
                mut engine,
                mut intake,
                mut wing,
                mut gear,
                mut res,
            )) = parts.get_mut(sel)
            else {
                ui.label("(invalid selection)");
                return;
            };
            ui.label(format!("Entity: {entity:?}"));
            let is_root = Some(sel) == state.ship_root;

            if let Some(p) = pod.as_deref_mut() {
                ui.label(format!("Kind: Command Pod ({})", p.geometry.label()));
                ui.label(format!("Model: {}", p.model));
                ui.label(format!("Diameter: {:.2}m (fixed)", p.diameter));
                ui.label(format!("Dry mass: {:.0} kg (fixed)", p.dry_mass));
            } else if let Some(d) = dec.as_deref_mut() {
                ui.label("Kind: Decoupler");
                if is_root {
                    ui.add(egui::Slider::new(&mut d.diameter, 0.3..=6.0).text("Diameter"));
                } else {
                    ui.label(format!("Diameter: {:.2}m (from parent)", d.diameter));
                }
                // Ejection impulse and dry mass are catalog-derived from
                // diameter (`ejection_impulse_per_diameter`, `mass_per_diameter`).
                // Editing them here would just be overwritten by
                // `recompute::recompute_decoupler_state`.
                ui.label(format!("Ejection impulse: {:.0} N·s", d.ejection_impulse));
                ui.label(format!("Dry mass: {:.0} kg", d.dry_mass));
            } else if let Some(a) = adapter.as_deref_mut() {
                ui.label("Kind: Adapter");
                if is_root {
                    ui.add(egui::Slider::new(&mut a.diameter, 0.3..=6.0).text("Diameter"));
                } else {
                    ui.label(format!("Diameter: {:.2}m (from parent)", a.diameter));
                }
                ui.add(
                    egui::Slider::new(&mut a.target_diameter, 0.3..=6.0).text("Target diameter"),
                );
                // dry_mass scales with frustum surface area via the
                // catalog's `wall_mass_per_m2`; recomputed by
                // `recompute::recompute_adapter_state` on every change.
                ui.label(format!("Dry mass: {:.0} kg", a.dry_mass));
            } else if let Some(t) = tank.as_deref_mut() {
                ui.label("Kind: Fuel Tank");
                if is_root {
                    ui.add(egui::Slider::new(&mut t.diameter, 0.3..=6.0).text("Diameter"));
                } else {
                    ui.label(format!("Diameter: {:.2}m (from parent)", t.diameter));
                }
                let effective_d = nodes.get("top").map(|n| n.diameter).unwrap_or(t.diameter);
                let max_length = 8.0 * effective_d;
                ui.add(egui::Slider::new(&mut t.length, 0.5..=max_length).text("Length"));
                // dry_mass and pool capacities scale with cylinder
                // geometry via the catalog; recomputed by
                // `recompute::recompute_tank_state` on every change.
                ui.label(format!("Dry mass: {:.0} kg", t.dry_mass));
            } else if let Some(f) = fuselage.as_deref_mut() {
                ui.label("Kind: Fuselage (stationed loft)");
                ui.add(egui::Slider::new(&mut f.length, 2.0..=60.0).text("Length"));
                if is_root {
                    ui.add(egui::Slider::new(&mut f.max_width, 0.5..=8.0).text("Width (Ø)"));
                } else {
                    ui.label(format!("Width: {:.2}m (from parent)", f.max_width));
                }
                ui.add(egui::Slider::new(&mut f.max_height, 0.5..=8.0).text("Height"));
                ui.add(egui::Slider::new(&mut f.roundness, 0.0..=1.0).text("Roundness"));
                ui.add(egui::Slider::new(&mut f.nose_fraction, 0.0..=0.45).text("Nose fraction"));
                ui.add(egui::Slider::new(&mut f.nose_bluntness, 0.0..=1.0).text("Nose shape (cone→radome)"));
                ui.add(egui::Slider::new(&mut f.tail_fraction, 0.0..=0.9).text("Tail fraction"));
                ui.add(egui::Slider::new(&mut f.nose_droop, 0.0..=2.0).text("Nose droop"));
                ui.add(egui::Slider::new(&mut f.tail_upsweep, 0.0..=3.0).text("Tail upsweep"));
                ui.add(
                    egui::Slider::new(&mut f.tail_tip_diameter, 0.0..=3.0).text("Tail tip Ø"),
                );
                ui.add(
                    egui::Slider::new(&mut f.tail_bluntness, 0.0..=1.0)
                        .text("Tail shape (cone→dome)"),
                );
                // dry_mass tracks lofted skin area via `recompute_fuselage_state`.
                ui.label(format!("Dry mass: {:.0} kg", f.dry_mass));
            } else if let Some(e) = engine.as_deref_mut() {
                let optimized_for = catalog
                    .resolve(&catalog_ref.id)
                    .ok()
                    .and_then(|entry| match entry {
                        CatalogEntry::Engine(spec) => Some(spec.optimized_for.label()),
                        _ => None,
                    })
                    .unwrap_or("Unknown");
                ui.label(format!("Kind: Engine ({optimized_for})"));
                ui.label(format!("Model: {}", e.model));
                ui.label(format!("Geometry: {}", e.geometry.label()));
                if e.requires_atmosphere {
                    ui.label("Requires atmosphere");
                }
                ui.label(format!("Diameter: {:.2}m (fixed)", e.diameter));
                ui.label(format!("Thrust: {:.1} kN (fixed)", e.thrust / 1000.0));
                ui.label(format!("Isp: {:.0} s (fixed)", e.isp));
                ui.label(format!("Dry mass: {:.0} kg (fixed)", e.dry_mass));
                if e.power_draw_kw > 0.0 {
                    ui.label(format!("Power draw: {:.1} kW (fixed)", e.power_draw_kw));
                }
                ui.label("Reactants:");
                for r in &e.reactants {
                    ui.label(format!(
                        "  {}: {:.1}%",
                        r.resource.display_name(),
                        r.mass_fraction * 100.0,
                    ));
                }
                if let Some(requirement) = e.intake_requirement {
                    ui.label(format!(
                        "Intake required: {:.2} m² {}",
                        requirement.area_m2,
                        requirement.kind.label()
                    ));
                }
                if let Some(capture) = e.builtin_intake {
                    ui.label(format!(
                        "Built-in intake: {:.2} m² {} (eff {:.0}%)",
                        capture.area_m2,
                        capture.kind.label(),
                        capture.efficiency * 100.0
                    ));
                }
            } else if let Some(i) = intake.as_deref_mut() {
                ui.label("Kind: Air Intake");
                ui.label(format!("Model: {}", i.model));
                ui.label(format!("Diameter: {:.2}m (fixed)", i.diameter));
                ui.label(format!("Length: {:.2}m (fixed)", i.length));
                ui.label(format!(
                    "Capture: {:.2} m² {} (eff {:.0}%)",
                    i.capture.area_m2,
                    i.capture.kind.label(),
                    i.capture.efficiency * 100.0
                ));
                ui.label(format!("Dry mass: {:.0} kg (fixed)", i.dry_mass));
            } else if let Some(w) = wing.as_deref_mut() {
                ui.label("Kind: Wing");
                ui.add(egui::Slider::new(&mut w.span, 0.5..=30.0).text("Span (per side)"));
                ui.add(egui::Slider::new(&mut w.root_chord, 0.3..=15.0).text("Root chord"));
                ui.add(egui::Slider::new(&mut w.tip_chord, 0.1..=15.0).text("Tip chord"));
                // Angles authored in degrees, stored in radians.
                let mut sweep_deg = w.sweep.to_degrees();
                if ui
                    .add(egui::Slider::new(&mut sweep_deg, -10.0..=60.0).text("Sweep °"))
                    .changed()
                {
                    w.sweep = sweep_deg.to_radians();
                }
                let mut dihedral_deg = w.dihedral.to_degrees();
                if ui
                    .add(egui::Slider::new(&mut dihedral_deg, -15.0..=15.0).text("Dihedral °"))
                    .changed()
                {
                    w.dihedral = dihedral_deg.to_radians();
                }
                let mut incidence_deg = w.incidence.to_degrees();
                if ui
                    .add(egui::Slider::new(&mut incidence_deg, -5.0..=10.0).text("Incidence °"))
                    .changed()
                {
                    w.incidence = incidence_deg.to_radians();
                }
                ui.add(egui::Slider::new(&mut w.thickness, 0.04..=0.25).text("Thickness t/c"));
                // dry_mass tracks planform area via `recompute_wing_state`.
                ui.label(format!("Dry mass: {:.0} kg/panel", w.dry_mass));
                // `sel` was resolved to the group primary above, so a grouped
                // wing always lands here as the primary — editing either side
                // updates both.
                match groups.get(sel).ok() {
                    Some(_) => {
                        ui.label("Symmetry: mirrored pair");
                        ui.label(
                            egui::RichText::new(
                                "Editing either side updates both; deleting either removes both.",
                            )
                            .small()
                            .weak(),
                        );
                    }
                    None => {
                        ui.label("Symmetry: single");
                    }
                }
            } else if let Some(g) = gear.as_deref_mut() {
                ui.label(if g.track_fraction > 0.0 {
                    "Kind: Landing Gear (main, L/R)"
                } else {
                    "Kind: Landing Gear (nose)"
                });
                ui.add(egui::Slider::new(&mut g.strut_length, 0.3..=4.0).text("Strut length"));
                ui.add(egui::Slider::new(&mut g.wheel_radius, 0.1..=1.2).text("Wheel radius"));
                if g.track_fraction > 0.0 {
                    ui.label(format!(
                        "Track: ±{:.0}% of host radius (fixed)",
                        g.track_fraction * 100.0
                    ));
                }
                // dry_mass tracks strut length × leg count via `recompute_gear_state`.
                ui.label(format!("Dry mass: {:.0} kg", g.dry_mass));
                ui.label(
                    egui::RichText::new(
                        "Self-contained gearbox — draws its own legs, not mirrored.",
                    )
                    .small()
                    .weak(),
                );
            }

            ui.separator();
            ui.label("Attach nodes:");
            for (id, node) in &nodes.nodes {
                ui.label(format!("  {id}: Ø{:.2}m", node.diameter));
            }

            ui.separator();
            ui.label("Resources:");
            if let Some(r) = res.as_deref_mut() {
                let params = inspector_params(
                    dec.as_deref(),
                    adapter.as_deref(),
                    tank.as_deref(),
                    fuselage.as_deref(),
                    wing.as_deref(),
                    gear.as_deref(),
                );
                let mut any_resource_row = false;
                let mut remove_resource = Vec::new();
                let mut add_resource = Vec::new();
                if let Ok(entry) = catalog.resolve(&catalog_ref.id) {
                    for option in entry.storage_options() {
                        let Some(capacity) = resource_capacity_for(entry, &params, option.resource)
                        else {
                            continue;
                        };
                        any_resource_row = true;
                        if let Some(pool) = r.pools.get_mut(&option.resource) {
                            ui.horizontal(|ui| {
                                if ui.small_button("Remove").clicked() {
                                    remove_resource.push(option.resource);
                                }
                                ui.label(format!(
                                    "{}: {:.0}/{:.0} {}",
                                    option.resource.display_name(),
                                    pool.amount,
                                    pool.capacity,
                                    option.resource.unit_label(),
                                ));
                            });
                            ui.add(
                                egui::Slider::new(&mut pool.amount, 0.0..=pool.capacity)
                                    .text("amount"),
                            );
                        } else if ui
                            .button(format!(
                                "Add {} ({:.0} {})",
                                option.resource.display_name(),
                                capacity,
                                option.resource.unit_label()
                            ))
                            .clicked()
                        {
                            add_resource.push((
                                option.resource,
                                ResourcePool {
                                    capacity,
                                    amount: capacity * option.default_fill_fraction.clamp(0.0, 1.0),
                                },
                            ));
                        }
                    }
                }
                for resource in remove_resource {
                    r.pools.remove(&resource);
                }
                for (resource, pool) in add_resource {
                    r.pools.insert(resource, pool);
                }
                if !any_resource_row {
                    ui.label("  (none)");
                }
            }

            ui.separator();
            ui.add_enabled_ui(!is_root, |ui| {
                if ui.button("Set as root").clicked() {
                    state.set_as_root = true;
                }
            });
            if ui.button("Delete part").clicked() {
                state.delete_selected = true;
            }
            }); // scroll area
        });

    // -------- Staging preview (right, left of the inspector) --------
    egui::SidePanel::right("staging")
        .resizable(true)
        .default_width(210.0)
        .show(&ctx, |ui| {
            ui.heading("Staging");
            ui.label(
                egui::RichText::new("Derived from decoupler position")
                    .small()
                    .weak(),
            );
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| match &stage_summaries {
                Some(Ok(summaries)) => draw_staging(ui, summaries),
                Some(Err(e)) => {
                    ui.colored_label(egui::Color32::from_rgb(220, 110, 60), format!("{e}"));
                }
                None => {
                    ui.label("(no ship)");
                }
            });
        });

    // -------- Bottom: ship hierarchy & placement picker --------
    egui::TopBottomPanel::bottom("hierarchy")
        .default_height(180.0)
        .show(&ctx, |ui| {
            ui.horizontal(|ui| {
                // Hierarchy list
                ui.vertical(|ui| {
                    ui.heading("Ship");
                    let Some(root) = state.ship_root else {
                        return;
                    };
                    let mut child_map: HashMap<Entity, Vec<Entity>> = HashMap::new();
                    for (e, att) in attachments.iter() {
                        child_map.entry(att.parent).or_default().push(e);
                    }
                    // Surface-mounted wings are part of the ship tree too.
                    for (e, sm) in surface_mounts.iter() {
                        child_map.entry(sm.parent).or_default().push(e);
                    }
                    draw_hierarchy(ui, root, &child_map, &mut state, 0);
                });

                ui.separator();

                // Placement picker. Surface parts click a compatible body;
                // stack parts use a free attach node listed here.
                if let Some(pending) = state.pending.clone() {
                    let pending_wing = matches!(pending.params, PartParams::Wing { .. });
                    let pending_gear = matches!(pending.params, PartParams::Gear { .. });
                    let pending_nacelle = catalog.resolve(&pending.catalog_id).is_ok_and(|entry| {
                        matches!(
                            entry,
                            CatalogEntry::Engine(e) if e.geometry == EngineGeometry::JetNacelle
                        )
                    });
                    ui.vertical(|ui| {
                        if pending_wing {
                            ui.heading("Mount wing");
                            ui.label("Click a hull body where the wing root should sit.");
                            ui.label(
                                egui::RichText::new(
                                    "Side hit → mirrored pair · top/bottom hit → single fin",
                                )
                                .small()
                                .weak(),
                            );
                            return;
                        }
                        if pending_gear {
                            ui.heading("Mount landing gear");
                            ui.label("Click the fuselage belly where the gear should sit.");
                            ui.label(
                                egui::RichText::new(
                                    "Self-contained gearbox — main draws both legs; never mirrored",
                                )
                                .small()
                                .weak(),
                            );
                            return;
                        }
                        if pending_nacelle {
                            ui.heading("Mount nacelle");
                            ui.label("Click a wing where the pylon should sit.");
                            ui.label(
                                egui::RichText::new(
                                    "A mirrored wing creates a mirrored nacelle pair",
                                )
                                .small()
                                .weak(),
                            );
                            return;
                        }
                        ui.heading("Place at…");
                        let occupied: std::collections::HashSet<(Entity, String)> = attachments
                            .iter()
                            .map(|(_, a)| (a.parent, a.parent_node.clone()))
                            .collect();
                        let mut rows: Vec<(Entity, String, f32)> = Vec::new();
                        let parts = part_queries.p0();
                        for (e, _, nodes, _, _, _, _, _, _, _, _, _, _) in parts.iter() {
                            for (nid, node) in &nodes.nodes {
                                if occupied.contains(&(e, nid.clone())) {
                                    continue;
                                }
                                // Skip command pod's implicit "top" — but
                                // we don't store one, so nothing to skip.
                                rows.push((e, nid.clone(), node.diameter));
                            }
                        }
                        for (e, nid, d) in rows {
                            if ui.button(format!("{e:?} / {nid} (Ø{d:.2}m)")).clicked() {
                                state.place_at = Some((e, nid));
                            }
                        }
                    });
                }
            });
        });
}

fn draw_hierarchy(
    ui: &mut egui::Ui,
    entity: Entity,
    child_map: &HashMap<Entity, Vec<Entity>>,
    state: &mut EditorState,
    depth: usize,
) {
    let indent = "  ".repeat(depth);
    let selected = state.selected == Some(entity);
    let label = format!("{indent}{entity:?}");
    if ui.selectable_label(selected, label).clicked() {
        state.selected = Some(entity);
    }
    if let Some(kids) = child_map.get(&entity) {
        for c in kids {
            draw_hierarchy(ui, *c, child_map, state, depth + 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Celestial backdrop
// ---------------------------------------------------------------------------
//
// Duplicated from `thalos_game::sky_render` with the game-specific bits
// (CameraExposure, SimStage, OrbitCamera) stripped out. Keep until sky
// rendering is extracted into its own crate.

#[derive(Resource, Default)]
struct SkyBackdropEnabled(bool);

#[derive(Component)]
struct SkyBackdrop;

#[derive(Clone, Copy, ShaderType)]
struct StarsParams {
    pixel_radius: f32,
    brightness: f32,
    size_gamma: f32,
    _pad0: f32,
}

impl Default for StarsParams {
    fn default() -> Self {
        Self {
            pixel_radius: 4.0,
            brightness: 140.0,
            size_gamma: 0.50,
            _pad0: 0.0,
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct StarsMaterial {
    #[uniform(0)]
    params: StarsParams,
}

impl Material for StarsMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/stars.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/stars.wgsl".into()
    }
    fn prepass_vertex_shader() -> ShaderRef {
        "shaders/stars_prepass.wgsl".into()
    }
    fn prepass_fragment_shader() -> ShaderRef {
        "shaders/stars_prepass.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }

    fn specialize(
        _: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(1),
            Mesh::ATTRIBUTE_COLOR.at_shader_location(2),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = false;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, ShaderType)]
struct GalaxyParams {
    pixel_radius_scale: f32,
    min_pixel_radius: f32,
    brightness: f32,
    _pad0: f32,
}

impl Default for GalaxyParams {
    fn default() -> Self {
        Self {
            pixel_radius_scale: 2000.0,
            min_pixel_radius: 1.2,
            brightness: 1_500.0,
            _pad0: 0.0,
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct GalaxyMaterial {
    #[uniform(0)]
    params: GalaxyParams,
}

impl Material for GalaxyMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/galaxy.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/galaxy.wgsl".into()
    }
    fn prepass_vertex_shader() -> ShaderRef {
        "shaders/galaxy_prepass.wgsl".into()
    }
    fn prepass_fragment_shader() -> ShaderRef {
        "shaders/galaxy_prepass.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }

    fn specialize(
        _: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(1),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(2),
            Mesh::ATTRIBUTE_TANGENT.at_shader_location(3),
            Mesh::ATTRIBUTE_COLOR.at_shader_location(4),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = false;
        }
        Ok(())
    }
}

struct SkyBackdropPlugin;

impl Plugin for SkyBackdropPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<StarsMaterial>::default())
            .add_plugins(MaterialPlugin::<GalaxyMaterial>::default())
            .add_systems(Startup, spawn_sky_backdrop)
            .add_systems(Update, (update_sky_visibility, update_galaxy_uniform));
    }
}

fn spawn_sky_backdrop(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut stars_materials: ResMut<Assets<StarsMaterial>>,
    mut galaxy_materials: ResMut<Assets<GalaxyMaterial>>,
) {
    let universe = generate_default(&DefaultGenParams::default());

    commands.spawn((
        SkyBackdrop,
        Mesh3d(meshes.add(build_star_mesh(&universe))),
        MeshMaterial3d(stars_materials.add(StarsMaterial {
            params: StarsParams::default(),
        })),
        Transform::IDENTITY,
        Visibility::Hidden,
        NoFrustumCulling,
    ));

    commands.spawn((
        SkyBackdrop,
        Mesh3d(meshes.add(build_galaxy_mesh(&universe))),
        MeshMaterial3d(galaxy_materials.add(GalaxyMaterial {
            params: GalaxyParams::default(),
        })),
        Transform::IDENTITY,
        Visibility::Hidden,
        NoFrustumCulling,
    ));
}

fn update_sky_visibility(
    enabled: Res<SkyBackdropEnabled>,
    mut q: Query<&mut Visibility, With<SkyBackdrop>>,
) {
    let target = if enabled.0 {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut v in q.iter_mut() {
        if *v != target {
            *v = target;
        }
    }
}

fn update_galaxy_uniform(
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<&Projection, With<Camera3d>>,
    handles: Query<&MeshMaterial3d<GalaxyMaterial>>,
    mut materials: ResMut<Assets<GalaxyMaterial>>,
) {
    let Ok(window) = windows.single() else { return };
    let Ok(projection) = cameras.single() else {
        return;
    };
    let Projection::Perspective(p) = projection else {
        return;
    };
    let px_per_rad = window.resolution.physical_height() as f32 / p.fov;

    for handle in &handles {
        if let Some(mat) = materials.get_mut(&handle.0) {
            mat.params.pixel_radius_scale = px_per_rad;
        }
    }
}

fn build_star_mesh(universe: &Universe) -> Mesh {
    let n = universe.stars.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n * 4);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 6);

    const CORNERS: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

    for (i, star) in universe.stars.iter().enumerate() {
        let dir = star.position.normalize();
        let rgb = star.linear_srgb();
        let flux = star.magnitude_flux();
        for corner in CORNERS {
            positions.push([dir.x, dir.y, dir.z]);
            uvs.push(corner);
            colors.push([rgb[0], rgb[1], rgb[2], flux]);
        }
        let base = (i * 4) as u32;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn build_galaxy_mesh(universe: &Universe) -> Mesh {
    let n = universe.galaxies.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n * 4);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut tangents: Vec<[f32; 4]> = Vec::with_capacity(n * 4);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 6);

    const CORNERS: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

    for (i, galaxy) in universe.galaxies.iter().enumerate() {
        let dir = galaxy.position.normalize();
        let rgb = galaxy.linear_srgb();
        let flux = galaxy.magnitude_flux();
        let (sin_pa, cos_pa) = galaxy.position_angle_rad.sin_cos();
        for corner in CORNERS {
            positions.push([dir.x, dir.y, dir.z]);
            uvs.push(corner);
            normals.push([galaxy.effective_radius_rad, galaxy.sersic_n, 0.0]);
            tangents.push([galaxy.axis_ratio, cos_pa, sin_pa, 0.0]);
            colors.push([rgb[0], rgb[1], rgb[2], flux]);
        }
        let base = (i * 4) as u32;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

// ---------------------------------------------------------------------------
// Shrouds (auto-generated cones wrapping a [`Shroudable`] above a
// [`ShroudProvider`])
// ---------------------------------------------------------------------------

/// Attached to a shroud entity — a mesh child of the provider
/// (e.g. a decoupler) that wraps the shrouded part above. Spawned and
/// reconciled by [`sync_shrouds`]; not part of the persisted blueprint
/// and not user-spawnable.
#[derive(Component, Debug, Clone, Copy)]
struct Shroud {
    provider: Entity,
    shrouded: Entity,
    // Cached spec, compared each frame so we only rebuild the mesh /
    // material when geometry actually changed.
    bottom_radius: f32,
    top_radius: f32,
    height: f32,
}

/// Marker on the shroud entity's body. Kept distinct from [`PartBody`] so
/// part-level highlight systems (tint, material swap) don't fire on
/// hovered shrouds — the shroud manages its own hover feedback
/// (transparency) in [`update_shroud_transparency`].
#[derive(Component, Debug, Clone, Copy)]
struct ShroudBody;

/// Expected geometry for a shroud covering a given attachment. `None`
/// when no shroud should exist for this pair (misconfigured attachment,
/// shrouded part missing [`Shroudable`], or provider not wider than the
/// shrouded's top — the cone would degenerate).
struct ShroudSpec {
    bottom_radius: f32,
    top_radius: f32,
    height: f32,
    shrouded: Entity,
}

fn compute_shroud_spec(
    attachment: &Attachment,
    provider_nodes: &AttachNodes,
    shroudables: &Query<(&Engine, Has<Shroudable>)>,
) -> Option<ShroudSpec> {
    // Only the canonical "provider sits below shroudable" orientation
    // gets a shroud: provider's `top` mates with shroudable's `bottom`.
    if attachment.my_node != "top" || attachment.parent_node != "bottom" {
        return None;
    }
    let (engine, is_shroudable) = shroudables.get(attachment.parent).ok()?;
    if !is_shroudable {
        return None;
    }
    let provider_top_d = provider_nodes.get("top")?.diameter;
    // Shroud top matches the shrouded part's *attach* diameter — the
    // interface the stage above would mate with. That sits outside the
    // engine's narrowing visual silhouette, so the shroud stays clear
    // of the engine body instead of hugging (and z-fighting) it.
    let bottom_r = provider_top_d * 0.5;
    let top_r = engine.diameter * 0.5;
    let (_, _, height) = engine_visual_profile(engine.diameter);
    // Only generate when the provider is at least as wide as the
    // shrouded part at its top — a narrower provider would invert the
    // cone. Equal diameter gives a clean cylindrical interstage.
    if bottom_r + 1.0e-4 < top_r {
        return None;
    }
    Some(ShroudSpec {
        bottom_radius: bottom_r,
        top_radius: top_r,
        height,
        shrouded: attachment.parent,
    })
}

fn spec_matches(s: &Shroud, spec: &ShroudSpec) -> bool {
    s.shrouded == spec.shrouded
        && (s.bottom_radius - spec.bottom_radius).abs() < 1.0e-4
        && (s.top_radius - spec.top_radius).abs() < 1.0e-4
        && (s.height - spec.height).abs() < 1.0e-4
}

/// Reconcile shroud entities against current attachment state: spawn
/// missing shrouds, update ones whose geometry changed, and despawn
/// orphans. Idempotent per frame; cheap when attachment is stable.
fn sync_shrouds(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    providers: Query<(Entity, &Attachment, &AttachNodes), With<ShroudProvider>>,
    shroudables: Query<(&Engine, Has<Shroudable>)>,
    existing: Query<(Entity, &Shroud)>,
) {
    // Map provider -> (shroud_entity, current Shroud component).
    let mut current_by_provider: HashMap<Entity, (Entity, Shroud)> = HashMap::new();
    for (entity, shroud) in existing.iter() {
        current_by_provider.insert(shroud.provider, (entity, *shroud));
    }

    let mut kept: HashSet<Entity> = HashSet::new();

    for (provider, attachment, provider_nodes) in providers.iter() {
        let Some(spec) = compute_shroud_spec(attachment, provider_nodes, &shroudables) else {
            continue;
        };
        kept.insert(provider);

        // Reuse in-place if the cached spec still matches.
        if let Some((_, current)) = current_by_provider.get(&provider)
            && spec_matches(current, &spec)
        {
            continue;
        }
        if let Some((old, _)) = current_by_provider.get(&provider) {
            commands.entity(*old).despawn();
        }

        let shroud_mesh: Mesh = ConicalFrustum {
            radius_top: spec.top_radius,
            radius_bottom: spec.bottom_radius,
            height: spec.height,
        }
        .mesh()
        .resolution(PART_RESOLUTION)
        .into();
        let mesh_handle = meshes.add(shroud_mesh);

        // Slant length — the actual surface distance v = 0 → v = 1.
        // Matches the vertical height only when the two radii agree.
        let dr = spec.bottom_radius - spec.top_radius;
        let slant_length = (spec.height * spec.height + dr * dr).sqrt();
        // Blend mode is set once here; we only vary base-color alpha
        // from the hover system so the pipeline stays hot.
        let material = ship_materials.add(ShipPartMaterial {
            base: StandardMaterial {
                alpha_mode: AlphaMode::Blend,
                ..stainless_steel_base()
            },
            extension: ShipPartExtension {
                params: ShipPartParams {
                    length: slant_length,
                    radius_top: spec.top_radius,
                    radius_bottom: spec.bottom_radius,
                    // Mix provider index with a fixed mask so shroud
                    // detail doesn't look identical to the decoupler's.
                    seed: provider.index_u32() ^ 0x5A5A_5A5A,
                    ..default()
                },
            },
        });

        // Shroud mesh center sits at +height/2 in the provider's local
        // frame, since the provider's "top" node (y = 0) meets the
        // shrouded's base and the shroud extends upward from there.
        let shroud_entity = commands
            .spawn((
                Mesh3d(mesh_handle),
                MeshMaterial3d(material),
                Transform::from_xyz(0.0, spec.height * 0.5, 0.0),
                Visibility::default(),
                Shroud {
                    provider,
                    shrouded: spec.shrouded,
                    bottom_radius: spec.bottom_radius,
                    top_radius: spec.top_radius,
                    height: spec.height,
                },
                ShroudBody,
                Pickable::default(),
            ))
            .observe(on_shroud_click)
            .id();
        commands.entity(provider).add_child(shroud_entity);
    }

    // Despawn shrouds whose provider no longer qualifies (detachment,
    // geometry change below threshold, shrouded part removed, etc.).
    for (provider, (entity, _)) in &current_by_provider {
        if !kept.contains(provider) {
            commands.entity(*entity).despawn();
        }
    }
}

/// Drive the shroud's base-color alpha from hover: opaque by default
/// (engine hidden inside), partial transparency while hovered so the
/// shrouded silhouette reads through.
fn update_shroud_transparency(
    hover_map: Res<HoverMap>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    shrouds: Query<(Entity, &MeshMaterial3d<ShipPartMaterial>), With<ShroudBody>>,
) {
    let hovered: HashSet<Entity> = hover_map
        .0
        .values()
        .flat_map(|hovers| hovers.keys().copied())
        .collect();

    for (entity, mesh_mat) in shrouds.iter() {
        let target_alpha: f32 = if hovered.contains(&entity) { 0.18 } else { 1.0 };
        let Some(mat) = ship_materials.get_mut(&mesh_mat.0) else {
            continue;
        };
        let srgba = mat.base.base_color.to_srgba();
        if (srgba.alpha - target_alpha).abs() > 1.0e-3 {
            mat.base.base_color = Color::srgba(srgba.red, srgba.green, srgba.blue, target_alpha);
        }
    }
}

/// Click on a shroud selects the provider that owns it — the shroud is
/// a visual extension of the decoupler, not an independent part.
fn on_shroud_click(
    click: On<Pointer<Click>>,
    shrouds: Query<&Shroud>,
    mut state: ResMut<EditorState>,
    mut contexts: EguiContexts,
) {
    if pointer_over_egui(&mut contexts) {
        return;
    }
    if let Ok(shroud) = shrouds.get(click.entity) {
        state.selected = Some(shroud.provider);
    }
}

/// Propagate the coupled neighbor's [`MaterialKind`] onto parts that
/// visually continue with whatever is attached to their `bottom` node —
/// currently [`Decoupler`] (so the decoupler + its shroud read as part
/// of the stage below on staging) and [`Adapter`] (so a diameter
/// transition inherits from the narrower stage it feeds into). Parts
/// with nothing attached below keep their default [`MaterialKind`].
fn propagate_coupled_material(
    attachments: Query<(Entity, &Attachment)>,
    mut params: ParamSet<(
        Query<(Entity, &PartMaterial)>,
        Query<(Entity, &mut PartMaterial), Or<(With<Decoupler>, With<Adapter>)>>,
    )>,
) {
    // Build parent → bottom-attached-child entity map.
    let mut coupled: HashMap<Entity, Entity> = HashMap::new();
    for (child, att) in attachments.iter() {
        if att.parent_node == "bottom" {
            coupled.insert(att.parent, child);
        }
    }

    // Snapshot every part's current MaterialKind so read + write on
    // PartMaterial can both run in this system without conflicting
    // mutable borrows.
    let kinds: HashMap<Entity, MaterialKind> =
        params.p0().iter().map(|(e, m)| (e, m.kind)).collect();

    for (entity, mut my_mat) in params.p1().iter_mut() {
        let Some(coupled_entity) = coupled.get(&entity).copied() else {
            continue;
        };
        let Some(&kind) = kinds.get(&coupled_entity) else {
            continue;
        };
        if my_mat.kind != kind {
            my_mat.kind = kind;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugifies_ship_names_for_ron_filenames() {
        assert_eq!(
            slugify_ship_name("  Lunar Transfer Mk II  "),
            "lunar-transfer-mk-ii"
        );
        assert_eq!(slugify_ship_name("A__B / C"), "a-b-c");
        assert_eq!(slugify_ship_name("***"), "unnamed");
    }

    #[test]
    fn reads_ship_name_from_schema_header() {
        let text = r#"(
            name: "Lunar Transfer Vehicle",
            root: 0,
            parts: [],
            connections: [],
        )"#;

        assert_eq!(
            ship_name_from_ron(text).as_deref(),
            Some("Lunar Transfer Vehicle")
        );
    }

    #[test]
    fn schema_ship_names_are_trimmed_and_non_empty() {
        assert_eq!(schema_ship_name("  Apollo  "), "Apollo");
        assert_eq!(schema_ship_name("   "), "Unnamed");
    }
}
