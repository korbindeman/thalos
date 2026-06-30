//! In-world surface base editor — a Cities:Skylines-style placement tool for
//! buildings on a planetary surface.
//!
//! Unlike the shipyard editor (a separate hangar *scene* that hides the flight
//! world), the base editor is an **in-world overlay**: the real planet stays
//! visible, the sim pauses, a god-view camera looks down at the build site, and
//! buildings are placed on the actual flattened terrain. It is a modal pause
//! mode like the shipyard — [`BaseEditor::open`] is a sim-clock pause source
//! (see [`crate::sim_clock`]), not an `AppState`. While open:
//!
//! - the three `SimStage` sets are gated off (`base_editor_closed` in
//!   `main.rs`), freezing flight logic and the flight camera so the editor's own
//!   ungated god-view camera owns the view (the world is frozen-but-visible);
//! - all gameplay input contexts deactivate (see
//!   `crate::input::gate_enhanced_input_sources`) so stick/keys don't drive the
//!   ship while building; the editor reads raw mouse/keyboard directly;
//! - the flight HUD hides (the navball/MFD are meaningless in the god-view) and
//!   the editor's own Bevy-UI panels show.
//!
//! Entry: the pause menu's SURFACE BASE button. Escape closes (owned by
//! `pause_menu::handle_escape_input`'s priority chain).
//!
//! The workflow has two [`BaseEditorMode`]s: **pick a site** (aim at the surface,
//! confirm → the land flattens), then **place buildings** on the flattened pad.
//! Placed buildings and the flattened site are [`crate::structures`] records, so
//! they survive the session and (later) save to disk.

mod camera;
mod connections;
mod pick;
mod place;
mod ui;

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::rendering::{SimulationState, SolarSystemState};
use crate::structures::{StructureId, StructureKind, StructurePlacement, StructureRegistry};
use crate::view::ViewMode;

pub use place::BaseBuildState;

/// `tile_lod_m` for the focus / surface-height queries. The editor never needs
/// sub-metre LOD for camera framing, so a coarse 2 m floor is plenty.
pub(crate) const FOCUS_HEIGHT_LOD_M: f32 = 2.0;

/// In-world base editor state. A sim-clock pause source.
///
/// **Sole writer of `open`:** the pause menu's SURFACE BASE button and Escape
/// via `pause_menu::handle_escape_input`.
#[derive(Resource, Debug, Default, Clone)]
pub struct BaseEditor {
    pub open: bool,
    pub mode: BaseEditorMode,
    /// The site whose flattened pad we're currently building on (set when a
    /// pick is confirmed). `None` while picking, or before any site exists.
    pub active_site: Option<StructureId>,
}

/// The two phases of the editor workflow.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum BaseEditorMode {
    /// Aim at the surface and confirm a building site; confirming flattens it.
    #[default]
    PickSite,
    /// Place / move / delete buildings on the active site's flattened pad.
    PlaceBuildings,
}

/// Run condition: the base editor is open.
pub fn base_editor_open(editor: Option<Res<BaseEditor>>) -> bool {
    editor.map(|e| e.open).unwrap_or(false)
}

/// Run condition: the base editor is closed.
pub fn base_editor_closed(editor: Option<Res<BaseEditor>>) -> bool {
    editor.map(|e| !e.open).unwrap_or(true)
}

pub struct BaseEditorPlugin;

impl Plugin for BaseEditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BaseEditor>()
            .add_plugins(camera::BaseEditorCameraPlugin)
            .add_plugins(pick::BaseEditorPickPlugin)
            .add_plugins(place::BaseEditorPlacePlugin)
            .add_plugins(connections::BaseEditorConnectionsPlugin)
            .add_plugins(ui::BaseEditorUiPlugin)
            .add_systems(Update, apply_open_state);
    }
}

/// Runway-edge bounding radius used as the runway's connection node, so the
/// authored tarmac meets the runway's side rather than its centreline.
const RUNWAY_EDGE_M: f64 = 18.0;

/// Author the **default base**: a launch complex laid out beside the runway on
/// the shared flat basin (everything coplanar — `Drape` at `pad_r`). Two large
/// launchpads with clearing around them, each flanked by a flame diverter and a
/// small tank farm; a VAB and a pair of hangars set back along the far edge;
/// blockhouses and an operations building near the strip. A tarmac MST links the
/// runway, pads, big buildings and blockhouses (the satellite tanks/diverters
/// stay off the road network). Called by the runway scenario after it installs
/// the basin BaseSite. `pad_r = radius_m + E`.
///
/// Layout is in runway-frame `(along, off)` metres from the runway centre, with
/// `+off` = the launch-complex side (`heading × center_dir`) — the same side the
/// basin is offset toward in [`crate::runway`], so the complex falls inside the
/// flattened basin.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_default_base(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    registry: &mut StructureRegistry,
    root: Entity,
    body_id: BodyId,
    basin_site_id: StructureId,
    center_dir: DVec3,
    heading: DVec3,
    pad_r: f64,
) {
    let mats = place::BaseMaterials::create(materials);
    let across = heading.cross(center_dir).normalize();
    let dir = |along: f64, off: f64| (center_dir * pad_r + heading * along + across * off).normalize();

    let launchpad = |r: f32| StructureKind::Launchpad { radius_m: r };
    let building = |hx: f32, hz: f32, h: f32| StructureKind::Building {
        half_x_m: hx,
        half_z_m: hz,
        height_m: h,
    };
    let tank = |r: f32, h: f32| StructureKind::Tank {
        radius_m: r,
        height_m: h,
    };

    // Pads sit ~600 m off the centreline, 1.7 km apart, with a wide blast-clear
    // ring — well off the strip. Big buildings (VAB, hangars) line the far edge
    // of the basin; ops/blockhouses sit near the strip.
    const PAD_ALONG: f64 = 850.0;
    const PAD_OFF: f64 = 600.0;

    // Road structures: linked by the tarmac MST (runway ↔ pads ↔ buildings).
    let road: &[(f64, f64, StructureKind)] = &[
        // Two large launch pads with clearing.
        (PAD_ALONG, PAD_OFF, launchpad(50.0)),
        (-PAD_ALONG, PAD_OFF, launchpad(50.0)),
        // Operations / terminal building near the runway centre.
        (0.0, 300.0, building(16.0, 12.0, 12.0)),
        // A blockhouse beside each pad (between strip and pad).
        (PAD_ALONG, 330.0, building(10.0, 10.0, 8.0)),
        (-PAD_ALONG, 330.0, building(10.0, 10.0, 8.0)),
        // VAB-scale assembly building, set back on the far edge.
        (0.0, 1040.0, building(68.0, 46.0, 96.0)),
        // Two long hangars flanking it.
        (520.0, 980.0, building(58.0, 24.0, 26.0)),
        (-520.0, 980.0, building(58.0, 24.0, 26.0)),
    ];

    // Satellite structures: placed but kept off the road network (they cluster
    // around their pad). Flame diverters just outboard of each pad, plus a small
    // tank farm beyond.
    let satellites: &[(f64, f64, StructureKind)] = &[
        // Flame diverters / trenches (low, wide concrete) outboard of each pad.
        (PAD_ALONG, 712.0, building(14.0, 44.0, 4.0)),
        (-PAD_ALONG, 712.0, building(14.0, 44.0, 4.0)),
        // Propellant tank farm beyond each pad (three tanks).
        (PAD_ALONG - 42.0, 845.0, tank(9.0, 26.0)),
        (PAD_ALONG, 875.0, tank(9.0, 26.0)),
        (PAD_ALONG + 42.0, 845.0, tank(9.0, 26.0)),
        (-PAD_ALONG + 42.0, 845.0, tank(9.0, 26.0)),
        (-PAD_ALONG, 875.0, tank(9.0, 26.0)),
        (-PAD_ALONG - 42.0, 845.0, tank(9.0, 26.0)),
    ];

    let mut place_one = |along: f64, off: f64, kind: StructureKind| {
        place::place_structure(
            commands,
            meshes,
            &mats,
            registry,
            root,
            body_id,
            Some(basin_site_id),
            dir(along, off),
            heading,
            across,
            pad_r,
            kind,
            0.0,
        );
    };

    // Node 0 is the runway itself (runway centre), so the tarmac links the
    // complex to the runway edge.
    let mut nodes: Vec<(f64, f64, f64)> = vec![(0.0, 0.0, RUNWAY_EDGE_M)];
    for &(along, off, kind) in road {
        place_one(along, off, kind);
        nodes.push((along, off, place::kind_bounding_m(&kind)));
    }
    for &(along, off, kind) in satellites {
        place_one(along, off, kind);
    }

    connections::spawn_authored(
        commands,
        meshes,
        &mats,
        root,
        body_id,
        basin_site_id,
        center_dir,
        heading,
        pad_r,
        &nodes,
    );
}

/// Ray-vs-sphere intersection for a sphere centred at the origin; returns the
/// unit hit direction, or `None` on a miss. Shared by site-pick and building
/// placement (both raycast the cursor against the body sphere in render space,
/// where directions equal world directions). Mirrors the helper in `debug.rs`.
pub(crate) fn ray_vs_sphere_dir(origin: Vec3, dir: Vec3, radius: f32) -> Option<Vec3> {
    let b = origin.dot(dir);
    let c = origin.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    let near = -b - root;
    let far = -b + root;
    let t = if near >= 0.0 {
        near
    } else if far >= 0.0 {
        far
    } else {
        return None;
    };
    Some((origin + dir * t).normalize_or_zero())
}

/// The world-space frame the editor is currently looking at: a point on the
/// dominant body's surface plus its local vertical. The god-view camera orbits
/// it; site-pick and placement reuse it. All positions are heliocentric metres
/// (the big_space absolute frame — see `rendering::real_space`).
pub(crate) struct EditorFocus {
    /// Focus point in heliocentric metres.
    pub center_world: DVec3,
    /// Local vertical at the focus (world-space, unit).
    pub up_world: DVec3,
}

/// Resolve the editor's current focus: the active site's flattened centre when
/// placing buildings, otherwise the surface point directly under the player
/// ship (the natural spot to start picking a site). `None` if body state isn't
/// available yet.
pub(crate) fn compute_focus(
    editor: &BaseEditor,
    sim: &SimulationState,
    solar: &SolarSystemState,
    height_sources: &HeightSourceRegistry,
    registry: &StructureRegistry,
) -> Option<EditorFocus> {
    let states = solar.states.as_deref()?;
    let body_id = sim.simulation.dominant_body();
    let body_state = states.get(body_id)?;
    let radius_m = sim.system.bodies.get(body_id)?.radius_m;

    if editor.mode == BaseEditorMode::PlaceBuildings
        && let Some(site_id) = editor.active_site
        && let Some(site) = registry.get(site_id)
    {
        let up_world = (body_state.orientation * site.anchor_dir).normalize();
        let elevation_m = match site.placement {
            StructurePlacement::FlattenTo { elevation_m, .. } => elevation_m,
            StructurePlacement::Drape => 0.0,
        };
        return Some(EditorFocus {
            center_world: body_state.position + up_world * (radius_m + elevation_m),
            up_world,
        });
    }

    // Pick mode (or no active site yet): the surface point under the ship.
    let ship_pos = sim.simulation.ship_state().position;
    let dir_world = (ship_pos - body_state.position).normalize_or_zero();
    if dir_world == DVec3::ZERO {
        return None;
    }
    let dir_body = (body_state.orientation.inverse() * dir_world).normalize();
    let height_m = height_sources
        .get(body_id)
        .and_then(|hs| hs.sample_height_m(dir_body.as_vec3(), FOCUS_HEIGHT_LOD_M))
        .unwrap_or(0.0) as f64;
    Some(EditorFocus {
        center_world: body_state.position + dir_world * (radius_m + height_m.max(0.0)),
        up_world: dir_world,
    })
}

/// React to the open/close *edge*: force ship view (the god-view is a 3D view,
/// so the editor must not run over the orbital map) and hide the flight HUD.
///
/// Unlike the shipyard editor this does **not** swap to a dedicated camera or
/// hide the world — the planet stays visible and the god-view camera (gated
/// `base_editor_open`) repositions the ship camera in place. `apply_active_camera`
/// (ungated, keyed on `ViewMode` change) activates the ship camera when we force
/// `ViewMode::Ship`; on close the previous view is restored and the flight
/// camera systems (`SimStage::Camera`) un-gate and take the camera back.
///
/// Edge-detected via a `Local` so mode/active-site changes (which also dirty the
/// resource) don't re-capture the saved view or re-run the open transition.
fn apply_open_state(
    editor: Res<BaseEditor>,
    mut view: ResMut<ViewMode>,
    mut last_open: Local<bool>,
    mut prev_view: Local<Option<ViewMode>>,
    mut hud: ParamSet<(
        Query<&mut Visibility, With<crate::hud::HudPanel>>,
        Query<&mut Visibility, With<crate::photo_mode::HideInPhotoMode>>,
    )>,
) {
    if editor.open == *last_open {
        return;
    }
    *last_open = editor.open;

    if editor.open {
        *prev_view = Some(*view);
        if *view != ViewMode::Ship {
            *view = ViewMode::Ship;
        }
    } else if let Some(prev) = prev_view.take()
        && *view != prev
    {
        *view = prev;
    }

    let target = if editor.open {
        Visibility::Hidden
    } else {
        Visibility::Inherited
    };
    for mut vis in hud.p0().iter_mut() {
        if *vis != target {
            *vis = target;
        }
    }
    for mut vis in hud.p1().iter_mut() {
        if *vis != target {
            *vis = target;
        }
    }
}
