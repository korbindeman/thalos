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
mod launch_select;
mod pick;
mod place;
mod ui;

pub use launch_select::SpaceportLaunchRequest;

use bevy::math::{DQuat, DVec2, DVec3};
use bevy::prelude::*;
use big_space::prelude::{CellCoord, Grid};
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::rendering::{SimulationState, SolarSystemState};
use crate::structures::{
    Facility, StructureId, StructureKind, StructurePlacement, StructureRegistry,
};
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
    /// Pick a launch point (one of the base's runways or launchpads) to place +
    /// launch the current craft from. Opened programmatically by the
    /// launch-select flow (the shipyard's LAUNCH), not the pause-menu SURFACE
    /// BASE button; `active_site` is the spaceport basin. See
    /// [`launch_select`](self::launch_select).
    SelectLaunch,
}

/// Run condition: the base editor is open.
pub fn base_editor_open(editor: Option<Res<BaseEditor>>) -> bool {
    editor.map(|e| e.open).unwrap_or(false)
}

pub struct BaseEditorPlugin;

impl Plugin for BaseEditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BaseEditor>()
            .add_plugins(camera::BaseEditorCameraPlugin)
            .add_plugins(pick::BaseEditorPickPlugin)
            .add_plugins(place::BaseEditorPlacePlugin)
            .add_plugins(connections::BaseEditorConnectionsPlugin)
            .add_plugins(launch_select::BaseEditorLaunchSelectPlugin)
            .add_plugins(ui::BaseEditorUiPlugin)
            .add_systems(Update, apply_open_state);
    }
}

/// Author the **default base**: a spaceport laid out on the shared flat basin
/// (everything coplanar — `Drape` at `pad_r`), with the paving links between the
/// placed features generated automatically as a **typed connection network**
/// (taxiways / aprons / roads / crawlerways). The two numbered runways (the
/// primary plus the angled crosswind secondary forming a **V** off its `−along`
/// threshold) are spawned by [`crate::runway`]; this lays out everything else
/// and the pavement between it:
///
/// - **Airside** (`ConnectionKind::Taxiway`) — one **core campus** on the same
///   (`+off`) side as the launch complex, reading outward from the strip:
///   runway → full-length parallel taxiway with evenly-spaced connectors → a
///   **large apron** (`ConnectionKind::Apron`, auto-derived from the hangar
///   row, which stands *on* it, ramp all around) → landside → launch complex.
///   Nothing is authored between the runways. The angled secondary hangs off
///   the core taxiway through three **curved link taxiways**
///   ([`connections::spawn_authored_path`] fillets) that cross the primary
///   strip — normal runway crossings, rendered under the runway's higher
///   paving — and sweep tangentially onto the secondary's parallel-taxiway
///   line. Run-up/holding aprons sit at both primary thresholds.
/// - **Launch complex** (`+off`, behind the core) — the **VAB**-scale assembly
///   building (the enterable shipyard [`Facility::Vab`]) nearest the runway,
///   then two large **launchpads** set furthest out (well back from the runway,
///   beyond the VAB), each flanked by a **flame diverter** and a **fuel/tank
///   farm**. A single **crawlerway** (`ConnectionKind::Crawlerway`) leaves the
///   VAB's one pad-facing (`+off`) door, runs straight out, then forks to the
///   two pads — so both pad routes share the same VAB exit (the future
///   crawler-transporter route).
/// - **Landside** (`ConnectionKind::Road`) — one curved service road behind the
///   apron: ops → east blockhouse → behind the VAB (its landside/runway-side
///   face) → west blockhouse. The VAB's pad-facing door and crawlerway are on
///   the far (`+off`) side, so the road stays clear of them.
///
/// Called by the runway scenario after it installs the basin `BaseSite`.
/// `pad_r = radius_m + E`; `sec_heading` is the second runway's takeoff heading
/// (the secondary taxiway is laid out in its rotated frame). Layout is in
/// runway-frame `(along, off)` metres from the runway centre, `+off` = the
/// launch-complex side (`heading × center_dir`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_default_base(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<thalos_body_render::ShadowedStandardMaterial>,
    registry: &mut StructureRegistry,
    root: Entity,
    body_id: BodyId,
    basin_site_id: StructureId,
    center_dir: DVec3,
    heading: DVec3,
    sec_heading: DVec3,
    pad_r: f64,
) {
    let mats = place::BaseMaterials::create(materials);
    let across = heading.cross(center_dir).normalize();
    let dir =
        |along: f64, off: f64| (center_dir * pad_r + heading * along + across * off).normalize();

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

    // Launch-complex geometry (`+off` side), in runway-frame (along, off) metres.
    // The VAB sits nearest the runway of the complex; the two launchpads sit
    // furthest out — well back from the runway, beyond the VAB — each flanking
    // the centreline. The VAB has a single crawler exit facing the pads
    // (`+off`): its crawlerway runs straight out that door to `CRAWLER_FORK_OFF`,
    // then forks to the two pads (see the crawlerway network below). Pushing the
    // pads (and their beyond-pad flame diverters / tank farms at `+112` / `+235`)
    // this far out still keeps the tank farm inside the flattened basin, whose
    // `+off` reach is ~1.4 km (see `runway::BASIN_*`).
    const PAD_ALONG: f64 = 700.0;
    const PAD_OFF: f64 = 1050.0;
    const VAB_OFF: f64 = 720.0;
    // Where the VAB's single crawlerway forks toward the two pads (`+off`,
    // between the VAB's pad-facing door and the pads).
    const CRAWLER_FORK_OFF: f64 = 900.0;
    // Airside geometry, on the SAME (`+off`) side as the launch complex — one
    // core campus reading outward from the strip: runway → parallel taxiway →
    // large apron with the hangar row standing ON it → landside (ops /
    // blockhouses) → launch complex (pads / VAB / tanks) behind. Nothing is
    // authored between the runways; the angled secondary (`−off` side) is
    // reached by the taxiway wrapping around the primary's `−along` threshold
    // (a curved end-around — taxi to the secondary goes through the primary's
    // system, never across a strip).
    let pri_half_len: f64 = crate::runway::RUNWAY_HALF_LENGTH_M;
    // Runway-edge station connectors end here: 1 m *inside* the strip edge, so
    // the joint tucks under the runway's higher paving (seamless, no grass
    // sliver) while nothing paved ever spans the strip itself.
    let pri_edge_off: f64 = crate::runway::RUNWAY_HALF_WIDTH_M - 1.0;
    const TWY_OFF: f64 = 190.0; // primary parallel taxiway centreline
    /// Half the `ConnectionKind::Taxiway` strip width (44 m — see
    /// `connections::ConnectionKind::style`), for abutting aprons to it.
    const TWY_HALF_WIDTH: f64 = 22.0;
    const HANGAR_OFF: f64 = 500.0; // hangar row, standing on the apron
    const HANGAR_HALF_X: f64 = 30.0;
    const HANGAR_HALF_Z: f32 = 20.0;
    let vab_kind = building(68.0, 46.0, 96.0);
    let ops_kind = building(16.0, 12.0, 12.0);

    let mut place_one = |along: f64, off: f64, kind: StructureKind| -> StructureId {
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
        )
    };

    // --- Launch complex ---
    place_one(PAD_ALONG, PAD_OFF, launchpad(50.0));
    place_one(-PAD_ALONG, PAD_OFF, launchpad(50.0));
    let vab = place_one(0.0, VAB_OFF, vab_kind);
    // Flame diverters just outboard of each pad.
    place_one(PAD_ALONG, PAD_OFF + 112.0, building(14.0, 44.0, 4.0));
    place_one(-PAD_ALONG, PAD_OFF + 112.0, building(14.0, 44.0, 4.0));
    // Fuel / tank farm beyond each pad (three tanks each).
    for side in [1.0, -1.0] {
        for dx in [-42.0, 0.0, 42.0] {
            place_one(side * PAD_ALONG + dx, PAD_OFF + 235.0, tank(9.0, 26.0));
        }
    }

    // --- Landside buildings ---
    // Ops/tower set well past the apron along the strip so it never sits on the
    // paving.
    place_one(1100.0, 300.0, ops_kind);
    place_one(PAD_ALONG, PAD_OFF - 240.0, building(10.0, 10.0, 8.0));
    place_one(-PAD_ALONG, PAD_OFF - 240.0, building(10.0, 10.0, 8.0));

    // --- Hangars, standing on the core apron. The apron below is auto-derived
    // from this row, so adding/moving hangars grows it to match. ---
    let hangar_alongs = [-750.0, -250.0, 250.0, 750.0];
    for &h_along in &hangar_alongs {
        place_one(
            h_along,
            HANGAR_OFF,
            building(HANGAR_HALF_X as f32, HANGAR_HALF_Z, 22.0),
        );
    }

    // The VAB is the enterable shipyard facility for the space-center hub.
    registry.set_facility(vab, Facility::Vab);

    // `place_one`'s last use is above, so its `commands`/`meshes`/`registry`
    // borrows end here — freeing them for the connection spawners below.

    // --- Taxiways. The core-side parallel taxiway runs the primary's full
    // length, straight. The secondary's system hangs off it through three
    // curved **link taxiways** sweeping across the V interior. Each crossing
    // is split at the strip: a straight stub from the core taxiway stops at
    // the near runway edge, and the curved link resumes at the far edge — no
    // paving ever spans the runway; the strip + its markings stay untouched
    // between the two ends (taxi across is over the runway's own asphalt).
    // The link then curves tangentially onto the secondary's parallel-taxiway
    // line: the first link, at the primary threshold, *is* that line's start
    // and runs its full length; the other two merge into it at a hair-lower
    // lift so the overlap renders as one strip. `sd` is the secondary strip
    // direction in (along, off), `sn` its perpendicular toward the primary;
    // `sp(s, t)` maps stations along the strip (`s` from the near threshold)
    // + offsets toward the primary (`t`) into the base frame. ---
    let sd = DVec2::new(sec_heading.dot(heading), sec_heading.dot(across));
    let sn = DVec2::new(-sd.y, sd.x);
    let sec_near = DVec2::new(
        crate::runway::SEC_NEAR_ALONG_M,
        -crate::runway::SEC_NEAR_ACROSS_M,
    );
    let sp = |s: f64, t: f64| sec_near + sd * s + sn * t;
    const SEC_TWY_T: f64 = 170.0; // secondary parallel taxiway offset
    // 1 m inside the secondary's strip edge (half-width 40), tucked under its
    // paving like `pri_edge_off`.
    const SEC_EDGE_T: f64 = 39.0;
    const LINK_FILLET_M: f64 = 300.0; // sweep radius of the link taxiways
    let sec_len = crate::runway::SECONDARY_LENGTH_M;
    let q0 = sp(0.0, SEC_TWY_T);
    // Intersection of the constant-`along` line at `c` with the secondary
    // taxiway centreline — where a link dropping straight down from the core
    // taxiway meets it.
    let line_at = |c: f64| {
        let s = (c - q0.x) / sd.x;
        DVec2::new(c, q0.y + s * sd.y)
    };
    // Core parallel taxiway: straight, full length.
    connections::spawn_authored_path(
        commands,
        meshes,
        &mats,
        root,
        body_id,
        basin_site_id,
        center_dir,
        heading,
        pad_r,
        connections::ConnectionKind::Taxiway,
        &[
            DVec2::new(pri_half_len - 30.0, TWY_OFF),
            DVec2::new(-(pri_half_len - 30.0), TWY_OFF),
        ],
        0.0,
        0.0,
    );
    // Link 1, at the primary threshold: resume at the far runway edge, sweep
    // onto the secondary line and run it out to the far end.
    let c1 = -(pri_half_len - 70.0);
    connections::spawn_authored_path(
        commands,
        meshes,
        &mats,
        root,
        body_id,
        basin_site_id,
        center_dir,
        heading,
        pad_r,
        connections::ConnectionKind::Taxiway,
        &[
            DVec2::new(c1, -pri_edge_off),
            line_at(c1),
            sp(sec_len, SEC_TWY_T),
        ],
        LINK_FILLET_M,
        0.0,
    );
    // Links 2–3, midfield: same shape, ending just past tangency (the short
    // straight tail hides under link 1's strip via the lower lift).
    for c in [-1400.0, -700.0] {
        let merge = line_at(c);
        connections::spawn_authored_path(
            commands,
            meshes,
            &mats,
            root,
            body_id,
            basin_site_id,
            center_dir,
            heading,
            pad_r,
            connections::ConnectionKind::Taxiway,
            &[
                DVec2::new(c, -pri_edge_off),
                merge,
                merge + sd * (LINK_FILLET_M + 60.0),
            ],
            LINK_FILLET_M,
            -0.006,
        );
    }
    // Straight perpendicular connectors, all stopping at the near runway edge:
    // the core-side halves of the three link crossings, a threshold connector
    // at the east primary end (through the holding pad, for full-length
    // departures), evenly spaced exits between them, and the secondary's
    // thresholds + midfield.
    let mut cn: Vec<(f64, f64, f64)> = Vec::new();
    let mut ce: Vec<(usize, usize)> = Vec::new();
    let mut stub = |a: DVec2, b: DVec2| {
        cn.push((a.x, a.y, 0.0));
        cn.push((b.x, b.y, 0.0));
        ce.push((cn.len() - 2, cn.len() - 1));
    };
    for c in [
        c1, -2100.0, -1400.0, -700.0, 0.0, 700.0, 1400.0, 2100.0, 2430.0,
    ] {
        stub(DVec2::new(c, TWY_OFF), DVec2::new(c, pri_edge_off));
    }
    for s in [60.0, sec_len * 0.5, sec_len - 60.0] {
        stub(sp(s, SEC_TWY_T), sp(s, SEC_EDGE_T));
    }
    connections::spawn_authored_network(
        commands,
        meshes,
        &mats,
        root,
        body_id,
        basin_site_id,
        center_dir,
        heading,
        pad_r,
        connections::ConnectionKind::Taxiway,
        &cn,
        Some(&ce),
    );

    // --- The core apron (auto-generated from the hangar row): one large ramp
    // the hangars stand ON — it spans the row plus a parking margin each end
    // and fills from the parallel taxiway to behind the hangars' rear wall, so
    // pavement surrounds them entirely like a real airport ramp. ---
    const APRON_END_MARGIN_M: f64 = 70.0;
    const APRON_BACK_M: f64 = 40.0; // pavement behind the hangar rear wall
    let (row_min, row_max) = hangar_alongs
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &a| (lo.min(a), hi.max(a)));
    let apron_a0 = row_min - HANGAR_HALF_X - APRON_END_MARGIN_M;
    let apron_a1 = row_max + HANGAR_HALF_X + APRON_END_MARGIN_M;
    // Tuck the apron under the taxiway strip so they abut with no sliver gap
    // (the taxiway's slightly higher lift renders over it).
    let apron_off0 = TWY_OFF;
    let apron_off1 = HANGAR_OFF + f64::from(HANGAR_HALF_Z) + APRON_BACK_M;
    connections::spawn_authored_apron(
        commands,
        meshes,
        &mats,
        root,
        body_id,
        basin_site_id,
        center_dir,
        heading,
        pad_r,
        (apron_a0 + apron_a1) * 0.5,
        (apron_off0 + apron_off1) * 0.5,
        (apron_a1 - apron_a0) * 0.5,
        (apron_off1 - apron_off0).abs() * 0.5,
    );

    // --- Run-up / holding aprons at each primary threshold: fill the whole
    // band between the runway edge and the parallel taxiway (abutting both) —
    // the paved holding pads real fields keep beside the runway ends. ---
    let bay_inner = pri_edge_off;
    let bay_outer = TWY_OFF - TWY_HALF_WIDTH;
    for side in [-1.0, 1.0] {
        connections::spawn_authored_apron(
            commands,
            meshes,
            &mats,
            root,
            body_id,
            basin_site_id,
            center_dir,
            heading,
            pad_r,
            side * (pri_half_len - 200.0),
            (bay_inner + bay_outer) * 0.5,
            140.0,
            (bay_outer - bay_inner) * 0.5,
        );
    }

    // --- Road (landside): one curved service road behind the apron — ops →
    // east blockhouse → behind the VAB (its landside/runway-side face) → west
    // blockhouse. The VAB's pad-facing door and crawlerway are on the far
    // (`+off`) side, so this stays clear of them. ---
    connections::spawn_authored_path(
        commands,
        meshes,
        &mats,
        root,
        body_id,
        basin_site_id,
        center_dir,
        heading,
        pad_r,
        connections::ConnectionKind::Road,
        &[
            DVec2::new(1100.0, 300.0),               // ops
            DVec2::new(PAD_ALONG, PAD_OFF - 240.0),  // east blockhouse
            DVec2::new(0.0, VAB_OFF - 150.0),        // behind the VAB (landside)
            DVec2::new(-PAD_ALONG, PAD_OFF - 240.0), // west blockhouse
        ],
        120.0,
        0.0,
    );

    // --- Crawlerway: the VAB has one exit, facing the pads (`+off`). A single
    // trunk leaves that door and runs straight out to a fork, which then splits
    // to each launchpad — so both pad routes share the same VAB door instead of
    // fanning off different faces. Explicit edges (trunk + two forks), not an
    // MST. ---
    let crawler_nodes: Vec<(f64, f64, f64)> = vec![
        (0.0, VAB_OFF, place::kind_bounding_m(&vab_kind)), // 0: VAB door (+off face)
        (0.0, CRAWLER_FORK_OFF, 0.0),                      // 1: fork junction
        (PAD_ALONG, PAD_OFF, 50.0),                        // 2: east pad
        (-PAD_ALONG, PAD_OFF, 50.0),                       // 3: west pad
    ];
    connections::spawn_authored_network(
        commands,
        meshes,
        &mats,
        root,
        body_id,
        basin_site_id,
        center_dir,
        heading,
        pad_r,
        connections::ConnectionKind::Crawlerway,
        &crawler_nodes,
        Some(&[(0, 1), (1, 2), (1, 3)]),
    );
}

/// Cast the screen cursor onto a body-centred sphere of radius `pad_r`
/// (heliocentric metres) and return the **body-fixed** unit direction of the
/// nearest hit, or `None` on a miss / missing input. The single canonical pick
/// used by every god-view cursor picker (site pick, building placement, launch
/// point, hub building select).
///
/// Rather than raycasting in render space off the camera's `GlobalTransform`
/// (as the older per-picker helpers did), this reads the god-view camera's
/// **current-frame** pose straight from its `CellCoord` + `Transform`. The
/// god-view driver
/// ([`crate::god_view::drive_god_view`]) writes the camera `Transform` in
/// `Update`, but `GlobalTransform` is only propagated in `PostUpdate`, so a
/// same-frame raycast off `GlobalTransform` uses *last* frame's camera pose — the
/// pick then visibly desyncs from the rendered buildings when the camera pans
/// fast (the body's render position is stable frame-to-frame; only the floating
/// origin's own render pose lags). Every god-view picker must therefore (a) call
/// this and (b) run `.after(`[`crate::god_view::GodViewCameraSet`]`)` so the
/// `Transform` it reads is the pose the driver just wrote this frame.
///
/// Works entirely in the heliocentric f64 frame — big_space cells are pure
/// translations and `SHIP_SCALE == 1`, so it is exact across floating-origin cell
/// recentering (which a render-space reconstruction from the fresh local
/// `Transform` would not be, since the body's `GlobalTransform` is anchored to
/// the previous frame's origin cell).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cursor_body_dir(
    camera: &Camera,
    cam_cell: &CellCoord,
    cam_transform: &Transform,
    root_grid: &Grid,
    cursor: Vec2,
    body_position: DVec3,
    body_orientation: DQuat,
    pad_r: f64,
) -> Option<DVec3> {
    // Cursor ray *direction* in world space. `viewport_to_world` needs a
    // `GlobalTransform`, but only its rotation affects the direction, and the
    // camera's world rotation equals its local `Transform.rotation` (it is a
    // direct child of the identity-transform BigSpace root). A rotation-only
    // transform therefore yields the correct world direction; its origin is
    // discarded — we supply the exact f64 origin below.
    let rot_only = GlobalTransform::from(Transform::from_rotation(cam_transform.rotation));
    let ray = camera.viewport_to_world(&rot_only, cursor).ok()?;
    let dir = (*ray.direction).as_dvec3();

    // Camera world position from its fresh grid cell + local offset.
    let origin = root_grid.grid_position_double(cam_cell, cam_transform);

    // Ray–sphere against the body sphere, in heliocentric metres (nearest hit:
    // the near root if in front of the camera, else the far root).
    let oc = origin - body_position;
    let b = oc.dot(dir);
    let c = oc.length_squared() - pad_r * pad_r;
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
    let hit = origin + dir * t;
    Some((body_orientation.inverse() * (hit - body_position)).normalize())
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

    if matches!(
        editor.mode,
        BaseEditorMode::PlaceBuildings | BaseEditorMode::SelectLaunch
    ) && let Some(site_id) = editor.active_site
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
