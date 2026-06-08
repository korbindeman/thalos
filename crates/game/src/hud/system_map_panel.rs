//! Burn-time local-system schematic (top-right HUD panel).
//!
//! While the ship is burning in space (or has a pending maneuver node), this
//! panel shows a simplified top-down plot of the ship's local system —
//! concentric orbit rings for the dominant body's satellites, the dominant
//! body at the centre, the ship, its current ballistic trajectory (solid), and
//! the maneuver-node projected trajectory (dotted). It surfaces, in the ship
//! ("main") view, the trajectory information that is otherwise only visible in
//! the map view, so a burn can be reasoned about without leaving the cockpit.
//!
//! Everything is drawn by `assets/shaders/system_map.wgsl`; this module
//! projects the simulation state into the shader's normalised [-1, 1] space.
//! The [`SystemMapData`] layout must stay in lock-step with the WGSL
//! `SystemMapData` struct.

use bevy::math::{DVec2, DVec3};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use thalos_physics_canonical::trajectory::TrajectoryBranchStack;
use thalos_physics_canonical::types::BodyState;
use thalos_world::BodyId;

use crate::fuel::ThrottleState;
use crate::hud::HudPanel;
use crate::hud::theme::{HudTheme, label, panel_frame, panel_node};
use crate::rendering::{SimulationState, SolarSystemState};
use crate::view::ViewMode;

/// Max orbit rings (dominant-body satellites) drawn. Mirror in the shader.
const MAX_RINGS: usize = 8;
/// Max sample points per trajectory line. Mirror in the shader.
const MAX_TRAJ: usize = 96;

/// Side length of the square plot, in logical px.
const MAP_SIZE_PX: f32 = 200.0;
/// Fraction of the half-extent the fit radius maps to (leaves a margin).
const NDC_FIT: f64 = 0.82;
/// Cap on how far the plot zooms out relative to the ship's current radius, so
/// an escape hyperbola can't shrink the useful near-field to a dot.
const MAX_ZOOM_RATIO: f64 = 16.0;
/// Keep the panel up briefly after a burn ends so throttle blips don't flicker.
const BURN_LINGER_SECS: f32 = 1.0;

// Shape sizes / line widths, in the shader's normalised half-extent units
// (1.0 == half the plot, i.e. MAP_SIZE_PX * 0.5 px).
const CENTRAL_R: f32 = 0.06;
const SHIP_R: f32 = 0.035;
const NODE_R: f32 = 0.04;
const LINE_HW: f32 = 0.014;
const RING_HW: f32 = 0.008;
const BODY_DOT_R: f32 = 0.025;
const DASH_PERIOD: f32 = 0.09;
const DASH_DUTY: f32 = 0.5;

/// Uniform mirror of the WGSL `SystemMapData`. All fields are `Vec4`/arrays of
/// `Vec4` so the std140 layout has no scalar-padding surprises.
#[derive(Clone, ShaderType)]
pub(super) struct SystemMapData {
    /// x = ring_count, y = solid_count, z = dotted_count, w = node_flag.
    params: Vec4,
    /// x = central radius, y = ship radius, z = node radius, w = line half-width.
    geom: Vec4,
    /// x = ring half-width, y = dash period, z = dash duty, w = body-dot radius.
    style: Vec4,
    /// xy = ship marker, zw = maneuver-node marker.
    markers: Vec4,
    col_central: Vec4,
    col_ring: Vec4,
    col_body: Vec4,
    col_solid: Vec4,
    col_dotted: Vec4,
    col_ship: Vec4,
    col_node: Vec4,
    /// per ring: xy = body pos, z = ring radius, w = unused.
    rings: [Vec4; MAX_RINGS],
    /// per point: xy = position, z = cumulative arc-length, w = valid.
    solid: [Vec4; MAX_TRAJ],
    dotted: [Vec4; MAX_TRAJ],
}

impl Default for SystemMapData {
    fn default() -> Self {
        Self {
            params: Vec4::ZERO,
            geom: Vec4::ZERO,
            style: Vec4::ZERO,
            markers: Vec4::ZERO,
            col_central: Vec4::ZERO,
            col_ring: Vec4::ZERO,
            col_body: Vec4::ZERO,
            col_solid: Vec4::ZERO,
            col_dotted: Vec4::ZERO,
            col_ship: Vec4::ZERO,
            col_node: Vec4::ZERO,
            rings: [Vec4::ZERO; MAX_RINGS],
            solid: [Vec4::ZERO; MAX_TRAJ],
            dotted: [Vec4::ZERO; MAX_TRAJ],
        }
    }
}

#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub(super) struct SystemMapMaterial {
    #[uniform(0)]
    data: SystemMapData,
}

impl UiMaterial for SystemMapMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/system_map.wgsl".into()
    }
}

/// Marker on the panel root (drives visibility).
#[derive(Component)]
pub(super) struct SystemMapRoot;

/// Marker on the `MaterialNode` canvas (carries the live material handle).
#[derive(Component)]
pub(super) struct SystemMapCanvas;

pub(super) fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<SystemMapMaterial>>,
    theme: Res<HudTheme>,
) {
    let mut root = panel_node();
    // Right edge, between the top-right FPS overlay and the bottom-right
    // staging stack.
    root.right = Val::Px(20.0);
    root.top = Val::Px(140.0);
    root.row_gap = Val::Px(6.0);

    let (bg, border) = panel_frame(&theme);
    let material = materials.add(SystemMapMaterial {
        data: SystemMapData::default(),
    });

    commands
        .spawn((
            root,
            bg,
            border,
            Visibility::Hidden,
            SystemMapRoot,
            HudPanel,
            Name::new("HudSystemMap"),
        ))
        .with_children(|p| {
            p.spawn(label(&theme, "TRAJECTORY"));
            p.spawn((
                Node {
                    width: Val::Px(MAP_SIZE_PX),
                    height: Val::Px(MAP_SIZE_PX),
                    ..default()
                },
                MaterialNode(material),
                SystemMapCanvas,
                Name::new("HudSystemMapCanvas"),
            ));
        });
}

#[allow(clippy::too_many_arguments)]
pub(super) fn update(
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    throttle: Res<ThrottleState>,
    view: Res<ViewMode>,
    time: Res<Time>,
    theme: Res<HudTheme>,
    mut materials: ResMut<Assets<SystemMapMaterial>>,
    mut root_q: Query<&mut Visibility, With<SystemMapRoot>>,
    canvas_q: Query<&MaterialNode<SystemMapMaterial>, With<SystemMapCanvas>>,
    mut last_burn: Local<Option<f32>>,
) {
    let Ok(mut vis) = root_q.single_mut() else {
        return;
    };

    // A live branch stack only exists for an in-space ballistic craft — the
    // prediction is cleared while landed / grounded / in terrain contact (see
    // `bridge::update_prediction`), so this doubles as the "in space" gate.
    let branches = sim.simulation.trajectory_branches();
    let has_nodes = branches.is_some_and(|b| !b.branches.is_empty());

    let now = time.elapsed_secs();
    if throttle.effective > 0.02 {
        *last_burn = Some(now);
    }
    let recently_burning = last_burn.is_some_and(|t| now - t < BURN_LINGER_SECS);

    let show =
        matches!(*view, ViewMode::Ship) && branches.is_some() && (recently_burning || has_nodes);

    if !show {
        if *vis != Visibility::Hidden {
            *vis = Visibility::Hidden;
        }
        return;
    }
    if *vis != Visibility::Inherited {
        *vis = Visibility::Inherited;
    }

    let (Some(branches), Some(states)) = (branches, solar.states.as_deref()) else {
        return;
    };
    let Ok(canvas) = canvas_q.single() else {
        return;
    };
    let Some(material) = materials.get_mut(canvas) else {
        return;
    };

    build_data(&mut material.data, &sim, branches, states, &theme);
}

fn build_data(
    data: &mut SystemMapData,
    sim: &SimulationState,
    branches: &TrajectoryBranchStack,
    states: &[BodyState],
    theme: &HudTheme,
) {
    let dominant = sim.simulation.dominant_body();
    let center = states.get(dominant).map_or(DVec3::ZERO, |s| s.position);

    // Project an inertial sample into the dominant-body-centred ecliptic plane
    // (Thalos uses XZ with Y up). Pins to the anchor body's *current* position
    // so the trajectory stays glued to where the body is now, matching the map.
    let project = |pos: DVec3, anchor: BodyId, ref_pos: DVec3| -> DVec2 {
        let anchor_now = states.get(anchor).map_or(ref_pos, |s| s.position);
        let rel = (pos - ref_pos) + anchor_now - center;
        DVec2::new(rel.x, rel.z)
    };

    let has_nodes = !branches.branches.is_empty();

    // Current (no-maneuver) path, drawn solid.
    let mut solid_pts: Vec<DVec2> = Vec::new();
    for seg in branches.actual.plan.segments() {
        for s in &seg.samples {
            solid_pts.push(project(s.position, s.anchor_body, s.ref_pos));
        }
    }

    // Projected (all-maneuvers-applied) path, drawn dotted — only when nodes
    // actually fork the trajectory.
    let mut dotted_pts: Vec<DVec2> = Vec::new();
    if has_nodes {
        for seg in branches.active_plan.segments() {
            for s in &seg.samples {
                dotted_pts.push(project(s.position, s.anchor_body, s.ref_pos));
            }
        }
    }

    let ship_rel = sim.simulation.ship_state().position - center;
    let ship2d = DVec2::new(ship_rel.x, ship_rel.z);

    // Maneuver-node marker: start of the first leg that carries a burn.
    let mut node2d: Option<DVec2> = None;
    if has_nodes {
        for leg in branches.active_plan.legs() {
            if leg.applied_delta_v.is_some() {
                let seg = leg.burn_segment.as_ref().unwrap_or(&leg.coast_segment);
                if let Some(s) = seg.samples.first() {
                    node2d = Some(project(s.position, s.anchor_body, s.ref_pos));
                    break;
                }
            }
        }
    }

    // Satellites of the dominant body → concentric rings + dots.
    let mut children: Vec<(f64, DVec2)> = Vec::new();
    for body in &sim.system.bodies {
        if body.parent == Some(dominant) {
            if let Some(bs) = states.get(body.id) {
                let r = bs.position - center;
                let p2 = DVec2::new(r.x, r.z);
                children.push((p2.length(), p2));
            }
        }
    }

    // Fit to the trajectory (so the burn-relevant orbits dominate the frame),
    // capped relative to the ship's radius so escapes don't collapse the view.
    let ship_radius = ship2d.length().max(1.0);
    let mut fit = ship_radius;
    for p in solid_pts.iter().chain(dotted_pts.iter()) {
        fit = fit.max(p.length());
    }
    fit = fit.min(ship_radius * MAX_ZOOM_RATIO).max(1.0);

    // Keep only rings that fall (roughly) inside the framed region.
    children.retain(|(r, _)| *r <= fit * 1.3);
    children.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    children.truncate(MAX_RINGS);

    let scale = NDC_FIT / fit; // world metres → normalised plot units

    *data = SystemMapData::default();
    data.geom = Vec4::new(CENTRAL_R, SHIP_R, NODE_R, LINE_HW);
    data.style = Vec4::new(RING_HW, DASH_PERIOD, DASH_DUTY, BODY_DOT_R);
    data.col_central = lin(Color::srgb(0.46, 0.44, 0.40));
    data.col_ring = lin(Color::srgba(0.56, 0.58, 0.52, 0.30));
    data.col_body = lin(Color::srgba(0.74, 0.76, 0.72, 0.90));
    data.col_solid = lin_with_alpha(theme.text_accent, 0.92); // amber = current path
    data.col_dotted = lin_with_alpha(theme.text_datum_sea, 0.95); // cyan = planned path
    data.col_ship = lin(Color::srgb(0.55, 0.92, 0.50));
    data.col_node = lin(Color::srgb(0.95, 0.75, 0.30));

    let ship_ndc = (ship2d * scale).as_vec2();
    let node_ndc = node2d.map_or(Vec2::ZERO, |n| (n * scale).as_vec2());
    data.markers = Vec4::new(ship_ndc.x, ship_ndc.y, node_ndc.x, node_ndc.y);

    for (i, (r, p)) in children.iter().enumerate() {
        let pn = (*p * scale).as_vec2();
        data.rings[i] = Vec4::new(pn.x, pn.y, (*r * scale) as f32, BODY_DOT_R);
    }

    let solid_count = fill_line(&mut data.solid, &solid_pts, scale);
    let dotted_count = fill_line(&mut data.dotted, &dotted_pts, scale);

    data.params = Vec4::new(
        children.len() as f32,
        solid_count as f32,
        dotted_count as f32,
        if node2d.is_some() { 1.0 } else { 0.0 },
    );
}

/// Decimate `pts` to at most [`MAX_TRAJ`], scale into plot units, and record a
/// per-point cumulative arc-length (used for the dotted line's dashes).
/// Returns the number of points written.
fn fill_line(out: &mut [Vec4; MAX_TRAJ], pts: &[DVec2], scale: f64) -> usize {
    if pts.len() < 2 {
        return 0;
    }
    let n = pts.len();
    let count = n.min(MAX_TRAJ);
    let mut arc = 0.0f32;
    let mut prev = Vec2::ZERO;
    for (i, slot) in out.iter_mut().take(count).enumerate() {
        // Even stride that always keeps the first and last point.
        let idx = if n <= MAX_TRAJ {
            i
        } else {
            (i * (n - 1)) / (MAX_TRAJ - 1)
        };
        let p = (pts[idx] * scale).as_vec2();
        if i > 0 {
            arc += p.distance(prev);
        }
        prev = p;
        *slot = Vec4::new(p.x, p.y, arc, 1.0);
    }
    count
}

fn lin(color: Color) -> Vec4 {
    color.to_linear().to_vec4()
}

fn lin_with_alpha(color: Color, alpha: f32) -> Vec4 {
    let s = color.to_srgba();
    Color::srgba(s.red, s.green, s.blue, alpha)
        .to_linear()
        .to_vec4()
}
