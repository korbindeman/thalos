//! Trajectory widget: a top-down (XZ ecliptic) schematic of the ship's local
//! system — concentric orbit rings for the dominant body's satellites, the
//! dominant body at the centre, the ship, its current ballistic trajectory
//! (solid), and the maneuver-node projected trajectory (dotted).
//!
//! This is the ship-view surfacing of the trajectory information otherwise
//! only visible in the map view, so a burn can be reasoned about without
//! leaving the cockpit. Ported from the old `system_map_panel`; the only
//! behavioural change is that it is **not relevant in atmosphere** (the old
//! panel popped up during jet cruise), and the MFD selector — not this
//! module — owns the root's visibility.
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

use crate::hud::theme::{HudTheme, label};
use crate::rendering::{SimulationState, SolarSystemState};

use super::super::{ActiveWidget, FlightContext, MfdWidgetRoot, WidgetKind};

/// Max orbit rings (dominant-body satellites) drawn. Mirror in the shader.
const MAX_RINGS: usize = 8;
/// Max sample points per trajectory line. Mirror in the shader.
const MAX_TRAJ: usize = 96;

/// Side length of the square plot, in logical px.
const MAP_SIZE_PX: f32 = 200.0;
/// Fraction of the half-extent a zoom level's radius maps to (leaves a margin).
const NDC_FIT: f64 = 0.82;

/// Discrete zoom ladder: each entry is the view radius (panel half-extent) in
/// metres. The plot snaps to the smallest level that contains the relevant
/// content, so the scale changes in fixed notches instead of drifting
/// continuously as a burn proceeds. 1-2-5 ladder from 1 000 km to ~6.7 AU.
const ZOOM_LEVELS_M: [f64; 19] = [
    1.0e6, 2.0e6, 5.0e6, 1.0e7, 2.0e7, 5.0e7, 1.0e8, 2.0e8, 5.0e8, 1.0e9, 2.0e9, 5.0e9, 1.0e10,
    2.0e10, 5.0e10, 1.0e11, 2.0e11, 5.0e11, 1.0e12,
];
/// Step down a notch only when content shrinks to well inside the smaller
/// level, so content hovering near a boundary doesn't oscillate the zoom.
const ZOOM_DOWN_HYSTERESIS: f64 = 0.8;

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
pub struct SystemMapData {
    /// x = ring_count, y = solid_count, z = dotted_count, w = node_flag.
    params: Vec4,
    /// x = central radius, y = ship radius, z = node radius, w = line half-width.
    geom: Vec4,
    /// x = ring half-width, y = dash period, z = dash duty, w = body-dot radius.
    style: Vec4,
    /// xy = ship marker, zw = maneuver-node marker.
    markers: Vec4,
    /// x = dominant-body disc radius (to scale), yz = sun direction (plot
    /// space, normalized), w = is-star flag (1 = dominant body is the star).
    extra: Vec4,
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
            extra: Vec4::ZERO,
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
pub struct SystemMapMaterial {
    #[uniform(0)]
    data: SystemMapData,
}

impl UiMaterial for SystemMapMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/system_map.wgsl".into()
    }
}

/// Marker on the `MaterialNode` canvas (carries the live material handle).
#[derive(Component)]
pub(crate) struct SystemMapCanvas;

/// Marker on the scale-readout text under the plot.
#[derive(Component)]
pub(crate) struct SystemMapScale;

/// Persistent zoom selection, kept across frames so the ladder snap has
/// hysteresis (held in a `Local` on [`update`]). Reset when the dominant body
/// changes, since the appropriate scale shifts by orders of magnitude.
#[derive(Default)]
pub(crate) struct ZoomState {
    idx: usize,
    dominant: Option<BodyId>,
}

/// Relevant only outside an atmosphere with a live prediction and a reason to
/// look at it (a recent burn or a pending maneuver node). The `!in_atmosphere`
/// gate is the fix for the panel popping up during jet cruise.
pub(crate) fn relevance(ctx: &FlightContext) -> Option<i32> {
    (!ctx.in_atmosphere && ctx.prediction_shown && (ctx.recently_burning || ctx.has_nodes))
        .then_some(60)
}

pub(crate) fn build(
    area: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    materials: &mut Assets<SystemMapMaterial>,
) {
    let material = materials.add(SystemMapMaterial {
        data: SystemMapData::default(),
    });
    area.spawn((
        Node {
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            row_gap: Val::Px(6.0),
            ..default()
        },
        Visibility::Hidden,
        MfdWidgetRoot {
            kind: WidgetKind::Trajectory,
        },
        Name::new("MfdTrajectory"),
    ))
    .with_children(|p| {
        p.spawn(label(theme, "TRAJECTORY"));
        p.spawn((
            Node {
                width: Val::Px(MAP_SIZE_PX),
                height: Val::Px(MAP_SIZE_PX),
                ..default()
            },
            MaterialNode(material),
            SystemMapCanvas,
            Name::new("MfdTrajectoryCanvas"),
        ));
        p.spawn((
            label(theme, "—"),
            SystemMapScale,
            Name::new("MfdTrajectoryScale"),
        ));
    });
}

pub(crate) fn update(
    active: Res<ActiveWidget>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    theme: Res<HudTheme>,
    mut materials: ResMut<Assets<SystemMapMaterial>>,
    canvas_q: Query<&MaterialNode<SystemMapMaterial>, With<SystemMapCanvas>>,
    mut scale_q: Query<&mut Text, With<SystemMapScale>>,
    mut zoom: Local<ZoomState>,
) {
    if active.0 != Some(WidgetKind::Trajectory) {
        return;
    }
    let Ok(canvas) = canvas_q.single() else {
        return;
    };
    let Some(mut material) = materials.get_mut(canvas) else {
        return;
    };

    // Pinned with no live prediction (e.g. landed): blank rather than show a
    // frozen pre-collapse orbit.
    let (Some(branches), Some(states)) = (
        sim.simulation.trajectory_branches(),
        solar.states.as_deref(),
    ) else {
        material.data = SystemMapData::default();
        return;
    };

    build_data(
        &mut material.data,
        &sim,
        branches,
        states,
        &theme,
        &mut zoom,
    );

    if let Ok(mut text) = scale_q.single_mut() {
        let s = fmt_view_radius(ZOOM_LEVELS_M[zoom.idx]);
        if text.0 != s {
            text.0 = s;
        }
    }
}

fn build_data(
    data: &mut SystemMapData,
    sim: &SimulationState,
    branches: &TrajectoryBranchStack,
    states: &[BodyState],
    theme: &HudTheme,
    zoom: &mut ZoomState,
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
        if body.parent == Some(dominant)
            && let Some(bs) = states.get(body.id)
        {
            let r = bs.position - center;
            let p2 = DVec2::new(r.x, r.z);
            children.push((p2.length(), p2));
        }
    }

    // Orientation reference: the dominant body's true radius + colour, and the
    // sun direction (toward the star). `dominant == star` (interplanetary) has
    // no meaningful sun direction → full-bright.
    let (body_radius_m, body_color) = sim
        .system
        .bodies
        .get(dominant)
        .map_or((1.0, [0.46, 0.44, 0.40]), |b| (b.radius_m, b.color));
    let star_pos = sim
        .system
        .bodies
        .iter()
        .position(|b| b.parent.is_none())
        .and_then(|id| states.get(id))
        .map(|s| s.position);
    let is_star = star_pos.is_some_and(|sp| (sp - center).length() < 1.0);
    let sun_dir = match star_pos {
        Some(sp) if !is_star => {
            let d = sp - center;
            DVec2::new(d.x, d.z).normalize_or_zero().as_vec2()
        }
        _ => Vec2::ZERO,
    };

    // Content the view must contain: the current orbit, the planned (maneuver)
    // orbit when a node exists — so the level fits the *maneuver* and stays put
    // through its execution — plus the ship and the body itself.
    let mut content = ship2d.length().max(body_radius_m * 1.1);
    for p in solid_pts.iter() {
        content = content.max(p.length());
    }
    if has_nodes {
        for p in dotted_pts.iter() {
            content = content.max(p.length());
        }
    }

    // Snap to the discrete zoom ladder with hysteresis: step up the instant
    // content overflows the current notch, step down only once it shrinks well
    // inside a smaller notch, so the scale holds steady during a burn.
    let smallest_fit = ZOOM_LEVELS_M
        .iter()
        .position(|&l| l >= content)
        .unwrap_or(ZOOM_LEVELS_M.len() - 1);
    if zoom.dominant != Some(dominant) {
        zoom.dominant = Some(dominant);
        zoom.idx = smallest_fit;
    } else if content > ZOOM_LEVELS_M[zoom.idx]
        || (zoom.idx > 0 && content < ZOOM_LEVELS_M[zoom.idx - 1] * ZOOM_DOWN_HYSTERESIS)
    {
        zoom.idx = smallest_fit;
    }
    let scale = NDC_FIT / ZOOM_LEVELS_M[zoom.idx]; // world metres → plot units

    // Keep only rings that fall inside the framed region.
    children.retain(|(r, _)| (*r * scale) as f32 <= 1.05);
    children.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    children.truncate(MAX_RINGS);

    let body_disc_r = ((body_radius_m * scale) as f32).clamp(CENTRAL_R, 0.95);

    *data = SystemMapData::default();
    data.geom = Vec4::new(CENTRAL_R, SHIP_R, NODE_R, LINE_HW);
    data.style = Vec4::new(RING_HW, DASH_PERIOD, DASH_DUTY, BODY_DOT_R);
    data.extra = Vec4::new(
        body_disc_r,
        sun_dir.x,
        sun_dir.y,
        if is_star { 1.0 } else { 0.0 },
    );
    data.col_central = lin(Color::srgb(body_color[0], body_color[1], body_color[2]));
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

/// Format a zoom-level view radius (metres) as the panel's scale readout.
fn fmt_view_radius(m: f64) -> String {
    let km = m / 1000.0;
    if km >= 1.0e6 {
        format!("\u{00b1}{:.1} Mkm", km / 1.0e6)
    } else {
        format!("\u{00b1}{:.0} km", km)
    }
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
