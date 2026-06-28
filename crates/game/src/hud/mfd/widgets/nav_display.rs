//! Navigation Display (ND) widget: an airliner-style heading-up top-down map.
//!
//! The craft sits fixed at the centre pointing up; a compass rose rotates
//! beneath it so the top of the plot is always the current heading. Runways
//! on the dominant body are projected into the craft-centred ground plane and
//! drawn as oriented symbols with a dashed extended-centerline approach path.
//!
//! Relevant in atmospheric flight (and low over a runway), so it is the
//! widget the MFD auto-selects in place of the orbital trajectory plot once
//! the craft drops into the atmosphere.
//!
//! Drawn by `assets/shaders/nav_display.wgsl`; [`NavDisplayData`] mirrors that
//! shader's `NavDisplayData` struct field-for-field. The ground-plane
//! projection reuses the shared [`crate::hud::geo::local_enu_basis`] so ND and
//! PFD headings agree by construction.

use bevy::math::{DVec2, DVec3};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use thalos_world::BodyId;

use crate::hud::geo::local_enu_basis;
use crate::hud::theme::{HudTheme, label};
use crate::rendering::{SimulationState, SolarSystemState};
use crate::structures::{StructureKind, StructureRegistry};

use super::super::{ActiveWidget, FlightContext, MfdWidgetRoot, WidgetKind, runway_surface_inertial};

/// Body nose axis, matching the navball / PFD conventions.
const BODY_NOSE: DVec3 = DVec3::Y;

/// Side length of the square plot, in logical px.
const MAP_SIZE_PX: f32 = 200.0;
/// Max runways drawn. Mirror in the shader.
const MAX_RUNWAYS: usize = 8;

// Geometry in the shader's normalised half-extent units (1.0 == plot edge).
const RING_R: f32 = 0.9;
const CRAFT_R: f32 = 0.06;
const RWY_HALF_LEN: f32 = 0.06;
const RWY_HALF_WIDTH: f32 = 0.018;
const LINE_HW: f32 = 0.006;
const TICK_HW: f32 = 0.006;
const TICK_LEN: f32 = 0.06;
const NORTH_TICK_LEN: f32 = 0.12;
const APPROACH_LEN: f32 = 0.6;
const DASH_PERIOD: f32 = 0.05;
const DASH_DUTY: f32 = 0.5;

/// View-radius ladder (metres). The plot snaps to the smallest level that
/// comfortably contains the nearest runway, with hysteresis so the scale
/// holds steady on approach.
const RANGE_LEVELS_M: [f64; 7] = [2.0e3, 5.0e3, 1.0e4, 2.0e4, 5.0e4, 1.0e5, 1.5e5];
const RANGE_DOWN_HYSTERESIS: f64 = 0.8;

/// Uniform mirror of the WGSL `NavDisplayData`.
#[derive(Clone, ShaderType)]
pub struct NavDisplayData {
    /// x = runway_count.
    params: Vec4,
    /// x = ring radius, y = craft radius, z = runway half-length, w = line half-width.
    geom: Vec4,
    /// x = tick half-width, y = dash period, z = dash duty, w = runway half-width.
    style: Vec4,
    /// x = heading (rad), y = approach length, z = tick length, w = north-tick length.
    nav: Vec4,
    col_ring: Vec4,
    col_tick: Vec4,
    col_north: Vec4,
    col_craft: Vec4,
    col_runway: Vec4,
    col_approach: Vec4,
    /// per runway: xy = centre (plot), zw = heading-up unit direction.
    runways: [Vec4; MAX_RUNWAYS],
}

impl Default for NavDisplayData {
    fn default() -> Self {
        Self {
            params: Vec4::ZERO,
            geom: Vec4::ZERO,
            style: Vec4::ZERO,
            nav: Vec4::ZERO,
            col_ring: Vec4::ZERO,
            col_tick: Vec4::ZERO,
            col_north: Vec4::ZERO,
            col_craft: Vec4::ZERO,
            col_runway: Vec4::ZERO,
            col_approach: Vec4::ZERO,
            runways: [Vec4::ZERO; MAX_RUNWAYS],
        }
    }
}

#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub struct NavDisplayMaterial {
    #[uniform(0)]
    data: NavDisplayData,
}

impl UiMaterial for NavDisplayMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/nav_display.wgsl".into()
    }
}

#[derive(Component)]
pub(crate) struct NavCanvas;

#[derive(Component)]
pub(crate) struct NavInfo;

/// Persistent range selection (hysteresis), reset when the dominant body
/// changes.
#[derive(Default)]
pub(crate) struct RangeState {
    idx: usize,
    dominant: Option<BodyId>,
}

/// Relevant in atmosphere, or low over a runway just above the Kármán line.
pub(crate) fn relevance(ctx: &FlightContext) -> Option<i32> {
    if ctx.in_atmosphere {
        Some(100)
    } else if ctx.altitude_m < 100_000.0 && ctx.nearest_runway_m.is_some_and(|d| d < 150_000.0) {
        Some(90)
    } else {
        None
    }
}

pub(crate) fn build(
    area: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    materials: &mut Assets<NavDisplayMaterial>,
) {
    let material = materials.add(NavDisplayMaterial {
        data: NavDisplayData::default(),
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
            kind: WidgetKind::NavDisplay,
        },
        Name::new("MfdNavDisplay"),
    ))
    .with_children(|p| {
        p.spawn(label(theme, "NAV"));
        p.spawn((
            Node {
                width: Val::Px(MAP_SIZE_PX),
                height: Val::Px(MAP_SIZE_PX),
                ..default()
            },
            MaterialNode(material),
            NavCanvas,
            Name::new("MfdNavCanvas"),
        ));
        p.spawn((
            Text::new("—"),
            TextFont {
                font: theme.font.clone(),
                font_size: 11.0,
                ..default()
            },
            TextColor(theme.text_dim),
            NavInfo,
            Name::new("MfdNavInfo"),
        ));
    });
}

/// A runway projected into the craft's local ground plane.
struct ProjectedRunway {
    /// Ground-plane offset (east, north) of the runway centre, metres.
    east_m: f64,
    north_m: f64,
    /// Heading-tangent direction in (east, north), metres (unnormalised).
    dir_east: f64,
    dir_north: f64,
    /// 3D slant distance to the craft.
    distance_m: f64,
    /// Runway compass heading (deg, 0 = north).
    heading_deg: f64,
}

pub(crate) fn update(
    active: Res<ActiveWidget>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    structures: Res<StructureRegistry>,
    mut materials: ResMut<Assets<NavDisplayMaterial>>,
    canvas_q: Query<&MaterialNode<NavDisplayMaterial>, With<NavCanvas>>,
    mut info_q: Query<&mut Text, With<NavInfo>>,
    mut range_state: Local<RangeState>,
) {
    if active.0 != Some(WidgetKind::NavDisplay) {
        return;
    }
    let Ok(canvas) = canvas_q.single() else {
        return;
    };
    let Some(material) = materials.get_mut(canvas) else {
        return;
    };

    let s = &sim.simulation;
    let craft = s.craft_state();
    let q = craft.attitude.orientation;
    let craft_pos = craft.translation.position;
    let dominant = s.dominant_body();
    let body_radius_m = s.bodies()[dominant].radius_m;

    let Some(states) = solar.states.as_deref() else {
        material.data = NavDisplayData::default();
        return;
    };
    let Some(bs) = states.get(dominant) else {
        material.data = NavDisplayData::default();
        return;
    };
    let Some((_up, north, east)) = local_enu_basis(craft_pos, bs.position) else {
        return;
    };

    let nose = q * BODY_NOSE;
    // Heading: 0 = north, 90° = east (matches the PFD).
    let psi = nose.dot(east).atan2(nose.dot(north));

    // Project every runway on the dominant body into the ground plane.
    let mut runways: Vec<ProjectedRunway> = Vec::new();
    for site in structures.sites_on(dominant) {
        if site.kind != StructureKind::Runway {
            continue;
        }
        let surf = runway_surface_inertial(site, body_radius_m, bs.position, bs.orientation);
        let rel = surf - craft_pos;
        let t_world = bs.orientation * site.heading_tangent;
        let te = t_world.dot(east);
        let tn = t_world.dot(north);
        runways.push(ProjectedRunway {
            east_m: rel.dot(east),
            north_m: rel.dot(north),
            dir_east: te,
            dir_north: tn,
            distance_m: rel.length(),
            heading_deg: te.atan2(tn).to_degrees().rem_euclid(360.0),
        });
    }
    runways.sort_by(|a, b| {
        a.distance_m
            .partial_cmp(&b.distance_m)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let nearest = runways.first().map(|r| r.distance_m);
    let range = select_range(&mut range_state, dominant, nearest);

    let mut data = NavDisplayData {
        geom: Vec4::new(RING_R, CRAFT_R, RWY_HALF_LEN, LINE_HW),
        style: Vec4::new(TICK_HW, DASH_PERIOD, DASH_DUTY, RWY_HALF_WIDTH),
        nav: Vec4::new(psi as f32, APPROACH_LEN, TICK_LEN, NORTH_TICK_LEN),
        col_ring: lin(Color::srgba(0.55, 0.58, 0.52, 0.50)),
        col_tick: lin(Color::srgba(0.72, 0.74, 0.66, 0.85)),
        col_north: lin(Color::srgba(0.95, 0.45, 0.30, 0.95)),
        col_craft: lin(Color::srgb(0.55, 0.92, 0.50)),
        col_runway: lin(Color::srgba(0.86, 0.87, 0.80, 0.95)),
        col_approach: lin(Color::srgba(0.42, 0.74, 0.88, 0.85)),
        ..Default::default()
    };

    let mut count = 0;
    for r in &runways {
        if count >= MAX_RUNWAYS {
            break;
        }
        let (sx, sy) = heading_up_point(r.east_m, r.north_m, psi, range);
        // Cull runways outside the compass ring.
        if (sx * sx + sy * sy).sqrt() > RING_R as f64 {
            continue;
        }
        let (dx, dy) = heading_up_dir(r.dir_east, r.dir_north, psi);
        data.runways[count] = Vec4::new(sx as f32, sy as f32, dx as f32, dy as f32);
        count += 1;
    }
    data.params = Vec4::new(count as f32, 0.0, 0.0, 0.0);

    material.data = data;

    if let Ok(mut text) = info_q.single_mut() {
        let hdg = psi.to_degrees().rem_euclid(360.0).round() as i32;
        let line = match runways.first() {
            Some(r) => format!(
                "HDG {:03}  R {}  RWY {} {}",
                hdg,
                fmt_range(range),
                runway_number(r.heading_deg),
                fmt_distance(r.distance_m),
            ),
            None => format!("HDG {:03}  R {}  NO RWY", hdg, fmt_range(range)),
        };
        if text.0 != line {
            text.0 = line;
        }
    }
}

/// Project a ground-plane offset `(east, north)` (metres) into heading-up plot
/// coordinates in `[-1, 1]` (y points down, matching the shader).
fn heading_up_point(east_m: f64, north_m: f64, psi: f64, range_m: f64) -> (f64, f64) {
    let (sin, cos) = psi.sin_cos();
    let x_right = east_m * cos - north_m * sin;
    let y_fwd = east_m * sin + north_m * cos;
    (x_right / range_m, -y_fwd / range_m)
}

/// Heading-up unit direction for a ground-plane vector `(east, north)`.
fn heading_up_dir(east: f64, north: f64, psi: f64) -> (f64, f64) {
    let (sin, cos) = psi.sin_cos();
    let x_right = east * cos - north * sin;
    let y_fwd = east * sin + north * cos;
    let v = DVec2::new(x_right, -y_fwd).normalize_or_zero();
    (v.x, v.y)
}

/// Smallest range level that contains `1.2 ×` the nearest runway, with
/// step-down hysteresis. Defaults to a mid level when no runway is in view.
fn select_range(state: &mut RangeState, dominant: BodyId, nearest_m: Option<f64>) -> f64 {
    let content = nearest_m.map(|d| d * 1.2).unwrap_or(2.0e4);
    let smallest_fit = RANGE_LEVELS_M
        .iter()
        .position(|&l| l >= content)
        .unwrap_or(RANGE_LEVELS_M.len() - 1);
    if state.dominant != Some(dominant) {
        state.dominant = Some(dominant);
        state.idx = smallest_fit;
    } else if content > RANGE_LEVELS_M[state.idx]
        || (state.idx > 0 && content < RANGE_LEVELS_M[state.idx - 1] * RANGE_DOWN_HYSTERESIS)
    {
        state.idx = smallest_fit;
    }
    RANGE_LEVELS_M[state.idx]
}

/// Runway designator (01–36) from its compass heading.
fn runway_number(heading_deg: f64) -> String {
    let mut n = (heading_deg / 10.0).round() as i32;
    if n <= 0 {
        n += 36;
    } else if n > 36 {
        n -= 36;
    }
    format!("{n:02}")
}

fn fmt_range(m: f64) -> String {
    if m >= 1000.0 {
        format!("{:.0}km", m / 1000.0)
    } else {
        format!("{m:.0}m")
    }
}

fn fmt_distance(m: f64) -> String {
    if m >= 1000.0 {
        format!("{:.1}km", m / 1000.0)
    } else {
        format!("{m:.0}m")
    }
}

fn lin(color: Color) -> Vec4 {
    color.to_linear().to_vec4()
}
