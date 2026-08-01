//! Navigation Display (ND) widget: an airliner-style heading-up top-down map.
//!
//! The craft sits fixed at the centre pointing up; a compass rose rotates
//! beneath it so the top of the plot is always the current heading. Runways on
//! the dominant body are drawn **at true scale** with their landing threshold
//! marked, and the armed approach route — the flyable, bank-limited path
//! computed by the runtime's `route` module — is drawn as a real polyline with its final
//! approach segment highlighted.
//!
//! Relevant in atmospheric flight (and low over a runway), so it is the widget
//! the MFD auto-selects in place of the orbital trajectory plot once the craft
//! drops into the atmosphere.
//!
//! # Two ways to select, one authority
//!
//! Clicking a runway on the plot arms an approach to it (clicking the armed one
//! again lands the other way); the `◀ ▶` / `FLIP` / `CLR` buttons do the same
//! for a runway that is off-plot. Neither writes the selection directly — both
//! send a [`RouteRequest`], so the runtime's `route` module stays the sole writer and the
//! two paths cannot disagree.
//!
//! # Projection
//!
//! One [`RouteFrame`] anchored at the craft does all the work: runways, route
//! points, and waypoints are projected through it into local east/north metres,
//! then rotated heading-up and divided by the plot range. That frame's basis is
//! built exactly like the shared [`crate::geo::local_enu_basis`], so ND and
//! PFD headings agree by construction.
//!
//! Drawn by `assets/shaders/nav_display.wgsl`; [`NavDisplayData`] mirrors that
//! shader's struct field-for-field. Assembly is the pure
//! [`nav_display_data`] over a [`NavScene`], so the headless preview
//! (`cargo run -p thalos_runtime --example nav_preview`) can render states that
//! are hard to fly to.

use bevy::ecs::system::SystemParam;
use bevy::math::{DVec2, DVec3};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use bevy::ui::RelativeCursorPosition;

use thalos_navigation::approach::strip_in_frame;
use thalos_navigation::{
    ApproachPhase, Guidance, RouteFrame, RunwayEnd, RunwayStrip, WaypointKind, theta_of,
    theta_to_heading,
};
use thalos_physics_canonical::body_fixed::inertial_to_body_fixed;
use thalos_world::BodyId;

use crate::format;
use crate::theme::HudTheme;
use thalos_game_state::{SimulationState, SolarSystemState};
use thalos_game_state::nav::{RouteRequest, RouteState, RouteStatus};
use thalos_game_state::units::{UnitDomain, UnitSystem};

use super::super::{ActiveWidget, FlightContext, MfdWidgetRoot, WidgetKind};

/// Body nose axis, matching the navball / PFD conventions.
const BODY_NOSE: DVec3 = DVec3::Y;

/// Side length of the square plot, in logical px.
const MAP_SIZE_PX: f32 = 240.0;
/// Max runways drawn. Mirror in the shader.
pub(crate) const MAX_RUNWAYS: usize = 8;
/// Max route polyline points. Mirror in the shader (packed two per `vec4`).
pub(crate) const MAX_ROUTE_POINTS: usize = 48;
/// Max waypoint symbols. Mirror in the shader.
pub(crate) const MAX_WAYPOINTS: usize = 4;
/// Max rejoin polyline points. Mirror in the shader (packed two per `vec4`).
pub(crate) const MAX_REJOIN_POINTS: usize = 24;
/// Sentinel for an absent angular marker. Mirror in the shader.
const NO_ANGLE: f32 = 1.0e8;

// Geometry in the shader's normalised half-extent units (1.0 == plot edge).
const RING_R: f32 = 0.9;
const CRAFT_R: f32 = 0.06;
const LINE_HW: f32 = 0.006;
const TICK_HW: f32 = 0.006;
const TICK_LEN: f32 = 0.06;
const NORTH_TICK_LEN: f32 = 0.12;
const ROUTE_HW: f32 = 0.009;
const DASH_PERIOD: f32 = 0.05;
const DASH_DUTY: f32 = 0.5;
/// Minimum drawn runway half-width, in plot units. A 90 m strip is 0.0003 plot
/// units wide at 150 km range, and even at 20 km it is 0.002 — a sub-pixel
/// scratch on a 200 px plot. Length stays true (it is the dimension a pilot
/// judges); only the width gets this floor, set so the strip is ~4 px wide and
/// reads as a runway rather than a hairline.
const RWY_MIN_HALF_WIDTH: f32 = 0.010;
/// Radius of the dashed half-range ring.
const RANGE_RING_R: f32 = 0.45;
/// Click tolerance, in plot units, for picking a runway on the plot.
const PICK_TOLERANCE: f32 = 0.08;
/// Ground speed (m/s) below which the craft has no meaningful ground track and
/// the track marker is hidden rather than pointing somewhere arbitrary.
const MIN_TRACK_SPEED_M_S: f64 = 2.0;

/// View-radius ladder (metres). In `AUTO` the plot snaps to the smallest level
/// that contains what is still ahead; the zoom controls step through the same
/// ladder, so manual and automatic ranges are always the same set of scales.
///
/// The bottom two rungs exist for the last mile: at 500 m the 5 km strip runs
/// well off the plot, which is exactly what you want when you are looking at
/// where you will touch down rather than where the airfield is.
const RANGE_LEVELS_M: [f64; 10] = [
    500.0, 1.0e3, 2.0e3, 5.0e3, 1.0e4, 2.0e4, 5.0e4, 1.0e5, 1.5e5, 3.0e5,
];
const RANGE_DOWN_HYSTERESIS: f64 = 0.8;
/// Default AUTO level before anything is armed (index into [`RANGE_LEVELS_M`]).
const DEFAULT_RANGE_INDEX: usize = 5;

/// Side of the square deviation indicator, in logical px.
const DEV_BOX_PX: f32 = 78.0;
/// Diameter of the moving deviation dot (px).
const DEV_DOT_PX: f32 = 9.0;
/// Distance from the box centre to a full-scale deflection (px).
const DEV_HALF_PX: f32 = 30.0;

/// Uniform mirror of the WGSL `NavDisplayData`.
#[derive(Clone, ShaderType)]
pub struct NavDisplayData {
    /// x = runway_count, y = route_point_count, z = waypoint_count,
    /// w = first route point index on the final segment.
    params: Vec4,
    /// x = ring radius, y = craft radius, z = line half-width, w = tick half-width.
    geom: Vec4,
    /// x = dash period, y = dash duty, z = route half-width, w = min runway half-width.
    style: Vec4,
    /// x = heading (rad), y = tick length, z = north-tick length,
    /// w = bearing-to-destination marker (heading-up rad, or [`NO_ANGLE`]).
    nav: Vec4,
    /// x = range-ring radius, y = ground-track marker (heading-up rad, or
    /// [`NO_ANGLE`]), z = rejoin point count, w reserved.
    extra: Vec4,
    col_ring: Vec4,
    col_tick: Vec4,
    col_north: Vec4,
    col_craft: Vec4,
    col_runway: Vec4,
    col_runway_armed: Vec4,
    col_route: Vec4,
    col_route_final: Vec4,
    col_waypoint: Vec4,
    col_rejoin: Vec4,
    /// per runway: xy = centre (plot), zw = along-strip unit direction.
    runways: [Vec4; MAX_RUNWAYS],
    /// per runway: x = half-length, y = half-width (plot), z = armed,
    /// w = threshold end sign along `zw`.
    runway_ext: [Vec4; MAX_RUNWAYS],
    /// Route polyline, two points per element.
    route: [Vec4; MAX_ROUTE_POINTS / 2],
    /// Rejoin polyline, same packing.
    rejoin: [Vec4; MAX_REJOIN_POINTS / 2],
    /// per waypoint: xy = position (plot), z = kind.
    waypoints: [Vec4; MAX_WAYPOINTS],
}

impl Default for NavDisplayData {
    fn default() -> Self {
        Self {
            params: Vec4::ZERO,
            geom: Vec4::new(RING_R, CRAFT_R, LINE_HW, TICK_HW),
            style: Vec4::new(DASH_PERIOD, DASH_DUTY, ROUTE_HW, RWY_MIN_HALF_WIDTH),
            nav: Vec4::new(0.0, TICK_LEN, NORTH_TICK_LEN, NO_ANGLE),
            extra: Vec4::new(RANGE_RING_R, NO_ANGLE, 0.0, 0.0),
            col_ring: lin(Color::srgba(0.55, 0.58, 0.52, 0.50)),
            col_tick: lin(Color::srgba(0.72, 0.74, 0.66, 0.85)),
            col_north: lin(Color::srgba(0.95, 0.45, 0.30, 0.95)),
            col_craft: lin(Color::srgb(0.55, 0.92, 0.50)),
            col_runway: lin(Color::srgba(0.86, 0.87, 0.80, 0.80)),
            col_runway_armed: lin(Color::srgba(1.0, 0.95, 0.75, 1.0)),
            col_route: lin(Color::srgba(0.42, 0.74, 0.88, 0.85)),
            col_route_final: lin(Color::srgba(0.99, 0.75, 0.37, 0.95)),
            col_waypoint: lin(Color::srgba(0.80, 0.86, 0.92, 0.90)),
            col_rejoin: lin(Color::srgba(0.55, 0.90, 0.62, 0.85)),
            runways: [Vec4::ZERO; MAX_RUNWAYS],
            runway_ext: [Vec4::ZERO; MAX_RUNWAYS],
            route: [Vec4::ZERO; MAX_ROUTE_POINTS / 2],
            rejoin: [Vec4::ZERO; MAX_REJOIN_POINTS / 2],
            waypoints: [Vec4::ZERO; MAX_WAYPOINTS],
        }
    }
}

#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub struct NavDisplayMaterial {
    #[uniform(0)]
    data: NavDisplayData,
}

impl NavDisplayMaterial {
    pub fn new(data: NavDisplayData) -> Self {
        Self { data }
    }
}

impl UiMaterial for NavDisplayMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/nav_display.wgsl".into()
    }
}

#[derive(Component)]
pub(crate) struct NavCanvas;

/// The armed runway designator in the header (the panel's primary readout).
#[derive(Component)]
pub(crate) struct NavHeaderRunway;

/// The approach-phase chip beside it (`INTC` / `FNL` / `TDZ`).
#[derive(Component)]
pub(crate) struct NavHeaderPhase;

/// Distance-to-go, right-aligned in the header.
#[derive(Component)]
pub(crate) struct NavHeaderDistance;

/// The current plot range, between the zoom controls.
#[derive(Component)]
pub(crate) struct NavRangeLabel;

/// The moving dot in the deviation indicator.
#[derive(Component)]
pub(crate) struct NavDeviationDot;

/// The deviation indicator's frame — hidden when no approach is armed, since a
/// centring dot with nothing to centre on is worse than no dot.
#[derive(Component)]
pub(crate) struct NavDeviationBox;

/// One secondary readout in the data column.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NavDatum {
    Heading,
    Track,
    CrossTrack,
    Glideslope,
}

/// The value half of a [`NavDatum`] row.
#[derive(Component)]
pub(crate) struct NavDatumValue(NavDatum);

/// Zoom controls under the plot.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NavZoomButton {
    /// Tighter range (fewer metres across the plot).
    In,
    /// Wider range.
    Out,
    /// Hand the range back to AUTO.
    Auto,
}

/// A runway designator label pinned over the plot.
#[derive(Component)]
pub(crate) struct NavRunwayLabel(usize);

/// Selector buttons under the plot.
#[derive(Component, Clone, Copy)]
pub(crate) enum NavSelectButton {
    Prev,
    Next,
    Flip,
    Clear,
}

/// The automatic range selection: which rung of [`RANGE_LEVELS_M`] AUTO has
/// settled on, plus the body it was settled for (so it resets on a body change).
///
/// A **resource**, not a `Local`, because two systems need the same value:
/// [`update`] computes it, and [`handle_zoom`] seeds the first manual step from
/// it so zooming out of AUTO starts where the plot already is instead of jumping
/// to the end of the ladder. As a `Local` each system would have kept its own
/// copy and the zoom would have stepped from a rung the plot was never on.
///
/// **Sole writer:** [`update`].
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct NavRangeState {
    idx: usize,
    dominant: Option<BodyId>,
}

impl Default for NavRangeState {
    fn default() -> Self {
        Self {
            idx: DEFAULT_RANGE_INDEX,
            dominant: None,
        }
    }
}

/// The pilot's zoom override. `None` = AUTO (the plot frames what is still
/// ahead); `Some(i)` pins a rung of [`RANGE_LEVELS_M`].
///
/// **Sole writer:** [`handle_zoom`]. A resource rather than a `Local` because
/// three inputs drive it — the two buttons, the AUTO button, and the scroll
/// wheel — and because the selection must survive the widget being swapped out
/// and back.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NavZoom {
    pub manual: Option<usize>,
}

impl NavZoom {
    /// Step the manual range, seeding from `auto_index` on the first step so
    /// zooming from AUTO starts where the plot already is instead of jumping.
    fn step(&mut self, delta: i32, auto_index: usize) {
        let current = self.manual.unwrap_or(auto_index) as i32;
        let next = (current + delta).clamp(0, RANGE_LEVELS_M.len() as i32 - 1);
        self.manual = Some(next as usize);
    }
}

// ---------------------------------------------------------------------------
// Scene: the display-independent description of what to draw
// ---------------------------------------------------------------------------

/// One strip, already projected into the craft frame's local metres.
#[derive(Debug, Clone, Copy)]
pub struct NavStrip {
    /// Registry id of the strip, so a click can resolve back to it.
    pub id: u64,
    /// Strip centre in local `(east, north)` metres.
    pub center_m: DVec2,
    /// Unit along-strip direction in local metres.
    pub along: DVec2,
    pub half_length_m: f64,
    pub half_width_m: f64,
    /// Is this the armed strip?
    pub armed: bool,
    /// `+1` if the landing threshold is at `center + along · half_length`,
    /// `−1` if at the other end. Only meaningful when `armed`; otherwise it
    /// marks the end nearer the craft so the plot still reads.
    pub threshold_sign: f64,
    /// Designator to label it with (`1..=36`), if known.
    pub designator: Option<u8>,
}

/// Everything the ND draws, in one display-independent value.
///
/// Built from resources by [`update`] in the game, and hand-built by the
/// headless preview — which is the point: symbology can be verified in states
/// (way off course, behind the field, high on the slope) that would take a long
/// flight to reach.
#[derive(Debug, Clone, Default)]
pub struct NavScene {
    /// Craft compass heading (rad).
    pub heading_rad: f64,
    /// Craft ground track (compass rad), if it has one.
    pub track_rad: Option<f64>,
    pub strips: Vec<NavStrip>,
    /// Planned route in local metres, in fly order.
    pub route_m: Vec<DVec2>,
    /// Index of the first route point on the final approach segment.
    pub route_final_index: usize,
    pub waypoints_m: Vec<(DVec2, WaypointKind)>,
    /// The flyable rejoin in local metres, in fly order. Empty when the craft is
    /// close enough to the route that drawing it says nothing.
    pub rejoin_m: Vec<DVec2>,
    /// Bearing to the armed threshold (compass rad).
    pub bearing_to_dest_rad: Option<f64>,
    /// Plot half-extent in metres.
    pub range_m: f64,
}

/// Build the shader uniform for a scene. Pure — no resources, no world.
pub fn nav_display_data(scene: &NavScene) -> NavDisplayData {
    let mut data = NavDisplayData {
        nav: Vec4::new(
            scene.heading_rad as f32,
            TICK_LEN,
            NORTH_TICK_LEN,
            scene
                .bearing_to_dest_rad
                .map(|b| heading_up_angle(b, scene.heading_rad) as f32)
                .unwrap_or(NO_ANGLE),
        ),
        extra: Vec4::new(
            RANGE_RING_R,
            scene
                .track_rad
                .map(|t| heading_up_angle(t, scene.heading_rad) as f32)
                .unwrap_or(NO_ANGLE),
            0.0,
            0.0,
        ),
        ..Default::default()
    };

    let range = scene.range_m.max(1.0);

    // Runways. Culled when the whole strip is outside the ring — a strip whose
    // centre is off-plot but whose near half is inside must still draw, so the
    // test is on the nearest point of the strip, not its centre.
    let mut count = 0;
    for strip in &scene.strips {
        if count >= MAX_RUNWAYS {
            break;
        }
        let centre = heading_up_point(strip.center_m, scene.heading_rad, range);
        let along = heading_up_dir(strip.along, scene.heading_rad);
        let half_len = strip.half_length_m / range;
        let nearest = (centre.length() - half_len).max(0.0);
        if nearest > RING_R as f64 {
            continue;
        }
        data.runways[count] = Vec4::new(
            centre.x as f32,
            centre.y as f32,
            along.x as f32,
            along.y as f32,
        );
        data.runway_ext[count] = Vec4::new(
            half_len as f32,
            (strip.half_width_m / range) as f32,
            if strip.armed { 1.0 } else { 0.0 },
            strip.threshold_sign as f32,
        );
        count += 1;
    }

    // Route polyline, decimated to the uniform's capacity. Decimation keeps the
    // first and last points and the final-segment boundary, so the highlighted
    // final approach never shifts.
    let route = decimate_route(&scene.route_m, scene.route_final_index, MAX_ROUTE_POINTS);
    for (i, p) in route.points.iter().enumerate() {
        let plot = heading_up_point(*p, scene.heading_rad, range);
        let slot = &mut data.route[i / 2];
        if i % 2 == 0 {
            slot.x = plot.x as f32;
            slot.y = plot.y as f32;
        } else {
            slot.z = plot.x as f32;
            slot.w = plot.y as f32;
        }
    }

    let rejoin = decimate_route(&scene.rejoin_m, 0, MAX_REJOIN_POINTS);
    for (i, p) in rejoin.points.iter().enumerate() {
        let plot = heading_up_point(*p, scene.heading_rad, range);
        let slot = &mut data.rejoin[i / 2];
        if i % 2 == 0 {
            slot.x = plot.x as f32;
            slot.y = plot.y as f32;
        } else {
            slot.z = plot.x as f32;
            slot.w = plot.y as f32;
        }
    }
    data.extra.z = rejoin.points.len() as f32;

    let mut waypoint_count = 0;
    for (p, kind) in &scene.waypoints_m {
        if waypoint_count >= MAX_WAYPOINTS {
            break;
        }
        let plot = heading_up_point(*p, scene.heading_rad, range);
        if plot.length() > RING_R as f64 {
            continue;
        }
        data.waypoints[waypoint_count] =
            Vec4::new(plot.x as f32, plot.y as f32, waypoint_kind_code(*kind), 0.0);
        waypoint_count += 1;
    }

    data.params = Vec4::new(
        count as f32,
        route.points.len() as f32,
        waypoint_count as f32,
        route.final_index as f32,
    );
    data
}

/// A route reduced to at most `capacity` points.
struct DecimatedRoute {
    points: Vec<DVec2>,
    final_index: usize,
}

/// Reduce a polyline to `capacity` points, preserving the endpoints and the
/// index where the final approach segment starts.
///
/// The transition legs (arcs) are what need points; the final approach is a
/// straight line and survives any decimation, so the budget is spent on the
/// transition and the final contributes its two endpoints.
fn decimate_route(points: &[DVec2], final_index: usize, capacity: usize) -> DecimatedRoute {
    if points.len() <= capacity {
        return DecimatedRoute {
            points: points.to_vec(),
            final_index,
        };
    }
    let final_index = final_index.min(points.len().saturating_sub(1));
    // Reserve the tail (final segment) verbatim, decimate the head.
    let tail = &points[final_index..];
    let tail_len = tail.len().min(capacity / 2);
    let head_budget = capacity - tail_len;
    let head = &points[..final_index];
    let mut out: Vec<DVec2> = Vec::with_capacity(capacity);
    if head_budget >= 2 && !head.is_empty() {
        let stride = (head.len() as f64 - 1.0) / (head_budget as f64 - 1.0);
        for i in 0..head_budget {
            let idx = ((i as f64 * stride).round() as usize).min(head.len() - 1);
            out.push(head[idx]);
        }
    }
    let new_final = out.len();
    let tail_stride = if tail_len > 1 {
        (tail.len() as f64 - 1.0) / (tail_len as f64 - 1.0)
    } else {
        0.0
    };
    for i in 0..tail_len {
        let idx = ((i as f64 * tail_stride).round() as usize).min(tail.len() - 1);
        out.push(tail[idx]);
    }
    DecimatedRoute {
        points: out,
        final_index: new_final,
    }
}

fn waypoint_kind_code(kind: WaypointKind) -> f32 {
    match kind {
        WaypointKind::Fix => 0.0,
        WaypointKind::FinalApproach => 1.0,
        WaypointKind::Threshold => 2.0,
        WaypointKind::Aim => 3.0,
    }
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

/// Small helper for a readout label (dim, uppercase, small).
fn datum_label(theme: &HudTheme, text: &str) -> impl Bundle {
    (
        Text::new(text.to_string()),
        TextFont {
            font: theme.font.clone(),
            font_size: FontSize::Px(9.0),
            ..default()
        },
        TextColor(theme.text_dim),
    )
}

/// A 1 px separator, which is what gives the stacked blocks their edges.
fn separator(theme: &HudTheme) -> impl Bundle {
    (
        Node {
            width: Val::Px(MAP_SIZE_PX),
            height: Val::Px(1.0),
            ..default()
        },
        BackgroundColor(theme.panel_border),
    )
}

/// A small control button used by the zoom and selector rows.
fn control_button(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    label_text: &str,
    marker: impl Bundle,
) {
    parent
        .spawn((
            Button,
            Node {
                min_width: Val::Px(22.0),
                padding: UiRect::axes(Val::Px(6.0), Val::Px(2.0)),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            marker,
            Name::new(format!("MfdNavBtn{label_text}")),
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(label_text.to_string()),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(9.0),
                    ..default()
                },
                TextColor(theme.text_dim),
            ));
        });
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
        // --- Header: the one line a pilot should be able to read at a glance —
        // what is armed, what phase it is in, and how far. Everything below it
        // is deliberately smaller and dimmer.
        p.spawn((
            Node {
                width: Val::Px(MAP_SIZE_PX),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(6.0),
                ..default()
            },
            Name::new("MfdNavHeader"),
        ))
        .with_children(|header| {
            header.spawn((
                Text::new("SELECT RWY"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(15.0),
                    ..default()
                },
                TextColor(theme.text_dim),
                NavHeaderRunway,
                Name::new("MfdNavHeaderRunway"),
            ));
            header
                .spawn((
                    Node {
                        padding: UiRect::axes(Val::Px(4.0), Val::Px(1.0)),
                        border: UiRect::all(Val::Px(1.0)),
                        border_radius: BorderRadius::all(Val::Px(2.0)),
                        ..default()
                    },
                    BorderColor::all(theme.panel_border),
                    Visibility::Hidden,
                    NavHeaderPhase,
                    Name::new("MfdNavHeaderPhase"),
                ))
                .with_children(|chip| {
                    chip.spawn((
                        Text::new(""),
                        TextFont {
                            font: theme.font.clone(),
                            font_size: FontSize::Px(9.0),
                            ..default()
                        },
                        TextColor(theme.text_accent),
                    ));
                });
            // Spacer pushes the distance to the right edge.
            header.spawn(Node {
                flex_grow: 1.0,
                ..default()
            });
            header.spawn((
                Text::new(""),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(theme.text_primary),
                NavHeaderDistance,
                Name::new("MfdNavHeaderDistance"),
            ));
        });

        // --- The plot.
        p.spawn((
            Node {
                width: Val::Px(MAP_SIZE_PX),
                height: Val::Px(MAP_SIZE_PX),
                ..default()
            },
            MaterialNode(material),
            NavCanvas,
            // Clicking the plot arms an approach; the relative position is what
            // turns a click into plot coordinates. Hover also gates the scroll
            // wheel, so the wheel only zooms the plot the cursor is over.
            Interaction::None,
            RelativeCursorPosition::default(),
            Name::new("MfdNavCanvas"),
        ))
        .with_children(|canvas| {
            // Designator labels, pre-spawned and repositioned each frame — one
            // per drawable runway slot.
            for i in 0..MAX_RUNWAYS {
                canvas.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        ..default()
                    },
                    Text::new(""),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(9.0),
                        ..default()
                    },
                    TextColor(theme.text_dim),
                    Visibility::Hidden,
                    NavRunwayLabel(i),
                    Name::new(format!("MfdNavRunwayLabel{i}")),
                ));
            }
        });

        // --- Zoom row.
        p.spawn((
            Node {
                width: Val::Px(MAP_SIZE_PX),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                column_gap: Val::Px(5.0),
                ..default()
            },
            Name::new("MfdNavZoomRow"),
        ))
        .with_children(|row| {
            control_button(row, theme, "-", NavZoomButton::Out);
            row.spawn((
                Node {
                    min_width: Val::Px(64.0),
                    justify_content: JustifyContent::Center,
                    ..default()
                },
                Text::new("--"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(11.0),
                    ..default()
                },
                TextColor(theme.text_primary),
                NavRangeLabel,
                Name::new("MfdNavRangeLabel"),
            ));
            control_button(row, theme, "+", NavZoomButton::In);
            control_button(row, theme, "AUTO", NavZoomButton::Auto);
        });

        p.spawn(separator(theme));

        // --- Guidance block: the centring dot beside the secondary readouts.
        p.spawn((
            Node {
                width: Val::Px(MAP_SIZE_PX),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(10.0),
                ..default()
            },
            Name::new("MfdNavGuidance"),
        ))
        .with_children(|block| {
            // The deviation indicator: keep the dot in the middle and you are on
            // the centreline and on the glideslope.
            block
                .spawn((
                    Node {
                        width: Val::Px(DEV_BOX_PX),
                        height: Val::Px(DEV_BOX_PX),
                        border: UiRect::all(Val::Px(1.0)),
                        border_radius: BorderRadius::all(Val::Px(3.0)),
                        ..default()
                    },
                    BorderColor::all(theme.panel_border),
                    BackgroundColor(theme.panel_bg),
                    Visibility::Hidden,
                    NavDeviationBox,
                    Name::new("MfdNavDeviation"),
                ))
                .with_children(|box_node| {
                    let centre = DEV_BOX_PX * 0.5 - 1.0;
                    // Cross hairs through the centre: the target the dot is
                    // flown onto.
                    box_node.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(centre - 9.0),
                            top: Val::Px(centre),
                            width: Val::Px(18.0),
                            height: Val::Px(1.0),
                            ..default()
                        },
                        BackgroundColor(theme.text_dim),
                    ));
                    box_node.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(centre),
                            top: Val::Px(centre - 9.0),
                            width: Val::Px(1.0),
                            height: Val::Px(18.0),
                            ..default()
                        },
                        BackgroundColor(theme.text_dim),
                    ));
                    // Full-scale ticks on both axes, so the dot's travel has a
                    // readable scale rather than being a free-floating blob.
                    for (dx, dy) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
                        box_node.spawn((
                            Node {
                                position_type: PositionType::Absolute,
                                left: Val::Px(centre + dx * DEV_HALF_PX - 1.0),
                                top: Val::Px(centre + dy * DEV_HALF_PX - 1.0),
                                width: Val::Px(3.0),
                                height: Val::Px(3.0),
                                border_radius: BorderRadius::all(Val::Px(1.5)),
                                ..default()
                            },
                            BackgroundColor(theme.panel_border),
                        ));
                    }
                    // The dot itself.
                    box_node.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(centre - DEV_DOT_PX * 0.5),
                            top: Val::Px(centre - DEV_DOT_PX * 0.5),
                            width: Val::Px(DEV_DOT_PX),
                            height: Val::Px(DEV_DOT_PX),
                            border_radius: BorderRadius::all(Val::Px(DEV_DOT_PX * 0.5)),
                            ..default()
                        },
                        BackgroundColor(theme.text_accent),
                        NavDeviationDot,
                        Name::new("MfdNavDeviationDot"),
                    ));
                });

            // Secondary readouts, in a fixed order so the eye learns the rows.
            block
                .spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        flex_grow: 1.0,
                        row_gap: Val::Px(1.0),
                        ..default()
                    },
                    Name::new("MfdNavData"),
                ))
                .with_children(|column| {
                    for (datum, label_text) in [
                        (NavDatum::Heading, "HDG"),
                        (NavDatum::Track, "TRK"),
                        (NavDatum::CrossTrack, "XTK"),
                        (NavDatum::Glideslope, "G/S"),
                    ] {
                        column
                            .spawn((
                                Node {
                                    flex_direction: FlexDirection::Row,
                                    align_items: AlignItems::Center,
                                    column_gap: Val::Px(6.0),
                                    ..default()
                                },
                                Name::new(format!("MfdNavRow{label_text}")),
                            ))
                            .with_children(|row| {
                                row.spawn((
                                    Node {
                                        min_width: Val::Px(24.0),
                                        ..default()
                                    },
                                    datum_label(theme, label_text),
                                ));
                                row.spawn((
                                    Text::new("---"),
                                    TextFont {
                                        font: theme.font.clone(),
                                        font_size: FontSize::Px(11.0),
                                        ..default()
                                    },
                                    TextColor(theme.text_primary),
                                    NavDatumValue(datum),
                                ));
                            });
                    }
                });
        });

        // --- Runway selector, for a strip that is off the plot.
        p.spawn((
            Node {
                width: Val::Px(MAP_SIZE_PX),
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::Center,
                column_gap: Val::Px(5.0),
                ..default()
            },
            Name::new("MfdNavSelector"),
        ))
        .with_children(|row| {
            control_button(row, theme, "<", NavSelectButton::Prev);
            control_button(row, theme, ">", NavSelectButton::Next);
            control_button(row, theme, "FLIP", NavSelectButton::Flip);
            control_button(row, theme, "CLR", NavSelectButton::Clear);
        });
    });
}

/// Everything [`build_nav_scene`] needs, with no Bevy in sight.
///
/// The game fills this from its resources; the headless preview fills it from
/// hand-built craft states and real [`plan_approach`](thalos_navigation::plan_approach)
/// output. One projection implementation serves both, so a symbology check in
/// the preview is a check of what the game actually draws.
pub struct NavSceneInputs<'a> {
    /// Craft position in the body-fixed frame (m from the body centre).
    pub craft_body_fixed: DVec3,
    /// Craft nose direction, body-fixed.
    pub nose_body_fixed: DVec3,
    /// Surface-relative velocity, body-fixed (zero for a parked craft).
    pub velocity_body_fixed: DVec3,
    pub body_radius_m: f64,
    /// Every strip to draw.
    pub strips: &'a [RunwayStrip],
    /// The armed end, if an approach is active.
    pub armed: Option<RunwayEnd>,
    /// Planned route in body-fixed metres.
    pub route_points: &'a [DVec3],
    /// Rejoin path in body-fixed metres (empty when there is nothing to show).
    pub rejoin_points: &'a [DVec3],
    /// Index into `route_points` of the first point on the final approach leg.
    /// An index, not a distance — see `RouteDisplay::final_start_index` for why
    /// deriving it from along-path distances is wrong.
    pub final_start_index: usize,
    pub waypoints: &'a [(DVec3, WaypointKind)],
    /// Plot half-extent (m).
    pub range_m: f64,
}

/// Project everything into the craft-anchored plot frame.
///
/// Returns `None` only for a degenerate craft position (at the body centre).
pub fn build_nav_scene(inputs: &NavSceneInputs<'_>) -> Option<NavScene> {
    // One frame at the craft does every projection on this plot.
    let frame = RouteFrame::new(
        inputs.craft_body_fixed.try_normalize()?,
        inputs.body_radius_m,
        inputs.craft_body_fixed.length() - inputs.body_radius_m,
    )?;

    let heading_rad = theta_to_heading(theta_of(frame.direction_to_local(inputs.nose_body_fixed)));

    let up = frame.origin_dir;
    let velocity = inputs.velocity_body_fixed;
    let horizontal = velocity - up * velocity.dot(up);
    let track_rad = (horizontal.length() > MIN_TRACK_SPEED_M_S)
        .then(|| theta_to_heading(theta_of(frame.direction_to_local(horizontal))));

    let mut strips: Vec<NavStrip> = Vec::new();
    for strip in inputs.strips {
        let Some(geometry) = strip_in_frame(strip, &frame) else {
            continue;
        };
        let armed_here = inputs.armed.filter(|a| a.strip.id == strip.id);
        // Which end is the landing threshold, expressed along the strip's own
        // `along` direction: the armed end when armed, else the end nearer the
        // craft so an unarmed strip still shows a plausible approach end.
        let threshold_sign = match armed_here {
            Some(end) => {
                if end.reciprocal {
                    1.0
                } else {
                    -1.0
                }
            }
            None => {
                let (minus_end, plus_end) = geometry.ends();
                if plus_end.length() < minus_end.length() {
                    1.0
                } else {
                    -1.0
                }
            }
        };
        // Designator of the end that is drawn as the threshold — computed from
        // the strip itself, so no registry lookup is needed here (which is what
        // lets the preview build a scene from nothing but geometry).
        let designator = strip
            .ends()
            .iter()
            .find(|end| {
                (end.reciprocal && threshold_sign > 0.0)
                    || (!end.reciprocal && threshold_sign < 0.0)
            })
            .and_then(|end| end.route_frame().map(|f| end.designator(&f)));
        strips.push(NavStrip {
            id: strip.id,
            center_m: geometry.center,
            along: geometry.along,
            half_length_m: geometry.half_length_m,
            half_width_m: geometry.half_width_m,
            armed: armed_here.is_some(),
            threshold_sign,
            designator,
        });
    }
    // Armed strip last so it wins the draw order and keeps its label slot.
    strips.sort_by_key(|s| s.armed);

    let route_m: Vec<DVec2> = inputs
        .route_points
        .iter()
        .map(|p| frame.to_local(*p))
        .collect();
    let rejoin_m: Vec<DVec2> = inputs
        .rejoin_points
        .iter()
        .map(|p| frame.to_local(*p))
        .collect();
    let route_final_index = inputs.final_start_index.min(route_m.len());
    let waypoints_m = inputs
        .waypoints
        .iter()
        .map(|(p, kind)| (frame.to_local(*p), *kind))
        .collect();

    let bearing_to_dest_rad = inputs.armed.and_then(|end| {
        let to_threshold = frame.to_local(end.threshold_point());
        (to_threshold.length() > 1.0).then(|| theta_to_heading(theta_of(to_threshold)))
    });

    Some(NavScene {
        heading_rad,
        track_rad,
        strips,
        route_m,
        route_final_index,
        waypoints_m,
        rejoin_m,
        bearing_to_dest_rad,
        range_m: inputs.range_m,
    })
}

/// Gather the frame's [`NavScene`] from the world, or `None` when the craft/body
/// state needed to project anything is unavailable.
fn scene_from_world(
    sim: &SimulationState,
    solar: &SolarSystemState,
    route: &RouteState,
    zoom: &NavZoom,
    range_state: &mut NavRangeState,
) -> Option<NavScene> {
    let s = &sim.simulation;
    let dominant = s.dominant_body();
    let body_radius_m = s.bodies()[dominant].radius_m;
    let body_state = solar.states.as_deref()?.get(dominant)?;

    let craft = s.craft_state();
    let frame_state = inertial_to_body_fixed(body_state, craft.translation, craft.attitude);
    let position_bf = frame_state.translation_body.position;

    // De-duplicate the enumerated ends back to physical strips (each strip
    // appears twice, once per landable direction).
    let mut strips: Vec<RunwayStrip> = Vec::new();
    for entry in &route.ends {
        if !strips.iter().any(|s| s.id == entry.end.strip.id) {
            strips.push(entry.end.strip);
        }
    }

    // Range is picked from body-fixed distances, before projecting: what has to
    // fit does not depend on the plot.
    //
    // It must be framed on what is **still ahead**, not on the whole plan. The
    // plan freezes once established on final (re-planning from there would fly
    // you away from the runway), so its points still include the original
    // intercept from tens of km out — framing all of them keeps the plot at
    // 50 km while the threshold is 700 m away, which is exactly the "small and
    // zoomed out" the display was reported as.
    let along_now = route.guidance.map(|g| g.along_m).unwrap_or(0.0);
    let ahead_m = route
        .display
        .path_points
        .iter()
        .zip(route.display.path_along_m.iter())
        .filter(|(_, along)| **along >= along_now)
        .map(|(p, _)| position_bf.distance(*p))
        .fold(0.0_f64, f64::max);
    let armed_range_m = route
        .plan
        .as_ref()
        .map(|plan| position_bf.distance(plan.end.threshold_point()));
    let content_m = match (armed_range_m, ahead_m) {
        // Armed: frame the rest of the route and the runway itself.
        (Some(threshold), ahead) => threshold.max(ahead),
        // Idle: frame the nearest runway so there is something to click.
        (None, _) => route
            .ends
            .first()
            .map(|e| e.threshold_range_m)
            .unwrap_or(2.0e4),
    }
    .max(RANGE_LEVELS_M[0]);
    let range_m = select_range(range_state, dominant, content_m, zoom.manual);

    build_nav_scene(&NavSceneInputs {
        craft_body_fixed: position_bf,
        nose_body_fixed: frame_state.orientation_body * BODY_NOSE,
        velocity_body_fixed: frame_state.translation_body.velocity,
        body_radius_m,
        strips: &strips,
        armed: route.plan.as_ref().map(|p| p.end),
        route_points: &route.display.path_points,
        rejoin_points: &route.display.rejoin_points,
        final_start_index: route.display.final_start_index,
        waypoints: &route.display.waypoints,
        range_m,
    })
}

/// Queries the ND update writes into, bundled to stay inside Bevy's system
/// parameter limit. Each is disjoint by marker component.
#[derive(SystemParam)]
pub(crate) struct NavWidgets<'w, 's> {
    header_runway: Query<
        'w,
        's,
        (&'static mut Text, &'static mut TextColor),
        (
            With<NavHeaderRunway>,
            Without<NavHeaderPhase>,
            Without<NavHeaderDistance>,
            Without<NavRangeLabel>,
            Without<NavDatumValue>,
            Without<NavRunwayLabel>,
        ),
    >,
    header_phase: Query<
        'w,
        's,
        (&'static Children, &'static mut Visibility),
        (With<NavHeaderPhase>, Without<NavRunwayLabel>),
    >,
    header_distance: Query<
        'w,
        's,
        &'static mut Text,
        (
            With<NavHeaderDistance>,
            Without<NavHeaderRunway>,
            Without<NavRangeLabel>,
            Without<NavDatumValue>,
            Without<NavRunwayLabel>,
        ),
    >,
    range_label: Query<
        'w,
        's,
        (&'static mut Text, &'static mut TextColor),
        (
            With<NavRangeLabel>,
            Without<NavHeaderRunway>,
            Without<NavHeaderDistance>,
            Without<NavDatumValue>,
            Without<NavRunwayLabel>,
        ),
    >,
    data: Query<
        'w,
        's,
        (
            &'static NavDatumValue,
            &'static mut Text,
            &'static mut TextColor,
        ),
        (
            Without<NavHeaderRunway>,
            Without<NavHeaderDistance>,
            Without<NavRangeLabel>,
            Without<NavRunwayLabel>,
        ),
    >,
    deviation_box: Query<
        'w,
        's,
        &'static mut Visibility,
        (
            With<NavDeviationBox>,
            Without<NavHeaderPhase>,
            Without<NavRunwayLabel>,
        ),
    >,
    deviation_dot: Query<
        'w,
        's,
        (&'static mut Node, &'static mut BackgroundColor),
        (With<NavDeviationDot>, Without<NavRunwayLabel>),
    >,
    /// Chip text lives on a child of the phase node.
    chip_text: Query<
        'w,
        's,
        &'static mut Text,
        (
            Without<NavHeaderRunway>,
            Without<NavHeaderDistance>,
            Without<NavRangeLabel>,
            Without<NavDatumValue>,
            Without<NavRunwayLabel>,
        ),
    >,
    labels: Query<
        'w,
        's,
        (
            &'static NavRunwayLabel,
            &'static mut Node,
            &'static mut Text,
            &'static mut Visibility,
        ),
    >,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update(
    active: Res<ActiveWidget>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    route: Res<RouteState>,
    zoom: Res<NavZoom>,
    theme: Res<HudTheme>,
    units: Res<thalos_game_state::units::UnitsSettings>,
    mut materials: ResMut<Assets<NavDisplayMaterial>>,
    canvas_q: Query<&MaterialNode<NavDisplayMaterial>, With<NavCanvas>>,
    mut widgets: NavWidgets,
    mut range_state: ResMut<NavRangeState>,
) {
    if active.0 != Some(WidgetKind::NavDisplay) {
        return;
    }
    let Ok(canvas) = canvas_q.single() else {
        return;
    };
    let Some(mut material) = materials.get_mut(canvas) else {
        return;
    };

    let Some(scene) = scene_from_world(&sim, &solar, &route, &zoom, &mut range_state) else {
        material.data = NavDisplayData::default();
        return;
    };
    material.data = nav_display_data(&scene);

    // The ND is an airliner navigation display, so its distances read in
    // nautical miles and its altitude deviations in feet unless the player has
    // asked the flight instruments to follow the global system.
    let system = units.system_for(UnitDomain::Aviation);

    // --- Runway designator labels, pinned just off each strip's threshold end.
    for (slot, mut node, mut text, mut visibility) in &mut widgets.labels {
        let entry = scene.strips.get(slot.0).and_then(|strip| {
            let designator = strip.designator?;
            let centre = heading_up_point(strip.center_m, scene.heading_rad, scene.range_m);
            let along = heading_up_dir(strip.along, scene.heading_rad);
            let half_len = strip.half_length_m / scene.range_m;
            let at = centre + along * (half_len * strip.threshold_sign);
            (at.length() <= RING_R as f64).then_some((designator, at))
        });
        match entry {
            Some((designator, at)) => {
                // Plot units → px inside the square canvas node.
                let half = MAP_SIZE_PX * 0.5;
                node.left = Val::Px(half + at.x as f32 * half + 4.0);
                node.top = Val::Px(half + at.y as f32 * half - 5.0);
                let label = format!("{designator:02}");
                if text.0 != label {
                    text.0 = label;
                }
                if *visibility != Visibility::Inherited {
                    *visibility = Visibility::Inherited;
                }
            }
            None => {
                if *visibility != Visibility::Hidden {
                    *visibility = Visibility::Hidden;
                }
            }
        }
    }

    let guidance = route.guidance.as_ref();
    let plan = route.plan.as_ref();

    // --- Header.
    if let Ok((mut text, mut color)) = widgets.header_runway.single_mut() {
        let (line, tint) = match (route.status, plan) {
            (RouteStatus::NoRunways, _) => ("NO RUNWAY".to_string(), theme.text_dim),
            (RouteStatus::Unavailable, _) => ("NAV UNAVAIL".to_string(), theme.text_dim),
            (_, Some(plan)) => (format!("RWY {:02}", plan.designator), theme.text_accent),
            _ => ("SELECT RWY".to_string(), theme.text_dim),
        };
        if text.0 != line {
            text.0 = line;
        }
        if color.0 != tint {
            color.0 = tint;
        }
    }

    let phase_text = guidance.map(|g| match g.phase {
        ApproachPhase::Transition => "INTC",
        ApproachPhase::Final => "FNL",
        ApproachPhase::Touchdown => "TDZ",
    });
    if let Ok((children, mut visibility)) = widgets.header_phase.single_mut() {
        let target = if phase_text.is_some() {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *visibility != target {
            *visibility = target;
        }
        if let Some(phase) = phase_text
            && let Some(&child) = children.first()
            && let Ok(mut text) = widgets.chip_text.get_mut(child)
            && text.0 != phase
        {
            text.0 = phase.to_string();
        }
    }

    if let Ok(mut text) = widgets.header_distance.single_mut() {
        let line = guidance
            .map(|g| format::ground_distance(g.dtg_m, system))
            .unwrap_or_default();
        if text.0 != line {
            text.0 = line;
        }
    }

    // --- Range readout: AUTO is the default, so a manual range says so.
    if let Ok((mut text, mut color)) = widgets.range_label.single_mut() {
        let line = if zoom.manual.is_some() {
            format::ground_range(scene.range_m, system)
        } else {
            format!("{} AUTO", format::ground_range(scene.range_m, system))
        };
        let tint = if zoom.manual.is_some() {
            theme.text_accent
        } else {
            theme.text_dim
        };
        if text.0 != line {
            text.0 = line;
        }
        if color.0 != tint {
            color.0 = tint;
        }
    }

    // --- The centring dot. Its two axes carry opposite screen signs on purpose:
    // see `deviation_offsets_px`.
    if let Ok(mut visibility) = widgets.deviation_box.single_mut() {
        let target = if guidance.is_some() {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *visibility != target {
            *visibility = target;
        }
    }
    if let Ok((mut node, mut color)) = widgets.deviation_dot.single_mut()
        && let Some(g) = guidance
    {
        let (dx, dy) = deviation_offsets_px(g);
        let centre = DEV_BOX_PX * 0.5 - 1.0;
        node.left = Val::Px(centre + dx - DEV_DOT_PX * 0.5);
        node.top = Val::Px(centre + dy - DEV_DOT_PX * 0.5);
        // Amber while centred, warning colour once either axis pegs — the dot
        // should not look the same when it has run out of scale.
        let pegged = g.director_lateral().abs() >= 1.0 || g.director_vertical().abs() >= 1.0;
        let tint = if pegged {
            theme.text_warn
        } else {
            theme.text_accent
        };
        if color.0 != tint {
            color.0 = tint;
        }
    }

    // --- Secondary readouts.
    for (datum, mut text, mut color) in &mut widgets.data {
        let (value, tint) = match datum.0 {
            NavDatum::Heading => (
                format!(
                    "{:03}",
                    scene.heading_rad.to_degrees().rem_euclid(360.0).round() as i32
                ),
                theme.text_primary,
            ),
            NavDatum::Track => (
                scene
                    .track_rad
                    .map(|t| format!("{:03}", t.to_degrees().rem_euclid(360.0).round() as i32))
                    .unwrap_or_else(|| "---".to_string()),
                theme.text_primary,
            ),
            NavDatum::CrossTrack => match guidance {
                Some(g) => {
                    let side = if g.cross_track_m >= 0.0 { 'R' } else { 'L' };
                    (
                        format!(
                            "{} {}",
                            side,
                            format::ground_distance(g.cross_track_m.abs(), system)
                        ),
                        if g.loc_deflection().abs() >= 1.0 {
                            theme.text_warn
                        } else {
                            theme.text_primary
                        },
                    )
                }
                None => ("---".to_string(), theme.text_dim),
            },
            NavDatum::Glideslope => match guidance {
                Some(g) => (
                    fmt_altitude_error(g, system),
                    if g.gs_deflection().abs() >= 1.0 {
                        theme.text_warn
                    } else {
                        theme.text_primary
                    },
                ),
                None => ("---".to_string(), theme.text_dim),
            },
        };
        if text.0 != value {
            text.0 = value;
        }
        if color.0 != tint {
            color.0 = tint;
        }
    }
}

/// Screen offsets (px) of the steering dot, from the box centre.
///
/// **The dot is where to point the aircraft, and it follows the route — not the
/// runway.** It reads the route-relative director
/// ([`Guidance::director_lateral`] / [`Guidance::director_vertical`]), which
/// steers along the planned *path* on whatever leg the craft is on, and along a
/// flyable rejoin when the craft is off course. The runway is only where today's
/// route happens to end; the identical cue will fly a waypoint route.
///
/// This is deliberately **not** the ILS localizer/glideslope pair. Those are
/// beam deviations measured against the final approach centreline: correct for
/// the PFD's needles, meaningless on a base leg, and silent about *how* to get
/// back — which is the question a steering cue exists to answer.
///
/// Screen signs: `x` is positive right, so a right turn puts the dot right.
/// Screen `y` grows **downward**, so a climb command is a *negative* offset —
/// the axes carry opposite signs and that is not a bug. Pinned by
/// `the_dot_leads_where_the_craft_should_steer`.
fn deviation_offsets_px(guidance: &Guidance) -> (f32, f32) {
    let x = (guidance.director_lateral() as f32) * DEV_HALF_PX;
    let y = -(guidance.director_vertical() as f32) * DEV_HALF_PX;
    (x, y)
}

/// Zoom controls and the scroll wheel. **Sole writer** of [`NavZoom`].
pub(crate) fn handle_zoom(
    active: Res<ActiveWidget>,
    buttons: Query<(&Interaction, &NavZoomButton), Changed<Interaction>>,
    canvas_q: Query<&Interaction, With<NavCanvas>>,
    mut wheel: MessageReader<bevy::input::mouse::MouseWheel>,
    mut zoom: ResMut<NavZoom>,
    range_state: Res<NavRangeState>,
) {
    if active.0 != Some(WidgetKind::NavDisplay) {
        // Drain the wheel so a scroll made elsewhere does not apply late.
        wheel.clear();
        return;
    }
    let auto_index = range_state.idx;

    for (interaction, button) in &buttons {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        match button {
            // Zooming *in* means a smaller range, i.e. a lower rung.
            NavZoomButton::In => zoom.step(-1, auto_index),
            NavZoomButton::Out => zoom.step(1, auto_index),
            NavZoomButton::Auto => zoom.manual = None,
        }
    }

    // The wheel only zooms while the cursor is over the plot, so scrolling past
    // the HUD does not silently rescale it.
    let over_plot = canvas_q
        .iter()
        .any(|i| matches!(i, Interaction::Hovered | Interaction::Pressed));
    let mut steps = 0i32;
    for event in wheel.read() {
        if !over_plot {
            continue;
        }
        if event.y > 0.0 {
            steps -= 1;
        } else if event.y < 0.0 {
            steps += 1;
        }
    }
    if steps != 0 {
        zoom.step(steps, auto_index);
    }
}

/// Altitude deviation from the vertical profile, with the sense a pilot reads
/// (`+` = high).
///
/// The "on glideslope" band is a **physical** 10 m, not a display-unit count, so
/// the annunciation triggers at the same real deviation in either system.
fn fmt_altitude_error(guidance: &Guidance, system: UnitSystem) -> String {
    let err = guidance.altitude_error_m;
    if err.abs() < 10.0 {
        "ON GS".to_string()
    } else {
        format::altitude_delta(err, system)
    }
}

/// Clicking the plot arms an approach to the runway under the cursor.
pub(crate) fn handle_canvas_click(
    active: Res<ActiveWidget>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    route: Res<RouteState>,
    zoom: Res<NavZoom>,
    canvas_q: Query<
        (&Interaction, &RelativeCursorPosition),
        (With<NavCanvas>, Changed<Interaction>),
    >,
    mut requests: MessageWriter<RouteRequest>,
    range_state: Res<NavRangeState>,
) {
    if active.0 != Some(WidgetKind::NavDisplay) {
        return;
    }
    for (interaction, cursor) in &canvas_q {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        let Some(normalized) = cursor.normalized else {
            continue;
        };
        // Node-normalised [0, 1] → the shader's centred [-1, 1] (y down).
        let plot = DVec2::new(
            (normalized.x as f64 - 0.5) * 2.0,
            (normalized.y as f64 - 0.5) * 2.0,
        );
        // A copy: hit-testing must not nudge the hysteresis state that `update`
        // owns, or clicking the plot could change its scale.
        let mut range_snapshot = *range_state;
        let Some(scene) = scene_from_world(&sim, &solar, &route, &zoom, &mut range_snapshot) else {
            continue;
        };
        // Nearest strip whose drawn rectangle is within tolerance of the click.
        let mut best: Option<(f64, u64)> = None;
        for strip in &scene.strips {
            let centre = heading_up_point(strip.center_m, scene.heading_rad, scene.range_m);
            let along = heading_up_dir(strip.along, scene.heading_rad);
            let half_len = strip.half_length_m / scene.range_m;
            let rel = plot - centre;
            let t = rel.dot(along).clamp(-half_len, half_len);
            let distance = (rel - along * t).length();
            if distance <= PICK_TOLERANCE as f64
                && best.is_none_or(|(best_distance, _)| distance < best_distance)
            {
                best = Some((distance, strip.id));
            }
        }
        if let Some((_, strip_id)) = best
            && let Some(entry) = route.ends.iter().find(|e| e.end.strip.id == strip_id)
        {
            requests.write(RouteRequest::Pick(entry.armed_end.strip));
        }
    }
}

/// Selector buttons → selection requests.
pub(crate) fn handle_select_buttons(
    buttons: Query<(&Interaction, &NavSelectButton), Changed<Interaction>>,
    mut requests: MessageWriter<RouteRequest>,
) {
    for (interaction, button) in &buttons {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        requests.write(match button {
            NavSelectButton::Prev => RouteRequest::Cycle(-1),
            NavSelectButton::Next => RouteRequest::Cycle(1),
            NavSelectButton::Flip => RouteRequest::Flip,
            NavSelectButton::Clear => RouteRequest::Clear,
        });
    }
}

/// Project a local `(east, north)` offset (metres) into heading-up plot
/// coordinates in `[-1, 1]` (y points down, matching the shader).
fn heading_up_point(local_m: DVec2, heading_rad: f64, range_m: f64) -> DVec2 {
    let (sin, cos) = heading_rad.sin_cos();
    let x_right = local_m.x * cos - local_m.y * sin;
    let y_fwd = local_m.x * sin + local_m.y * cos;
    DVec2::new(x_right / range_m, -y_fwd / range_m)
}

/// Heading-up unit direction for a local-plane vector.
fn heading_up_dir(local: DVec2, heading_rad: f64) -> DVec2 {
    let (sin, cos) = heading_rad.sin_cos();
    let x_right = local.x * cos - local.y * sin;
    let y_fwd = local.x * sin + local.y * cos;
    DVec2::new(x_right, -y_fwd).normalize_or_zero()
}

/// A compass angle expressed relative to the plot's up direction, so the shader
/// can place a marker without knowing the heading convention.
fn heading_up_angle(compass_rad: f64, heading_rad: f64) -> f64 {
    compass_rad - heading_rad
}

/// The plot's half-extent in metres.
///
/// `manual` pins a rung of [`RANGE_LEVELS_M`]; otherwise the smallest rung that
/// contains `1.2 ×` the content wins, with step-down hysteresis so the scale
/// does not flap on approach. The automatic state keeps tracking even while a
/// manual range is pinned, so releasing back to AUTO lands on the right rung
/// instead of wherever the pilot last left it.
fn select_range(
    state: &mut NavRangeState,
    dominant: BodyId,
    content_m: f64,
    manual: Option<usize>,
) -> f64 {
    let content = content_m * 1.2;
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
    match manual {
        Some(index) => RANGE_LEVELS_M[index.min(RANGE_LEVELS_M.len() - 1)],
        None => RANGE_LEVELS_M[state.idx],
    }
}

fn lin(color: Color) -> Vec4 {
    color.to_linear().to_vec4()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(center: DVec2, along: DVec2, half_len: f64) -> NavStrip {
        NavStrip {
            id: 1,
            center_m: center,
            along,
            half_length_m: half_len,
            half_width_m: 45.0,
            armed: false,
            threshold_sign: -1.0,
            designator: Some(3),
        }
    }

    #[test]
    fn runway_length_is_drawn_to_scale() {
        // A 5 km strip (2.5 km half-length) at 10 km range must cover a quarter
        // of the plot half-extent — the whole point of the revamp, and the thing
        // the old fixed 0.06 symbol got wrong by 20x.
        let scene = NavScene {
            heading_rad: 0.0,
            strips: vec![strip(
                DVec2::new(0.0, 3_000.0),
                DVec2::new(0.0, 1.0),
                2_500.0,
            )],
            range_m: 10_000.0,
            ..Default::default()
        };
        let data = nav_display_data(&scene);
        assert_eq!(data.params.x as i32, 1);
        assert!(
            (data.runway_ext[0].x - 0.25).abs() < 1e-5,
            "half-length should be 0.25 plot units, got {}",
            data.runway_ext[0].x
        );
    }

    #[test]
    fn a_strip_longer_than_the_plot_still_draws() {
        // At 2 km range a 5 km strip overflows the plot entirely. It must be
        // kept (the near half is visible), not culled on its centre.
        let scene = NavScene {
            strips: vec![strip(
                DVec2::new(0.0, 2_400.0),
                DVec2::new(0.0, 1.0),
                2_500.0,
            )],
            range_m: 2_000.0,
            ..Default::default()
        };
        let data = nav_display_data(&scene);
        assert_eq!(data.params.x as i32, 1, "overflowing strip was culled");
        assert!(data.runway_ext[0].x > 1.0);
    }

    #[test]
    fn a_strip_entirely_off_plot_is_culled() {
        let scene = NavScene {
            strips: vec![strip(
                DVec2::new(0.0, 80_000.0),
                DVec2::new(0.0, 1.0),
                2_500.0,
            )],
            range_m: 5_000.0,
            ..Default::default()
        };
        assert_eq!(nav_display_data(&scene).params.x as i32, 0);
    }

    #[test]
    fn heading_up_rotation_puts_the_nose_at_the_top() {
        // Something due east, while flying east, must plot straight ahead (up).
        let p = heading_up_point(
            DVec2::new(1_000.0, 0.0),
            std::f64::consts::FRAC_PI_2,
            2_000.0,
        );
        assert!(p.x.abs() < 1e-9, "should be centred laterally: {p:?}");
        assert!(p.y < 0.0, "ahead is up, i.e. negative y: {p:?}");
        // And something due north while flying east plots to the left.
        let p = heading_up_point(
            DVec2::new(0.0, 1_000.0),
            std::f64::consts::FRAC_PI_2,
            2_000.0,
        );
        assert!(p.x < 0.0, "north should be on the left when heading east");
    }

    #[test]
    fn route_is_decimated_without_losing_the_final_boundary() {
        let points: Vec<DVec2> = (0..200).map(|i| DVec2::new(i as f64 * 10.0, 0.0)).collect();
        let route = decimate_route(&points, 150, MAX_ROUTE_POINTS);
        assert!(route.points.len() <= MAX_ROUTE_POINTS);
        assert!(route.final_index < route.points.len());
        // The point where the final begins must survive as the boundary.
        assert!(
            (route.points[route.final_index].x - 1_500.0).abs() < 60.0,
            "final boundary drifted to {}",
            route.points[route.final_index].x
        );
        // Endpoints preserved.
        assert!((route.points[0].x - 0.0).abs() < 1e-9);
        assert!((route.points.last().expect("non-empty").x - 1_990.0).abs() < 1e-9);
    }

    #[test]
    fn a_short_route_is_passed_through_unchanged() {
        let points = vec![DVec2::ZERO, DVec2::new(10.0, 0.0), DVec2::new(20.0, 0.0)];
        let route = decimate_route(&points, 2, MAX_ROUTE_POINTS);
        assert_eq!(route.points.len(), 3);
        assert_eq!(route.final_index, 2);
    }

    #[test]
    fn route_points_pack_two_per_uniform_slot() {
        let scene = NavScene {
            route_m: vec![
                DVec2::new(0.0, 0.0),
                DVec2::new(0.0, 1_000.0),
                DVec2::new(500.0, 1_000.0),
            ],
            route_final_index: 2,
            range_m: 2_000.0,
            ..Default::default()
        };
        let data = nav_display_data(&scene);
        assert_eq!(data.params.y as i32, 3);
        // Point 0 in .xy of slot 0, point 1 in .zw of slot 0, point 2 in .xy of 1.
        assert_eq!(data.route[0].x, 0.0);
        assert!((data.route[0].w - (-0.5)).abs() < 1e-6, "point 1 y in .w");
        assert!((data.route[1].x - 0.25).abs() < 1e-6, "point 2 x in slot 1");
    }

    #[test]
    fn absent_markers_use_the_sentinel() {
        let data = nav_display_data(&NavScene {
            range_m: 2_000.0,
            ..Default::default()
        });
        assert_eq!(data.nav.w, NO_ANGLE, "no bearing marker");
        assert_eq!(data.extra.y, NO_ANGLE, "no track marker");
    }

    #[test]
    fn the_dot_leads_where_the_craft_should_steer() {
        // Steer right: the dot goes right. This is the *route* director, so it
        // reads the same on a base leg as on final — unlike a localizer needle,
        // which has no opinion off the centreline.
        let turn_right = guidance_steering(0.5, 0.0);
        let (x, _) = deviation_offsets_px(&turn_right);
        assert!(x > 0.0, "dot should lead right, got x = {x}");
        let turn_left = guidance_steering(-0.5, 0.0);
        assert!(deviation_offsets_px(&turn_left).0 < 0.0);

        // Climb command: the dot goes UP, and screen y grows downward, so that
        // is a NEGATIVE offset. The axes carry opposite signs on purpose.
        let climb = guidance_steering(0.0, 0.5);
        let (_, y) = deviation_offsets_px(&climb);
        assert!(y < 0.0, "a climb cue should sit above centre, got y = {y}");
        let descend = guidance_steering(0.0, -0.5);
        assert!(deviation_offsets_px(&descend).1 > 0.0);

        // On the path, on the profile: centred.
        let (x, y) = deviation_offsets_px(&guidance_steering(0.0, 0.0));
        assert!(x.abs() < 1e-6 && y.abs() < 1e-6);
    }

    #[test]
    fn the_dot_stays_inside_its_box_at_any_deviation() {
        // Full scale is a clamp, so a wildly off-course craft parks the dot on
        // the edge instead of drawing it outside the frame.
        for (lat, vert) in [(9.0, 0.0), (-9.0, 0.0), (0.0, 12.0), (-12.0, -12.0)] {
            let (x, y) = deviation_offsets_px(&guidance_steering(lat, vert));
            assert!(x.abs() <= DEV_HALF_PX + 1e-6, "x {x} escaped the box");
            assert!(y.abs() <= DEV_HALF_PX + 1e-6, "y {y} escaped the box");
        }
    }

    #[test]
    fn the_dot_ignores_the_ils_beams() {
        // Pegged localizer and glideslope with the steering answer centred: the
        // dot must stay centred, because it follows the route, not the runway.
        let mut g = guidance_steering(0.0, 0.0);
        g.loc_deviation_rad = 10.0 * thalos_navigation::guidance::LOC_FULL_SCALE_RAD;
        g.gs_deviation_rad = 10.0 * thalos_navigation::guidance::GS_FULL_SCALE_RAD;
        let (x, y) = deviation_offsets_px(&g);
        assert!(
            x.abs() < 1e-6 && y.abs() < 1e-6,
            "the beams must not move the steering dot: ({x}, {y})"
        );
    }

    #[test]
    fn zoom_steps_from_where_the_plot_already_is() {
        let mut zoom = NavZoom::default();
        // First step seeds from the automatic rung rather than jumping to 0.
        zoom.step(-1, 5);
        assert_eq!(zoom.manual, Some(4));
        zoom.step(-1, 5);
        assert_eq!(zoom.manual, Some(3));
        zoom.step(1, 5);
        assert_eq!(zoom.manual, Some(4));
    }

    #[test]
    fn zoom_clamps_to_the_ladder() {
        let mut zoom = NavZoom::default();
        for _ in 0..40 {
            zoom.step(-1, 5);
        }
        assert_eq!(zoom.manual, Some(0));
        for _ in 0..40 {
            zoom.step(1, 5);
        }
        assert_eq!(zoom.manual, Some(RANGE_LEVELS_M.len() - 1));
    }

    #[test]
    fn a_manual_range_wins_but_auto_keeps_tracking() {
        let mut state = NavRangeState::default();
        let body: BodyId = 0;
        // Pinned to the tightest rung while the content wants a wide one.
        let range = select_range(&mut state, body, 40_000.0, Some(0));
        assert_eq!(range, RANGE_LEVELS_M[0]);
        // Releasing to AUTO lands on the rung the content needs — not on
        // whatever the manual pin left behind.
        let range = select_range(&mut state, body, 40_000.0, None);
        assert!(
            range >= 40_000.0,
            "auto should contain the content, got {range}"
        );
    }

    /// A `Guidance` carrying a steering answer: `lateral` in units of full-scale
    /// heading error (+ = turn right), `vertical` in units of full-scale
    /// flight-path-angle error (+ = climb).
    fn guidance_steering(lateral: f64, vertical: f64) -> Guidance {
        let track = 1.0_f64;
        Guidance {
            desired_heading_rad: track
                + lateral * thalos_navigation::guidance::DIRECTOR_HEADING_FULL_SCALE_RAD,
            track_heading_rad: track,
            fpa_rad: 0.0,
            fpa_command_rad: vertical * thalos_navigation::guidance::DIRECTOR_FPA_FULL_SCALE_RAD,
            ..guidance_with(0.0, 0.0)
        }
    }

    /// A `Guidance` carrying only the ILS beam deviations.
    fn guidance_with(loc_scale: f64, gs_scale: f64) -> Guidance {
        Guidance {
            phase: ApproachPhase::Final,
            cross_track_m: 0.0,
            course_heading_rad: 0.0,
            track_heading_rad: 0.0,
            desired_heading_rad: 0.0,
            fpa_rad: 0.0,
            fpa_command_rad: 0.0,
            dtg_m: 5_000.0,
            along_m: 0.0,
            threshold_range_m: 5_000.0,
            altitude_m: 0.0,
            target_altitude_m: 0.0,
            altitude_error_m: 0.0,
            target_speed_m_s: None,
            next_gate: None,
            loc_deviation_rad: loc_scale * thalos_navigation::guidance::LOC_FULL_SCALE_RAD,
            gs_deviation_rad: gs_scale * thalos_navigation::guidance::GS_FULL_SCALE_RAD,
            bank_command_rad: 0.0,
            vertical_speed_command_m_s: 0.0,
            established: true,
        }
    }

    #[test]
    fn range_ladder_snaps_up_and_holds_with_hysteresis() {
        let mut state = NavRangeState::default();
        let body: BodyId = 0;
        // 8 km of content needs the 10 km level.
        assert_eq!(select_range(&mut state, body, 8_000.0, None), 1.0e4);
        // Closing slightly must NOT drop a level (hysteresis).
        assert_eq!(select_range(&mut state, body, 7_000.0, None), 1.0e4);
        // Closing a lot does — and now reaches the short-final rungs the old
        // ladder did not have.
        assert_eq!(select_range(&mut state, body, 300.0, None), 500.0);
        // Growing past the level steps up.
        assert_eq!(select_range(&mut state, body, 40_000.0, None), 5.0e4);
    }
}
