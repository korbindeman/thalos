//! Headless navigation-display preview: renders the ND in a grid of approach
//! situations and writes one PNG, so ND symbology can be checked in seconds
//! without launching the game (`just nd-preview`).
//!
//! # Why this exists
//!
//! The ND's job is to be *correct* in situations that are slow and fiddly to
//! fly to: 60 km out where a 5 km runway is three pixels long, overflown and
//! turning back, established on short final, landing the reciprocal end. Testing
//! that through `just game runway-approach` costs a boot, a flight, and a human;
//! testing it here costs one process and produces an image an agent can read.
//!
//! # What it is evidence of, and what it is not
//!
//! Every panel is built from a **real** [`plan_approach`] result, tessellated
//! through the game's own `route::plan_display`, projected through the game's own
//! `build_nav_scene`, and drawn by the real `nav_display.wgsl`. So this is
//! genuine evidence about planner geometry, projection, scale, and shader
//! symbology.
//!
//! It is **not** evidence about the ECS wiring: resource plumbing, widget
//! auto-selection, click handling, and the PFD deviation scales are not
//! exercised here and still need an in-game check.

use std::time::Duration;

use bevy::app::{AppExit, ScheduleRunnerPlugin};
use bevy::asset::{AssetPlugin, RenderAssetUsages};
use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::math::{DVec2, DVec3};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::ui::IsDefaultUiCamera;
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;

use thalos_navigation::{
    ApproachParams, ApproachPlan, Pose2, RejoinParams, RouteFrame, RunwayEnd, RunwayStrip,
    VnavParams, plan_approach, plan_rejoin, theta_of,
};
use thalos_runtime::display_preview::{
    NavDisplayMaterial, NavSceneInputs, build_nav_scene, nav_display_data,
};
use thalos_runtime::route::plan_display;

/// Plot size per panel (px). Larger than the in-game 200 px so thin symbology is
/// legible in the contact sheet.
const PLOT_PX: f32 = 250.0;
const COLUMNS: usize = 5;
const OUT_PATH: &str = "artifacts/visual/latest/nav_preview.png";
const WIDTH: u32 = 1460;
const HEIGHT: u32 = 1010;
/// Frames before the capture: pipeline compile + font atlas fill.
const WARMUP_FRAMES: u32 = 90;
const TAIL_FRAMES: u32 = 16;

// --- The spaceport this preview mirrors (see `crate::runway` for the real one:
// a 5 km × 90 m primary at heading 30°, plus a 3.6 km × 80 m crosswind strip
// diverging 30° from near the primary's south threshold).
const BODY_RADIUS_M: f64 = 6.0e6;
const ELEVATION_M: f64 = 700.0;
const SITE_LAT_DEG: f64 = 7.6;
const SITE_LON_DEG: f64 = 178.0;
const PRIMARY_HEADING_DEG: f64 = 30.0;
const PRIMARY_HALF_LENGTH_M: f64 = 2_500.0;
const PRIMARY_HALF_WIDTH_M: f64 = 45.0;
const SECONDARY_HALF_LENGTH_M: f64 = 1_800.0;
const SECONDARY_HALF_WIDTH_M: f64 = 40.0;
const SECONDARY_OFFSET_DEG: f64 = 30.0;
const SEC_NEAR_ALONG_M: f64 = -2_400.0;
const SEC_NEAR_ACROSS_M: f64 = 420.0;

fn main() {
    std::fs::create_dir_all("artifacts/visual/latest").ok();

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                close_when_requested: false,
                ..default()
            })
            .set(AssetPlugin {
                // Relative to CARGO_MANIFEST_DIR (crates/runtime/game).
                file_path: "../../../assets".to_string(),
                ..default()
            })
            .disable::<WinitPlugin>(),
    )
    .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
        1.0 / 60.0,
    )))
    .add_plugins(UiMaterialPlugin::<NavDisplayMaterial>::default())
    .add_systems(Startup, setup)
    .add_systems(Update, drive_capture)
    .run();
}

// ---------------------------------------------------------------------------
// Scenario construction
// ---------------------------------------------------------------------------

/// Body-fixed direction for a latitude/longitude, matching `runway::latlon_dir`.
fn latlon_dir(lat_deg: f64, lon_deg: f64) -> DVec3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()).normalize()
}

/// The two strips of the preview spaceport.
fn strips() -> Vec<RunwayStrip> {
    let center_dir = latlon_dir(SITE_LAT_DEG, SITE_LON_DEG);
    let site = RouteFrame::new(center_dir, BODY_RADIUS_M, ELEVATION_M).expect("valid site");
    let heading = local_dir(&site, PRIMARY_HEADING_DEG.to_radians());
    let across = center_dir.cross(heading).normalize();

    let primary = RunwayStrip {
        id: 1,
        center_dir,
        heading_tangent: heading,
        half_length_m: PRIMARY_HALF_LENGTH_M,
        half_width_m: PRIMARY_HALF_WIDTH_M,
        elevation_m: ELEVATION_M,
        body_radius_m: BODY_RADIUS_M,
    };

    let sec_heading = {
        let a = SECONDARY_OFFSET_DEG.to_radians();
        (heading * a.cos() + across * a.sin()).normalize()
    };
    let sec_center_offset = heading * SEC_NEAR_ALONG_M
        + across * SEC_NEAR_ACROSS_M
        + sec_heading * SECONDARY_HALF_LENGTH_M;
    let sec_center_dir =
        (center_dir * (BODY_RADIUS_M + ELEVATION_M) + sec_center_offset).normalize();
    let secondary = RunwayStrip {
        id: 2,
        center_dir: sec_center_dir,
        heading_tangent: sec_heading,
        half_length_m: SECONDARY_HALF_LENGTH_M,
        half_width_m: SECONDARY_HALF_WIDTH_M,
        elevation_m: ELEVATION_M,
        body_radius_m: BODY_RADIUS_M,
    };
    vec![primary, secondary]
}

/// Panel caption range, which has to stay honest below a kilometre — the
/// close-in rungs are exactly what these panels exist to show.
fn fmt_range(range_m: f64) -> String {
    if range_m >= 1_000.0 {
        format!("{:.0} km", range_m / 1_000.0)
    } else {
        format!("{range_m:.0} m")
    }
}

/// Body-fixed unit direction for a compass heading in a frame.
fn local_dir(frame: &RouteFrame, heading_rad: f64) -> DVec3 {
    (frame.north * heading_rad.cos() + frame.east * heading_rad.sin()).normalize()
}

/// One preview panel.
struct Panel {
    caption: String,
    scene_inputs: PanelInputs,
}

/// Owned inputs for a panel (the borrowed `NavSceneInputs` is built from these
/// at draw time).
struct PanelInputs {
    craft_body_fixed: DVec3,
    nose_body_fixed: DVec3,
    velocity_body_fixed: DVec3,
    strips: Vec<RunwayStrip>,
    armed: Option<RunwayEnd>,
    route_points: Vec<DVec3>,
    rejoin_points: Vec<DVec3>,
    final_start_index: usize,
    waypoints: Vec<(DVec3, thalos_navigation::WaypointKind)>,
    range_m: f64,
}

fn params() -> ApproachParams {
    ApproachParams {
        maneuver_speed_m_s: 110.0,
        vnav: VnavParams {
            approach_speed_m_s: 80.0,
            final_dtg_m: 9_300.0,
            ..VnavParams::default()
        },
        ..ApproachParams::default()
    }
}

/// Build a panel: place the craft at `(along_m, across_m)` relative to the armed
/// threshold (negative `along` = before it), on `heading_deg`, and plan.
///
/// `armed_index` selects the strip and `reciprocal` the landing direction;
/// `None` leaves nothing armed (the idle state).
#[allow(clippy::too_many_arguments)]
fn panel(
    caption: &str,
    armed_index: Option<usize>,
    reciprocal: bool,
    along_m: f64,
    across_m: f64,
    altitude_m: f64,
    heading_offset_deg: f64,
    range_m: f64,
) -> Panel {
    panel_drifted(
        caption,
        armed_index,
        reciprocal,
        along_m,
        across_m,
        altitude_m,
        heading_offset_deg,
        range_m,
        0.0,
    )
}

/// Like [`panel`], but the craft is displaced `drift_across_m` to the right
/// *after* the route was planned — which is the situation a rejoin exists for:
/// a committed route and a craft no longer on it.
#[allow(clippy::too_many_arguments)]
fn panel_drifted(
    caption: &str,
    armed_index: Option<usize>,
    reciprocal: bool,
    along_m: f64,
    across_m: f64,
    altitude_m: f64,
    heading_offset_deg: f64,
    range_m: f64,
    drift_across_m: f64,
) -> Panel {
    let all = strips();
    let strip = all[armed_index.unwrap_or(0)];
    let end = RunwayEnd { strip, reciprocal };
    let frame = end.route_frame().expect("valid threshold");
    let landing_local = frame
        .direction_to_local(end.landing_dir())
        .normalize_or_zero();
    let right = DVec2::new(landing_local.y, -landing_local.x);
    let local = landing_local * along_m + right * across_m;
    let craft_body_fixed = frame.to_body_fixed(local, altitude_m);

    let landing_heading = end.landing_heading_rad(&frame);
    let craft_heading = landing_heading + heading_offset_deg.to_radians();
    let craft_frame = RouteFrame::new(craft_body_fixed.normalize(), BODY_RADIUS_M, altitude_m)
        .expect("valid craft frame");
    let nose_body_fixed = local_dir(&craft_frame, craft_heading);
    // A flying craft tracks its nose here (no wind in this preview).
    let velocity_body_fixed = nose_body_fixed * 90.0;

    let (armed, plan): (Option<RunwayEnd>, Option<ApproachPlan>) = match armed_index {
        Some(_) => {
            let plan = plan_approach(end, craft_body_fixed, nose_body_fixed, &params());
            (Some(end), plan)
        }
        None => (None, None),
    };

    // Displace the craft off the planned route, then plan the way back.
    let (craft_body_fixed, nose_body_fixed, velocity_body_fixed) = if drift_across_m.abs() > 1.0 {
        let right = DVec2::new(landing_local.y, -landing_local.x);
        let drifted_local = local + right * drift_across_m;
        let drifted = frame.to_body_fixed(drifted_local, altitude_m);
        let drifted_frame =
            RouteFrame::new(drifted.normalize(), BODY_RADIUS_M, altitude_m).expect("valid");
        let nose = local_dir(&drifted_frame, craft_heading);
        (drifted, nose, nose * 90.0)
    } else {
        (craft_body_fixed, nose_body_fixed, velocity_body_fixed)
    };

    let rejoin_points = plan
        .as_ref()
        .filter(|_| drift_across_m.abs() > 1.0)
        .and_then(|plan| {
            let plan_frame = plan.frame;
            let here = plan_frame.to_local(craft_body_fixed);
            let track = plan_frame
                .direction_to_local(nose_body_fixed)
                .try_normalize()?;
            let closest = plan.path.closest(here)?;
            let rejoin = plan_rejoin(
                &plan.path,
                Pose2::new(here, theta_of(track)),
                closest.along_m,
                &RejoinParams::for_radius(plan.turn_radius_m),
                None,
            )?;
            let elevation = plan_frame.origin_altitude_m;
            Some(
                rejoin
                    .path
                    .polyline(20.0)
                    .into_iter()
                    .map(|q| plan_frame.to_body_fixed(q, elevation))
                    .collect::<Vec<_>>(),
            )
        })
        .unwrap_or_default();

    let display = plan.as_ref().map(plan_display);
    Panel {
        caption: caption.to_string(),
        scene_inputs: PanelInputs {
            craft_body_fixed,
            nose_body_fixed,
            velocity_body_fixed,
            strips: all,
            armed,
            route_points: display
                .as_ref()
                .map(|d| d.path_points.clone())
                .unwrap_or_default(),
            rejoin_points,
            final_start_index: display.as_ref().map(|d| d.final_start_index).unwrap_or(0),
            waypoints: display
                .as_ref()
                .map(|d| d.waypoints.clone())
                .unwrap_or_default(),
            range_m,
        },
    }
}

/// The situations worth looking at. Each one is a question the old ND could not
/// answer.
fn panels() -> Vec<Panel> {
    vec![
        // Is the runway drawn at its real 5 km, and does the route run straight in?
        panel(
            "straight-in 12 km",
            Some(0),
            false,
            -12_000.0,
            0.0,
            1_400.0,
            0.0,
            20_000.0,
        ),
        // Does an offset approach produce a real S-turn intercept?
        panel(
            "offset 6 km right",
            Some(0),
            false,
            -20_000.0,
            6_000.0,
            2_200.0,
            0.0,
            50_000.0,
        ),
        // Overflown the field: does it plan the turn-around rather than a straight line?
        panel(
            "overflown, turning back",
            Some(0),
            false,
            8_000.0,
            1_500.0,
            1_600.0,
            0.0,
            20_000.0,
        ),
        // Short final: does the strip fill the plot and the threshold bar read?
        panel(
            "short final 3 km",
            Some(0),
            false,
            -3_000.0,
            -150.0,
            850.0,
            -4.0,
            5_000.0,
        ),
        // Landing the other way: the threshold bar and designator must flip ends.
        panel(
            "reciprocal end",
            Some(0),
            true,
            -11_000.0,
            0.0,
            1_300.0,
            0.0,
            20_000.0,
        ),
        // The crosswind strip armed, with the primary still drawn unarmed.
        panel(
            "crosswind strip",
            Some(1),
            false,
            -9_000.0,
            3_000.0,
            1_200.0,
            20.0,
            20_000.0,
        ),
        // Long range: a 90 m-wide strip must still be visible, at true length.
        panel(
            "60 km out",
            Some(0),
            false,
            -60_000.0,
            12_000.0,
            6_000.0,
            -30.0,
            150_000.0,
        ),
        // Nothing armed: two strips, no route, no bearing pointer.
        panel(
            "idle, none armed",
            None,
            false,
            -6_000.0,
            2_000.0,
            900.0,
            15.0,
            20_000.0,
        ),
        // Zoomed in for the last mile — the rungs the range ladder gained. The
        // 5 km strip runs off the plot on purpose: at this range you are looking
        // at the touchdown zone, not at the airfield.
        panel(
            "1.5 km out, 1 km range",
            Some(0),
            false,
            -1_500.0,
            -60.0,
            780.0,
            -2.0,
            1_000.0,
        ),
        panel(
            "over the threshold, 500 m",
            Some(0),
            false,
            -200.0,
            -10.0,
            720.0,
            0.0,
            500.0,
        ),
        // Blown off a committed route: the dashed rejoin is the flyable way
        // back, meeting the route tangentially rather than cutting at it.
        panel_drifted(
            "drifted 2 km right",
            Some(0),
            false,
            -14_000.0,
            0.0,
            1_500.0,
            0.0,
            20_000.0,
            2_000.0,
        ),
        panel_drifted(
            "drifted 5 km left",
            Some(0),
            false,
            -18_000.0,
            0.0,
            1_900.0,
            0.0,
            50_000.0,
            -5_000.0,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<NavDisplayMaterial>>,
    assets: Res<AssetServer>,
) {
    // Headless render target + UI camera.
    let mut target = Image::new_fill(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[8, 10, 12, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    target.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT;
    let target = images.add(target);
    commands.spawn((
        Camera2d,
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb(0.03, 0.04, 0.05)),
            ..default()
        },
        RenderTarget::Image(ImageRenderTarget::from(target.clone())),
        IsDefaultUiCamera,
    ));
    commands.insert_resource(CaptureTarget(target));
    commands.init_resource::<CaptureState>();

    // 0.19: `TextFont.font` is a `FontSource`, not a raw handle.
    let font = FontSource::Handle(assets.load("fonts/FiraCode-Regular.ttf"));

    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                row_gap: Val::Px(6.0),
                padding: UiRect::all(Val::Px(10.0)),
                ..default()
            },
            Name::new("NavPreviewRoot"),
        ))
        .with_children(|root| {
            root.spawn((
                Text::new("ND preview — real approach plans, real shader"),
                TextFont {
                    font: font.clone(),
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(Color::srgb(0.85, 0.87, 0.82)),
            ));
            let all = panels();
            for row in all.chunks(COLUMNS) {
                root.spawn((
                    Node {
                        flex_direction: FlexDirection::Row,
                        column_gap: Val::Px(8.0),
                        ..default()
                    },
                    Name::new("NavPreviewRow"),
                ))
                .with_children(|row_node| {
                    for panel in row {
                        let inputs = &panel.scene_inputs;
                        let scene = build_nav_scene(&NavSceneInputs {
                            craft_body_fixed: inputs.craft_body_fixed,
                            nose_body_fixed: inputs.nose_body_fixed,
                            velocity_body_fixed: inputs.velocity_body_fixed,
                            body_radius_m: BODY_RADIUS_M,
                            strips: &inputs.strips,
                            armed: inputs.armed,
                            route_points: &inputs.route_points,
                            rejoin_points: &inputs.rejoin_points,
                            final_start_index: inputs.final_start_index,
                            waypoints: &inputs.waypoints,
                            range_m: inputs.range_m,
                        })
                        .expect("scene projects");
                        let material =
                            materials.add(NavDisplayMaterial::new(nav_display_data(&scene)));
                        row_node
                            .spawn((
                                Node {
                                    flex_direction: FlexDirection::Column,
                                    align_items: AlignItems::Center,
                                    row_gap: Val::Px(2.0),
                                    ..default()
                                },
                                Name::new("NavPreviewPanel"),
                            ))
                            .with_children(|cell| {
                                cell.spawn((
                                    Node {
                                        width: Val::Px(PLOT_PX),
                                        height: Val::Px(PLOT_PX),
                                        border: UiRect::all(Val::Px(1.0)),
                                        ..default()
                                    },
                                    BorderColor::all(Color::srgba(0.4, 0.42, 0.4, 0.5)),
                                    MaterialNode(material),
                                ));
                                cell.spawn((
                                    Text::new(panel.caption.clone()),
                                    TextFont {
                                        font: font.clone(),
                                        font_size: FontSize::Px(11.0),
                                        ..default()
                                    },
                                    TextColor(Color::srgb(0.72, 0.75, 0.7)),
                                ));
                                cell.spawn((
                                    Text::new(format!(
                                        "R {}  HDG {:03.0}",
                                        fmt_range(inputs.range_m),
                                        scene.heading_rad.to_degrees().rem_euclid(360.0)
                                    )),
                                    TextFont {
                                        font: font.clone(),
                                        font_size: FontSize::Px(10.0),
                                        ..default()
                                    },
                                    TextColor(Color::srgb(0.5, 0.53, 0.5)),
                                ));
                            });
                    }
                });
            }
        });
}

#[derive(Resource)]
struct CaptureTarget(Handle<Image>);

#[derive(Resource, Default)]
struct CaptureState {
    frames: u32,
    captured: bool,
    tail: u32,
}

fn drive_capture(
    mut state: ResMut<CaptureState>,
    target: Res<CaptureTarget>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    if state.captured {
        state.tail += 1;
        if state.tail >= TAIL_FRAMES {
            println!("nav preview written to {OUT_PATH}");
            exit.write(AppExit::Success);
        }
        return;
    }
    state.frames += 1;
    if state.frames < WARMUP_FRAMES {
        return;
    }
    commands
        .spawn(Screenshot::image(target.0.clone()))
        .observe(save_to_disk(OUT_PATH));
    state.captured = true;
}
