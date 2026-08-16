//! PFD-style HUD overlay — the navigation UI's "HUD mode".
//!
//! An aircraft primary-flight-display drawn over the centre of the ship
//! view, replacing the navball cluster when active: a roll-rotating pitch
//! ladder with heading ticks riding the horizon line, the navball's
//! direction markers (prograde, retrograde, normal, radial, target,
//! maneuver) projected into the view, a boresight at the craft's nose, a
//! speed tape (left, active velocity frame, click to cycle) and an
//! altitude tape (right, SEA/GND datum, click to toggle) with vertical
//! speed, throttle, and heading readouts, plus FBW / A.PROT flight-assist
//! annunciators.
//!
//! **Approach guidance** (localizer + glideslope deviation scales and a
//! two-axis flight director) appears whenever the runtime's `route` module has an armed
//! approach, and is driven entirely by that module's published guidance — the
//! PFD never computes a deviation of its own, so the needle the pilot follows
//! and the route the ND draws cannot disagree.
//!
//! **Projection model:** the PFD is the view from the craft's nose,
//! independent of the actual orbit camera. Pitch/bank/heading come from
//! the craft attitude expressed in the local ENU frame at the craft
//! (same construction as the navball, [`attitude_angles`]); markers are
//! placed by their body-frame angular offsets from the nose at
//! [`PX_PER_DEG`] pixels per degree, the same scale the ladder uses, so
//! ladder and markers agree.
//!
//! **Mode switching:** [`NavDisplayMode`] (sole writer:
//! [`handle_mode_clicks`]) selects BALL (classic navball) or HUD (this
//! overlay) via a small selector in the top-left row.
//! [`sync_visibility`] applies it every frame with diff-writes so it
//! coexists with the photo-mode / shipyard-editor visibility writers
//! (which also flip these roots; values converge within a frame).

use bevy::math::{DQuat, DVec3};
use bevy::prelude::*;
use bevy::ui::Val2;
use thalos_physics_canonical::velocity_frame::{VelocityReferenceFrame, nav_basis};

use crate::HudPanel;
use crate::format;
use crate::navball::markers::{
    MarkerIconState, MarkerKind, MarkerVariants, compute_marker_directions, marker_icon_image,
    orientation_icon_image,
};
use crate::navball::ui::NavballFrameRoot;
use crate::theme::{HudTheme, panel_frame, panel_node};
use crate::velocity_frame::VelocityFrameState;
use thalos_game_state::flight::RealizedControl;
use thalos_game_state::flight::ThrottleState;
use thalos_game_state::maneuver_plan::ManeuverPlan;
use thalos_game_state::nav::TargetBody;
use thalos_game_state::units::UnitDomain;
use thalos_game_state::{SimulationState, SolarSystemState};

use super::TopLeftRowAnchor;
use super::flight_panel::VelocityPanel;
use super::orbital_panel::{AltitudeDatum, AltitudeDisplay, AltitudePanel};

// ---------------------------------------------------------------------------
// Geometry + style constants
// ---------------------------------------------------------------------------

/// Angular scale of the whole display: ladder rungs, heading ticks, and
/// marker projection all share it so the overlay stays self-consistent.
const PX_PER_DEG: f32 = 7.0;

/// Rungs farther than this from the current pitch hide (no UI clipping —
/// rotated-node clip rects are unreliable, so the ladder culls itself).
const LADDER_HALF_RANGE_DEG: f32 = 26.0;

const RUNG_BAR_LEN: f32 = 64.0;
const RUNG_GAP_HALF: f32 = 46.0;
const RUNG_MINOR_BAR_LEN: f32 = 24.0;
const RUNG_MINOR_GAP_HALF: f32 = 52.0;
const HORIZON_BAR_LEN: f32 = 250.0;
const HORIZON_GAP_HALF: f32 = 60.0;

const MARKER_ICON_SIZE: u32 = 30;
/// Markers clamp to this radius from the boresight; a clamped (or
/// behind-the-nose) marker shows the dimmed icon variant, mirroring the
/// navball's back-hemisphere treatment.
const MARKER_CLAMP_PX: f32 = 195.0;

/// Distance from screen centre to each tape's inner edge, and tape size.
const TAPE_INNER_X: f32 = 320.0;
const TAPE_W: f32 = 84.0;
const TAPE_H: f32 = 320.0;
/// Vertical pixels per tape tick step (the step itself adapts, see
/// [`nice_step`]).
const TAPE_PX_PER_STEP: f32 = 34.0;
const TAPE_TICK_SLOTS: i32 = 9;
const HEADING_TICK_SLOTS: i32 = 4;

/// The vertical-speed tape: a narrower, shorter column outboard of the
/// altitude tape.
const VS_TAPE_W: f32 = 56.0;
const VS_TAPE_H: f32 = 240.0;
const VS_TAPE_GAP: f32 = 10.0;
const VS_TICK_SLOTS: i32 = 7;

// Warm-amber HUD palette. The PFD used to render in a bright, saturated
// green phosphor — that read as too bright/obnoxious and clashed with the
// rest of the HUD's warm-amber accent identity (`HudTheme::text_accent`).
// These colours are amber, dimmer, and a touch more translucent so the
// overlay reads as a thin, softly-glowing real HUD instead.
const HUD_AMBER: Color = Color::srgba(0.99, 0.75, 0.37, 0.85);
const HUD_AMBER_DIM: Color = Color::srgba(0.96, 0.69, 0.32, 0.42);
const HUD_TAPE_BG: Color = Color::srgba(0.05, 0.035, 0.012, 0.20);
const HUD_BOX_BG: Color = Color::srgba(0.07, 0.048, 0.016, 0.66);
/// Zero-offset blurred halo behind the thin lines and readout boxes — the
/// "slight bloom". Bevy composites the UI pass *after* the camera's bloom
/// node (`EndMainPass → Bloom → Tonemapping → … → UiPass`), so HDR UI
/// colours never reach that bloom; a box-shadow glow is the closest a real
/// HUD bloom the UI layer can produce. [`HUD_GLOW_SOFT`] is the fainter
/// variant for the larger tape frames.
const HUD_GLOW: Color = Color::srgba(1.0, 0.73, 0.32, 0.5);
const HUD_GLOW_SOFT: Color = Color::srgba(1.0, 0.73, 0.32, 0.22);

// ---------------------------------------------------------------------------
// Mode state + components
// ---------------------------------------------------------------------------

/// Which navigation display is active. **Sole writer:**
/// [`handle_mode_clicks`]. Reflect-registered (for a future debug UI).
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Resource)]
pub enum NavDisplayMode {
    /// Classic navball cluster (ball + velocity readout + throttle arc).
    #[default]
    Ball,
    /// The PFD overlay in this module.
    Hud,
}

#[derive(Component, Clone, Copy)]
pub(super) struct NavDisplayButton {
    pub target: NavDisplayMode,
}

/// Root of the whole PFD overlay (fullscreen wrapper).
#[derive(Component)]
pub(super) struct PfdRoot;

/// The roll-rotated ladder container, centred on the boresight.
#[derive(Component)]
pub(super) struct PfdLadder;

/// Child of [`PfdLadder`], translated vertically by pitch each frame.
#[derive(Component)]
pub(super) struct PfdPitchShift;

#[derive(Component)]
pub(super) struct PfdRung {
    pitch_deg: f32,
}

/// Heading label riding the horizon rung; `slot` is the offset (in 10°
/// steps) from the rounded current heading.
#[derive(Component)]
pub(super) struct PfdHeadingTick {
    slot: i32,
}

/// A navball direction marker projected onto the PFD.
#[derive(Component)]
pub(super) struct PfdMarker {
    kind: MarkerKind,
}

/// Marker for the PFD speed tape root, so [`sync_visibility`] can tell it
/// apart from the classic velocity panel (both carry [`VelocityPanel`] for
/// the shared click-to-cycle handler).
#[derive(Component)]
pub(super) struct PfdSpeedTape;

#[derive(Component)]
pub(super) struct PfdSpeedTick {
    slot: i32,
}

#[derive(Component)]
pub(super) struct PfdAltTick {
    slot: i32,
}

#[derive(Component)]
pub(super) struct PfdVsTick {
    slot: i32,
}

/// Single-line text readouts, all updated through one query.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(super) enum PfdReadout {
    SpeedValue,
    AltValue,
    SpeedFrame,
    AltDatum,
    VerticalSpeed,
    /// Unit suffix under the `V/S` label, filled from the resolved aviation
    /// unit (the V/S box itself carries no unit).
    VerticalSpeedUnit,
    Throttle,
    Heading,
}

#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(super) enum PfdAnnunciator {
    Fbw,
    AlphaProt,
}

// ---------------------------------------------------------------------------
// Approach guidance (localizer / glideslope / flight director)
// ---------------------------------------------------------------------------

/// Half-length of each deviation scale (px): the distance from centre to a
/// full-scale deflection.
const DEV_SCALE_HALF_PX: f32 = 96.0;
/// Localizer scale centre, below the pitch ladder.
const LOC_SCALE_Y: f32 = 216.0;
/// Glideslope scale centre-x, inboard of the altitude tape (`TAPE_INNER_X`).
const GS_SCALE_X: f32 = 246.0;
/// Dot radius on the deviation scales (px).
const DEV_DOT_PX: f32 = 5.0;
/// Deviation index (diamond) size (px).
const DEV_INDEX_PX: f32 = 13.0;
/// Flight-director bar length / thickness (px).
const FD_BAR_LEN_PX: f32 = 150.0;
const FD_BAR_THICK_PX: f32 = 3.0;
/// Flight-director travel limit (px) — the cue saturates rather than flying off
/// the display.
const FD_LIMIT_PX: f32 = 110.0;
/// Roll-cue scale: screen px per degree of bank error.
const FD_PX_PER_BANK_DEG: f32 = 4.0;

/// Root of the localizer (horizontal) deviation scale.
#[derive(Component)]
pub(super) struct PfdLocScale;

/// Root of the glideslope (vertical) deviation scale.
#[derive(Component)]
pub(super) struct PfdGsScale;

/// The moving localizer index.
#[derive(Component)]
pub(super) struct PfdLocIndex;

/// The moving glideslope index.
#[derive(Component)]
pub(super) struct PfdGsIndex;

/// The flight-director roll cue (a vertical bar you centre by rolling).
#[derive(Component)]
pub(super) struct PfdDirectorRoll;

/// The flight-director pitch cue (a horizontal bar you fly to).
#[derive(Component)]
pub(super) struct PfdDirectorPitch;

/// The approach annunciation text (`APPR RWY 03 · 12.4 km`).
#[derive(Component)]
pub(super) struct PfdApproachLabel;

/// Horizontal (localizer) index offset in px for a deflection in `[-1, 1]`.
///
/// **The index moves toward where the course is**, which is the universal
/// "fly toward the needle" convention: being right of course (positive
/// deflection) puts the course to your *left*, so the index sits left of centre
/// and you steer to it.
fn loc_index_offset_px(deflection: f64) -> f32 {
    -(deflection.clamp(-1.0, 1.0) as f32) * DEV_SCALE_HALF_PX
}

/// Vertical (glideslope) index offset in **screen** px for a deflection in
/// `[-1, 1]`.
///
/// Same "fly toward the needle" rule, but the two axes end up with *opposite*
/// signs and that is not a mistake: being high (positive deflection) puts the
/// glideslope **below** you, and screen `y` grows downward, so the index takes a
/// *positive* offset. Negating this instead — the natural thing to do if you
/// copy the lateral case — yields a perfectly plausible instrument that tells a
/// high aircraft to descend by pointing up. Pinned by
/// `deviation_indices_point_toward_the_course_and_the_slope`.
fn gs_index_offset_px(deflection: f64) -> f32 {
    (deflection.clamp(-1.0, 1.0) as f32) * DEV_SCALE_HALF_PX
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

pub fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>, theme: Res<HudTheme>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            PfdRoot,
            HudPanel,
            Visibility::Hidden,
            Name::new("HudPfdRoot"),
        ))
        .with_children(|root| {
            // Zero-size anchor at exact screen centre; every PFD element is
            // an absolute-positioned child with offsets from this point.
            root.spawn((Node::default(), Name::new("PfdAnchor")))
                .with_children(|anchor| {
                    spawn_ladder(anchor, &theme);
                    spawn_markers(anchor, &mut images);
                    spawn_boresight(anchor, &mut images);
                    spawn_speed_tape(anchor, &theme);
                    spawn_alt_tape(anchor, &theme);
                    spawn_vs_tape(anchor, &theme);
                    spawn_heading_readout(anchor, &theme);
                    spawn_annunciators(anchor, &theme);
                    spawn_deviation_scales(anchor, &theme);
                    spawn_flight_director(anchor);
                });
        });
}

fn spawn_ladder(anchor: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    anchor
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(0.0),
                ..default()
            },
            UiTransform {
                translation: Val2::ZERO,
                scale: Vec2::ONE,
                rotation: Rot2::IDENTITY,
            },
            PfdLadder,
            Name::new("PfdLadder"),
        ))
        .with_children(|ladder| {
            ladder
                .spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Px(0.0),
                        top: Val::Px(0.0),
                        ..default()
                    },
                    PfdPitchShift,
                    Name::new("PfdPitchShift"),
                ))
                .with_children(|shift| {
                    spawn_horizon(shift, theme);
                    for pitch_deg in (-85..=85).step_by(5) {
                        if pitch_deg == 0 {
                            continue;
                        }
                        spawn_rung(shift, theme, pitch_deg);
                    }
                });
        });
}

fn spawn_horizon(shift: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    shift
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(0.0),
                ..default()
            },
            PfdRung { pitch_deg: 0.0 },
            Visibility::Inherited,
            Name::new("PfdHorizon"),
        ))
        .with_children(|h| {
            spawn_bar(
                h,
                -(HORIZON_GAP_HALF + HORIZON_BAR_LEN),
                HORIZON_BAR_LEN,
                1.5,
                HUD_AMBER,
                true,
            );
            spawn_bar(h, HORIZON_GAP_HALF, HORIZON_BAR_LEN, 1.5, HUD_AMBER, true);
            // Heading labels below the line; left offsets driven per frame.
            for slot in -HEADING_TICK_SLOTS..=HEADING_TICK_SLOTS {
                h.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Px(0.0),
                        top: Val::Px(7.0),
                        ..default()
                    },
                    Text::new(""),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(12.0),
                        ..default()
                    },
                    TextColor(HUD_AMBER_DIM),
                    PfdHeadingTick { slot },
                    Visibility::Hidden,
                    Name::new(format!("PfdHeadingTick_{slot}")),
                ));
            }
        });
}

fn spawn_rung(shift: &mut ChildSpawnerCommands<'_>, theme: &HudTheme, pitch_deg: i32) {
    let major = pitch_deg % 10 == 0;
    let (bar_len, gap_half, thickness) = if major {
        (RUNG_BAR_LEN, RUNG_GAP_HALF, 1.5)
    } else {
        (RUNG_MINOR_BAR_LEN, RUNG_MINOR_GAP_HALF, 1.0)
    };
    // Sky rungs solid green, ground rungs dimmed (stand-in for the
    // conventional dashed negative rungs).
    let color = if pitch_deg >= 0 {
        HUD_AMBER
    } else {
        HUD_AMBER_DIM
    };

    shift
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(-(pitch_deg as f32) * PX_PER_DEG),
                ..default()
            },
            PfdRung {
                pitch_deg: pitch_deg as f32,
            },
            Visibility::Inherited,
            Name::new(format!("PfdRung_{pitch_deg}")),
        ))
        .with_children(|r| {
            spawn_bar(r, -(gap_half + bar_len), bar_len, thickness, color, major);
            spawn_bar(r, gap_half, bar_len, thickness, color, major);
            if major {
                for x in [-(gap_half + bar_len + 34.0), gap_half + bar_len + 8.0] {
                    r.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(x),
                            top: Val::Px(-8.0),
                            ..default()
                        },
                        Text::new(format!("{pitch_deg}")),
                        TextFont {
                            font: theme.font.clone(),
                            font_size: FontSize::Px(11.0),
                            ..default()
                        },
                        TextColor(color),
                    ));
                }
            }
        });
}

/// A soft, zero-offset blurred halo behind a node — the HUD's "slight
/// bloom". `blur` is the halo spread in logical px. See [`HUD_GLOW`] for
/// why the glow is faked with a box shadow rather than the camera bloom.
fn hud_glow(color: Color, blur: f32) -> BoxShadow {
    BoxShadow(vec![ShadowStyle {
        color,
        x_offset: Val::Px(0.0),
        y_offset: Val::Px(0.0),
        spread_radius: Val::Px(0.0),
        blur_radius: Val::Px(blur),
    }])
}

fn spawn_bar(
    parent: &mut ChildSpawnerCommands<'_>,
    x: f32,
    len: f32,
    thickness: f32,
    color: Color,
    glow: bool,
) {
    let mut bar = parent.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(x),
            top: Val::Px(-thickness * 0.5),
            width: Val::Px(len),
            height: Val::Px(thickness),
            ..default()
        },
        BackgroundColor(color),
    ));
    // Only the prominent, persistent lines (horizon + major rungs) glow;
    // the minor rungs stay crisp so the ladder doesn't smear into a haze.
    if glow {
        bar.insert(hud_glow(HUD_GLOW, 5.0));
    }
}

fn spawn_markers(anchor: &mut ChildSpawnerCommands<'_>, images: &mut Assets<Image>) {
    let half = MARKER_ICON_SIZE as f32 * 0.5;
    for kind in MarkerKind::ALL {
        let variants = MarkerVariants {
            front: images.add(marker_icon_image(
                kind,
                MARKER_ICON_SIZE,
                MarkerIconState::Visible,
            )),
            back: images.add(marker_icon_image(
                kind,
                MARKER_ICON_SIZE,
                MarkerIconState::Occluded,
            )),
        };
        anchor.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(-half),
                top: Val::Px(-half),
                width: Val::Px(MARKER_ICON_SIZE as f32),
                height: Val::Px(MARKER_ICON_SIZE as f32),
                ..default()
            },
            ImageNode::new(variants.front.clone()),
            PfdMarker { kind },
            variants,
            Visibility::Hidden,
            ZIndex(3),
            Name::new(format!("PfdMarker_{kind:?}")),
        ));
    }
}

fn spawn_boresight(anchor: &mut ChildSpawnerCommands<'_>, images: &mut Assets<Image>) {
    let image = images.add(orientation_icon_image(40, 16));
    anchor.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(-20.0),
            top: Val::Px(-8.0),
            width: Val::Px(40.0),
            height: Val::Px(16.0),
            ..default()
        },
        ImageNode::new(image),
        ZIndex(4),
        Name::new("PfdBoresight"),
    ));
}

fn spawn_speed_tape(anchor: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    let left = -(TAPE_INNER_X + TAPE_W);
    anchor
        .spawn((
            Button,
            tape_root_node(left),
            BackgroundColor(HUD_TAPE_BG),
            BorderColor::all(HUD_AMBER_DIM),
            Interaction::None,
            // Shared with the classic velocity readout: clicking cycles the
            // speed reference frame via `flight_panel::handle_velocity_frame_click`.
            VelocityPanel,
            PfdSpeedTape,
            Name::new("PfdSpeedTape"),
        ))
        .with_children(|tape| {
            for slot in -TAPE_TICK_SLOTS..=TAPE_TICK_SLOTS {
                tape.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        right: Val::Px(18.0),
                        top: Val::Px(0.0),
                        ..default()
                    },
                    Text::new(""),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(12.0),
                        ..default()
                    },
                    TextColor(HUD_AMBER),
                    PfdSpeedTick { slot },
                    Visibility::Hidden,
                ))
                .with_children(|t| {
                    t.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            right: Val::Px(-16.0),
                            top: Val::Px(6.0),
                            width: Val::Px(12.0),
                            height: Val::Px(2.0),
                            ..default()
                        },
                        BackgroundColor(HUD_AMBER),
                    ));
                });
            }
            spawn_value_box(tape, theme, PfdReadout::SpeedValue, TAPE_H, 26.0, 15.0);
        });

    spawn_tape_labels(
        anchor,
        theme,
        left,
        &[
            (PfdReadout::SpeedFrame, HUD_AMBER),
            (PfdReadout::Throttle, HUD_AMBER_DIM),
        ],
        "PfdSpeedLabels",
    );
}

fn spawn_alt_tape(anchor: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    let left = TAPE_INNER_X;
    anchor
        .spawn((
            Button,
            tape_root_node(left),
            BackgroundColor(HUD_TAPE_BG),
            BorderColor::all(HUD_AMBER_DIM),
            Interaction::None,
            // Shared with the top altitude panel: clicking toggles SEA/GND
            // via `orbital_panel::handle_click`.
            AltitudePanel,
            Name::new("PfdAltTape"),
        ))
        .with_children(|tape| {
            for slot in -TAPE_TICK_SLOTS..=TAPE_TICK_SLOTS {
                tape.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Px(18.0),
                        top: Val::Px(0.0),
                        ..default()
                    },
                    Text::new(""),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(12.0),
                        ..default()
                    },
                    TextColor(HUD_AMBER),
                    PfdAltTick { slot },
                    Visibility::Hidden,
                ))
                .with_children(|t| {
                    t.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(-16.0),
                            top: Val::Px(6.0),
                            width: Val::Px(12.0),
                            height: Val::Px(2.0),
                            ..default()
                        },
                        BackgroundColor(HUD_AMBER),
                    ));
                });
            }
            spawn_value_box(tape, theme, PfdReadout::AltValue, TAPE_H, 26.0, 15.0);
        });

    spawn_tape_labels(
        anchor,
        theme,
        left,
        &[(PfdReadout::AltDatum, HUD_AMBER)],
        "PfdAltLabels",
    );
}

/// Vertical-speed tape: same scrolling-tape mechanics as speed/altitude,
/// narrower and shorter, signed values, with a static V/S label below.
fn spawn_vs_tape(anchor: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    let left = TAPE_INNER_X + TAPE_W + VS_TAPE_GAP;
    anchor
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(left),
                top: Val::Px(-VS_TAPE_H * 0.5),
                width: Val::Px(VS_TAPE_W),
                height: Val::Px(VS_TAPE_H),
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BackgroundColor(HUD_TAPE_BG),
            BorderColor::all(HUD_AMBER_DIM),
            hud_glow(HUD_GLOW_SOFT, 7.0),
            Name::new("PfdVsTape"),
        ))
        .with_children(|tape| {
            for slot in -VS_TICK_SLOTS..=VS_TICK_SLOTS {
                tape.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Px(14.0),
                        top: Val::Px(0.0),
                        ..default()
                    },
                    Text::new(""),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(11.0),
                        ..default()
                    },
                    TextColor(HUD_AMBER),
                    PfdVsTick { slot },
                    Visibility::Hidden,
                ))
                .with_children(|t| {
                    t.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(-12.0),
                            top: Val::Px(6.0),
                            width: Val::Px(8.0),
                            height: Val::Px(2.0),
                            ..default()
                        },
                        BackgroundColor(HUD_AMBER),
                    ));
                });
            }
            spawn_value_box(
                tape,
                theme,
                PfdReadout::VerticalSpeed,
                VS_TAPE_H,
                22.0,
                13.0,
            );
        });

    anchor
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(left),
                top: Val::Px(VS_TAPE_H * 0.5 + 8.0),
                width: Val::Px(VS_TAPE_W),
                // Two stacked lines: the fixed "V/S" caption and, under it, the
                // unit the tape is currently in. Stacked rather than inline
                // because "V/S ft/min" overflows VS_TAPE_W at this font size.
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                ..default()
            },
            Name::new("PfdVsLabel"),
        ))
        .with_children(|c| {
            c.spawn((
                Text::new("V/S"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(HUD_AMBER_DIM),
            ));
            c.spawn((
                Text::new("—"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(10.0),
                    ..default()
                },
                TextColor(HUD_AMBER_DIM),
                PfdReadout::VerticalSpeedUnit,
            ));
        });
}

fn tape_root_node(left: f32) -> impl Bundle {
    (
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(left),
            top: Val::Px(-TAPE_H * 0.5),
            width: Val::Px(TAPE_W),
            height: Val::Px(TAPE_H),
            border: UiRect::all(Val::Px(1.0)),
            ..default()
        },
        hud_glow(HUD_GLOW_SOFT, 7.0),
    )
}

fn spawn_value_box(
    tape: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    readout: PfdReadout,
    tape_h: f32,
    box_h: f32,
    font_size: f32,
) {
    tape.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(3.0),
            right: Val::Px(3.0),
            top: Val::Px((tape_h - box_h) * 0.5),
            height: Val::Px(box_h),
            border: UiRect::all(Val::Px(1.0)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(HUD_BOX_BG),
        BorderColor::all(HUD_AMBER),
        hud_glow(HUD_GLOW, 4.0),
        ZIndex(2),
    ))
    .with_children(|b| {
        b.spawn((
            Text::new("—"),
            TextFont {
                font: theme.font.clone(),
                font_size: FontSize::Px(font_size),
                ..default()
            },
            TextColor(HUD_AMBER),
            readout,
        ));
    });
}

fn spawn_tape_labels(
    anchor: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    left: f32,
    rows: &[(PfdReadout, Color)],
    name: &str,
) {
    anchor
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(left),
                top: Val::Px(TAPE_H * 0.5 + 8.0),
                width: Val::Px(TAPE_W),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                row_gap: Val::Px(2.0),
                ..default()
            },
            Name::new(name.to_string()),
        ))
        .with_children(|c| {
            for &(readout, color) in rows {
                c.spawn((
                    Text::new("—"),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(12.0),
                        ..default()
                    },
                    TextColor(color),
                    readout,
                ));
            }
        });
}

fn spawn_heading_readout(anchor: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    anchor
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(-60.0),
                top: Val::Px(TAPE_H * 0.5 + 40.0),
                width: Val::Px(120.0),
                justify_content: JustifyContent::Center,
                ..default()
            },
            Name::new("PfdHeading"),
        ))
        .with_children(|c| {
            c.spawn((
                Text::new("HDG —"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(HUD_AMBER),
                PfdReadout::Heading,
            ));
        });
}

fn spawn_annunciators(anchor: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    let rows: [(PfdAnnunciator, &str, Color, f32); 2] = [
        (PfdAnnunciator::Fbw, "FBW", HUD_AMBER, -150.0),
        (PfdAnnunciator::AlphaProt, "A.PROT", theme.text_warn, 60.0),
    ];
    for (kind, label, color, x) in rows {
        anchor.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(x),
                top: Val::Px(-(TAPE_H * 0.5 + 36.0)),
                ..default()
            },
            Text::new(label),
            TextFont {
                font: theme.font.clone(),
                font_size: FontSize::Px(14.0),
                ..default()
            },
            TextColor(color),
            kind,
            Visibility::Hidden,
            Name::new(format!("PfdAnnunciator_{label}")),
        ));
    }
}

// ---------------------------------------------------------------------------
// Mode toggle panel (top-left row, next to the SHIP/MAP selector)
// ---------------------------------------------------------------------------

const TOGGLE_BUTTON_WIDTH: f32 = 50.0;
const TOGGLE_BUTTON_HEIGHT: f32 = 28.0;

pub fn setup_toggle(mut commands: Commands, theme: Res<HudTheme>, anchor: Res<TopLeftRowAnchor>) {
    let mut root = panel_node();
    root.position_type = PositionType::Relative;
    root.padding = UiRect::axes(Val::Px(8.0), Val::Px(6.0));
    root.row_gap = Val::Px(4.0);
    let (bg, border) = panel_frame(&theme);

    commands.entity(anchor.0).with_children(|row_parent| {
        row_parent
            .spawn((root, bg, border, HudPanel, Name::new("HudNavDisplayMode")))
            .with_children(|p| {
                p.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(6.0),
                    ..default()
                })
                .with_children(|row| {
                    spawn_toggle_button(row, &theme, NavDisplayMode::Ball, "BALL");
                    spawn_toggle_button(row, &theme, NavDisplayMode::Hud, "HUD");
                });
                p.spawn((
                    Text::new("NAV"),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(11.0),
                        ..default()
                    },
                    TextColor(theme.text_subtitle),
                ));
            });
    });
}

fn spawn_toggle_button(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    target: NavDisplayMode,
    label: &str,
) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Px(TOGGLE_BUTTON_WIDTH),
                height: Val::Px(TOGGLE_BUTTON_HEIGHT),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            NavDisplayButton { target },
            Name::new(format!("NavDisplay_{label}")),
        ))
        .with_children(|c| {
            c.spawn((
                Text::new(label),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(13.0),
                    ..default()
                },
                TextColor(theme.text_dim),
            ));
        });
}

pub fn handle_mode_clicks(
    interactions: Query<(&Interaction, &NavDisplayButton), Changed<Interaction>>,
    mut mode: ResMut<NavDisplayMode>,
) {
    for (interaction, button) in &interactions {
        if matches!(interaction, Interaction::Pressed) && *mode != button.target {
            *mode = button.target;
        }
    }
}

pub fn update_mode_button_visuals(
    mode: Res<NavDisplayMode>,
    theme: Res<HudTheme>,
    mut buttons: Query<(
        &NavDisplayButton,
        &Interaction,
        &mut BorderColor,
        &mut BackgroundColor,
        &Children,
    )>,
    mut text_q: Query<&mut TextColor>,
) {
    for (button, interaction, mut border, mut bg, children) in &mut buttons {
        let active = *mode == button.target;
        let (border_color, bg_color) = match (active, interaction) {
            (true, _) => (theme.text_accent, theme.panel_bg),
            (false, Interaction::Pressed) => (theme.text_primary, theme.panel_border),
            (false, Interaction::Hovered) => (theme.text_primary, theme.panel_bg),
            (false, Interaction::None) => (theme.panel_border, theme.panel_bg),
        };
        let new_border = BorderColor::all(border_color);
        if border.top != new_border.top {
            *border = new_border;
        }
        if bg.0 != bg_color {
            bg.0 = bg_color;
        }
        let label_color = if active {
            theme.text_accent
        } else {
            theme.text_dim
        };
        if let Some(&child) = children.first()
            && let Ok(mut tc) = text_q.get_mut(child)
            && tc.0 != label_color
        {
            tc.0 = label_color;
        }
    }
}

/// Swap the navball cluster and the PFD according to [`NavDisplayMode`].
///
/// Runs every frame with diff-writes (not change-gated) because photo mode
/// and the shipyard editor also write these roots' `Visibility` on their own
/// transitions; re-asserting converges within a frame without ordering
/// constraints against those private systems.
pub fn sync_visibility(
    mode: Res<NavDisplayMode>,
    mut queries: ParamSet<(
        Query<&mut Visibility, With<PfdRoot>>,
        Query<&mut Visibility, With<NavballFrameRoot>>,
        Query<&mut Visibility, (With<VelocityPanel>, Without<PfdSpeedTape>)>,
    )>,
) {
    let hud_on = *mode == NavDisplayMode::Hud;
    let pfd_target = if hud_on {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    let ball_target = if hud_on {
        Visibility::Hidden
    } else {
        Visibility::Inherited
    };
    for mut vis in queries.p0().iter_mut() {
        if *vis != pfd_target {
            *vis = pfd_target;
        }
    }
    for mut vis in queries.p1().iter_mut() {
        if *vis != ball_target {
            *vis = ball_target;
        }
    }
    for mut vis in queries.p2().iter_mut() {
        if *vis != ball_target {
            *vis = ball_target;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame display updates
// ---------------------------------------------------------------------------

/// Body axes, matching the navball's conventions (`navball::attitude`).
const BODY_NOSE: DVec3 = DVec3::Y;
const BODY_RIGHT: DVec3 = DVec3::X;
const BODY_DORSAL: DVec3 = DVec3::Z;

#[allow(clippy::too_many_arguments)]
pub fn update_attitude_display(
    mode: Res<NavDisplayMode>,
    sim_state: Res<SimulationState>,
    solar_system: Res<SolarSystemState>,
    velocity_frame: Res<VelocityFrameState>,
    target: Res<TargetBody>,
    plan: thalos_game_state::ActiveCraftRef<ManeuverPlan>,
    mut ladder_q: Query<&mut UiTransform, With<PfdLadder>>,
    mut shift_q: Query<&mut Node, With<PfdPitchShift>>,
    mut rung_q: Query<(&PfdRung, &mut Visibility), Without<PfdHeadingTick>>,
    mut hdg_tick_q: Query<
        (&PfdHeadingTick, &mut Node, &mut Text, &mut Visibility),
        (
            Without<PfdPitchShift>,
            Without<PfdRung>,
            Without<PfdReadout>,
        ),
    >,
    mut marker_q: Query<
        (
            &PfdMarker,
            &MarkerVariants,
            &mut ImageNode,
            &mut Node,
            &mut Visibility,
        ),
        (
            Without<PfdPitchShift>,
            Without<PfdHeadingTick>,
            Without<PfdRung>,
        ),
    >,
    mut readout_q: Query<(&PfdReadout, &mut Text), Without<PfdHeadingTick>>,
) {
    let Some(plan) = plan.get() else {
        return;
    };
    if *mode != NavDisplayMode::Hud {
        return;
    }

    let sim = &sim_state.simulation;
    let craft = sim.craft_state();
    let q_body_to_world = craft.attitude.orientation;
    let craft_pos = craft.translation.position;

    let soi_body_id = sim.dominant_body();
    let Some(states) = solar_system.states.as_deref() else {
        return;
    };
    let Some(body_state) = states.get(soi_body_id) else {
        return;
    };
    let Some(angles) = attitude_angles(q_body_to_world, craft_pos, body_state.position) else {
        return;
    };

    let pitch_deg = angles.pitch_rad.to_degrees();
    let heading_deg = angles.heading_rad.to_degrees().rem_euclid(360.0);

    // Ladder: roll about the boresight, then shift along the rolled
    // vertical by pitch. Positive bank (right wing down) tilts the horizon
    // right-end-up, which in UI rotation terms (positive = clockwise) is a
    // negative angle.
    if let Ok(mut transform) = ladder_q.single_mut() {
        transform.rotation = Rot2::radians(-(angles.bank_rad as f32));
    }
    if let Ok(mut node) = shift_q.single_mut() {
        node.top = Val::Px(pitch_deg as f32 * PX_PER_DEG);
    }

    for (rung, mut vis) in &mut rung_q {
        let target_vis = if (pitch_deg as f32 - rung.pitch_deg).abs() <= LADDER_HALF_RANGE_DEG {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *vis != target_vis {
            *vis = target_vis;
        }
    }

    // Heading labels ride the horizon line, one per 10°.
    let base = (heading_deg / 10.0).round() as i32;
    for (tick, mut node, mut text, mut vis) in &mut hdg_tick_q {
        let tick_heading = (base + tick.slot) * 10;
        let dx = wrap_deg(tick_heading as f64 - heading_deg) as f32 * PX_PER_DEG;
        node.left = Val::Px(dx - 7.0);
        let label = format!("{:02}", (tick_heading.rem_euclid(360)) / 10);
        if text.0 != label {
            text.0 = label;
        }
        let target_vis = if dx.abs() < HORIZON_GAP_HALF + HORIZON_BAR_LEN {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *vis != target_vis {
            *vis = target_vis;
        }
    }

    // Direction markers, projected by body-frame angles from the nose.
    let nose = q_body_to_world * BODY_NOSE;
    let right = q_body_to_world * BODY_RIGHT;
    let dorsal = q_body_to_world * BODY_DORSAL;
    let directions = compute_marker_directions(
        velocity_frame.active,
        &sim_state,
        &solar_system,
        &target,
        plan,
    );
    let icon_half = MARKER_ICON_SIZE as f32 * 0.5;
    for (marker, variants, mut image, mut node, mut vis) in &mut marker_q {
        let Some(d_world) = directions.for_kind(marker.kind) else {
            *vis = Visibility::Hidden;
            continue;
        };
        let forward = d_world.dot(nose);
        let az = d_world.dot(right).atan2(forward);
        let el = d_world.dot(dorsal).atan2(d_world.dot(right).hypot(forward));
        let mut x = az.to_degrees() as f32 * PX_PER_DEG;
        let mut y = -el.to_degrees() as f32 * PX_PER_DEG;
        let dist = x.hypot(y);
        let clamped = dist > MARKER_CLAMP_PX;
        if clamped {
            let scale = MARKER_CLAMP_PX / dist;
            x *= scale;
            y *= scale;
        }
        node.left = Val::Px(x - icon_half);
        node.top = Val::Px(y - icon_half);
        let target_handle = if clamped || forward < 0.0 {
            &variants.back
        } else {
            &variants.front
        };
        if image.image != *target_handle {
            image.image = target_handle.clone();
        }
        if *vis != Visibility::Inherited {
            *vis = Visibility::Inherited;
        }
    }

    for (readout, mut text) in &mut readout_q {
        if *readout == PfdReadout::Heading {
            let s = format!("HDG {:03}", (heading_deg.round() as i32).rem_euclid(360));
            if text.0 != s {
                text.0 = s;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn update_tapes(
    mode: Res<NavDisplayMode>,
    sim_state: Res<SimulationState>,
    solar_system: Res<SolarSystemState>,
    velocity_frame: Res<VelocityFrameState>,
    target: Res<TargetBody>,
    throttle: Res<ThrottleState>,
    display: Res<AltitudeDisplay>,
    units: Res<thalos_game_state::units::UnitsSettings>,
    mut speed_ticks: Query<
        (&PfdSpeedTick, &mut Node, &mut Text, &mut Visibility),
        Without<PfdAltTick>,
    >,
    mut alt_ticks: Query<
        (&PfdAltTick, &mut Node, &mut Text, &mut Visibility),
        (Without<PfdSpeedTick>, Without<PfdVsTick>),
    >,
    mut vs_ticks: Query<
        (&PfdVsTick, &mut Node, &mut Text, &mut Visibility),
        (Without<PfdSpeedTick>, Without<PfdAltTick>),
    >,
    mut readout_q: Query<
        (&PfdReadout, &mut Text),
        (
            Without<PfdSpeedTick>,
            Without<PfdAltTick>,
            Without<PfdVsTick>,
        ),
    >,
) {
    if *mode != NavDisplayMode::Hud {
        return;
    }

    let sim = &sim_state.simulation;
    let ship = sim.ship_state();
    let body_id = sim.dominant_body();
    let body = &sim.bodies()[body_id];
    let Some(states) = solar_system.states.as_deref() else {
        return;
    };
    let Some(body_state) = states.get(body_id) else {
        return;
    };
    let target_state = target.target.and_then(|id| states.get(id));

    // Speed in the active frame (same source as the classic readout).
    let basis = nav_basis(velocity_frame.active, ship, body_state, target_state);
    let speed = basis.map(|b| b.speed);

    // Altitude: the resolved datum the top altitude panel published this
    // frame; fall back to raw ASL before its first update.
    let (datum, altitude) = display.resolved.unwrap_or_else(|| {
        let asl = (ship.position - body_state.position).length() - body.radius_m;
        (AltitudeDatum::Sea, asl)
    });

    // Vertical speed: surface-frame velocity projected on local up.
    let up = (ship.position - body_state.position).try_normalize();
    let vertical_speed = match (
        nav_basis(VelocityReferenceFrame::Surface, ship, body_state, None),
        up,
    ) {
        (Some(surf), Some(up)) => (ship.velocity - surf.reference_vel).dot(up),
        _ => 0.0,
    };

    // The PFD is an aviation instrument, so its tapes read feet and knots even
    // when the global preference is metric. Resolved once: every tick, readout,
    // and threshold below must agree on one unit or the tape lies.
    let system = units.system_for(UnitDomain::Aviation);

    // Tick columns. The tapes render in the active display unit so their ticks
    // match the readouts below (m/ft, m/s/kn, m/s / ft·min⁻¹).
    if let Some(speed) = speed.map(|v| format::speed_tape_value(v, system)) {
        let step = nice_step((speed * 0.04).max(4.0));
        for (tick, mut node, mut text, mut vis) in &mut speed_ticks {
            apply_tape_tick(
                speed, step, tick.slot, false, TAPE_H, &mut node, &mut text, &mut vis,
            );
        }
    } else {
        for (_, _, _, mut vis) in &mut speed_ticks {
            if *vis != Visibility::Hidden {
                *vis = Visibility::Hidden;
            }
        }
    }
    let altitude_disp = format::altitude_tape_value(altitude, system);
    let alt_step = nice_step((altitude_disp.abs() * 0.04).max(10.0));
    for (tick, mut node, mut text, mut vis) in &mut alt_ticks {
        apply_tape_tick(
            altitude_disp,
            alt_step,
            tick.slot,
            true,
            TAPE_H,
            &mut node,
            &mut text,
            &mut vis,
        );
    }
    let vs_disp = format::vertical_speed_value(vertical_speed, system);
    let vs_step = nice_step((vs_disp.abs() * 0.08).max(2.0));
    for (tick, mut node, mut text, mut vis) in &mut vs_ticks {
        apply_tape_tick(
            vs_disp, vs_step, tick.slot, true, VS_TAPE_H, &mut node, &mut text, &mut vis,
        );
    }

    // Text readouts.
    for (readout, mut text) in &mut readout_q {
        let s = match readout {
            PfdReadout::SpeedValue => match speed {
                Some(v) => format::speed(v, system),
                None => "—".to_string(),
            },
            PfdReadout::AltValue => format::altitude(altitude, system),
            PfdReadout::SpeedFrame => match velocity_frame.active {
                VelocityReferenceFrame::Orbit => "ORB".to_string(),
                VelocityReferenceFrame::Surface => "SRF".to_string(),
                VelocityReferenceFrame::Target => "TGT".to_string(),
            },
            PfdReadout::AltDatum => match datum {
                AltitudeDatum::Sea => "SEA".to_string(),
                AltitudeDatum::Ground => "GND".to_string(),
            },
            PfdReadout::VerticalSpeed => signed_speed(vertical_speed, system),
            // The V/S box shows a bare signed number, so the unit has to be
            // stated somewhere — and it now varies independently of the global
            // setting, which makes an unlabelled tape actively misleading.
            PfdReadout::VerticalSpeedUnit => format::vertical_speed_unit(system).to_string(),
            PfdReadout::Throttle => {
                format!("THR {:3.0}%", throttle.commanded.clamp(0.0, 1.0) * 100.0)
            }
            PfdReadout::Heading => continue, // owned by update_attitude_display
        };
        if text.0 != s {
            text.0 = s;
        }
    }
}

pub fn update_annunciators(
    mode: Res<NavDisplayMode>,
    realized: thalos_game_state::ActiveCraftRef<RealizedControl>,
    mut q: Query<(&PfdAnnunciator, &mut Visibility)>,
) {
    let Some(realized) = realized.get() else {
        return;
    };
    if *mode != NavDisplayMode::Hud {
        return;
    }
    for (kind, mut vis) in &mut q {
        let on = match kind {
            PfdAnnunciator::Fbw => realized.assist.fbw_active,
            PfdAnnunciator::AlphaProt => realized.assist.protection_active,
        };
        let target = if on {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *vis != target {
            *vis = target;
        }
    }
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

struct AttitudeAngles {
    pitch_rad: f64,
    bank_rad: f64,
    heading_rad: f64,
}

/// Craft attitude expressed as aviation pitch / bank / heading in the local
/// ENU frame at the craft (up = radial-out from the dominant body, north =
/// world-Y projected, matching `navball::attitude`).
///
/// Heading: 0 = north, 90° = east. Pitch: nose above horizon positive.
/// Bank: right wing down positive, `(−π, π]`. Returns `None` only when the
/// craft sits at the body centre. A vertical nose makes bank/heading
/// degenerate; bank falls back to 0 there (gimbal-lock pole).
fn attitude_angles(
    q_body_to_world: DQuat,
    craft_pos: DVec3,
    body_pos: DVec3,
) -> Option<AttitudeAngles> {
    let (up, north, east) = super::geo::local_enu_basis(craft_pos, body_pos)?;

    let nose = q_body_to_world * BODY_NOSE;
    let right = q_body_to_world * BODY_RIGHT;

    let heading_rad = nose.dot(east).atan2(nose.dot(north));
    let pitch_rad = nose.dot(up).clamp(-1.0, 1.0).asin();

    // Zero-roll right = nose × up; bank is the actual right vector's angle
    // from it, signed right-wing-down positive.
    let bank_rad = match nose.cross(up).try_normalize() {
        Some(r0) => {
            let u0 = r0.cross(nose);
            (-right.dot(u0)).atan2(right.dot(r0))
        }
        None => 0.0,
    };

    Some(AttitudeAngles {
        pitch_rad,
        bank_rad,
        heading_rad,
    })
}

/// Wrap a degree difference into `(−180, 180]`.
fn wrap_deg(deg: f64) -> f64 {
    let wrapped = (deg + 180.0).rem_euclid(360.0) - 180.0;
    if wrapped == -180.0 { 180.0 } else { wrapped }
}

/// Smallest "nice" tick step (1/2/5 × 10ⁿ) at or above `target`.
fn nice_step(target: f64) -> f64 {
    let t = target.max(1e-9);
    let pow10 = 10f64.powf(t.log10().floor());
    let m = t / pow10;
    let mult = if m <= 1.0 {
        1.0
    } else if m <= 2.0 {
        2.0
    } else if m <= 5.0 {
        5.0
    } else {
        10.0
    };
    mult * pow10
}

/// Tick value + top offset (px in the tape rect) for `slot`, or `None`
/// when the tick falls outside the tape (or below zero where the quantity
/// can't go negative).
fn tape_tick_layout(
    value: f64,
    step: f64,
    slot: i32,
    allow_negative: bool,
    tape_h: f32,
) -> Option<(f64, f32)> {
    let base = (value / step).round();
    let tick_value = (base + slot as f64) * step;
    if tick_value < 0.0 && !allow_negative {
        return None;
    }
    let top = tape_h * 0.5 - ((tick_value - value) / step) as f32 * TAPE_PX_PER_STEP - 7.0;
    if !(4.0..=tape_h - 18.0).contains(&top) {
        return None;
    }
    Some((tick_value, top))
}

#[allow(clippy::too_many_arguments)]
fn apply_tape_tick(
    value: f64,
    step: f64,
    slot: i32,
    allow_negative: bool,
    tape_h: f32,
    node: &mut Node,
    text: &mut Text,
    vis: &mut Visibility,
) {
    match tape_tick_layout(value, step, slot, allow_negative, tape_h) {
        Some((tick_value, top)) => {
            node.top = Val::Px(top);
            let label = tick_label(tick_value, step);
            if text.0 != label {
                text.0 = label;
            }
            if *vis != Visibility::Inherited {
                *vis = Visibility::Inherited;
            }
        }
        None => {
            if *vis != Visibility::Hidden {
                *vis = Visibility::Hidden;
            }
        }
    }
}

fn tick_label(value: f64, step: f64) -> String {
    if step >= 1000.0 {
        format!("{:.0}k", value / 1000.0)
    } else {
        format!("{:.0}", value)
    }
}

/// Compact signed rate for the V/S readout, in the active display unit (m/s or
/// ft/min). The `k`-suffix threshold is unit-relative so feet-per-minute values
/// (~200× larger) don't collapse to `k` during normal flight.
fn signed_speed(v: f64, system: thalos_game_state::units::UnitSystem) -> String {
    let v = format::vertical_speed_value(v, system);
    let k_threshold = if system.is_imperial() {
        10_000.0
    } else {
        1000.0
    };
    if v.abs() >= k_threshold {
        format!("{:+.1}k", v / 1000.0)
    } else {
        format!("{:+.0}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Craft on the body's +Z axis: local up = +Z, north = +Y, east = +X.
    const CRAFT_POS: DVec3 = DVec3::new(0.0, 0.0, 1000.0);
    const BODY_POS: DVec3 = DVec3::ZERO;

    fn angles(q: DQuat) -> AttitudeAngles {
        attitude_angles(q, CRAFT_POS, BODY_POS).expect("non-degenerate")
    }

    #[track_caller]
    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn identity_attitude_reads_level_north() {
        let a = angles(DQuat::IDENTITY);
        assert_close(a.pitch_rad, 0.0);
        assert_close(a.bank_rad, 0.0);
        assert_close(a.heading_rad, 0.0);
    }

    #[test]
    fn pitch_up_about_the_right_axis() {
        // +30° about body X tips the nose (body +Y) toward local up (+Z).
        let q = DQuat::from_axis_angle(DVec3::X, 30f64.to_radians());
        let a = angles(q);
        assert_close(a.pitch_rad.to_degrees(), 30.0);
        assert_close(a.bank_rad, 0.0);
    }

    #[test]
    fn bank_right_about_the_nose() {
        // +40° about body Y dips the right wing (body +X) below the horizon.
        let q = DQuat::from_axis_angle(DVec3::Y, 40f64.to_radians());
        let a = angles(q);
        assert_close(a.bank_rad.to_degrees(), 40.0);
        assert_close(a.pitch_rad, 0.0);
    }

    #[test]
    fn heading_east_reads_90() {
        // −90° about local up (+Z) swings the nose from north to east.
        let q = DQuat::from_axis_angle(DVec3::Z, -90f64.to_radians());
        let a = angles(q);
        assert_close(a.heading_rad.to_degrees(), 90.0);
        assert_close(a.pitch_rad, 0.0);
        assert_close(a.bank_rad, 0.0);
    }

    #[test]
    fn nice_steps_snap_up_to_1_2_5() {
        assert_eq!(nice_step(4.0), 5.0);
        assert_eq!(nice_step(10.0), 10.0);
        assert_eq!(nice_step(11.0), 20.0);
        assert_eq!(nice_step(80.0), 100.0);
        assert_eq!(nice_step(308.0), 500.0);
    }

    #[test]
    fn wrap_deg_takes_the_short_way() {
        assert_close(wrap_deg(190.0), -170.0);
        assert_close(wrap_deg(-190.0), 170.0);
        assert_close(wrap_deg(180.0), 180.0);
    }

    #[test]
    fn tape_ticks_hide_outside_the_tape_and_below_zero() {
        // Centre slot is always inside.
        assert!(tape_tick_layout(100.0, 10.0, 0, false, TAPE_H).is_some());
        // Far slot is off the tape.
        assert!(tape_tick_layout(100.0, 10.0, 9, false, TAPE_H).is_none());
        // Negative speed ticks hide; negative altitude/VS ticks may show.
        assert!(tape_tick_layout(5.0, 10.0, -2, false, TAPE_H).is_none());
        assert!(tape_tick_layout(5.0, 10.0, -2, true, TAPE_H).is_some());
        // The shorter V/S tape culls sooner.
        assert!(tape_tick_layout(0.0, 2.0, 4, true, VS_TAPE_H).is_none());
        assert!(tape_tick_layout(0.0, 2.0, 3, true, VS_TAPE_H).is_some());
    }
}

// ---------------------------------------------------------------------------
// Approach guidance: spawners
// ---------------------------------------------------------------------------

/// Localizer (horizontal, under the ladder) and glideslope (vertical, inboard of
/// the altitude tape) deviation scales. Both start hidden and are revealed only
/// while an approach is armed.
fn spawn_deviation_scales(anchor: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    // --- Localizer: five dots in a row with a centre index mark.
    anchor
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(LOC_SCALE_Y),
                ..default()
            },
            Visibility::Hidden,
            PfdLocScale,
            Name::new("PfdLocScale"),
        ))
        .with_children(|scale| {
            for i in -2..=2i32 {
                if i == 0 {
                    // Centre: a short vertical tick, not a dot, so "on course"
                    // is unambiguous.
                    scale.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(-1.0),
                            top: Val::Px(-11.0),
                            width: Val::Px(2.0),
                            height: Val::Px(22.0),
                            ..default()
                        },
                        BackgroundColor(HUD_AMBER),
                    ));
                    continue;
                }
                let x = i as f32 * (DEV_SCALE_HALF_PX * 0.5);
                scale.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Px(x - DEV_DOT_PX * 0.5),
                        top: Val::Px(-DEV_DOT_PX * 0.5),
                        width: Val::Px(DEV_DOT_PX),
                        height: Val::Px(DEV_DOT_PX),
                        border_radius: BorderRadius::all(Val::Px(DEV_DOT_PX * 0.5)),
                        ..default()
                    },
                    BackgroundColor(HUD_AMBER_DIM),
                ));
            }
            scale.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(-DEV_INDEX_PX * 0.5),
                    top: Val::Px(-DEV_INDEX_PX * 0.5),
                    width: Val::Px(DEV_INDEX_PX),
                    height: Val::Px(DEV_INDEX_PX),
                    border: UiRect::all(Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(2.0)),
                    ..default()
                },
                BorderColor::all(HUD_AMBER),
                UiTransform {
                    translation: Val2::ZERO,
                    scale: Vec2::ONE,
                    // A diamond is a 45-degree-rotated square.
                    rotation: Rot2::degrees(45.0),
                },
                hud_glow(HUD_GLOW, 4.0),
                PfdLocIndex,
                Name::new("PfdLocIndex"),
            ));
            scale.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(-72.0),
                    top: Val::Px(16.0),
                    ..default()
                },
                Text::new(""),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(HUD_AMBER),
                PfdApproachLabel,
                Name::new("PfdApproachLabel"),
            ));
        });

    // --- Glideslope: five dots in a column.
    anchor
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(GS_SCALE_X),
                top: Val::Px(0.0),
                ..default()
            },
            Visibility::Hidden,
            PfdGsScale,
            Name::new("PfdGsScale"),
        ))
        .with_children(|scale| {
            for i in -2..=2i32 {
                if i == 0 {
                    scale.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(-11.0),
                            top: Val::Px(-1.0),
                            width: Val::Px(22.0),
                            height: Val::Px(2.0),
                            ..default()
                        },
                        BackgroundColor(HUD_AMBER),
                    ));
                    continue;
                }
                let y = i as f32 * (DEV_SCALE_HALF_PX * 0.5);
                scale.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Px(-DEV_DOT_PX * 0.5),
                        top: Val::Px(y - DEV_DOT_PX * 0.5),
                        width: Val::Px(DEV_DOT_PX),
                        height: Val::Px(DEV_DOT_PX),
                        border_radius: BorderRadius::all(Val::Px(DEV_DOT_PX * 0.5)),
                        ..default()
                    },
                    BackgroundColor(HUD_AMBER_DIM),
                ));
            }
            scale.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(-DEV_INDEX_PX * 0.5),
                    top: Val::Px(-DEV_INDEX_PX * 0.5),
                    width: Val::Px(DEV_INDEX_PX),
                    height: Val::Px(DEV_INDEX_PX),
                    border: UiRect::all(Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(2.0)),
                    ..default()
                },
                BorderColor::all(HUD_AMBER),
                UiTransform {
                    translation: Val2::ZERO,
                    scale: Vec2::ONE,
                    rotation: Rot2::degrees(45.0),
                },
                hud_glow(HUD_GLOW, 4.0),
                PfdGsIndex,
                Name::new("PfdGsIndex"),
            ));
        });
}

/// The two flight-director cues: a vertical bar you centre by rolling and a
/// horizontal bar you fly to in pitch. Both start hidden.
fn spawn_flight_director(anchor: &mut ChildSpawnerCommands<'_>) {
    anchor.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(-FD_BAR_THICK_PX * 0.5),
            top: Val::Px(-FD_BAR_LEN_PX * 0.5),
            width: Val::Px(FD_BAR_THICK_PX),
            height: Val::Px(FD_BAR_LEN_PX),
            ..default()
        },
        BackgroundColor(HUD_AMBER),
        hud_glow(HUD_GLOW, 6.0),
        Visibility::Hidden,
        PfdDirectorRoll,
        Name::new("PfdDirectorRoll"),
    ));
    anchor.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(-FD_BAR_LEN_PX * 0.5),
            top: Val::Px(-FD_BAR_THICK_PX * 0.5),
            width: Val::Px(FD_BAR_LEN_PX),
            height: Val::Px(FD_BAR_THICK_PX),
            ..default()
        },
        BackgroundColor(HUD_AMBER),
        hud_glow(HUD_GLOW, 6.0),
        Visibility::Hidden,
        PfdDirectorPitch,
        Name::new("PfdDirectorPitch"),
    ));
}

// ---------------------------------------------------------------------------
// Approach guidance: update
// ---------------------------------------------------------------------------

/// Drive the deviation scales, the flight director, and the approach label from
/// the guidance published by the runtime's `route` module.
///
/// Everything here is a *projection*: this system reads deviations and commands,
/// converts them to pixels, and writes nodes. It derives no navigation quantity
/// of its own.
#[allow(clippy::too_many_arguments)]
pub fn update_approach_guidance(
    mode: Res<NavDisplayMode>,
    route: Res<thalos_game_state::nav::RouteState>,
    sim_state: Res<SimulationState>,
    solar_system: Res<SolarSystemState>,
    units: Res<thalos_game_state::units::UnitsSettings>,
    // Disjointness has to be spelled out for every pair of mutable queries over
    // the same component, and a `ParamSet` only makes its OWN members exclusive —
    // not exclusive against the params beside it. Four `&mut Node` views and four
    // `&mut Visibility` views over marker components Bevy cannot prove disjoint is
    // a boot panic (B0001), which takes the whole app (and the capture host) down
    // rather than misbehaving quietly.
    mut scales: ParamSet<(
        Query<
            &mut Visibility,
            (
                With<PfdLocScale>,
                Without<PfdDirectorRoll>,
                Without<PfdDirectorPitch>,
            ),
        >,
        Query<
            &mut Visibility,
            (
                With<PfdGsScale>,
                Without<PfdDirectorRoll>,
                Without<PfdDirectorPitch>,
            ),
        >,
    )>,
    mut loc_index: Query<
        &mut Node,
        (
            With<PfdLocIndex>,
            Without<PfdGsIndex>,
            Without<PfdDirectorRoll>,
            Without<PfdDirectorPitch>,
        ),
    >,
    mut gs_index: Query<
        &mut Node,
        (
            With<PfdGsIndex>,
            Without<PfdLocIndex>,
            Without<PfdDirectorRoll>,
            Without<PfdDirectorPitch>,
        ),
    >,
    mut director: ParamSet<(
        Query<
            (&mut Node, &mut Visibility),
            (
                With<PfdDirectorRoll>,
                Without<PfdLocIndex>,
                Without<PfdGsIndex>,
                Without<PfdLocScale>,
                Without<PfdGsScale>,
            ),
        >,
        Query<
            (&mut Node, &mut Visibility),
            (
                With<PfdDirectorPitch>,
                Without<PfdLocIndex>,
                Without<PfdGsIndex>,
                Without<PfdLocScale>,
                Without<PfdGsScale>,
            ),
        >,
    )>,
    mut label_q: Query<&mut Text, With<PfdApproachLabel>>,
) {
    let armed = *mode == NavDisplayMode::Hud && route.plan.is_some() && route.guidance.is_some();
    let target = if armed {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut vis in scales.p0().iter_mut() {
        if *vis != target {
            *vis = target;
        }
    }
    for mut vis in scales.p1().iter_mut() {
        if *vis != target {
            *vis = target;
        }
    }

    // The flight-director cue: `None` whenever it must not be shown, so there is
    // exactly one place that decides its visibility.
    let mut cue: Option<(f32, f32)> = None;
    let plan_and_guidance = route
        .plan
        .as_ref()
        .zip(route.guidance.as_ref())
        .filter(|_| armed);

    if let Some((plan, guidance)) = plan_and_guidance {
        // --- Deviation indices.
        let loc_px = loc_index_offset_px(guidance.loc_deflection());
        for mut node in &mut loc_index {
            node.left = Val::Px(loc_px - DEV_INDEX_PX * 0.5);
        }
        let gs_px = gs_index_offset_px(guidance.gs_deflection());
        for mut node in &mut gs_index {
            node.top = Val::Px(gs_px - DEV_INDEX_PX * 0.5);
        }

        // --- Flight director. The roll cue is bank error; the pitch cue is
        // flight-path-angle error at the ladder's own px/degree, so the two read
        // consistently against the pitch ladder behind them.
        let sim = &sim_state.simulation;
        let craft = sim.craft_state();
        let angles = solar_system
            .states
            .as_deref()
            .and_then(|st| st.get(sim.dominant_body()))
            .and_then(|bs| {
                attitude_angles(
                    craft.attitude.orientation,
                    craft.translation.position,
                    bs.position,
                )
            });

        let speed = craft.translation.velocity.length();
        let gamma_command = if speed > 1.0 {
            (guidance.vertical_speed_command_m_s / speed)
                .clamp(-1.0, 1.0)
                .asin()
        } else {
            0.0
        };

        if let Some(angles) = angles {
            // The same cue the ND's steering dot shows, from the same function,
            // so the two instruments cannot command different rolls. Rescaled
            // from its normalised [-1, 1] into this panel's pixel travel.
            let roll_px = (guidance.director_lateral() as f32
                * thalos_navigation::guidance::DIRECTOR_BANK_FULL_SCALE_RAD.to_degrees() as f32
                * FD_PX_PER_BANK_DEG)
                .clamp(-FD_LIMIT_PX, FD_LIMIT_PX);
            // Flight-path angle is pitch minus angle of attack; the PFD does not
            // carry AoA and on a stabilised approach the difference is small and
            // near-constant, so the cue is driven against pitch attitude and
            // reads as a trim target rather than an exact path angle.
            let pitch_px = (-(gamma_command - angles.pitch_rad).to_degrees() as f32 * PX_PER_DEG)
                .clamp(-FD_LIMIT_PX, FD_LIMIT_PX);
            cue = Some((roll_px, pitch_px));
        }

        // --- Annunciation. A trailing asterisk means not yet established
        // (outside full-scale on either axis).
        if let Ok(mut text) = label_q.single_mut() {
            // Distance-to-go was hardcoded in km while the tapes right next to
            // it were unit-aware; it is part of the same instrument.
            let dtg =
                format::ground_distance(guidance.dtg_m, units.system_for(UnitDomain::Aviation));
            let line = format!(
                "APPR RWY {:02}  {}{}",
                plan.designator,
                dtg,
                if guidance.established { "" } else { " *" }
            );
            if text.0 != line {
                text.0 = line;
            }
        }
    }

    // One pass, one decision: the cue is shown only when it exists.
    let cue_target = if cue.is_some() {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for (mut node, mut vis) in director.p0().iter_mut() {
        if let Some((roll_px, _)) = cue {
            node.left = Val::Px(roll_px - FD_BAR_THICK_PX * 0.5);
        }
        if *vis != cue_target {
            *vis = cue_target;
        }
    }
    for (mut node, mut vis) in director.p1().iter_mut() {
        if let Some((_, pitch_px)) = cue {
            node.top = Val::Px(pitch_px - FD_BAR_THICK_PX * 0.5);
        }
        if *vis != cue_target {
            *vis = cue_target;
        }
    }
}

#[cfg(test)]
mod approach_guidance_tests {
    use super::*;

    #[test]
    fn deviation_indices_point_toward_the_course_and_the_slope() {
        // Right of course (positive) → the course is to the LEFT → index left.
        assert!(loc_index_offset_px(1.0) < 0.0);
        assert!(loc_index_offset_px(-1.0) > 0.0);
        // High (positive) → the slope is BELOW → index down, and screen y grows
        // downward, so that is a POSITIVE offset. The opposite sign to the
        // lateral case, deliberately: an instrument that points a high aircraft
        // upward is exactly as readable and exactly wrong.
        assert!(gs_index_offset_px(1.0) > 0.0);
        assert!(gs_index_offset_px(-1.0) < 0.0);
        assert_eq!(loc_index_offset_px(0.0), 0.0);
        assert_eq!(gs_index_offset_px(0.0), 0.0);
    }

    #[test]
    fn deviation_indices_saturate_at_full_scale() {
        assert_eq!(loc_index_offset_px(4.0), -DEV_SCALE_HALF_PX);
        assert_eq!(loc_index_offset_px(-4.0), DEV_SCALE_HALF_PX);
        assert_eq!(gs_index_offset_px(4.0), DEV_SCALE_HALF_PX);
        assert_eq!(gs_index_offset_px(-4.0), -DEV_SCALE_HALF_PX);
    }
}
