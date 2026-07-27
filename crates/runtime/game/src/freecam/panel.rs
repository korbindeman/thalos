//! The freecam's on-screen control surface.
//!
//! Freecam grew a keyboard-only vocabulary (wheel speed, roll, hidden modes)
//! that you had to read the source to discover, and a cruise speed you could
//! only nudge by scroll steps with no idea where it had landed. This panel is
//! the visible half of that: it *shows* the speed with a human reference for
//! scale, lets you drag it anywhere in the range in one gesture, and exposes
//! the two flight modes ([`FreeCam::level_to_up`], [`FreeCam::ground_collision`])
//! as switches rather than lore.
//!
//! The keys keep working — the panel and the keyboard are **two surfaces on one
//! state**, not two states. Controls are pushed from [`FreeCam`] every frame
//! ([`refresh_controls`]) and pulled back on user interaction
//! ([`apply_controls`]); every write on both sides is value-guarded, so wheeling
//! the speed moves the slider and dragging the slider moves the speed without
//! the two chasing each other.
//!
//! Placement is the empty left flank, below the top-left warp/view row and above
//! the navball cluster — the freecam has no navball to fight with, and the
//! centre of frame is what the user is composing.
//!
//! Later (`docs/backlog.md`): viewpoint save/apply and teleport tools belong on
//! this surface too, which is why it is a module rather than a function.

use bevy::prelude::*;
use thalos_ui::{
    SliderFormat, UiCheckbox, UiSlider, UiTheme, spawn_checkbox_row, spawn_divider, spawn_heading,
    spawn_slider_row, spawn_value_row, tokens,
};

use super::{
    FREECAM_MAX_SPEED_M_S, FREECAM_MIN_SPEED_M_S, FreeCam, format_speed, speed_reference,
};
use crate::pause_menu::GamePause;
use crate::photo_mode::PhotoMode;
use crate::rendering::{SimulationState, view_anchor::ViewAnchor};

/// Panel root — its `Visibility` follows [`FreeCam::active`].
#[derive(Component)]
struct FreeCamPanelRoot;

/// The cruise-speed slider. Its stored value is **log₁₀(m/s)**: the range spans
/// seven decades, so a linear bar would put everything below a kilometre per
/// second in the first pixel. [`SliderFormat::Custom`] reads it back out in the
/// unit the user thinks in.
#[derive(Component)]
struct SpeedSliderControl;

#[derive(Component)]
struct LevelLockControl;

#[derive(Component)]
struct GroundFloorControl;

#[derive(Component)]
struct SpeedValueText;

#[derive(Component)]
struct SpeedReferenceText;

#[derive(Component)]
struct AnchorValueText;

#[derive(Component)]
struct AltitudeValueText;

const PANEL_WIDTH: f32 = 244.0;
/// Clear of the top-left warp/view row (16 px + its height).
const PANEL_TOP: f32 = 132.0;

pub struct FreeCamPanelPlugin;

impl Plugin for FreeCamPanelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup.after(thalos_ui::init_ui_theme))
            .add_systems(
                Update,
                (
                    sync_visibility,
                    // After the widget drivers, so a drag that lands this frame
                    // is read this frame; refresh last so the resource always
                    // wins a tie.
                    apply_controls.after(thalos_ui::drive_sliders),
                    refresh_controls.after(apply_controls),
                )
                    .chain(),
            );
    }
}

fn setup(mut commands: Commands, theme: Res<UiTheme>) {
    let freecam = FreeCam::default();
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(16.0),
                top: Val::Px(PANEL_TOP),
                width: Val::Px(PANEL_WIDTH),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(tokens::SPACE_SM),
                padding: UiRect::axes(Val::Px(tokens::SPACE_LG), Val::Px(tokens::SPACE_MD)),
                border_radius: BorderRadius::all(Val::Px(tokens::RADIUS_PANEL)),
                ..default()
            },
            theme.glass(),
            Visibility::Hidden,
            FreeCamPanelRoot,
            Name::new("FreeCamPanel"),
        ))
        .with_children(|panel| {
            panel
                .spawn(Node {
                    width: Val::Percent(100.0),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|row| {
                    row.spawn(theme.heading("FREECAM"));
                    row.spawn(theme.faint("F4"));
                });

            // Speed, large, with the human reference right under it — the
            // number alone doesn't tell you whether you're about to crawl or
            // overshoot the planet.
            let mut value = theme.mono(format_speed(freecam.base_speed_m_s));
            value.1.font_size = FontSize::Px(20.0);
            panel.spawn((value, SpeedValueText));
            panel.spawn((
                theme.faint(reference_line(freecam.base_speed_m_s)),
                SpeedReferenceText,
            ));

            spawn_slider_row(
                panel,
                &theme,
                "SPEED",
                UiSlider {
                    min: FREECAM_MIN_SPEED_M_S.log10() as f32,
                    max: FREECAM_MAX_SPEED_M_S.log10() as f32,
                    value: freecam.base_speed_m_s.log10() as f32,
                    step: 0.0,
                    format: SliderFormat::Custom(format_log_speed),
                },
                SpeedSliderControl,
            );
            panel.spawn(theme.faint("Wheel adjusts · Shift ×5 · Ctrl ×0.2"));

            spawn_divider(panel);

            spawn_heading(panel, &theme, "FLIGHT", false);
            spawn_checkbox_row(
                panel,
                &theme,
                "Level to planet up  (L)",
                freecam.level_to_up,
                LevelLockControl,
            );
            spawn_checkbox_row(
                panel,
                &theme,
                "Stop at the ground  (C)",
                freecam.ground_collision,
                GroundFloorControl,
            );

            spawn_divider(panel);

            spawn_value_row(panel, &theme, "Anchor", "—", AnchorValueText);
            spawn_value_row(panel, &theme, "Altitude", "—", AltitudeValueText);
        });
}

/// The panel exists for the duration of the app and shows only while freecam
/// owns the camera. Photo mode hides it like any other overlay (freecam is the
/// tool people frame photos with, so the two meet constantly), and the pause
/// menu owns the screen when it is up.
fn sync_visibility(
    freecam: Res<FreeCam>,
    photo: Res<PhotoMode>,
    pause: Res<GamePause>,
    mut root: Query<&mut Visibility, With<FreeCamPanelRoot>>,
) {
    let target = if freecam.active && !photo.active && !pause.active {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut visibility in &mut root {
        if *visibility != target {
            *visibility = target;
        }
    }
}

/// Widgets → [`FreeCam`]. Only `Changed` widgets are read, so this is the user
/// moving a control, not [`refresh_controls`]'s own write echoing back.
fn apply_controls(
    mut freecam: ResMut<FreeCam>,
    speed: Query<&UiSlider, (Changed<UiSlider>, With<SpeedSliderControl>)>,
    level: Query<&UiCheckbox, (Changed<UiCheckbox>, With<LevelLockControl>)>,
    ground: Query<&UiCheckbox, (Changed<UiCheckbox>, With<GroundFloorControl>)>,
) {
    if let Ok(slider) = speed.single() {
        let value = 10.0_f64
            .powf(slider.value as f64)
            .clamp(FREECAM_MIN_SPEED_M_S, FREECAM_MAX_SPEED_M_S);
        // Guarded in the *speed* domain, not the log domain: a refresh writes
        // log10(speed) back into the slider, and reading that straight through
        // must not count as a user edit.
        if (freecam.base_speed_m_s - value).abs() > freecam.base_speed_m_s * 1.0e-3 {
            freecam.base_speed_m_s = value;
        }
    }
    if let Ok(checkbox) = level.single()
        && freecam.level_to_up != checkbox.checked
    {
        freecam.level_to_up = checkbox.checked;
    }
    if let Ok(checkbox) = ground.single()
        && freecam.ground_collision != checkbox.checked
    {
        freecam.ground_collision = checkbox.checked;
    }
}

/// [`FreeCam`] (+ the view anchor) → widgets and readouts. Every write is
/// value-guarded so change detection keeps meaning "this actually moved".
fn refresh_controls(
    freecam: Res<FreeCam>,
    anchor: Res<ViewAnchor>,
    sim: Res<SimulationState>,
    root: Query<&Visibility, With<FreeCamPanelRoot>>,
    mut speed: Query<&mut UiSlider, With<SpeedSliderControl>>,
    mut level: Query<&mut UiCheckbox, (With<LevelLockControl>, Without<GroundFloorControl>)>,
    mut ground: Query<&mut UiCheckbox, (With<GroundFloorControl>, Without<LevelLockControl>)>,
    mut texts: ParamSet<(
        Query<&mut Text, With<SpeedValueText>>,
        Query<&mut Text, With<SpeedReferenceText>>,
        Query<&mut Text, With<AnchorValueText>>,
        Query<&mut Text, With<AltitudeValueText>>,
    )>,
) {
    // Hidden panel: nothing to keep fresh, and the readouts would burn a
    // terrain-anchored query every frame of normal flight for no viewer.
    if root.single().is_ok_and(|v| *v == Visibility::Hidden) {
        return;
    }

    if let Ok(mut slider) = speed.single_mut() {
        let value = freecam.base_speed_m_s.log10() as f32;
        if (slider.value - value).abs() > 1.0e-4 {
            slider.value = value;
        }
    }
    if let Ok(mut checkbox) = level.single_mut()
        && checkbox.checked != freecam.level_to_up
    {
        checkbox.checked = freecam.level_to_up;
    }
    if let Ok(mut checkbox) = ground.single_mut()
        && checkbox.checked != freecam.ground_collision
    {
        checkbox.checked = freecam.ground_collision;
    }

    set_text(&mut texts.p0(), format_speed(freecam.base_speed_m_s));
    set_text(&mut texts.p1(), reference_line(freecam.base_speed_m_s));

    // The *latched* body, not the nearest one: freecam deliberately never
    // re-selects, so naming the view anchor's body here would lie the moment
    // the camera flew closer to a neighbour.
    let anchor_label = match freecam.anchor_body() {
        Some(body) => sim
            .system
            .bodies
            .get(body)
            .map(|definition| definition.name.clone())
            .unwrap_or_else(|| "—".to_string()),
        None => "inertial".to_string(),
    };
    set_text(&mut texts.p2(), anchor_label);

    let altitude = anchor
        .resolved
        .filter(|resolved| Some(resolved.body) == freecam.anchor_body())
        .map(|resolved| format_altitude(resolved.agl_m))
        .unwrap_or_else(|| "—".to_string());
    set_text(&mut texts.p3(), altitude);
}

fn set_text<F: bevy::ecs::query::QueryFilter>(query: &mut Query<&mut Text, F>, value: String) {
    for mut text in query.iter_mut() {
        if **text != value {
            **text = value.clone();
        }
    }
}

/// The slider's own inline readout — its stored value is a log₁₀ exponent.
fn format_log_speed(log10_m_s: f32) -> String {
    format_speed(10.0_f64.powf(log10_m_s as f64))
}

/// The line under the big number: what this speed is *like*, plus km/h while
/// that is still a unit anyone has intuition for.
fn reference_line(speed_m_s: f64) -> String {
    let reference = speed_reference(speed_m_s);
    if speed_m_s < 1_000.0 {
        format!("≈ {reference} · {:.0} km/h", speed_m_s * 3.6)
    } else {
        format!("≈ {reference}")
    }
}

fn format_altitude(agl_m: f64) -> String {
    if agl_m.abs() < 10_000.0 {
        format!("{agl_m:.0} m AGL")
    } else {
        format!("{:.1} km AGL", agl_m / 1_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The slider's log mapping must round-trip the whole range, or wheeling
    /// the speed and dragging the bar would disagree at the ends.
    #[test]
    fn log_slider_round_trips_the_speed_range() {
        for speed in [
            FREECAM_MIN_SPEED_M_S,
            100.0,
            7_800.0,
            FREECAM_MAX_SPEED_M_S,
        ] {
            let slider_value = speed.log10() as f32;
            let round_trip = 10.0_f64.powf(slider_value as f64);
            assert!(
                (round_trip - speed).abs() <= speed * 1.0e-3,
                "speed={speed} round_trip={round_trip}"
            );
        }
    }

    #[test]
    fn reference_line_drops_km_h_once_it_stops_helping() {
        assert!(reference_line(30.0).contains("km/h"));
        assert!(!reference_line(7_800.0).contains("km/h"));
    }
}
