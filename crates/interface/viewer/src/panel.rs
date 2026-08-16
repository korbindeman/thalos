//! Shared freecam control surface.

use bevy::prelude::*;
use thalos_render_model::{MAX_FOCAL_LENGTH_MM, MIN_FOCAL_LENGTH_MM};
use thalos_ui::{
    SliderFormat, UiCheckbox, UiSlider, UiTheme, spawn_checkbox_row, spawn_divider, spawn_heading,
    spawn_slider_row, spawn_value_row, tokens,
};

use crate::{
    CameraOptics, VIEWER_MAX_SPEED_M_S, VIEWER_MIN_SPEED_M_S, ViewerCamera, ViewerPreferences,
    ViewerStatus, ViewerUiCapture, ViewerUiRoot, format_speed, speed_reference,
};

#[derive(Resource)]
struct ViewerPanelConfig {
    shortcut_hint: &'static str,
}

#[derive(Component)]
struct ViewerPanelRoot;

#[derive(Component)]
struct SpeedSliderControl;
#[derive(Component)]
struct LensSliderControl;
#[derive(Component)]
struct LevelLockControl;
#[derive(Component)]
struct GroundFloorControl;
#[derive(Component)]
struct SpeedValueText;
#[derive(Component)]
struct SpeedReferenceText;
#[derive(Component)]
struct LensValueText;
#[derive(Component)]
struct LensAovText;
#[derive(Component)]
struct AnchorValueText;
#[derive(Component)]
struct AltitudeValueText;

const PANEL_WIDTH: f32 = 264.0;
const PANEL_TOP: f32 = 132.0;

pub struct ViewerPanelPlugin {
    shortcut_hint: &'static str,
}

impl ViewerPanelPlugin {
    pub const fn new(shortcut_hint: &'static str) -> Self {
        Self { shortcut_hint }
    }
}

impl Plugin for ViewerPanelPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ViewerPanelConfig {
            shortcut_hint: self.shortcut_hint,
        })
        .add_systems(Startup, setup.after(thalos_ui::init_ui_theme))
        .add_systems(
            Update,
            (
                sync_visibility,
                sync_pointer_capture,
                apply_controls.after(thalos_ui::drive_sliders),
                refresh_controls.after(apply_controls),
            ),
        );
    }
}

fn setup(
    mut commands: Commands,
    theme: Res<UiTheme>,
    preferences: Res<ViewerPreferences>,
    config: Res<ViewerPanelConfig>,
) {
    let optics = CameraOptics::default();
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
            Interaction::None,
            Visibility::Hidden,
            ViewerUiRoot,
            thalos_photo_mode::HideInPhotoMode,
            ViewerPanelRoot,
            Name::new("ViewerPanel"),
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
                    row.spawn(theme.faint(config.shortcut_hint));
                });

            let mut value = theme.mono(format_speed(preferences.base_speed_m_s));
            value.1.font_size = FontSize::Px(20.0);
            panel.spawn((value, SpeedValueText));
            panel.spawn((
                theme.faint(reference_line(preferences.base_speed_m_s)),
                SpeedReferenceText,
            ));
            spawn_slider_row(
                panel,
                &theme,
                "SPEED",
                UiSlider {
                    min: VIEWER_MIN_SPEED_M_S.log10() as f32,
                    max: VIEWER_MAX_SPEED_M_S.log10() as f32,
                    value: preferences.base_speed_m_s.log10() as f32,
                    step: 0.0,
                    format: SliderFormat::Custom(format_log_speed),
                },
                SpeedSliderControl,
            );
            panel.spawn(theme.faint("Wheel adjusts · Shift ×5 · Ctrl ×0.2"));

            spawn_divider(panel);
            spawn_heading(panel, &theme, "LENS", false);
            let mut lens_value = theme.mono(format_lens_value(&optics));
            lens_value.1.font_size = FontSize::Px(20.0);
            panel.spawn((lens_value, LensValueText));
            panel.spawn((theme.faint(format_aov(&optics)), LensAovText));
            spawn_slider_row(
                panel,
                &theme,
                "FOCAL",
                UiSlider {
                    min: MIN_FOCAL_LENGTH_MM.log10(),
                    max: MAX_FOCAL_LENGTH_MM.log10(),
                    value: optics.base_focal_length_mm().log10(),
                    step: 0.0,
                    format: SliderFormat::Custom(format_log_focal_length),
                },
                LensSliderControl,
            );
            panel.spawn(theme.faint("14 · 24 · 35 · 50 · 85 · 135 · 200 mm"));
            panel.spawn(theme.faint("Hold Z for spring telephoto ×4"));

            spawn_divider(panel);
            spawn_heading(panel, &theme, "FLIGHT", false);
            spawn_checkbox_row(
                panel,
                &theme,
                "Level to local up  (L)",
                preferences.level_to_up,
                LevelLockControl,
            );
            spawn_checkbox_row(
                panel,
                &theme,
                "Stop at the ground  (C)",
                preferences.ground_collision,
                GroundFloorControl,
            );

            spawn_divider(panel);
            spawn_value_row(panel, &theme, "Location", "—", AnchorValueText);
            spawn_value_row(panel, &theme, "Altitude", "—", AltitudeValueText);
        });
}

fn sync_visibility(
    status: Res<ViewerStatus>,
    mut roots: Query<&mut Visibility, With<ViewerPanelRoot>>,
) {
    let target = if status.active && status.panel_visible {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut visibility in &mut roots {
        if *visibility != target {
            *visibility = target;
        }
    }
}

fn sync_pointer_capture(
    roots: Query<&Interaction, With<ViewerPanelRoot>>,
    mut capture: ResMut<ViewerUiCapture>,
) {
    let busy = roots
        .iter()
        .any(|interaction| *interaction != Interaction::None);
    if capture.pointer_busy != busy {
        capture.pointer_busy = busy;
    }
}

fn apply_controls(
    mut preferences: ResMut<ViewerPreferences>,
    mut camera_optics: Query<&mut CameraOptics, With<ViewerCamera>>,
    speed: Query<&UiSlider, (Changed<UiSlider>, With<SpeedSliderControl>)>,
    lens: Query<
        &UiSlider,
        (
            Changed<UiSlider>,
            With<LensSliderControl>,
            Without<SpeedSliderControl>,
        ),
    >,
    level: Query<&UiCheckbox, (Changed<UiCheckbox>, With<LevelLockControl>)>,
    ground: Query<&UiCheckbox, (Changed<UiCheckbox>, With<GroundFloorControl>)>,
) {
    if let Ok(slider) = speed.single() {
        let value = 10.0_f64
            .powf(slider.value as f64)
            .clamp(VIEWER_MIN_SPEED_M_S, VIEWER_MAX_SPEED_M_S);
        if (preferences.base_speed_m_s - value).abs() > preferences.base_speed_m_s * 1.0e-3 {
            preferences.base_speed_m_s = value;
        }
    }
    if let Ok(slider) = lens.single()
        && let Ok(mut optics) = camera_optics.single_mut()
    {
        let focal_length_mm = 10.0_f32.powf(slider.value);
        if (optics.base_focal_length_mm() - focal_length_mm).abs()
            > optics.base_focal_length_mm() * 1.0e-4
        {
            optics.set_base_focal_length_mm(focal_length_mm);
        }
    }
    if let Ok(checkbox) = level.single()
        && preferences.level_to_up != checkbox.checked
    {
        preferences.level_to_up = checkbox.checked;
    }
    if let Ok(checkbox) = ground.single()
        && preferences.ground_collision != checkbox.checked
    {
        preferences.ground_collision = checkbox.checked;
    }
}

#[allow(clippy::too_many_arguments)]
fn refresh_controls(
    preferences: Res<ViewerPreferences>,
    status: Res<ViewerStatus>,
    root: Query<&Visibility, With<ViewerPanelRoot>>,
    camera_optics: Query<&CameraOptics, With<ViewerCamera>>,
    mut speed: Query<&mut UiSlider, With<SpeedSliderControl>>,
    mut lens: Query<&mut UiSlider, (With<LensSliderControl>, Without<SpeedSliderControl>)>,
    mut level: Query<&mut UiCheckbox, (With<LevelLockControl>, Without<GroundFloorControl>)>,
    mut ground: Query<&mut UiCheckbox, (With<GroundFloorControl>, Without<LevelLockControl>)>,
    mut texts: ParamSet<(
        Query<&mut Text, With<SpeedValueText>>,
        Query<&mut Text, With<SpeedReferenceText>>,
        Query<&mut Text, With<LensValueText>>,
        Query<&mut Text, With<LensAovText>>,
        Query<&mut Text, With<AnchorValueText>>,
        Query<&mut Text, With<AltitudeValueText>>,
    )>,
) {
    if root
        .single()
        .is_ok_and(|visibility| *visibility == Visibility::Hidden)
    {
        return;
    }
    if let Ok(mut slider) = speed.single_mut() {
        let value = preferences.base_speed_m_s.log10() as f32;
        if (slider.value - value).abs() > 1.0e-4 {
            slider.value = value;
        }
    }
    if let Ok(optics) = camera_optics.single() {
        if let Ok(mut slider) = lens.single_mut() {
            let value = optics.base_focal_length_mm().log10();
            if (slider.value - value).abs() > 1.0e-4 {
                slider.value = value;
            }
        }
        set_text(&mut texts.p2(), format_lens_value(optics));
        set_text(&mut texts.p3(), format_aov(optics));
    }
    if let Ok(mut checkbox) = level.single_mut()
        && checkbox.checked != preferences.level_to_up
    {
        checkbox.checked = preferences.level_to_up;
    }
    if let Ok(mut checkbox) = ground.single_mut()
        && checkbox.checked != preferences.ground_collision
    {
        checkbox.checked = preferences.ground_collision;
    }
    set_text(&mut texts.p0(), format_speed(preferences.base_speed_m_s));
    set_text(&mut texts.p1(), reference_line(preferences.base_speed_m_s));
    set_text(&mut texts.p4(), status.anchor_label.clone());
    set_text(
        &mut texts.p5(),
        status
            .altitude_agl_m
            .map(format_altitude)
            .unwrap_or_else(|| "—".into()),
    );
}

fn set_text<F: bevy::ecs::query::QueryFilter>(query: &mut Query<&mut Text, F>, value: String) {
    for mut text in query.iter_mut() {
        if **text != value {
            **text = value.clone();
        }
    }
}

fn format_log_speed(log10_m_s: f32) -> String {
    format_speed(10.0_f64.powf(log10_m_s as f64))
}

fn format_log_focal_length(log10_mm: f32) -> String {
    format!("{:.0} mm", 10.0_f32.powf(log10_mm))
}

fn format_lens_value(optics: &CameraOptics) -> String {
    let base = optics.base_focal_length_mm();
    let effective = optics.effective_focal_length_mm();
    if (effective - base).abs() > 0.05 {
        format!(
            "{effective:.0} mm  ({base:.0} ×{:.1})",
            optics.zoom_multiplier()
        )
    } else {
        format!("{base:.0} mm")
    }
}

fn format_aov(optics: &CameraOptics) -> String {
    format!(
        "{:.1}° horizontal · {:.1}° vertical · {}:{} sensor",
        optics.horizontal_fov_rad().to_degrees(),
        optics.vertical_fov_rad().to_degrees(),
        optics.spec().sensor.aspect[0],
        optics.spec().sensor.aspect[1],
    )
}

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

    #[test]
    fn log_slider_round_trips_the_speed_range() {
        for speed in [VIEWER_MIN_SPEED_M_S, 100.0, 7_800.0, VIEWER_MAX_SPEED_M_S] {
            let round_trip = 10.0_f64.powf(speed.log10() as f32 as f64);
            assert!((round_trip - speed).abs() <= speed * 1.0e-3);
        }
    }
}
