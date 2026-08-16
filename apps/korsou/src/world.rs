use std::f64::consts::TAU;

use bevy::{
    camera::visibility::RenderLayers,
    light::{
        CascadeShadowConfigBuilder, DirectionalLightShadowMap, atmosphere::ScatteringMedium,
        light_consts::lux,
    },
    prelude::*,
};
use thalos_atmosphere::add_bevy_earth_atmosphere;
use thalos_runtime::{
    preferences::{SettingsMenuSet, SettingsPage, SettingsPageBuild, register_settings_page},
    ui::{
        SliderFormat, UiCheckbox, UiCycle, UiSlider, UiTheme, spawn_checkbox_row, spawn_cycle_row,
        spawn_slider_row, tokens,
    },
};

use crate::{cli::RunConfig, foliage::FOLIAGE_SHADOW_LAYER};

const CURACAO_LATITUDE_DEG: f64 = 12.1696;
const CURACAO_LONGITUDE_DEG: f64 = -68.9900;
const CURACAO_UTC_OFFSET_HOURS: f64 = -4.0;
const DEFAULT_DAY_OF_YEAR: u16 = 222;
const DEFAULT_LOCAL_TIME_HOURS: f64 = 15.5;
const DEFAULT_CLOCK_RATE: f64 = 60.0;
const SECONDS_PER_DAY: f64 = 86_400.0;
const DAYS_PER_YEAR: u16 = 365;
const SUN_TRANSFORM_DISTANCE_M: f32 = 50_000.0;

/// Shadow-only foliage geometry lives outside the scene camera's layer set.
/// The sun sees both layers; the camera sees only the normal world layer.
const SUN_RENDER_LAYERS: [usize; 2] = [0, FOLIAGE_SHADOW_LAYER];

const CLOCK_RATES: [f64; 4] = [1.0, 60.0, 300.0, 1_200.0];
const CLOCK_RATE_LABELS: [&str; 4] = ["Real time", "60×", "300×", "1,200×"];

pub struct WorldPlugin {
    interactive: bool,
}

impl WorldPlugin {
    pub const fn new(interactive: bool) -> Self {
        Self { interactive }
    }
}

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        let clock = SolarClock::from_config(app.world().resource::<RunConfig>());
        app.insert_resource(clock)
            .insert_resource(DirectionalLightShadowMap { size: 4_096 })
            .add_systems(Startup, setup_world)
            .add_systems(Update, (advance_solar_clock, update_sun).chain());

        if self.interactive {
            register_settings_page(
                app,
                SettingsPage {
                    id: "world",
                    label: "World",
                    order: 20,
                },
            );
            app.add_systems(
                Update,
                build_world_settings.in_set(SettingsMenuSet::BuildSections),
            )
            .add_systems(
                Update,
                (
                    apply_world_controls,
                    sync_world_controls.after(apply_world_controls),
                )
                    .in_set(SettingsMenuSet::Apply),
            );
        }
    }
}

/// Curaçao local civil clock. This is application state, not simulation time:
/// Kòrsou has no orbital/gameplay simulation, while the renderer only consumes
/// the resolved sun direction.
#[derive(Resource, Clone, Copy, Debug)]
pub(crate) struct SolarClock {
    pub(crate) day_of_year: u16,
    pub(crate) local_seconds: f64,
    pub(crate) running: bool,
    pub(crate) rate: f64,
}

impl SolarClock {
    fn from_config(config: &RunConfig) -> Self {
        Self {
            day_of_year: DEFAULT_DAY_OF_YEAR,
            local_seconds: config.local_time_hours.unwrap_or(DEFAULT_LOCAL_TIME_HOURS) * 3_600.0,
            // A headless capture is a deterministic instant. Interactive Kòrsou
            // starts the cycle so the world visibly changes without setup.
            running: !config.is_headless(),
            rate: DEFAULT_CLOCK_RATE,
        }
    }

    pub(crate) fn local_hours(self) -> f64 {
        self.local_seconds / 3_600.0
    }

    pub(crate) fn sun_direction(self) -> Vec3 {
        solar_direction(self.day_of_year, self.local_hours())
    }

    fn set_local_hours(&mut self, hours: f64) {
        self.local_seconds = hours.clamp(0.0, 24.0 - f64::EPSILON) * 3_600.0;
    }

    fn advance(&mut self, real_delta_s: f64) {
        if !self.running || real_delta_s <= 0.0 || !real_delta_s.is_finite() {
            return;
        }
        self.local_seconds += real_delta_s * self.rate.max(0.0);
        while self.local_seconds >= SECONDS_PER_DAY {
            self.local_seconds -= SECONDS_PER_DAY;
            self.day_of_year = if self.day_of_year >= DAYS_PER_YEAR {
                1
            } else {
                self.day_of_year + 1
            };
        }
    }
}

#[derive(Component)]
struct KorsouSun;

fn setup_world(
    mut commands: Commands,
    clock: Res<SolarClock>,
    mut scattering_media: ResMut<Assets<ScatteringMedium>>,
    mut ambient: ResMut<GlobalAmbientLight>,
) {
    // The atmosphere environment map on the camera supplies physically based
    // ambient light and reflections. Keep the global fallback from washing it
    // out with directionless light.
    *ambient = GlobalAmbientLight::NONE;

    // Preserve Earth-like scattering while compensating for this explorer's
    // flat local world, where long aerial sightlines never curve out of the
    // dense lower atmosphere as they would on the globe.
    commands.spawn((
        add_bevy_earth_atmosphere(&mut scattering_media, 0.45),
        Name::new("Earth atmosphere"),
    ));

    let direction = clock.sun_direction();
    commands.spawn((
        sun_light(direction),
        sun_transform(direction),
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 500.0,
            maximum_distance: 22_000.0,
            ..default()
        }
        .build(),
        RenderLayers::from_layers(&SUN_RENDER_LAYERS),
        KorsouSun,
        Name::new("Caribbean sun"),
    ));
}

fn sun_light(direction: Vec3) -> DirectionalLight {
    DirectionalLight {
        illuminance: lux::RAW_SUNLIGHT * direct_sun_factor(direction.y),
        color: Color::WHITE,
        shadow_maps_enabled: direction.y > 0.0,
        shadow_depth_bias: 0.015,
        // A smaller override caused the terrain to self-shadow in regular
        // bands at grazing view angles. Keep Bevy's scene-scaled default.
        shadow_normal_bias: DirectionalLight::DEFAULT_SHADOW_NORMAL_BIAS,
        ..default()
    }
}

fn advance_solar_clock(time: Res<Time<Real>>, mut clock: ResMut<SolarClock>) {
    clock.advance(time.delta_secs_f64());
}

fn update_sun(
    clock: Res<SolarClock>,
    mut sun: Single<(&mut Transform, &mut DirectionalLight), With<KorsouSun>>,
) {
    let direction = clock.sun_direction();
    *sun.0 = sun_transform(direction);
    sun.1.illuminance = lux::RAW_SUNLIGHT * direct_sun_factor(direction.y);
    sun.1.shadow_maps_enabled = direction.y > 0.0;
}

fn sun_transform(direction: Vec3) -> Transform {
    let up = if direction.y.abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    Transform::from_translation(direction * SUN_TRANSFORM_DISTANCE_M).looking_at(Vec3::ZERO, up)
}

/// Unit direction from the local observer toward the sun in Kòrsou render
/// coordinates: +X east, +Y up, -Z north.
///
/// The fractional-year, equation-of-time, and declination terms are NOAA's
/// compact solar-position approximation. They preserve Curaçao's real latitude,
/// longitude, UTC-4 civil clock, seasonal solar declination, and sunrise/sunset
/// motion without importing the planetary ephemeris into the lightweight app.
fn solar_direction(day_of_year: u16, local_hours: f64) -> Vec3 {
    let day = f64::from(day_of_year.clamp(1, DAYS_PER_YEAR));
    let hours = local_hours.rem_euclid(24.0);
    let gamma = TAU / 365.0 * (day - 1.0 + (hours - 12.0) / 24.0);
    let equation_of_time_min = 229.18
        * (0.000_075 + 0.001_868 * gamma.cos()
            - 0.032_077 * gamma.sin()
            - 0.014_615 * (2.0 * gamma).cos()
            - 0.040_849 * (2.0 * gamma).sin());
    let declination = 0.006_918 - 0.399_912 * gamma.cos() + 0.070_257 * gamma.sin()
        - 0.006_758 * (2.0 * gamma).cos()
        + 0.000_907 * (2.0 * gamma).sin()
        - 0.002_697 * (3.0 * gamma).cos()
        + 0.001_48 * (3.0 * gamma).sin();
    let time_offset_min =
        equation_of_time_min + 4.0 * CURACAO_LONGITUDE_DEG - 60.0 * CURACAO_UTC_OFFSET_HOURS;
    let true_solar_minutes = (hours * 60.0 + time_offset_min).rem_euclid(1_440.0);
    let hour_angle = (true_solar_minutes / 4.0 - 180.0).to_radians();
    let latitude = CURACAO_LATITUDE_DEG.to_radians();

    let east = -declination.cos() * hour_angle.sin();
    let north =
        latitude.cos() * declination.sin() - latitude.sin() * declination.cos() * hour_angle.cos();
    let up =
        latitude.sin() * declination.sin() + latitude.cos() * declination.cos() * hour_angle.cos();
    Vec3::new(east as f32, up as f32, -north as f32).normalize()
}

fn direct_sun_factor(sun_up: f32) -> f32 {
    smoothstep(
        (-6.0_f32).to_radians().sin(),
        2.0_f32.to_radians().sin(),
        sun_up,
    )
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[derive(Component)]
struct LocalTimeControl;
#[derive(Component)]
struct CalendarDayControl;
#[derive(Component)]
struct ClockRunningControl;
#[derive(Component)]
struct ClockRateControl;

type ChangedSlider<'w, 's, Marker> =
    Query<'w, 's, (&'static UiSlider, &'static Interaction), (Changed<UiSlider>, With<Marker>)>;

type ChangedCheckbox<'w, 's, Marker> =
    Query<'w, 's, (&'static UiCheckbox, &'static Interaction), (Changed<UiCheckbox>, With<Marker>)>;

type DisjointSlider<'w, 's, Marker, Excluded> =
    Query<'w, 's, (&'static mut UiSlider, &'static Interaction), (With<Marker>, Without<Excluded>)>;

fn build_world_settings(
    mut commands: Commands,
    mut builds: MessageReader<SettingsPageBuild>,
    theme: Res<UiTheme>,
    clock: Res<SolarClock>,
) {
    for build in builds.read() {
        if build.id != "world" {
            continue;
        }
        commands.entity(build.body).with_children(|body| {
            section_header(body, &theme, "CARIBBEAN SUN");
            spawn_slider_row(
                body,
                &theme,
                "Local time",
                UiSlider {
                    min: 0.0,
                    max: 24.0,
                    value: clock.local_hours() as f32,
                    step: 0.25,
                    format: SliderFormat::Custom(format_local_time),
                },
                LocalTimeControl,
            );
            spawn_slider_row(
                body,
                &theme,
                "Date",
                UiSlider {
                    min: 1.0,
                    max: f32::from(DAYS_PER_YEAR),
                    value: f32::from(clock.day_of_year),
                    step: 1.0,
                    format: SliderFormat::Custom(format_ordinal_date),
                },
                CalendarDayControl,
            );
            spawn_checkbox_row(
                body,
                &theme,
                "Run day/night cycle",
                clock.running,
                ClockRunningControl,
            );
            let rate_index = CLOCK_RATES
                .iter()
                .position(|rate| (*rate - clock.rate).abs() < f64::EPSILON)
                .unwrap_or(1);
            spawn_cycle_row(
                body,
                &theme,
                "Cycle speed",
                CLOCK_RATE_LABELS
                    .iter()
                    .map(|label| (*label).to_string())
                    .collect(),
                rate_index,
                ClockRateControl,
            );
            note(
                body,
                &theme,
                "Sunrise, sunset, shadows, sky colour, and reflected environment follow the same astronomical direction for Curaçao (UTC-4).",
            );
            note(
                body,
                &theme,
                "Headless captures freeze this clock; pass --time HH:MM to select a deterministic instant.",
            );
        });
    }
}

fn apply_world_controls(
    mut clock: ResMut<SolarClock>,
    time: ChangedSlider<LocalTimeControl>,
    day: ChangedSlider<CalendarDayControl>,
    running: ChangedCheckbox<ClockRunningControl>,
    rate: Query<&UiCycle, (Changed<UiCycle>, With<ClockRateControl>)>,
) {
    for (slider, interaction) in &time {
        if matches!(interaction, Interaction::Pressed) {
            clock.set_local_hours(f64::from(slider.value));
        }
    }
    for (slider, interaction) in &day {
        if matches!(interaction, Interaction::Pressed) {
            clock.day_of_year = slider.value.round().clamp(1.0, 365.0) as u16;
        }
    }
    for (checkbox, interaction) in &running {
        if matches!(interaction, Interaction::Pressed) {
            clock.running = checkbox.checked;
        }
    }
    for cycle in &rate {
        if let Some(rate) = CLOCK_RATES.get(cycle.index) {
            clock.rate = *rate;
        }
    }
}

fn sync_world_controls(
    clock: Res<SolarClock>,
    mut time: DisjointSlider<LocalTimeControl, CalendarDayControl>,
    mut day: DisjointSlider<CalendarDayControl, LocalTimeControl>,
    mut running: Query<(&mut UiCheckbox, &Interaction), With<ClockRunningControl>>,
    mut rate: Query<&mut UiCycle, With<ClockRateControl>>,
) {
    for (mut slider, interaction) in &mut time {
        let value = clock.local_hours() as f32;
        if !matches!(interaction, Interaction::Pressed) && (slider.value - value).abs() > 0.001 {
            slider.value = value;
        }
    }
    for (mut slider, interaction) in &mut day {
        let value = f32::from(clock.day_of_year);
        if !matches!(interaction, Interaction::Pressed) && slider.value != value {
            slider.value = value;
        }
    }
    for (mut checkbox, interaction) in &mut running {
        if !matches!(interaction, Interaction::Pressed) && checkbox.checked != clock.running {
            checkbox.checked = clock.running;
        }
    }
    let index = CLOCK_RATES
        .iter()
        .position(|value| (*value - clock.rate).abs() < f64::EPSILON)
        .unwrap_or(1);
    for mut cycle in &mut rate {
        if cycle.index != index {
            cycle.index = index;
        }
    }
}

pub(crate) fn format_local_time(hours: f32) -> String {
    let total_minutes = ((hours.rem_euclid(24.0) * 60.0).round() as u32) % 1_440;
    format!("{:02}:{:02}", total_minutes / 60, total_minutes % 60)
}

pub(crate) fn format_ordinal_date(day: f32) -> String {
    const MONTHS: [(&str, u16); 12] = [
        ("Jan", 31),
        ("Feb", 28),
        ("Mar", 31),
        ("Apr", 30),
        ("May", 31),
        ("Jun", 30),
        ("Jul", 31),
        ("Aug", 31),
        ("Sep", 30),
        ("Oct", 31),
        ("Nov", 30),
        ("Dec", 31),
    ];
    let mut remaining = day.round().clamp(1.0, 365.0) as u16;
    for (month, days) in MONTHS {
        if remaining <= days {
            return format!("{month} {remaining}");
        }
        remaining -= days;
    }
    "Dec 31".to_string()
}

fn section_header(body: &mut ChildSpawnerCommands<'_>, theme: &UiTheme, text: &str) {
    body.spawn((
        Node {
            margin: UiRect::top(Val::Px(4.0)),
            ..default()
        },
        Text::new(text.to_string()),
        TextFont {
            font: theme.font_ui.clone(),
            font_size: FontSize::Px(10.0),
            ..default()
        },
        TextColor(tokens::TEXT_FAINT),
    ));
}

fn note(body: &mut ChildSpawnerCommands<'_>, theme: &UiTheme, text: &str) {
    body.spawn((
        Text::new(text.to_string()),
        TextFont {
            font: theme.font_ui.clone(),
            font_size: FontSize::Px(9.0),
            ..default()
        },
        TextColor(tokens::TEXT_DIM),
    ));
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::{IntoSystem, System};

    use super::*;

    #[test]
    fn world_control_sync_has_valid_ecs_access() {
        let mut world = World::new();
        let mut system = IntoSystem::into_system(sync_world_controls);

        system.initialize(&mut world);
    }

    #[test]
    fn curacao_solar_path_has_day_and_night() {
        let noon = solar_direction(222, 12.75);
        let midnight = solar_direction(222, 0.75);

        assert!((noon.length() - 1.0).abs() < 1.0e-5);
        assert!(noon.y > 0.9, "August sun should be high at local noon");
        assert!(
            midnight.y < -0.8,
            "sun should be well below the horizon at night"
        );
    }

    #[test]
    fn solar_clock_rolls_across_the_year_boundary() {
        let mut clock = SolarClock {
            day_of_year: 365,
            local_seconds: 23.0 * 3_600.0 + 59.0 * 60.0,
            running: true,
            rate: 60.0,
        };

        clock.advance(61.0);

        assert_eq!(clock.day_of_year, 1);
        assert!((clock.local_hours() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn direct_light_stands_down_below_the_horizon() {
        assert_eq!(direct_sun_factor((-6.0_f32).to_radians().sin()), 0.0);
        assert_eq!(direct_sun_factor(12.0_f32.to_radians().sin()), 1.0);
    }

    #[test]
    fn sun_uses_directional_shadow_normal_bias() {
        let light = sun_light(Vec3::Y);

        assert_eq!(
            light.shadow_normal_bias,
            DirectionalLight::DEFAULT_SHADOW_NORMAL_BIAS
        );
        assert!(light.shadow_maps_enabled);
    }
}
