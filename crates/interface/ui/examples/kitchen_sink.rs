//! Kitchen-sink testbed for the Thalos UI kit.
//!
//! Lays out every token and widget over a colourful 3D scene so the frosted
//! glass has something to blur. Two modes, mirroring `object_preview`:
//!
//! - **Headless (default):** renders off-screen and writes
//!   `artifacts/visual/latest/ui_preview.png`, then exits — agents iterate on the
//!   kit by reading the PNG (`just ui-preview`).
//! - **Windowed (`--window` / `-w`):** interactive, for hover/press/typing
//!   feel (`just ui-preview-window`). `S` saves the same screenshot.

use std::time::Duration;

use bevy::app::{AppExit, ScheduleRunnerPlugin};
use bevy::asset::{AssetPlugin, RenderAssetUsages};
use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::ui::IsDefaultUiCamera;
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;

use thalos_ui::tokens::*;
use thalos_ui::widgets::toast::{self, ToastArea, ToastKind};
use thalos_ui::*;

const WIDTH: u32 = 1760;
const HEIGHT: u32 = 990;
const OUT_PATH: &str = "artifacts/visual/latest/ui_preview.png";
/// Frames before the capture: shader/pipeline compiles, font atlas fill,
/// backdrop copy warm-up.
const WARMUP_FRAMES: u32 = 90;
const TAIL_FRAMES: u32 = 24;

fn main() {
    std::fs::create_dir_all("artifacts/visual/latest").ok();
    let window_mode = std::env::args()
        .skip(1)
        .any(|a| matches!(a.as_str(), "window" | "--window" | "-w"));

    let mut app = App::new();
    let asset_plugin = AssetPlugin {
        // Relative to `CARGO_MANIFEST_DIR` (crates/interface/ui), not the cwd.
        file_path: "../../../assets".to_string(),
        ..default()
    };
    if window_mode {
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Thalos — UI Kitchen Sink".into(),
                        resolution: (WIDTH, HEIGHT).into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(asset_plugin),
        )
        .add_systems(Startup, setup_window_camera)
        .add_systems(Update, screenshot_key);
    } else {
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    close_when_requested: false,
                    ..default()
                })
                .set(asset_plugin)
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .add_systems(Startup, setup_headless_camera)
        .add_systems(Update, drive_capture);
    }

    // `THALOS_UI_SCALE` rasterises the kit at a chosen effective UI scale. The
    // game's on-screen scale is `window scale × UiScale`, so a headless run at
    // 1.5 reproduces what a 150 % display shows — which is how the fractional
    // -scale text question gets answered without launching the game.
    if let Ok(raw) = std::env::var("THALOS_UI_SCALE")
        && let Ok(scale) = raw.trim().parse::<f32>()
        && scale > 0.0
    {
        app.insert_resource(UiScale(scale));
    }

    app.add_plugins(ThalosUiPlugin)
        .add_systems(Startup, setup_scene)
        .add_systems(Startup, (setup_ui, spawn_demo_toast).after(init_ui_theme))
        .add_systems(Startup, focus_preselected_field.after(setup_ui))
        .run();
}

// ---------------------------------------------------------------------------
// Cameras + capture
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct CaptureTarget(Handle<Image>);

#[derive(Resource, Default)]
struct CaptureState {
    frames: u32,
    captured: bool,
    tail: u32,
}

fn make_target(images: &mut Assets<Image>) -> Handle<Image> {
    let mut target = Image::new_fill(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    target.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT;
    images.add(target)
}

fn camera_bundle() -> impl Bundle {
    (
        Camera3d::default(),
        Transform::from_xyz(-6.0, 4.5, 11.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        AmbientLight {
            color: Color::srgb(0.7, 0.8, 1.0),
            brightness: 400.0,
            ..default()
        },
        UiBackdropSource,
        IsDefaultUiCamera,
    )
}

fn setup_window_camera(mut commands: Commands) {
    commands.spawn(camera_bundle());
}

fn setup_headless_camera(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let target = make_target(&mut images);
    commands.spawn((
        camera_bundle(),
        RenderTarget::Image(ImageRenderTarget::from(target.clone())),
    ));
    commands.insert_resource(CaptureTarget(target));
    commands.init_resource::<CaptureState>();
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
            println!("kitchen sink written to {OUT_PATH}");
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

fn screenshot_key(mut commands: Commands, keys: Res<ButtonInput<KeyCode>>) {
    if keys.just_pressed(KeyCode::KeyS) {
        std::fs::create_dir_all("artifacts/visual/latest").ok();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(OUT_PATH));
    }
}

// ---------------------------------------------------------------------------
// Backdrop scene — colour and contrast for the frost to chew on
// ---------------------------------------------------------------------------

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(ClearColor(Color::srgb(0.48, 0.62, 0.80)));
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, 0.5, 0.0)),
    ));
    let ground = materials.add(StandardMaterial {
        base_color: Color::srgb(0.28, 0.42, 0.24),
        perceptual_roughness: 0.9,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(200.0, 200.0))),
        MeshMaterial3d(ground),
    ));
    let sphere = meshes.add(Sphere::new(1.0));
    let colors = [
        Color::srgb(0.9, 0.4, 0.3),
        Color::srgb(0.95, 0.75, 0.3),
        Color::srgb(0.4, 0.7, 0.9),
        Color::srgb(0.8, 0.8, 0.85),
        Color::srgb(0.5, 0.9, 0.5),
    ];
    for (i, color) in colors.iter().enumerate() {
        let mat = materials.add(StandardMaterial {
            base_color: *color,
            perceptual_roughness: 0.25,
            metallic: 0.1,
            ..default()
        });
        let x = -6.0 + i as f32 * 3.2;
        commands.spawn((
            Mesh3d(sphere.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(x, 1.0, -(i as f32) * 1.5).with_scale(Vec3::splat(1.4)),
        ));
    }
}

// ---------------------------------------------------------------------------
// The sink itself
// ---------------------------------------------------------------------------

#[derive(Component)]
struct DemoAction;

/// The demo field that shows the prefilled-and-selected state (the F9
/// viewpoint prompt's). Focused once at startup so the preview captures it.
#[derive(Component)]
struct PreselectedField;

fn focus_preselected_field(
    mut focus: ResMut<TextFieldFocus>,
    field: Single<Entity, With<PreselectedField>>,
) {
    focus.field = Some(*field);
}

fn setup_ui(mut commands: Commands, theme: Res<UiTheme>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                top: Val::Px(0.0),
                bottom: Val::Px(0.0),
                padding: UiRect::all(Val::Px(SPACE_XL)),
                column_gap: Val::Px(SPACE_XL),
                align_items: AlignItems::FlexStart,
                ..default()
            },
            Pickable::IGNORE,
            Name::new("KitchenSinkRoot"),
        ))
        .with_children(|root| {
            menu_panel(root, &theme);
            controls_panel(root, &theme);
            data_panel(root, &theme);
            dialog_panel(root, &theme);
        });
}

/// Column 1 — the menu-screen family: display type, menu rows, key hints.
fn menu_panel(root: &mut ChildSpawnerCommands<'_>, theme: &UiTheme) {
    root.spawn((
        Node {
            width: Val::Px(340.0),
            row_gap: Val::Px(SPACE_SM),
            ..panel_node()
        },
        theme.glass(),
        Name::new("MenuPanel"),
    ))
    .with_children(|panel| {
        panel.spawn(theme.display("THALOS"));
        panel.spawn(theme.faint("pre-alpha  v0.1.0 — display face, light"));
        spawn_divider(panel);
        spawn_menu_row(panel, theme, DemoAction, "PLAY", "enter the space center");
        spawn_menu_row(panel, theme, DemoAction, "SETTINGS", "window & input");
        spawn_menu_row(panel, theme, DemoAction, "QUIT", "");
        spawn_heading(panel, theme, "SECTION HEADING", true);
        panel.spawn(theme.body("Body text — the interface face at 12px."));
        panel.spawn(theme.small("Small text — descriptions and sublabels."));
        panel.spawn(theme.faint("Faint text — placeholders, fine print."));
        panel.spawn(theme.mono("MONO 123 456.78 km — Fira Code"));
        panel
            .spawn(Node {
                column_gap: Val::Px(SPACE_SM),
                align_items: AlignItems::Center,
                margin: UiRect::top(Val::Px(SPACE_SM)),
                ..default()
            })
            .with_children(|row| {
                row.spawn(theme.small("key hints"));
                spawn_key_hint(row, theme, "Esc");
                spawn_key_hint(row, theme, "F9");
                spawn_key_hint(row, theme, "Tab");
            });
    });
}

/// Column 2 — interactive controls: buttons, toggles, text field, sliders.
fn controls_panel(root: &mut ChildSpawnerCommands<'_>, theme: &UiTheme) {
    root.spawn((
        Node {
            width: Val::Px(430.0),
            ..panel_node()
        },
        theme.glass(),
        Name::new("ControlsPanel"),
    ))
    .with_children(|panel| {
        panel.spawn(theme.title("CONTROLS"));
        spawn_heading(panel, theme, "BUTTONS", true);
        panel
            .spawn(Node {
                column_gap: Val::Px(SPACE_SM),
                align_items: AlignItems::Center,
                ..default()
            })
            .with_children(|row| {
                spawn_button(
                    row,
                    theme,
                    DemoAction,
                    "▶ LAUNCH",
                    ButtonVariant::Primary,
                    CTRL_H,
                );
                spawn_button(
                    row,
                    theme,
                    DemoAction,
                    "GHOST",
                    ButtonVariant::Ghost,
                    CTRL_H,
                );
                spawn_button(row, theme, DemoAction, "BARE", ButtonVariant::Bare, CTRL_H);
                spawn_button(
                    row,
                    theme,
                    DemoAction,
                    "DELETE",
                    ButtonVariant::Danger,
                    CTRL_H,
                );
            });
        spawn_heading(panel, theme, "TOGGLES (LATCHED STATES)", true);
        panel
            .spawn(Node {
                column_gap: Val::Px(SPACE_SM),
                align_items: AlignItems::Center,
                ..default()
            })
            .with_children(|row| {
                let on = spawn_button(
                    row,
                    theme,
                    DemoAction,
                    "MIRROR 2×",
                    ButtonVariant::Ghost,
                    CTRL_H,
                );
                row.commands_mut()
                    .entity(on)
                    .entry::<UiButton>()
                    .and_modify(|mut b| b.latched = true);
                spawn_button(
                    row,
                    theme,
                    DemoAction,
                    "SNAP 15°",
                    ButtonVariant::Ghost,
                    CTRL_H,
                );
            });
        spawn_heading(panel, theme, "TEXT FIELD", true);
        panel
            .spawn(Node {
                column_gap: Val::Px(SPACE_SM),
                ..default()
            })
            .with_children(|row| {
                spawn_text_field(
                    row,
                    theme,
                    UiTextField::new("Meridian Mk II", "ship name"),
                    Val::Px(200.0),
                    DemoAction,
                );
                spawn_text_field(
                    row,
                    theme,
                    UiTextField::new("", "empty placeholder"),
                    Val::Px(180.0),
                    DemoAction,
                );
                // Focused below so the selection highlight is in the preview:
                // it only renders on the field holding the keyboard.
                spawn_text_field(
                    row,
                    theme,
                    UiTextField::new("Thalos 340 m", "viewpoint name").selected(),
                    Val::Px(180.0),
                    (DemoAction, PreselectedField),
                );
            });
        spawn_heading(panel, theme, "SLIDERS", true);
        spawn_slider_row(
            panel,
            theme,
            "LENGTH",
            UiSlider::new(1.0, 24.0, 14.0, SliderFormat::Meters),
            DemoAction,
        );
        spawn_slider_row(
            panel,
            theme,
            "SWEEP",
            UiSlider::new(0.0, 60.0, 32.0, SliderFormat::Degrees),
            DemoAction,
        );
        spawn_slider_row(
            panel,
            theme,
            "FUEL",
            UiSlider::new(0.0, 4800.0, 3200.0, SliderFormat::Amount("L")),
            DemoAction,
        );
        spawn_heading(panel, theme, "CHECKBOX + CYCLE", true);
        spawn_checkbox_row(panel, theme, "Volumetric clouds", true, DemoAction);
        spawn_checkbox_row(panel, theme, "Grass blades", false, DemoAction);
        spawn_cycle_row(
            panel,
            theme,
            "Anti-aliasing",
            vec!["Off".into(), "SMAA".into(), "MSAA 4×".into()],
            1,
            DemoAction,
        );
    });
}

/// Column 3 — data display: value rows, list with selection, scroll column.
fn data_panel(root: &mut ChildSpawnerCommands<'_>, theme: &UiTheme) {
    root.spawn((
        Node {
            width: Val::Px(360.0),
            max_height: Val::Px(560.0),
            ..panel_node()
        },
        theme.glass(),
        Name::new("DataPanel"),
    ))
    .with_children(|panel| {
        panel.spawn(theme.title("READOUTS"));
        spawn_heading(panel, theme, "VALUES", true);
        spawn_value_row(panel, theme, "Apoapsis", "312.4 km", ());
        spawn_value_row(panel, theme, "Periapsis", "298.1 km", ());
        // NB: Titillium lacks the Δ glyph — Δ-strings must use the mono font.
        spawn_mono_value_row(panel, theme, "Δv", "3,412 m/s");
        spawn_divider(panel);
        spawn_heading(panel, theme, "LIST ROWS", true);
        for (i, (name, meta)) in [
            ("Apollo", "12 parts · 8.4 t"),
            ("Meridian", "31 parts · 42.7 t"),
            ("Atlas Heavy", "58 parts · 214 t"),
        ]
        .iter()
        .enumerate()
        {
            let row = spawn_list_row(panel, theme, name, meta);
            if i == 1 {
                panel
                    .commands_mut()
                    .entity(row)
                    .entry::<UiButton>()
                    .and_modify(|mut b| b.selected = true);
            }
        }
    });
}

/// Like `spawn_value_row` but with a mono label (for Δ and friends).
fn spawn_mono_value_row(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    label: &str,
    value: &str,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|row| {
            row.spawn(theme.mono_dim(label));
            row.spawn(theme.mono(value));
        });
}

/// A list row: name + right-aligned metadata, selectable.
fn spawn_list_row(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    name: &str,
    meta: &str,
) -> Entity {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Percent(100.0),
                height: Val::Px(CTRL_H),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(RADIUS_CTRL)),
                padding: UiRect::horizontal(Val::Px(SPACE_MD)),
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::NONE),
            BorderColor::all(Color::NONE),
            Interaction::None,
            UiButton::new(ButtonVariant::Bare),
            DemoAction,
        ))
        .with_children(|row| {
            row.spawn((theme.body(name), ButtonLabel));
            row.spawn((theme.small(meta), ButtonDesc));
        })
        .id()
}

/// Column 4 — heavy-glass dialog with a button rail.
fn dialog_panel(root: &mut ChildSpawnerCommands<'_>, theme: &UiTheme) {
    root.spawn((
        Node {
            width: Val::Px(320.0),
            margin: UiRect::top(Val::Px(140.0)),
            ..panel_node()
        },
        theme.glass_heavy(),
        Name::new("DialogPanel"),
    ))
    .with_children(|panel| {
        panel.spawn(theme.title("OVERWRITE CRAFT?"));
        panel.spawn(theme.small(
            "A craft named 'Meridian' already exists in the hangar. Overwrite it with the current build?",
        ));
        panel
            .spawn(Node {
                justify_content: JustifyContent::FlexEnd,
                column_gap: Val::Px(SPACE_SM),
                margin: UiRect::top(Val::Px(SPACE_MD)),
                ..default()
            })
            .with_children(|row| {
                spawn_button(row, theme, DemoAction, "CANCEL", ButtonVariant::Ghost, CTRL_H);
                spawn_button(row, theme, DemoAction, "OVERWRITE", ButtonVariant::Primary, CTRL_H);
            });
    });
}

fn spawn_demo_toast(
    mut commands: Commands,
    theme: Res<UiTheme>,
    area: Query<Entity, With<ToastArea>>,
) {
    let Ok(area) = area.single() else {
        return;
    };
    // Long lifetime so the headless capture always includes it.
    toast::spawn_toast(
        &mut commands,
        area,
        &theme,
        "Saved 'Meridian Mk II' to the hangar",
        ToastKind::Success,
    );
}
