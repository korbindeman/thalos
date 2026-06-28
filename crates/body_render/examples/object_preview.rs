//! Procedural object preview — view the procedural objects (trees, conifers,
//! shrubs; rocks etc. later) on their own, in two modes:
//!
//! - **Headless gallery** (default): renders each object to a PNG under
//!   `tools/preview/out/` and exits. No window, off-screen render target — so it
//!   can run unattended (CI, an agent) and the images be inspected afterwards.
//!   `just preview` / `cargo run -p thalos_body_render --example object_preview`.
//! - **Interactive window** (`--window`): opens a window with an orbit camera so
//!   you can fly around the objects, cycle between them, and grab screenshots.
//!   `just preview-window` / `… --example object_preview -- --window`.
//!
//! Both light objects with the real `TreeMaterial` + `thalos::lighting` sky
//! model, so the preview matches the in-game appearance. Add an object by
//! extending [`objects`].

use std::time::Duration;

use bevy::app::{AppExit, ScheduleRunnerPlugin};
use bevy::asset::RenderAssetUsages;
use bevy::camera::{ClearColorConfig, ImageRenderTarget, RenderTarget};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::Hdr;
use bevy::render::view::window::screenshot::{Screenshot, save_to_disk};
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;

use thalos_body_render::{
    CanopyStyle, GrassClumpParams, GrassMaterial, GrassMaterialPlugin, GrassParams, TreeMaterial,
    TreeMaterialPlugin, TreeMeshParams, build_foliage_atlas, build_grass_clump_mesh, build_tree_mesh,
    fallback_shadow_map,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 960;
/// Frames to render before the first capture (pipeline compile + atlas upload).
const WARMUP: u32 = 64;
/// Frames between one object's capture and the next (pose → render → capture).
const DWELL: u32 = 6;
/// Frames to keep running after the last capture so the async readback flushes.
const TAIL: u32 = 24;
const OUT_DIR: &str = "tools/preview/out";
const OBJECT_SPACING: f32 = 40.0;

/// Sun direction (toward the star) and clear-sky atmosphere shared by every
/// object, so they light exactly like the game's vegetation. The flux matches
/// the game's `LIGHT_AT_1AU` (10.0) at ~1 AU with unit exposure gain.
const SUN_DIR: Vec3 = Vec3::new(0.52, 0.62, 0.42);
const SUN_FLUX: f32 = 9.0;
const SKY: Color = Color::srgb(0.55, 0.70, 0.86);

fn main() {
    std::fs::create_dir_all(OUT_DIR).ok();
    let window_mode = std::env::args()
        .skip(1)
        .any(|a| matches!(a.as_str(), "window" | "--window" | "-w"));

    let mut app = App::new();
    if window_mode {
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Thalos — Object Preview".into(),
                resolution: (WIDTH, HEIGHT).into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, (setup_scene, setup_window_camera).chain())
        .add_systems(Update, (orbit_camera, screenshot_key));
    } else {
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    close_when_requested: false,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .add_systems(Startup, (setup_scene, setup_headless_camera).chain())
        .add_systems(Update, drive_capture);
    }
    app.add_plugins((TreeMaterialPlugin, GrassMaterialPlugin)).run();
}

/// What an object is built from — a procedural tree/shrub mesh (`TreeMaterial`)
/// or a standalone grass tuft (`GrassMaterial`).
#[derive(Clone)]
enum AssetKind {
    Tree(TreeMeshParams),
    GrassClump(GrassClumpParams),
}

/// Camera angle a preview is framed from. The grass clump is the asset for the
/// "billboard a clump, not a blade" impostor, so it's framed from above (the
/// aerial regime), level (the grazing-horizon regime), and 3/4.
#[derive(Clone, Copy)]
enum ViewKind {
    ThreeQuarter,
    Top,
    Side,
}

/// One previewable object: a display name, what to build, how to frame it.
#[derive(Clone)]
struct Preview {
    name: &'static str,
    kind: AssetKind,
    view: ViewKind,
    /// Height (m) of the point the camera looks at, and the camera distance.
    focus_y: f32,
    distance: f32,
}

fn objects() -> Vec<Preview> {
    // The grass clump asset (same params from every angle), framed three ways so
    // the two viewing regimes can be judged independently: `_top` is the aerial
    // look (aircraft overhead), `_side` is the grazing horizon, `_3q` is general.
    let clump = GrassClumpParams::default();
    vec![
        Preview {
            name: "tree_broadleaf",
            kind: AssetKind::Tree(TreeMeshParams {
                trunk_height_m: 5.2,
                trunk_radius_m: 0.32,
                canopy_radius_m: 3.0,
                canopy_height_m: 2.8,
                trunk_color: Vec3::new(0.16, 0.090, 0.045),
                canopy_color: Vec3::new(0.92, 1.0, 0.82),
                style: CanopyStyle::Broadleaf,
                seed: 0xB1_05_50,
                lod: 0,
            }),
            view: ViewKind::ThreeQuarter,
            focus_y: 4.6,
            distance: 17.0,
        },
        Preview {
            name: "tree_conifer",
            kind: AssetKind::Tree(TreeMeshParams {
                trunk_height_m: 7.0,
                trunk_radius_m: 0.26,
                canopy_radius_m: 1.9,
                canopy_height_m: 3.4,
                trunk_color: Vec3::new(0.13, 0.080, 0.045),
                canopy_color: Vec3::new(0.78, 1.0, 0.90),
                style: CanopyStyle::Conifer,
                seed: 0xC0_1F_E5,
                lod: 0,
            }),
            view: ViewKind::ThreeQuarter,
            focus_y: 5.6,
            distance: 20.0,
        },
        Preview {
            name: "shrub",
            kind: AssetKind::Tree(TreeMeshParams {
                trunk_height_m: 0.35,
                trunk_radius_m: 0.06,
                canopy_radius_m: 0.78,
                canopy_height_m: 0.62,
                trunk_color: Vec3::new(0.13, 0.085, 0.050),
                canopy_color: Vec3::new(0.95, 1.0, 0.86),
                style: CanopyStyle::Round,
                seed: 0x5_417,
                lod: 0,
            }),
            view: ViewKind::ThreeQuarter,
            focus_y: 0.7,
            distance: 3.2,
        },
        Preview {
            name: "grass_clump_top",
            kind: AssetKind::GrassClump(clump),
            view: ViewKind::Top,
            focus_y: clump.height_m * 0.45,
            distance: 1.6,
        },
        Preview {
            name: "grass_clump_side",
            kind: AssetKind::GrassClump(clump),
            view: ViewKind::Side,
            focus_y: clump.height_m * 0.5,
            distance: 1.6,
        },
        Preview {
            name: "grass_clump_3q",
            kind: AssetKind::GrassClump(clump),
            view: ViewKind::ThreeQuarter,
            focus_y: clump.height_m * 0.5,
            distance: 1.5,
        },
    ]
}

/// The placed objects, shared by the camera/capture/orbit systems.
#[derive(Resource)]
struct Scene {
    objects: Vec<Preview>,
}

/// World focus point of object `i` (its spot along +X at its framing height).
fn object_focus(objects: &[Preview], i: usize) -> Vec3 {
    Vec3::new(i as f32 * OBJECT_SPACING, objects[i].focus_y, 0.0)
}

/// Camera transform framing object `index` from its `view` angle (headless).
fn frame_transform(obj: &Preview, index: usize) -> Transform {
    let focus = Vec3::new(index as f32 * OBJECT_SPACING, obj.focus_y, 0.0);
    let d = obj.distance;
    match obj.view {
        // Straight down (aerial). `up` is +Z since +Y is degenerate looking down.
        ViewKind::Top => {
            let eye = focus + Vec3::new(0.0, d, 0.0);
            Transform::from_translation(eye).looking_at(focus, Vec3::Z)
        }
        // Level eye-line (grazing horizon).
        ViewKind::Side => {
            let eye = focus + Vec3::new(0.0, 0.0, d);
            Transform::from_translation(eye).looking_at(focus, Vec3::Y)
        }
        ViewKind::ThreeQuarter => {
            let eye = focus + Vec3::new(d * 0.55, obj.focus_y * 0.6 + d * 0.18, d * 0.92);
            Transform::from_translation(eye).looking_at(focus, Vec3::Y)
        }
    }
}

/// Spawn the ground, sun, objects and the shared tree material (both modes).
fn setup_scene(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut tree_materials: ResMut<Assets<TreeMaterial>>,
    mut grass_materials: ResMut<Assets<GrassMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
) {
    // Shared sky/sun parameters — identical for tree and grass materials so the
    // preview lights both exactly like the game's vegetation.
    let veg_params = GrassParams {
        sun_dir: SUN_DIR.normalize().extend(SUN_FLUX),
        wind: Vec4::ZERO,
        // Always full size (no clipmap fade): near edge well below any
        // distance, far edge well above.
        time_fade: Vec4::new(0.0, -1.0e9, 1.0e9, 1.0),
        sky_up: Vec3::Y.extend(0.0),
        sky_tau: Vec4::new(0.06, 0.12, 0.28, 1.0),
        anchor: Vec4::ZERO,
    };

    // One shared tree material with the procedural foliage atlas + sky lighting.
    let tree_material = tree_materials.add(TreeMaterial {
        atlas: images.add(build_foliage_atlas()),
        params: veg_params,
        // No sun-shadow pass here; bind valid depth textures so the per-cascade
        // `texture_depth_2d` slots resolve. `shadow.config.x` stays 0 (default),
        // so they're never sampled.
        sun_shadow_map_0: images.add(fallback_shadow_map()),
        sun_shadow_map_1: images.add(fallback_shadow_map()),
        sun_shadow_map_2: images.add(fallback_shadow_map()),
        ..default()
    });
    // One shared grass material (vertex-coloured blades, same sky model).
    let grass_material = grass_materials.add(GrassMaterial {
        params: veg_params,
    });

    let objects = objects();

    let ground = std_materials.add(StandardMaterial {
        base_color: Color::srgb(0.30, 0.38, 0.24),
        perceptual_roughness: 0.95,
        ..default()
    });

    for (i, obj) in objects.iter().enumerate() {
        let transform = Transform::from_xyz(i as f32 * OBJECT_SPACING, 0.0, 0.0);
        match &obj.kind {
            AssetKind::Tree(params) => {
                // A ground patch under each tree for grounding/contact shadows.
                commands.spawn((
                    Mesh3d(meshes.add(Plane3d::default().mesh().size(30.0, 30.0))),
                    MeshMaterial3d(ground.clone()),
                    transform,
                ));
                commands.spawn((
                    Mesh3d(meshes.add(build_tree_mesh(params))),
                    MeshMaterial3d(tree_material.clone()),
                    transform,
                ));
            }
            AssetKind::GrassClump(params) => {
                // No ground plane: the clump is the impostor asset, judged
                // against the sky (transparent at bake time) so its silhouette
                // and coverage read instead of blending into matching ground.
                commands.spawn((
                    Mesh3d(meshes.add(build_grass_clump_mesh(params))),
                    MeshMaterial3d(grass_material.clone()),
                    transform,
                ));
            }
        }
    }

    // Sun (lights the ground; trees self-light via the sky model).
    commands.spawn((
        DirectionalLight {
            illuminance: 9000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_translation(Vec3::ZERO).looking_at(-SUN_DIR, Vec3::Y),
    ));

    commands.insert_resource(Scene { objects });
}

/// `AmbientLight` is a per-view component in this Bevy; lifts the shadowed side.
fn ambient() -> AmbientLight {
    AmbientLight {
        color: Color::srgb(0.6, 0.72, 0.9),
        brightness: 600.0,
        ..default()
    }
}

// ---------------------------------------------------------------------------
// Headless gallery mode
// ---------------------------------------------------------------------------

/// Drives the capture timeline: which object is framed, when each is shot.
#[derive(Resource)]
struct Capture {
    frame: u32,
    target: Handle<Image>,
}

#[derive(Component)]
struct PreviewCamera;

fn setup_headless_camera(mut commands: Commands, mut images: ResMut<Assets<Image>>, scene: Res<Scene>) {
    // Off-screen render target the camera draws into and the screenshot reads.
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
    target.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC
        | TextureUsages::RENDER_ATTACHMENT;
    let target = images.add(target);

    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(SKY),
            ..default()
        },
        RenderTarget::Image(ImageRenderTarget::from(target.clone())),
        Hdr,
        // Match the game's main camera (Bevy default) so the preview is
        // representative — AgX desaturates highlights much harder and faked a
        // grey "hole" on the sunlit canopy top.
        Tonemapping::TonyMcMapface,
        ambient(),
        frame_transform(&scene.objects[0], 0),
        PreviewCamera,
    ));

    commands.insert_resource(Capture { frame: 0, target });
}

/// Timeline: warm up, then frame + capture each object in turn, then exit.
fn drive_capture(
    mut commands: Commands,
    mut cap: ResMut<Capture>,
    scene: Res<Scene>,
    mut cam: Query<&mut Transform, With<PreviewCamera>>,
    mut exit: MessageWriter<AppExit>,
) {
    cap.frame += 1;
    let n = scene.objects.len() as u32;

    for i in 0..n {
        let pose = WARMUP + i * DWELL;
        let shot = pose + 2;
        if cap.frame == pose
            && let Ok(mut t) = cam.single_mut()
        {
            *t = frame_transform(&scene.objects[i as usize], i as usize);
        }
        if cap.frame == shot {
            let path = format!("{OUT_DIR}/{}.png", scene.objects[i as usize].name);
            commands
                .spawn(Screenshot::image(cap.target.clone()))
                .observe(save_to_disk(path.clone()));
            info!("captured {path}");
        }
    }

    if cap.frame >= WARMUP + (n.saturating_sub(1)) * DWELL + 2 + TAIL {
        exit.write(AppExit::Success);
    }
}

// ---------------------------------------------------------------------------
// Interactive window mode
// ---------------------------------------------------------------------------

/// Orbit-camera state: a focus point with yaw/pitch/distance, plus which object
/// is currently centred.
#[derive(Component)]
struct OrbitCamera {
    focus: Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
    current: usize,
}

fn apply_orbit(t: &mut Transform, o: &OrbitCamera) {
    let rot = Quat::from_euler(EulerRot::YXZ, o.yaw, o.pitch, 0.0);
    let eye = o.focus + rot * Vec3::new(0.0, 0.0, o.distance);
    *t = Transform::from_translation(eye).looking_at(o.focus, Vec3::Y);
}

fn setup_window_camera(mut commands: Commands, scene: Res<Scene>) {
    let orbit = OrbitCamera {
        focus: object_focus(&scene.objects, 0),
        yaw: 0.6,
        pitch: -0.28,
        distance: scene.objects[0].distance,
        current: 0,
    };
    let mut transform = Transform::default();
    apply_orbit(&mut transform, &orbit);

    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(SKY),
            ..default()
        },
        Hdr,
        Tonemapping::TonyMcMapface,
        ambient(),
        transform,
        orbit,
    ));

    info!(
        "Object preview — drag: orbit · scroll: zoom · ←/→: cycle object · S: screenshot → {OUT_DIR}/"
    );
}

fn orbit_camera(
    btn: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    scene: Res<Scene>,
    mut q: Query<(&mut Transform, &mut OrbitCamera)>,
) {
    let Ok((mut t, mut o)) = q.single_mut() else {
        return;
    };

    if btn.pressed(MouseButton::Left) {
        o.yaw -= motion.delta.x * 0.008;
        o.pitch = (o.pitch - motion.delta.y * 0.008).clamp(-1.45, 1.45);
    }
    if scroll.delta.y != 0.0 {
        o.distance = (o.distance * (1.0 - scroll.delta.y * 0.12)).clamp(0.5, 250.0);
    }

    let n = scene.objects.len();
    let mut pick = None;
    if keys.just_pressed(KeyCode::ArrowRight) {
        pick = Some((o.current + 1) % n);
    } else if keys.just_pressed(KeyCode::ArrowLeft) {
        pick = Some((o.current + n - 1) % n);
    }
    if let Some(i) = pick {
        o.current = i;
        o.focus = object_focus(&scene.objects, i);
        o.distance = scene.objects[i].distance;
    }

    apply_orbit(&mut t, &o);
}

fn screenshot_key(
    keys: Res<ButtonInput<KeyCode>>,
    scene: Res<Scene>,
    cam: Query<&OrbitCamera>,
    mut commands: Commands,
) {
    if !keys.just_pressed(KeyCode::KeyS) {
        return;
    }
    // Name the shot after whichever object is centred, with a "_view" suffix so it
    // doesn't clobber the headless gallery PNGs.
    let name = cam
        .single()
        .ok()
        .map(|o| scene.objects[o.current].name)
        .unwrap_or("view");
    let path = format!("{OUT_DIR}/{name}_view.png");
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path.clone()));
    info!("saved {path}");
}
