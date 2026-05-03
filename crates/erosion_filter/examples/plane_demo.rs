//! Interactive plane demo for `bevy_erosion_filter`.
//!
//! Renders a full-screen 2D quad whose fragment shader builds an fBm
//! heightmap, applies the erosion filter, and shades the result with a simple
//! altitude colormap. egui sliders on the right edit the parameters live.
//!
//! Run from the workspace root:
//!     cargo run --example plane_demo -p bevy_erosion_filter --release

use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{Material2d, Material2dPlugin, MeshMaterial2d};
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use bevy_erosion_filter::{ErosionFilterPlugin, cpu::ErosionParams};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_erosion_filter — plane demo".into(),
                resolution: (1280u32, 800u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .add_plugins(ErosionFilterPlugin)
        .add_plugins(Material2dPlugin::<DemoMaterial>::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (sync_quad_to_window, sliders))
        .run();
}

#[derive(Component)]
struct FullscreenQuad;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<DemoMaterial>>,
    windows: Query<&Window>,
) {
    let win = windows.single().expect("primary window");
    commands.spawn(Camera2d);

    let mesh = meshes.add(Rectangle::new(win.width(), win.height()));
    let material = materials.add(DemoMaterial { params: DemoParams::default() });

    commands.spawn((Mesh2d(mesh), MeshMaterial2d(material), FullscreenQuad));
}

fn sync_quad_to_window(
    mut meshes: ResMut<Assets<Mesh>>,
    windows: Query<&Window, Changed<Window>>,
    quads: Query<&Mesh2d, With<FullscreenQuad>>,
) {
    let Ok(win) = windows.single() else { return };
    for Mesh2d(handle) in &quads {
        if let Some(mesh) = meshes.get_mut(handle) {
            *mesh = Rectangle::new(win.width(), win.height()).into();
        }
    }
}

fn sliders(
    mut ctx: EguiContexts,
    mut mats: ResMut<Assets<DemoMaterial>>,
    quads: Query<&MeshMaterial2d<DemoMaterial>>,
) -> Result {
    let ctx = ctx.ctx_mut()?;

    egui::SidePanel::right("controls").default_width(280.0).show(ctx, |ui| {
        ui.heading("Erosion");
        for MeshMaterial2d(handle) in &quads {
            let Some(mat) = mats.get_mut(handle) else { continue };
            let p = &mut mat.params;

            let mut on = p.show_erosion >= 0.5;
            if ui.checkbox(&mut on, "Apply erosion").changed() {
                p.show_erosion = if on { 1.0 } else { 0.0 };
            }

            ui.separator();
            ui.label("Erosion params");
            ui.add(egui::Slider::new(&mut p.erosion.scale, 0.005..=0.5).logarithmic(true).text("scale"));
            ui.add(egui::Slider::new(&mut p.erosion.strength, 0.0..=0.5).text("strength"));
            ui.add(egui::Slider::new(&mut p.erosion.slope_power, 0.1..=2.0).text("slope_power"));
            ui.add(egui::Slider::new(&mut p.erosion.cell_scale, 0.25..=4.0).text("cell_scale"));
            ui.add(egui::Slider::new(&mut p.erosion.octaves, 1..=8).text("octaves"));
            ui.add(egui::Slider::new(&mut p.erosion.gain, 0.1..=0.9).text("gain"));
            ui.add(egui::Slider::new(&mut p.erosion.lacunarity, 1.5..=3.0).text("lacunarity"));
            ui.add(egui::Slider::new(&mut p.erosion.height_offset, -1.0..=1.0).text("height_offset"));

            ui.separator();
            ui.label("Base heightmap (fBm)");
            ui.add(egui::Slider::new(&mut p.base_freq, 0.5..=8.0).text("frequency"));
            ui.add(egui::Slider::new(&mut p.base_octaves, 1..=8).text("octaves"));
            ui.add(egui::Slider::new(&mut p.base_lacunarity, 1.5..=3.0).text("lacunarity"));
            ui.add(egui::Slider::new(&mut p.base_gain, 0.1..=0.9).text("gain"));
            ui.add(egui::Slider::new(&mut p.base_amplitude, 0.05..=1.0).text("amplitude"));

            ui.separator();
            ui.add(egui::Slider::new(&mut p.water_level, 0.0..=1.0).text("water_level"));

            if ui.button("Reset to defaults").clicked() {
                *p = DemoParams::default();
            }
        }
    });
    Ok(())
}

#[derive(Clone, Copy, Debug, ShaderType)]
#[repr(C)]
struct DemoParams {
    erosion: ErosionParamsGpu,
    show_erosion: f32,
    water_level: f32,
    base_freq: f32,
    base_octaves: i32,
    base_lacunarity: f32,
    base_gain: f32,
    base_amplitude: f32,
    _pad: Vec3,
}

#[derive(Clone, Copy, Debug, ShaderType)]
#[repr(C)]
struct ErosionParamsGpu {
    scale: f32,
    strength: f32,
    slope_power: f32,
    cell_scale: f32,
    octaves: i32,
    gain: f32,
    lacunarity: f32,
    height_offset: f32,
}

impl From<ErosionParams> for ErosionParamsGpu {
    fn from(p: ErosionParams) -> Self {
        Self {
            scale: p.scale,
            strength: p.strength,
            slope_power: p.slope_power,
            cell_scale: p.cell_scale,
            octaves: p.octaves,
            gain: p.gain,
            lacunarity: p.lacunarity,
            height_offset: p.height_offset,
        }
    }
}

impl Default for DemoParams {
    fn default() -> Self {
        Self {
            erosion: ErosionParams::default().into(),
            show_erosion: 1.0,
            water_level: 0.465,
            base_freq: 3.0,
            base_octaves: 3,
            base_lacunarity: 2.0,
            base_gain: 0.5,
            base_amplitude: 0.5,
            _pad: Vec3::ZERO,
        }
    }
}

#[derive(Asset, AsBindGroup, TypePath, Clone)]
struct DemoMaterial {
    #[uniform(0)]
    params: DemoParams,
}

impl Material2d for DemoMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/erosion_demo.wgsl".into()
    }
}
