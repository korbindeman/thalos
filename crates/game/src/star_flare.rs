//! Screen-space lens flare. Projects the star's world position to viewport
//! coordinates each frame and draws a chain of additive ghost sprites along
//! the sun→screen-center axis, mimicking ghost reflections inside a camera
//! lens. Fades out when the sun is occluded by a planet, drifts off-screen,
//! or passes behind the camera.

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, BlendComponent, BlendFactor, BlendOperation, BlendState, Extent3d,
    RenderPipelineDescriptor, TextureDimension, TextureFormat,
};
use bevy::shader::ShaderRef;
use bevy::window::PrimaryWindow;

use crate::rendering::CelestialBody;

pub struct LensFlarePlugin;

impl Plugin for LensFlarePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(UiMaterialPlugin::<LensFlareMaterial>::default())
            .add_plugins(UiMaterialPlugin::<LensFlareHaloMaterial>::default())
            .add_systems(Startup, spawn_lens_flare_ghosts)
            .add_systems(Update, update_lens_flare.after(crate::SimStage::Camera));
    }
}

#[derive(Asset, AsBindGroup, TypePath, Clone)]
struct LensFlareMaterial {
    #[uniform(0)]
    tint: LinearRgba,
    #[texture(1)]
    #[sampler(2)]
    texture: Handle<Image>,
}

impl UiMaterial for LensFlareMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/lens_flare.wgsl".into()
    }

    fn specialize(descriptor: &mut RenderPipelineDescriptor, _key: UiMaterialKey<Self>) {
        apply_additive_blend(descriptor);
    }
}

/// Analytic halo ring drawn in the shader so it can be asymmetric
/// (crescent-shaped, brightest on the side facing screen center) rather
/// than a perfect circle. `params.xy` is the screen-space sun→center
/// direction; `params.z` is crescent amount; `params.w` is thickness.
#[derive(Asset, AsBindGroup, TypePath, Clone)]
struct LensFlareHaloMaterial {
    #[uniform(0)]
    tint: LinearRgba,
    #[uniform(1)]
    params: Vec4,
}

impl UiMaterial for LensFlareHaloMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/lens_flare_halo.wgsl".into()
    }

    fn specialize(descriptor: &mut RenderPipelineDescriptor, _key: UiMaterialKey<Self>) {
        apply_additive_blend(descriptor);
    }
}

fn apply_additive_blend(descriptor: &mut RenderPipelineDescriptor) {
    if let Some(fragment) = descriptor.fragment.as_mut() {
        for target in fragment.targets.iter_mut().flatten() {
            target.blend = Some(BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
            });
        }
    }
}

#[derive(Component)]
struct HaloShape {
    thickness: f32,
    crescent: f32,
}

#[derive(Component)]
struct LensFlareGhost {
    /// Position along the sun→screen-center axis. `0` = at sun, `1` = center,
    /// `>1` = mirrored past center on the far side.
    axis_t: f32,
    size_px: f32,
    base_tint: LinearRgba,
}

enum GhostShape {
    Iris,
    Starburst,
    /// Asymmetric halo ring (analytic shader). Bright arc faces away from
    /// the sun — i.e., toward screen center — matching real lens halos.
    #[allow(dead_code)]
    Halo { thickness: f32, crescent: f32 },
}

/// Color gradient along the chain reads as optical dispersion: warm near
/// the sun → magenta → cool blue near center → pink past center → warm
/// far-trailing. Scale spread is wide so ghosts don't cluster into same-
/// size blobs that read as planets. Chromatic aberration is baked into
/// the iris textures and analytic in the halo shader.
const GHOSTS: &[(f32, f32, Color, GhostShape)] = &[
    // Halo disabled — revisit once the rainbow blur feels right.
    // (0.0, 220.0, Color::srgba(1.00, 1.00, 1.00, 0.09),
    //     GhostShape::Halo { thickness: 0.12, crescent: 0.0 }),
    // Starburst anchor at the source.
    (0.0, 280.0, Color::srgba(1.00, 0.88, 0.65, 0.35), GhostShape::Starburst),
    // Tiny hot pip just off the sun.
    (0.14, 18.0, Color::srgba(1.00, 0.70, 0.32, 0.12), GhostShape::Iris),
    // Amber mid ghost.
    (0.28, 64.0, Color::srgba(1.00, 0.58, 0.26, 0.07), GhostShape::Iris),
    // Small magenta — color break that sells the optics.
    (0.46, 30.0, Color::srgba(0.85, 0.35, 0.68, 0.10), GhostShape::Iris),
    // Big faint cool ghost — long tail of the chain.
    (0.64, 160.0, Color::srgba(0.38, 0.58, 1.00, 0.035), GhostShape::Iris),
    // Sharp cyan near center.
    (0.86, 24.0, Color::srgba(0.48, 0.82, 1.00, 0.11), GhostShape::Iris),
    // Main centered ghost — cool.
    (1.00, 100.0, Color::srgba(0.58, 0.78, 1.00, 0.06), GhostShape::Iris),
    // Past-center pink.
    (1.20, 42.0, Color::srgba(1.00, 0.52, 0.62, 0.08), GhostShape::Iris),
    // Huge faint warm wash far past center.
    (1.42, 240.0, Color::srgba(1.00, 0.72, 0.48, 0.025), GhostShape::Iris),
];

fn spawn_lens_flare_ghosts(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut tex_materials: ResMut<Assets<LensFlareMaterial>>,
    mut halo_materials: ResMut<Assets<LensFlareHaloMaterial>>,
) {
    let iris = images.add(bake_iris_ghost(192));
    let starburst = images.add(bake_starburst(512));

    for &(axis_t, size_px, tint, ref shape) in GHOSTS {
        let base_tint: LinearRgba = tint.into();
        let node = Node {
            position_type: PositionType::Absolute,
            width: Val::Px(size_px),
            height: Val::Px(size_px),
            left: Val::Px(-10_000.0),
            top: Val::Px(-10_000.0),
            ..default()
        };
        let ghost = LensFlareGhost { axis_t, size_px, base_tint };

        match shape {
            GhostShape::Halo { thickness, crescent } => {
                let material = halo_materials.add(LensFlareHaloMaterial {
                    tint: LinearRgba::NONE,
                    params: Vec4::new(1.0, 0.0, *crescent, *thickness),
                });
                commands.spawn((
                    MaterialNode(material),
                    node,
                    Name::new("LensFlareHalo"),
                    ghost,
                    HaloShape { thickness: *thickness, crescent: *crescent },
                ));
            }
            GhostShape::Iris | GhostShape::Starburst => {
                let texture = match shape {
                    GhostShape::Iris => iris.clone(),
                    GhostShape::Starburst => starburst.clone(),
                    _ => unreachable!(),
                };
                let material = tex_materials.add(LensFlareMaterial {
                    tint: LinearRgba::NONE,
                    texture,
                });
                commands.spawn((
                    MaterialNode(material),
                    node,
                    Name::new("LensFlareGhost"),
                    ghost,
                ));
            }
        }
    }
}

fn update_lens_flare(
    cameras: Query<(&Camera, &GlobalTransform)>,
    bodies: Query<(&GlobalTransform, &CelestialBody)>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut tex_ghosts: Query<
        (&LensFlareGhost, &mut Node, &MaterialNode<LensFlareMaterial>),
        Without<HaloShape>,
    >,
    mut halo_ghosts: Query<(
        &LensFlareGhost,
        &HaloShape,
        &mut Node,
        &MaterialNode<LensFlareHaloMaterial>,
    )>,
    mut tex_materials: ResMut<Assets<LensFlareMaterial>>,
    mut halo_materials: ResMut<Assets<LensFlareHaloMaterial>>,
) {
    let hide = |tex_ghosts: &mut Query<
        (&LensFlareGhost, &mut Node, &MaterialNode<LensFlareMaterial>),
        Without<HaloShape>,
    >,
                halo_ghosts: &mut Query<(
        &LensFlareGhost,
        &HaloShape,
        &mut Node,
        &MaterialNode<LensFlareHaloMaterial>,
    )>,
                tex_mats: &mut Assets<LensFlareMaterial>,
                halo_mats: &mut Assets<LensFlareHaloMaterial>| {
        for (_, _, handle) in tex_ghosts.iter() {
            if let Some(m) = tex_mats.get_mut(&handle.0) {
                m.tint = LinearRgba::NONE;
            }
        }
        for (_, _, _, handle) in halo_ghosts.iter() {
            if let Some(m) = halo_mats.get_mut(&handle.0) {
                m.tint = LinearRgba::NONE;
            }
        }
    };

    let Ok((camera, cam_tf)) = cameras.single() else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };

    let cam_pos = cam_tf.translation();
    let sun_world = bodies
        .iter()
        .find(|(_, body)| body.is_star)
        .map(|(tf, _)| tf.translation());
    let Some(sun_world) = sun_world else {
        hide(&mut tex_ghosts, &mut halo_ghosts, &mut tex_materials, &mut halo_materials);
        return;
    };

    // --- Occlusion: ray camera→sun vs every non-star sphere ---------------
    let to_sun = sun_world - cam_pos;
    let sun_dist = to_sun.length();
    if sun_dist < 1e-6 {
        hide(&mut tex_ghosts, &mut halo_ghosts, &mut tex_materials, &mut halo_materials);
        return;
    }
    let ray_dir = to_sun / sun_dist;

    let mut occlusion = 1.0_f32;
    for (body_tf, body) in bodies.iter() {
        if body.is_star {
            continue;
        }
        let body_pos = body_tf.translation();
        let to_body = body_pos - cam_pos;
        let t = to_body.dot(ray_dir);
        if t <= 0.0 || t >= sun_dist {
            continue;
        }
        let closest = cam_pos + ray_dir * t;
        let miss = (body_pos - closest).length();
        let radius = body.render_radius;
        let soft = ((miss - radius) / (radius * 0.4)).clamp(0.0, 1.0);
        occlusion = occlusion.min(soft);
        if occlusion <= 0.0 {
            break;
        }
    }

    // --- On-screen projection --------------------------------------------
    let sun_screen = match camera.world_to_viewport(cam_tf, sun_world) {
        Ok(p) => p,
        Err(_) => {
            hide(&mut tex_ghosts, &mut halo_ghosts, &mut tex_materials, &mut halo_materials);
            return;
        }
    };

    let screen = Vec2::new(window.width(), window.height());

    // Fade starts `inset` pixels inside the viewport and reaches zero at
    // the edge — so the flare is fully gone the moment the sun crosses
    // off-screen, never floating in empty space.
    let inset = 60.0;
    let dx = ((inset - sun_screen.x).max(sun_screen.x - (screen.x - inset))).max(0.0);
    let dy = ((inset - sun_screen.y).max(sun_screen.y - (screen.y - inset))).max(0.0);
    let on_screen = 1.0 - (dx.max(dy) / inset).clamp(0.0, 1.0);

    let center = screen * 0.5;
    let to_center = center - sun_screen;
    let axis_len = to_center.length();

    // Physically correct: the chain collapses to a point when the sun is
    // at screen center (reflections stack onto the light source). Fade
    // the chain out as axis length shrinks so the collapse reads as
    // "merge into the core" instead of a visible pile.
    let axis_dir = if axis_len > 1.0 {
        to_center / axis_len
    } else {
        Vec2::new(1.0, 0.0)
    };
    let center_fade = smoothstep(40.0, screen.length() * 0.12, axis_len);

    let global_intensity = occlusion * on_screen * center_fade;

    if global_intensity <= 0.0 {
        hide(&mut tex_ghosts, &mut halo_ghosts, &mut tex_materials, &mut halo_materials);
        return;
    }

    let tint_for = |ghost: &LensFlareGhost| -> LinearRgba {
        let k = global_intensity * ghost.base_tint.alpha;
        LinearRgba::new(
            ghost.base_tint.red * k,
            ghost.base_tint.green * k,
            ghost.base_tint.blue * k,
            1.0,
        )
    };

    // Textured ghosts (iris, starburst).
    for (ghost, mut node, handle) in &mut tex_ghosts {
        let pos = sun_screen + to_center * ghost.axis_t;
        node.left = Val::Px(pos.x - ghost.size_px * 0.5);
        node.top = Val::Px(pos.y - ghost.size_px * 0.5);
        if let Some(m) = tex_materials.get_mut(&handle.0) {
            m.tint = tint_for(ghost);
        }
    }

    // Halo (analytic crescent ring) — orients its bright arc toward
    // screen center each frame via axis_dir.
    for (ghost, halo, mut node, handle) in &mut halo_ghosts {
        let pos = sun_screen + to_center * ghost.axis_t;
        node.left = Val::Px(pos.x - ghost.size_px * 0.5);
        node.top = Val::Px(pos.y - ghost.size_px * 0.5);
        if let Some(m) = halo_materials.get_mut(&handle.0) {
            m.tint = tint_for(ghost);
            m.params = Vec4::new(axis_dir.x, axis_dir.y, halo.crescent, halo.thickness);
        }
    }
}

/// Iris bokeh: nearly-flat disc with a soft edge. Mimics a defocused
/// point source through a spherical aperture. Per-channel radial offset
/// bakes chromatic aberration directly into the sprite — R disc slightly
/// larger than B — so additive blending produces a colored fringe at the
/// rim without any extra draws or shader samples.
fn bake_iris_ghost(size: u32) -> Image {
    let mut data = vec![0u8; (size * size * 4) as usize];
    let s = size as f32;
    let disc = |r: f32| -> f32 {
        let edge = 1.0 - smoothstep(0.42, 0.56, r);
        let rim = (-((r - 0.46).powi(2)) / 0.0015).exp() * 0.08;
        let fill = 0.55 + (0.5 - r) * 0.05;
        (edge * fill + rim * edge).clamp(0.0, 1.0)
    };
    for y in 0..size {
        for x in 0..size {
            let dx = (x as f32 + 0.5) / s * 2.0 - 1.0;
            let dy = (y as f32 + 0.5) / s * 2.0 - 1.0;
            let r = (dx * dx + dy * dy).sqrt();
            let vr = disc(r * 1.03);
            let vg = disc(r);
            let vb = disc(r * 0.97);
            let idx = ((y * size + x) * 4) as usize;
            data[idx] = (vr * 255.0) as u8;
            data[idx + 1] = (vg * 255.0) as u8;
            data[idx + 2] = (vb * 255.0) as u8;
            data[idx + 3] = 255;
        }
    }
    Image::new(
        Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// Starburst anchor: hot core, smooth warm halo, 4 long primary spikes
/// (H/V) + 4 shorter secondary spikes (diagonals). Spikes taper toward
/// their tips by increasing angular sharpness with radius.
fn bake_starburst(size: u32) -> Image {
    let mut data = vec![0u8; (size * size * 4) as usize];
    let s = size as f32;
    for y in 0..size {
        for x in 0..size {
            let dx = (x as f32 + 0.5) / s * 2.0 - 1.0;
            let dy = (y as f32 + 0.5) / s * 2.0 - 1.0;
            let r = (dx * dx + dy * dy).sqrt();
            let theta = dy.atan2(dx);

            let core = (-r * 28.0).exp();
            let halo = (-(r * r) / 0.045).exp() * 0.55;

            let taper_p = (1.0 - (r / 0.9)).clamp(0.0, 1.0).powi(2);
            let sharp_p = 40.0 + r * 120.0;
            let arm_h = theta.cos().abs().powf(sharp_p);
            let arm_v = theta.sin().abs().powf(sharp_p);
            let primary = arm_h.max(arm_v) * taper_p * 0.22;

            let t2 = theta + std::f32::consts::FRAC_PI_4;
            let taper_s = (1.0 - (r / 0.6)).clamp(0.0, 1.0).powi(2);
            let sharp_s = 60.0 + r * 140.0;
            let arm_d1 = t2.cos().abs().powf(sharp_s);
            let arm_d2 = t2.sin().abs().powf(sharp_s);
            let secondary = arm_d1.max(arm_d2) * taper_s * 0.11;

            let v = (core + halo + primary + secondary).clamp(0.0, 1.0);
            let mask = (1.0 - r.clamp(0.0, 1.0).powi(2)).powi(2);
            let byte = (v * mask * 255.0) as u8;
            let idx = ((y * size + x) * 4) as usize;
            data[idx] = byte;
            data[idx + 1] = byte;
            data[idx + 2] = byte;
            data[idx + 3] = 255;
        }
    }
    Image::new(
        Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
