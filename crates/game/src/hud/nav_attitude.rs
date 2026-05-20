//! 3D attitude indicator for the navigation panel centre.
//!
//! A small spacecraft built from primitive cuboids is rendered into an
//! off-screen image and shown in the centre of the nav panel.
//!
//! The view uses an **isometric projection** so that the three
//! orthogonal orbital axes (prograde, normal, radial-out) project to
//! the hex button positions:
//!
//!   12 o'clock  Normal      (nav-world +Z, projects straight up)
//!    2          Prograde    (nav-world −X, projects upper-right)
//!    4          Radial-Out  (nav-world +Y, projects lower-right)
//!    6          Anti-Normal (nav-world −Z, projects straight down)
//!    8          Retrograde  (nav-world +X, projects lower-left)
//!   10          Radial-In   (nav-world −Y, projects upper-left)
//!
//! When the craft is in "neutral" orbital attitude (body +Y aligned
//! with prograde, +Z with normal, +X with radial-out), the rendered
//! plane shows its nose pointing toward the Prograde button at 2
//! o'clock and its dorsal pointing toward the Normal button at 12.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{ClearColorConfig, ImageRenderTarget, RenderTarget, ScalingMode};
use bevy::math::{DMat3, DQuat};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use thalos_physics_canonical::maneuver::orbital_frame;

use crate::rendering::{SimulationState, SolarSystemState};

/// Render layer reserved for this module's mesh + camera.
pub const NAV_ATTITUDE_LAYER: usize = 4;

const RENDER_TARGET_SIZE: u32 = 128;

#[derive(Resource, Clone)]
pub struct NavAttitudeRenderTarget {
    pub image: Handle<Image>,
}

/// Root entity of the assembled spacecraft model. Its
/// [`Transform::rotation`] is overwritten each frame.
#[derive(Component)]
pub struct NavAttitudeModel;

pub fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut target = Image::new_fill(
        Extent3d {
            width: RENDER_TARGET_SIZE,
            height: RENDER_TARGET_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    target.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    let target_handle = images.add(target);
    commands.insert_resource(NavAttitudeRenderTarget {
        image: target_handle.clone(),
    });

    let body_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.88, 0.86, 0.80),
        unlit: true,
        ..default()
    });
    let accent_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.58, 0.54, 0.47),
        unlit: true,
        ..default()
    });

    // Plane primitives are built in BODY frame: +Y = nose, +Z = dorsal,
    // +X = right wing. The per-frame rotation handles mapping into the
    // nav-attitude render world (where +X = panel right, +Y = panel up,
    // +Z = out of screen).
    commands
        .spawn((
            Transform::IDENTITY,
            Visibility::default(),
            NavAttitudeModel,
            Name::new("NavAttitudeModel"),
        ))
        .with_children(|p| {
            // Fuselage along Y.
            p.spawn((
                Mesh3d(meshes.add(Cuboid::new(0.18, 1.4, 0.18))),
                MeshMaterial3d(body_material.clone()),
                Transform::IDENTITY,
                RenderLayers::layer(NAV_ATTITUDE_LAYER),
            ));
            // Main wings along X.
            p.spawn((
                Mesh3d(meshes.add(Cuboid::new(1.5, 0.10, 0.32))),
                MeshMaterial3d(body_material.clone()),
                Transform::from_xyz(0.0, 0.05, 0.0),
                RenderLayers::layer(NAV_ATTITUDE_LAYER),
            ));
            // Tail wings.
            p.spawn((
                Mesh3d(meshes.add(Cuboid::new(0.6, 0.10, 0.16))),
                MeshMaterial3d(body_material.clone()),
                Transform::from_xyz(0.0, -0.55, 0.0),
                RenderLayers::layer(NAV_ATTITUDE_LAYER),
            ));
            // Vertical fin (up in +Z dorsal).
            p.spawn((
                Mesh3d(meshes.add(Cuboid::new(0.10, 0.35, 0.36))),
                MeshMaterial3d(accent_material.clone()),
                Transform::from_xyz(0.0, -0.55, 0.18),
                RenderLayers::layer(NAV_ATTITUDE_LAYER),
            ));
            // Nose cone accent.
            p.spawn((
                Mesh3d(meshes.add(Cuboid::new(0.16, 0.22, 0.16))),
                MeshMaterial3d(accent_material),
                Transform::from_xyz(0.0, 0.78, 0.0),
                RenderLayers::layer(NAV_ATTITUDE_LAYER),
            ));
        });

    // Isometric camera: positioned along the (1,1,1) direction with
    // world +Z as up. This makes the three nav-world axes project to:
    //   +Z → screen up (12 o'clock)
    //   ±X → screen lower-left / upper-right  (8 / 2 o'clock)
    //   ±Y → screen lower-right / upper-left  (4 / 10 o'clock)
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: -1,
            clear_color: ClearColorConfig::Custom(Color::NONE),
            ..default()
        },
        RenderTarget::Image(ImageRenderTarget::from(target_handle)),
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed {
                width: 2.6,
                height: 2.6,
            },
            near: 0.1,
            far: 20.0,
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(1.5, 1.5, 1.5).looking_at(Vec3::ZERO, Vec3::Z),
        RenderLayers::layer(NAV_ATTITUDE_LAYER),
        Name::new("NavAttitudeCamera"),
    ));
}

/// Rotate the model so its body axes line up with the panel axes
/// (prograde/normal/radial), reflecting the craft's current attitude.
pub fn update_attitude(
    sim_state: Res<SimulationState>,
    solar_system: Res<SolarSystemState>,
    mut model: Query<&mut Transform, With<NavAttitudeModel>>,
) {
    let Ok(mut transform) = model.single_mut() else {
        return;
    };

    let sim = &sim_state.simulation;
    let craft = sim.craft_state();
    let dominant = sim.dominant_body();
    let Some(states) = solar_system.states.as_deref() else { return; };
    let Some(body_state) = states.get(dominant) else { return; };

    let [prograde, normal, radial] = orbital_frame(
        craft.translation.position,
        craft.translation.velocity,
        body_state.position,
        body_state.velocity,
    );

    // Orbital basis (columns of `orbital_to_world`).
    let orbital_basis = DMat3::from_cols(prograde, normal, radial);
    let q_orbital_to_world = DQuat::from_mat3(&orbital_basis);
    let q_world_to_orbital = q_orbital_to_world.inverse();
    let q_body_to_orbital = q_world_to_orbital * craft.attitude.orientation;

    // Fixed permutation: orbital frame axes → nav-world axes such that
    // after the camera's isometric projection they land at the right
    // button positions.
    //   orbital +X (prograde)   → nav −X (projects upper-right, 2 o'clock)
    //   orbital +Y (normal)     → nav +Z (projects straight up, 12)
    //   orbital +Z (radial-out) → nav +Y (projects lower-right, 4 o'clock)
    let q_orbital_to_navworld = orbital_to_navworld_permutation();
    let rotation_d = q_orbital_to_navworld * q_body_to_orbital;
    transform.rotation = rotation_d.as_quat();
}

fn orbital_to_navworld_permutation() -> DQuat {
    let mat = DMat3::from_cols(
        bevy::math::DVec3::new(-1.0, 0.0, 0.0), // orbital prograde → nav -X
        bevy::math::DVec3::new(0.0, 0.0, 1.0),  // orbital normal   → nav +Z
        bevy::math::DVec3::new(0.0, 1.0, 0.0),  // orbital radial   → nav +Y
    );
    DQuat::from_mat3(&mat)
}
