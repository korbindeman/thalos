//! Bottom-left HUD panel cluster: orbital velocity above the navball and
//! a vector throttle arc along the navball's left side.

use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;
use crate::fuel::ThrottleState;
use crate::hud::HudPanel;
use crate::hud::format;
use crate::hud::theme::{HudTheme, emphasis, label, panel_frame, panel_node};
use crate::navball::ui::{
    FRAME_SIZE_PX, NAVBALL_BOTTOM_PX, NAVBALL_LEFT_PX, NAVBALL_SIZE_PX, NavballFrameRoot,
};
use crate::rendering::{SimulationState, SolarSystemState};

/// The navball cluster sits at the bottom-left (navball at x=40,
/// nav panel just to its right). The flight readouts sit ABOVE the
/// navball with a small gap.
const THROTTLE_FRAME_RADIUS: f32 = FRAME_SIZE_PX * 0.5;
const THROTTLE_INNER_RADIUS: f32 = THROTTLE_FRAME_RADIUS - 14.0;
const THROTTLE_OUTER_RADIUS: f32 = THROTTLE_FRAME_RADIUS + 16.0;
const THROTTLE_NODE_PADDING: f32 = THROTTLE_OUTER_RADIUS - THROTTLE_FRAME_RADIUS + 4.0;
const THROTTLE_NODE_SIZE: f32 = FRAME_SIZE_PX + THROTTLE_NODE_PADDING * 2.0;
const THROTTLE_HALF_ANGLE: f32 = std::f32::consts::FRAC_PI_2;
const THROTTLE_BORDER_WIDTH: f32 = 1.6;

#[derive(Component)]
pub(super) struct VelocityText;

#[derive(Component)]
pub(super) struct ThrottleBar {
    commanded: f32,
    effective: f32,
}

#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub(super) struct ThrottleArcMaterial {
    /// x = logical node size; y/z = logical inner/outer radii.
    #[uniform(0)]
    geometry: Vec4,
    /// x = commanded; y = effective; z = half-angle; w = border width.
    #[uniform(1)]
    levels: Vec4,
    #[uniform(2)]
    track_color: Vec4,
    #[uniform(3)]
    fill_color: Vec4,
    #[uniform(4)]
    warn_color: Vec4,
    #[uniform(5)]
    tick_color: Vec4,
    #[uniform(6)]
    tick_major_color: Vec4,
    #[uniform(7)]
    border_color: Vec4,
}

impl ThrottleArcMaterial {
    fn new(commanded: f32, effective: f32, theme: &HudTheme) -> Self {
        Self {
            geometry: Vec4::new(
                THROTTLE_NODE_SIZE,
                THROTTLE_INNER_RADIUS,
                THROTTLE_OUTER_RADIUS,
                0.0,
            ),
            levels: Vec4::new(
                commanded,
                effective,
                THROTTLE_HALF_ANGLE,
                THROTTLE_BORDER_WIDTH,
            ),
            track_color: with_alpha(theme.panel_bg_alt, 0.95),
            fill_color: Color::srgba(0.42, 0.74, 0.36, 0.95).to_linear().to_vec4(),
            warn_color: theme.text_warn.to_linear().to_vec4(),
            tick_color: with_alpha(theme.text_subtitle, 0.62),
            tick_major_color: with_alpha(theme.text_subtitle, 0.88),
            border_color: with_alpha(theme.panel_border, 0.95),
        }
    }
}

impl UiMaterial for ThrottleArcMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/throttle_arc.wgsl".into()
    }
}

pub fn setup(
    mut commands: Commands,
    mut throttle_materials: ResMut<Assets<ThrottleArcMaterial>>,
    theme: Res<HudTheme>,
    navball_frame_q: Query<Entity, With<NavballFrameRoot>>,
) {
    let mut root = panel_node();
    // Sit immediately above the navball, aligned with its left edge.
    root.left = Val::Px(NAVBALL_LEFT_PX);
    root.bottom = Val::Px(NAVBALL_BOTTOM_PX + NAVBALL_SIZE_PX + 10.0);
    root.min_width = Val::Px(NAVBALL_SIZE_PX);

    let (bg, border) = panel_frame(&theme);
    commands
        .spawn((root, bg, border, HudPanel, Name::new("HudFlight")))
        .with_children(|p| {
            p.spawn(label(&theme, "ORBITAL VELOCITY"));
            p.spawn((emphasis(&theme, "—"), VelocityText));
        });

    let throttle_material = throttle_materials.add(ThrottleArcMaterial::new(0.0, 0.0, &theme));
    let Ok(navball_frame) = navball_frame_q.single() else {
        warn!("navball frame missing; throttle arc not spawned");
        return;
    };

    commands.entity(navball_frame).with_children(|p| {
        p.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(-THROTTLE_NODE_PADDING),
                top: Val::Px(-THROTTLE_NODE_PADDING),
                width: Val::Px(THROTTLE_NODE_SIZE),
                height: Val::Px(THROTTLE_NODE_SIZE),
                ..default()
            },
            MaterialNode(throttle_material),
            ThrottleBar {
                commanded: 0.0,
                effective: 0.0,
            },
            HudPanel,
            ZIndex(2),
            Name::new("HudThrottleBar"),
        ));
    });
}

pub fn update(
    sim: Res<SimulationState>,
    solar_system: Res<SolarSystemState>,
    throttle: Res<ThrottleState>,
    mut throttle_materials: ResMut<Assets<ThrottleArcMaterial>>,
    mut vel_q: Query<&mut Text, With<VelocityText>>,
    mut throttle_q: Query<(&mut ThrottleBar, &MaterialNode<ThrottleArcMaterial>)>,
) {
    let ship = sim.simulation.ship_state();
    let body = sim.simulation.dominant_body();
    let Some(states) = solar_system.states.as_deref() else { return; };
    let Some(body_state) = states.get(body) else { return; };
    let rel_speed = (ship.velocity - body_state.velocity).length();

    if let Ok(mut t) = vel_q.single_mut() {
        let s = format::speed(rel_speed);
        if t.0 != s {
            t.0 = s;
        }
    }

    if let Ok((mut bar, material_node)) = throttle_q.single_mut() {
        let commanded = throttle.commanded.clamp(0.0, 1.0) as f32;
        let effective = throttle.effective.clamp(0.0, 1.0) as f32;
        if (bar.commanded - commanded).abs() > 0.002 || (bar.effective - effective).abs() > 0.002 {
            if let Some(material) = throttle_materials.get_mut(material_node) {
                material.levels.x = commanded;
                material.levels.y = effective;
            }
            bar.commanded = commanded;
            bar.effective = effective;
        }
    }
}

fn with_alpha(color: Color, alpha_scale: f32) -> Vec4 {
    let srgba = color.to_srgba();
    Color::srgba(
        srgba.red,
        srgba.green,
        srgba.blue,
        srgba.alpha * alpha_scale,
    )
    .to_linear()
    .to_vec4()
}
