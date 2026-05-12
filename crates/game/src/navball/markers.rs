//! Direction markers overlaid on the navball, plus the static
//! ship-orientation indicator at the centre.
//!
//! Each directional marker is a `bevy_ui` `ImageNode` child of the
//! navball UI root. Its 2D position is updated every frame from the
//! world-space direction it represents, projected via
//! [`NavballFrame::world_to_navball`]. Each marker carries two icon
//! handles ([`MarkerVariants`]); when the direction is on the back
//! hemisphere (`d_nav.z < 0`) the system swaps to the dimmed variant.
//! Markers whose direction is currently undefined (no target selected,
//! no maneuver node, stationary relative to SOI body) hide themselves.
//!
//! The ship-orientation indicator is a fixed UI node at the navball
//! centre — by construction the craft's nose always points there, so
//! it doesn't need projection.

use bevy::asset::RenderAssetUsages;
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use thalos_physics::canonical::Epoch;

use crate::maneuver::ManeuverPlan;
use crate::navball::attitude::NavballFrame;
use crate::navball::ui::NavballUiRoot;
use crate::navigation::maneuver_burn_direction;
use crate::rendering::SimulationState;
use crate::target::TargetBody;

const ICON_SIZE: u32 = 32;
const ORIENTATION_ICON_SIZE: u32 = 40;

/// Navball image radius and centre in UI pixels. Derived from the UI
/// root size (256) and the off-screen camera's `ScalingMode::Fixed { 2.4 }`.
const NAVBALL_DISPLAY_RADIUS_PX: f32 = 256.0 / 2.4;
const NAVBALL_CENTER_PX: f32 = 128.0;

/// Alpha multiplier for the "occluded" variant (direction on back hemisphere).
const OCCLUDED_ALPHA: f32 = 0.35;

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerKind {
    Prograde,
    Retrograde,
    Normal,
    AntiNormal,
    RadialOut,
    RadialIn,
    ManeuverNode,
    Target,
    AntiTarget,
}

impl MarkerKind {
    const ALL: [Self; 9] = [
        Self::Prograde,
        Self::Retrograde,
        Self::Normal,
        Self::AntiNormal,
        Self::RadialOut,
        Self::RadialIn,
        Self::ManeuverNode,
        Self::Target,
        Self::AntiTarget,
    ];

    pub fn color(self) -> [u8; 3] {
        match self {
            Self::Prograde | Self::Retrograde => [255, 220, 60],
            Self::Normal | Self::AntiNormal => [200, 100, 220],
            Self::RadialOut | Self::RadialIn => [80, 200, 230],
            Self::ManeuverNode => [100, 160, 255],
            Self::Target | Self::AntiTarget => [255, 100, 200],
        }
    }

    fn coverage(self, dx: f32, dy: f32, r: f32, half: f32) -> f32 {
        match self {
            Self::Prograde | Self::RadialOut => filled_disc_with_outward_spokes(dx, dy, r, half),
            Self::Retrograde | Self::RadialIn => hollow_ring_with_crosshair(dx, dy, r, half),
            Self::Normal => chevron_up(dx, dy, half),
            Self::AntiNormal => chevron_up(dx, -dy, half),
            Self::ManeuverNode => maneuver_target(dx, dy, r, half),
            Self::Target => target_square(dx, dy, half, false),
            Self::AntiTarget => target_square(dx, dy, half, true),
        }
    }
}

/// Front (visible-hemisphere) and back (occluded) icon handles.
#[derive(Component, Clone)]
pub struct MarkerVariants {
    pub front: Handle<Image>,
    pub back: Handle<Image>,
}

pub fn setup_navball_markers(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    root_q: Query<Entity, With<NavballUiRoot>>,
) {
    let Ok(root) = root_q.single() else {
        warn!("navball UI root missing; markers not spawned");
        return;
    };

    let mut commands_entity = commands.entity(root);
    commands_entity.with_children(|p| {
        for kind in MarkerKind::ALL {
            let variants = MarkerVariants {
                front: add_marker_image(kind, false, &mut images),
                back: add_marker_image(kind, true, &mut images),
            };
            p.spawn(marker_bundle(kind, &variants));
        }

        let orientation = images.add(image_from_rgba8(
            ORIENTATION_ICON_SIZE,
            generate_orientation_icon(ORIENTATION_ICON_SIZE),
        ));
        p.spawn(orientation_bundle(orientation));
    });
}

fn marker_bundle(kind: MarkerKind, variants: &MarkerVariants) -> impl Bundle {
    let icon_half = ICON_SIZE as f32 * 0.5;
    (
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(NAVBALL_CENTER_PX - icon_half),
            top: Val::Px(NAVBALL_CENTER_PX - icon_half),
            width: Val::Px(ICON_SIZE as f32),
            height: Val::Px(ICON_SIZE as f32),
            ..default()
        },
        ImageNode::new(variants.front.clone()),
        kind,
        variants.clone(),
        Visibility::Hidden,
        Name::new(format!("NavballMarker_{:?}", kind)),
    )
}

fn orientation_bundle(image: Handle<Image>) -> impl Bundle {
    let size = ORIENTATION_ICON_SIZE as f32;
    (
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(NAVBALL_CENTER_PX - size * 0.5),
            top: Val::Px(NAVBALL_CENTER_PX - size * 0.5),
            width: Val::Px(size),
            height: Val::Px(size),
            ..default()
        },
        ImageNode::new(image),
        ZIndex(1),
        Name::new("NavballOrientationMarker"),
    )
}

fn add_marker_image(kind: MarkerKind, occluded: bool, images: &mut Assets<Image>) -> Handle<Image> {
    let pixels = generate_marker_icon(kind, ICON_SIZE, occluded);
    images.add(image_from_rgba8(ICON_SIZE, pixels))
}

pub fn image_from_rgba8(size: u32, pixels: Vec<u8>) -> Image {
    Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}

/// Per-frame: project each marker's world-space direction onto the
/// navball, write its `Node` position, and swap to the occluded icon
/// when the direction is on the back hemisphere.
pub fn update_navball_markers(
    frame: Res<NavballFrame>,
    sim_state: Res<SimulationState>,
    target: Res<TargetBody>,
    plan: Res<ManeuverPlan>,
    mut markers: Query<(
        &MarkerKind,
        &MarkerVariants,
        &mut ImageNode,
        &mut Node,
        &mut Visibility,
    )>,
) {
    let directions = compute_marker_directions(&sim_state, &target, &plan);
    let icon_half = ICON_SIZE as f32 * 0.5;

    for (kind, variants, mut image, mut node, mut visibility) in &mut markers {
        let Some(d_world) = directions.for_kind(*kind) else {
            *visibility = Visibility::Hidden;
            continue;
        };

        let d_nav = frame.world_to_navball * d_world.as_vec3();
        let dx = d_nav.x * NAVBALL_DISPLAY_RADIUS_PX;
        let dy = d_nav.y * NAVBALL_DISPLAY_RADIUS_PX;
        node.left = Val::Px(NAVBALL_CENTER_PX + dx - icon_half);
        node.top = Val::Px(NAVBALL_CENTER_PX - dy - icon_half);

        let target_handle = if d_nav.z >= 0.0 {
            &variants.front
        } else {
            &variants.back
        };
        if image.image != *target_handle {
            image.image = target_handle.clone();
        }
        // Inherited (not Visible) so the navball root's Hidden state in
        // photo mode cascades to markers; Visible would override it.
        *visibility = Visibility::Inherited;
    }
}

struct MarkerDirections {
    prograde: Option<DVec3>,
    normal: Option<DVec3>,
    radial_out: Option<DVec3>,
    target_dir: Option<DVec3>,
    maneuver: Option<DVec3>,
}

impl MarkerDirections {
    fn for_kind(&self, kind: MarkerKind) -> Option<DVec3> {
        match kind {
            MarkerKind::Prograde => self.prograde,
            MarkerKind::Retrograde => self.prograde.map(|v| -v),
            MarkerKind::Normal => self.normal,
            MarkerKind::AntiNormal => self.normal.map(|v| -v),
            MarkerKind::RadialOut => self.radial_out,
            MarkerKind::RadialIn => self.radial_out.map(|v| -v),
            MarkerKind::ManeuverNode => self.maneuver,
            MarkerKind::Target => self.target_dir,
            MarkerKind::AntiTarget => self.target_dir.map(|v| -v),
        }
    }
}

fn compute_marker_directions(
    sim_state: &SimulationState,
    target: &TargetBody,
    plan: &ManeuverPlan,
) -> MarkerDirections {
    let sim = &sim_state.simulation;
    let craft = sim.craft_state();
    let sim_time = sim.sim_time();
    let soi_body_id = sim.dominant_body();
    let body_state = sim_state.ephemeris.state(soi_body_id, Epoch(sim_time));

    let rel_pos = craft.translation.position - body_state.position;
    let rel_vel = craft.translation.velocity - body_state.velocity;

    let prograde = rel_vel.try_normalize();
    let radial_out = rel_pos.try_normalize();
    let normal = rel_pos.cross(rel_vel).try_normalize();

    let target_dir = target.target.and_then(|target_id| {
        let state = sim_state.ephemeris.state(target_id, Epoch(sim_time));
        (state.position - craft.translation.position).try_normalize()
    });

    let maneuver = maneuver_burn_direction(sim, plan);

    MarkerDirections {
        prograde,
        normal,
        radial_out,
        target_dir,
        maneuver,
    }
}

// ---------------------------------------------------------------------------
// Procedural icon generation
// ---------------------------------------------------------------------------

pub fn generate_marker_icon(kind: MarkerKind, size: u32, occluded: bool) -> Vec<u8> {
    let mut pixels = vec![0u8; (size as usize) * (size as usize) * 4];
    let half = size as f32 * 0.5;
    let alpha_scale = if occluded { OCCLUDED_ALPHA } else { 1.0 };
    let color = kind.color();

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 + 0.5 - half;
            let dy = y as f32 + 0.5 - half;
            let r = (dx * dx + dy * dy).sqrt();

            let alpha = kind.coverage(dx, dy, r, half).clamp(0.0, 1.0) * alpha_scale;
            if alpha <= 0.0 {
                continue;
            }

            let i = ((y * size + x) * 4) as usize;
            pixels[i] = color[0];
            pixels[i + 1] = color[1];
            pixels[i + 2] = color[2];
            pixels[i + 3] = (alpha * 255.0) as u8;
        }
    }
    pixels
}

/// Prograde / radial-out: filled centre disc + 4 outward axis-aligned spokes.
/// Spokes reach near the icon edge for prominence at small render sizes.
fn filled_disc_with_outward_spokes(dx: f32, dy: f32, r: f32, half: f32) -> f32 {
    let disc_r = half * 0.28;
    let disc = soft_disc(r, disc_r);
    let spoke_inner = disc_r + 2.0;
    let spoke_outer = half * 0.96;
    let spoke = axis_spoke(dx, dy, r, spoke_inner, spoke_outer, 1.6);
    disc.max(spoke)
}

/// Retrograde / radial-in: hollow ring + 4-direction crosshair that
/// extends from inside the ring through it to slightly outside.
fn hollow_ring_with_crosshair(dx: f32, dy: f32, r: f32, half: f32) -> f32 {
    let ring_r = half * 0.42;
    let ring_thickness = 1.8;
    let ring = soft_ring(r, ring_r, ring_thickness);
    let spoke_inner = (ring_r - ring_thickness - 6.0).max(0.0);
    let spoke_outer = ring_r + ring_thickness + 6.0;
    let spoke = axis_spoke(dx, dy, r, spoke_inner, spoke_outer, 1.6);
    ring.max(spoke)
}

/// Filled triangular chevron with apex at top, wide base at bottom.
/// `chevron_up(dx, -dy, half)` flips it for the anti-normal variant.
fn chevron_up(dx: f32, dy: f32, half: f32) -> f32 {
    let apex_y = -half * 0.82;
    let base_y = half * 0.48;
    if dy < apex_y || dy > base_y {
        return 0.0;
    }
    let t = (dy - apex_y) / (base_y - apex_y);
    let half_width = half * 0.60 * t;
    smoothstep_falloff(half_width - dx.abs(), 1.0)
}

/// Maneuver node: small filled centre disc inside a hollow outer ring
/// (concentric). Visually distinct from prograde (no spokes) and
/// retrograde (a filled centre instead of a hollow one).
fn maneuver_target(dx: f32, dy: f32, r: f32, half: f32) -> f32 {
    let disc_r = half * 0.18;
    let disc = soft_disc(r, disc_r);
    let ring_r = half * 0.62;
    let ring_thickness = 1.8;
    let ring = soft_ring(r, ring_r, ring_thickness);
    disc.max(ring)
}

/// Maneuver node: filled disc + 4 diagonal spokes (offset 45° from axes
/// to read distinctly against prograde). Kept for reference; no longer
/// used.
#[allow(dead_code)]
fn filled_disc_with_diagonal_spokes(dx: f32, dy: f32, r: f32, half: f32) -> f32 {
    let disc_r = half * 0.34;
    let disc = soft_disc(r, disc_r);
    let spoke_inner = disc_r + 2.0;
    let spoke_outer = half * 0.95;
    if r < spoke_inner || r > spoke_outer {
        return disc;
    }
    let inv_sqrt2 = core::f32::consts::FRAC_1_SQRT_2;
    let d1 = (dx + dy).abs() * inv_sqrt2;
    let d2 = (dx - dy).abs() * inv_sqrt2;
    let spoke = smoothstep_falloff(1.2 - d1.min(d2), 1.0);
    disc.max(spoke)
}

/// Target / anti-target: square outline + inner `+` (target) or `×`
/// (anti-target). `anti=true` swaps the crosshair for an X.
fn target_square(dx: f32, dy: f32, half: f32, anti: bool) -> f32 {
    let outer = half * 0.72;
    let inner = half * 0.56;
    let max_axis = dx.abs().max(dy.abs());
    let frame = if max_axis <= outer && max_axis >= inner {
        let outer_edge = smoothstep_falloff(outer - max_axis, 1.0);
        let inner_edge = smoothstep_falloff(max_axis - inner, 1.0);
        outer_edge.min(inner_edge)
    } else {
        0.0
    };

    let cross_outer = inner * 0.92;
    let cross = if anti {
        if dx.abs() >= cross_outer || dy.abs() >= cross_outer {
            0.0
        } else {
            let inv_sqrt2 = core::f32::consts::FRAC_1_SQRT_2;
            let d1 = (dx + dy).abs() * inv_sqrt2;
            let d2 = (dx - dy).abs() * inv_sqrt2;
            smoothstep_falloff(1.6 - d1.min(d2), 1.0)
        }
    } else {
        let cross_inner = half * 0.04;
        let r = (dx * dx + dy * dy).sqrt();
        axis_spoke(dx, dy, r, cross_inner, cross_outer, 1.6)
    };
    frame.max(cross)
}

// ---------------------------------------------------------------------------
// Orientation indicator
// ---------------------------------------------------------------------------

const ORIENTATION_YELLOW: [u8; 3] = [255, 220, 60];

pub fn generate_orientation_icon(size: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (size as usize) * (size as usize) * 4];
    let half = size as f32 * 0.5;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 + 0.5 - half;
            let dy = y as f32 + 0.5 - half;
            let alpha = orientation_coverage(dx, dy, half).clamp(0.0, 1.0);
            if alpha <= 0.0 {
                continue;
            }
            let i = ((y * size + x) * 4) as usize;
            pixels[i] = ORIENTATION_YELLOW[0];
            pixels[i + 1] = ORIENTATION_YELLOW[1];
            pixels[i + 2] = ORIENTATION_YELLOW[2];
            pixels[i + 3] = (alpha * 255.0) as u8;
        }
    }
    pixels
}

fn orientation_coverage(dx: f32, dy: f32, half: f32) -> f32 {
    // Top-down aircraft silhouette:
    //   - vertical fuselage (thin) running most of the icon height
    //   - main wings (wide horizontal bar) at the upper third
    //   - tail wing (shorter horizontal bar) near the bottom
    //   - cockpit dot at the nose
    let fuselage_top = -half * 0.72;
    let fuselage_bottom = half * 0.62;
    let fuselage_half_width = 2.0;
    let fuselage = if dy >= fuselage_top && dy <= fuselage_bottom {
        smoothstep_falloff(fuselage_half_width - dx.abs(), 1.0)
    } else {
        0.0
    };

    let main_wing_y = -half * 0.08;
    let main_wing_half_height = 2.2;
    let main_wing_span = half * 0.85;
    let main_wing = if dx.abs() <= main_wing_span {
        smoothstep_falloff(main_wing_half_height - (dy - main_wing_y).abs(), 1.0)
    } else {
        0.0
    };

    let tail_wing_y = half * 0.48;
    let tail_wing_half_height = 1.8;
    let tail_wing_span = half * 0.32;
    let tail_wing = if dx.abs() <= tail_wing_span {
        smoothstep_falloff(tail_wing_half_height - (dy - tail_wing_y).abs(), 1.0)
    } else {
        0.0
    };

    // Cockpit/nose dot — sits at the front of the fuselage.
    let nose_dy = dy - (-half * 0.62);
    let cockpit = soft_disc((dx * dx + nose_dy * nose_dy).sqrt(), half * 0.09);

    fuselage.max(main_wing).max(tail_wing).max(cockpit)
}

// ---------------------------------------------------------------------------
// Coverage helpers
// ---------------------------------------------------------------------------

fn soft_disc(r: f32, r0: f32) -> f32 {
    smoothstep_falloff(r0 - r, 1.0)
}

fn soft_ring(r: f32, r0: f32, thickness: f32) -> f32 {
    smoothstep_falloff(thickness - (r - r0).abs(), 1.0)
}

fn axis_spoke(dx: f32, dy: f32, r: f32, inner: f32, outer: f32, half_width: f32) -> f32 {
    if r < inner || r > outer {
        return 0.0;
    }
    let edge_x = smoothstep_falloff(half_width - dx.abs(), 1.0);
    let edge_y = smoothstep_falloff(half_width - dy.abs(), 1.0);
    edge_x.max(edge_y)
}

/// Coverage ramp used for antialiasing all shape edges. `d` is the
/// signed distance from the shape's edge (positive = inside), `soft` is
/// the half-width of the smooth ramp in pixels. Uses Hermite smoothstep
/// (3t² − 2t³) rather than a linear ramp for visibly smoother edges.
fn smoothstep_falloff(d: f32, soft: f32) -> f32 {
    let t = ((d + soft) / (2.0 * soft)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
