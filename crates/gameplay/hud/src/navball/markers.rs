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

use crate::navball::attitude::NavballFrame;
use crate::navball::ui::NavballUiRoot;
use crate::velocity_frame::VelocityFrameState;
use bevy::asset::RenderAssetUsages;
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use resvg::{tiny_skia, usvg};
use thalos_game_state::maneuver_plan::ManeuverPlan;
use thalos_game_state::nav::TargetBody;
use thalos_game_state::nav::maneuver_burn_direction;
use thalos_game_state::{SimulationState, SolarSystemState};
use thalos_physics_canonical::velocity_frame::{VelocityReferenceFrame, nav_basis};

const ICON_SIZE: u32 = 32;
const ORIENTATION_ICON_WIDTH: u32 = 40;
const ORIENTATION_ICON_HEIGHT: u32 = 16;

/// Navball image radius and centre in UI pixels. Derived from the UI root
/// size and the off-screen camera's `ScalingMode::Fixed { 2.4 }` — read from
/// [`NAVBALL_SIZE_PX`] rather than restated, so resizing the navball carries
/// the markers with it instead of sliding them off the ball.
const NAVBALL_DISPLAY_RADIUS_PX: f32 = crate::navball::ui::NAVBALL_SIZE_PX / 2.4;
const NAVBALL_CENTER_PX: f32 = crate::navball::ui::NAVBALL_SIZE_PX * 0.5;

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
    pub(crate) const ALL: [Self; 9] = [
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
            Self::Prograde | Self::Retrograde => [0xD7, 0xFE, 0x00],
            Self::Normal | Self::AntiNormal | Self::Target | Self::AntiTarget => [0xD6, 0x00, 0xD6],
            Self::RadialOut | Self::RadialIn => [0x00, 0xD6, 0xD6],
            Self::ManeuverNode => [0x00, 0x00, 0xD6],
        }
    }

    fn svg_bytes(self) -> &'static [u8] {
        match self {
            Self::Prograde => {
                include_bytes!("../../../../../assets/markers/navigation/prograde.svg")
            }
            Self::Retrograde => {
                include_bytes!("../../../../../assets/markers/navigation/retrograde.svg")
            }
            Self::Normal => include_bytes!("../../../../../assets/markers/navigation/normal.svg"),
            Self::AntiNormal => {
                include_bytes!("../../../../../assets/markers/navigation/anti-normal.svg")
            }
            Self::RadialOut => {
                include_bytes!("../../../../../assets/markers/navigation/radial-out.svg")
            }
            Self::RadialIn => {
                include_bytes!("../../../../../assets/markers/navigation/radial-in.svg")
            }
            Self::ManeuverNode => {
                include_bytes!("../../../../../assets/markers/navigation/maneuver.svg")
            }
            Self::Target => {
                include_bytes!("../../../../../assets/markers/navigation/target-prograde.svg")
            }
            Self::AntiTarget => {
                include_bytes!("../../../../../assets/markers/navigation/target-retrograde.svg")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerIconState {
    Visible,
    Occluded,
    Disabled,
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
                front: add_marker_image(kind, MarkerIconState::Visible, &mut images),
                back: add_marker_image(kind, MarkerIconState::Occluded, &mut images),
            };
            p.spawn(marker_bundle(kind, &variants));
        }

        let orientation = images.add(orientation_icon_image(
            ORIENTATION_ICON_WIDTH,
            ORIENTATION_ICON_HEIGHT,
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
    let width = ORIENTATION_ICON_WIDTH as f32;
    let height = ORIENTATION_ICON_HEIGHT as f32;
    (
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(NAVBALL_CENTER_PX - width * 0.5),
            top: Val::Px(NAVBALL_CENTER_PX - height * 0.5),
            width: Val::Px(width),
            height: Val::Px(height),
            ..default()
        },
        ImageNode::new(image),
        ZIndex(1),
        Name::new("NavballOrientationMarker"),
    )
}

fn add_marker_image(
    kind: MarkerKind,
    state: MarkerIconState,
    images: &mut Assets<Image>,
) -> Handle<Image> {
    images.add(marker_icon_image(kind, ICON_SIZE, state))
}

pub fn image_from_rgba8(width: u32, height: u32, pixels: Vec<u8>) -> Image {
    Image::new(
        Extent3d {
            width,
            height,
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
    solar_system: Res<SolarSystemState>,
    target: Res<TargetBody>,
    plan: thalos_game_state::ActiveCraftRef<ManeuverPlan>,
    velocity_frame: Res<VelocityFrameState>,
    mut markers: Query<(
        &MarkerKind,
        &MarkerVariants,
        &mut ImageNode,
        &mut Node,
        &mut Visibility,
    )>,
) {
    let Some(plan) = plan.get() else {
        return;
    };
    let directions = compute_marker_directions(
        velocity_frame.active,
        &sim_state,
        &solar_system,
        &target,
        plan,
    );
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

pub(crate) struct MarkerDirections {
    prograde: Option<DVec3>,
    normal: Option<DVec3>,
    radial_out: Option<DVec3>,
    target_dir: Option<DVec3>,
    maneuver: Option<DVec3>,
}

impl MarkerDirections {
    pub(crate) fn for_kind(&self, kind: MarkerKind) -> Option<DVec3> {
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

/// World-space directions for every marker kind in the active velocity
/// frame. Shared by the navball overlay and the PFD HUD mode
/// (`hud::pfd_panel`).
pub(crate) fn compute_marker_directions(
    active: VelocityReferenceFrame,
    sim_state: &SimulationState,
    solar_system: &SolarSystemState,
    target: &TargetBody,
    plan: &ManeuverPlan,
) -> MarkerDirections {
    let sim = &sim_state.simulation;
    let craft = sim.ship_state();
    let soi_body_id = sim.dominant_body();
    let maneuver = maneuver_burn_direction(sim, plan);

    let Some(states) = solar_system.states.as_ref() else {
        return MarkerDirections {
            prograde: None,
            normal: None,
            radial_out: None,
            target_dir: None,
            maneuver,
        };
    };
    let Some(body_state) = states.get(soi_body_id) else {
        return MarkerDirections {
            prograde: None,
            normal: None,
            radial_out: None,
            target_dir: None,
            maneuver,
        };
    };

    let target_state = target.target.and_then(|id| states.get(id));

    // Prograde / normal / radial follow the active velocity frame.
    let basis = nav_basis(active, craft, body_state, target_state);

    // The pink Target / AntiTarget markers point AT the target (a
    // direction-to), independent of the velocity frame.
    let target_dir =
        target_state.and_then(|state| (state.position - craft.position).try_normalize());

    MarkerDirections {
        prograde: basis.and_then(|b| b.prograde),
        normal: basis.and_then(|b| b.normal),
        radial_out: basis.and_then(|b| b.radial),
        target_dir,
        maneuver,
    }
}

// ---------------------------------------------------------------------------
// SVG icon rendering
// ---------------------------------------------------------------------------

pub fn marker_icon_image(kind: MarkerKind, size: u32, state: MarkerIconState) -> Image {
    let pixels = render_svg_rgba8(kind.svg_bytes(), size, size, state);
    image_from_rgba8(size, size, pixels)
}

/// The ship-orientation ("level indicator") icon; also the PFD boresight.
pub(crate) fn orientation_icon_image(width: u32, height: u32) -> Image {
    let pixels = render_svg_rgba8(
        include_bytes!("../../../../../assets/markers/navigation/level-indicator.svg"),
        width,
        height,
        MarkerIconState::Visible,
    );
    image_from_rgba8(width, height, pixels)
}

fn render_svg_rgba8(svg: &[u8], width: u32, height: u32, state: MarkerIconState) -> Vec<u8> {
    let opt = usvg::Options::default();
    let tree = usvg::Tree::from_data(svg, &opt).expect("bundled navigation marker SVG must parse");
    let mut pixmap =
        tiny_skia::Pixmap::new(width, height).expect("marker dimensions must be valid");

    let svg_size = tree.size();
    let scale = (width as f32 / svg_size.width()).min(height as f32 / svg_size.height());
    let tx = (width as f32 - svg_size.width() * scale) * 0.5;
    let ty = (height as f32 - svg_size.height() * scale) * 0.5;
    let transform = tiny_skia::Transform {
        sx: scale,
        sy: scale,
        tx,
        ty,
        ..default()
    };

    resvg::render(&tree, transform, &mut pixmap.as_mut());

    let mut pixels = pixmap.data().to_vec();
    demultiply_rgba8(&mut pixels);
    apply_icon_state(&mut pixels, state);
    pixels
}

fn demultiply_rgba8(pixels: &mut [u8]) {
    for px in pixels.chunks_exact_mut(4) {
        let alpha = px[3] as u16;
        if alpha == 0 || alpha == 255 {
            continue;
        }
        px[0] = ((px[0] as u16 * 255 + alpha / 2) / alpha).min(255) as u8;
        px[1] = ((px[1] as u16 * 255 + alpha / 2) / alpha).min(255) as u8;
        px[2] = ((px[2] as u16 * 255 + alpha / 2) / alpha).min(255) as u8;
    }
}

fn apply_icon_state(pixels: &mut [u8], state: MarkerIconState) {
    match state {
        MarkerIconState::Visible => {}
        MarkerIconState::Occluded => {
            for px in pixels.chunks_exact_mut(4) {
                px[3] = (px[3] as f32 * OCCLUDED_ALPHA).round() as u8;
            }
        }
        MarkerIconState::Disabled => {
            for px in pixels.chunks_exact_mut(4) {
                let alpha = px[3] as f32 / 255.0;
                if alpha <= 0.0 {
                    continue;
                }
                let luminance = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
                let grey = (50.0 + luminance * 0.35).clamp(0.0, 135.0) as u8;
                px[0] = grey;
                px[1] = grey;
                px[2] = grey;
                px[3] = (px[3] as f32 * 0.58).round() as u8;
            }
        }
    }
}
