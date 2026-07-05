//! Hover-based building picker for the space-center hub.
//!
//! Each frame the cursor is raycast against the pad sphere and the nearest
//! hoverable structure under it becomes [`SpaceCenter::hovered`]. The hovered
//! structure is outlined with a coloured gizmo silhouette and a floating callout
//! shows its name; a **left-click** on an enterable [`Facility`] building (the
//! VAB) enters it. Non-facility structures (launchpads, tanks) still highlight
//! and label — you can read what they are — but there is nothing to enter yet.
//! This is the generic seam future facilities plug into via the
//! [`Facility`](crate::structures::Facility) tag; only the VAB is wired today.

use bevy::picking::prelude::Pickable;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::base_editor::ray_vs_sphere_dir;
use crate::camera::{ActiveCamera, ShipCamera};
use crate::coords::SHIP_SCALE;
use crate::god_view::GodViewGizmos;
use crate::hud::theme::HudTheme;
use crate::rendering::{RealSpaceBody, SimulationState, SolarSystemState};
use crate::shipyard_editor::ShipyardEditor;
use crate::spawn::Homeworld;
use crate::structures::{StructureId, StructureKind, StructurePlacement, StructureRegistry, StructureSite};
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use super::{
    ReturnToSpaceCenter, SpaceCenter, enter_facility, hub_context, kind_name, selectable_bound,
    space_center_open,
};

pub(super) struct SpaceCenterSelectPlugin;

impl Plugin for SpaceCenterSelectPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_hover_label.after(crate::hud::theme::init_theme))
            .add_systems(
                Update,
                (hover_and_enter, draw_hover).chain().run_if(space_center_open),
            )
            // Ungated: it also *hides* the callout when the hub closes (or the
            // cursor leaves every building), so it must run even while closed.
            .add_systems(Update, update_hover_label);
    }
}

/// The warm accent used to outline / label an enterable facility building.
const ENTERABLE_COLOR: Color = Color::srgb(1.0, 0.82, 0.28);
/// The cool tint used for a hoverable-but-not-enterable structure.
const PLAIN_COLOR: Color = Color::srgb(0.62, 0.82, 1.0);

/// Update [`SpaceCenter::hovered`] from the cursor and, on a left-click over an
/// enterable facility, enter it.
#[allow(clippy::too_many_arguments)]
fn hover_and_enter(
    mut sc: ResMut<SpaceCenter>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    height_sources: Res<HeightSourceRegistry>,
    registry: Res<StructureRegistry>,
    homeworld: Res<Homeworld>,
    ui_gate: Res<crate::hud::UiPointerGate>,
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), (With<ShipCamera>, With<ActiveCamera>)>,
    bodies: Query<(&RealSpaceBody, &GlobalTransform)>,
    mut shipyard: ResMut<ShipyardEditor>,
    mut return_flag: ResMut<ReturnToSpaceCenter>,
) {
    // Over a UI panel → no world hover (so the callout + highlight vanish and a
    // click on the panel isn't also read as a click on the ground).
    let hovered = if ui_gate.hovered {
        None
    } else {
        compute_hover(
            &sim,
            &solar,
            &height_sources,
            &registry,
            homeworld.0,
            &windows,
            &cameras,
            &bodies,
        )
    };
    if sc.hovered != hovered {
        sc.hovered = hovered;
    }

    // Click an enterable facility → enter it.
    if mouse.just_pressed(MouseButton::Left)
        && let Some(id) = hovered
        && let Some(facility) = registry.get(id).and_then(|s| s.facility)
    {
        enter_facility(facility, &mut sc, &mut shipyard, &mut return_flag);
    }
}

/// Raycast the cursor against the pad sphere and return the nearest hoverable
/// structure under it (a building / launchpad / tank — the runway and invisible
/// base site are skipped), or `None`.
#[allow(clippy::too_many_arguments)]
fn compute_hover(
    sim: &SimulationState,
    solar: &SolarSystemState,
    height_sources: &HeightSourceRegistry,
    registry: &StructureRegistry,
    body_id: BodyId,
    windows: &Query<&Window, With<PrimaryWindow>>,
    cameras: &Query<(&Camera, &GlobalTransform), (With<ShipCamera>, With<ActiveCamera>)>,
    bodies: &Query<(&RealSpaceBody, &GlobalTransform)>,
) -> Option<StructureId> {
    let ctx = hub_context(sim, solar, height_sources, registry, body_id)?;
    let states = solar.states.as_deref()?;
    let body_state = states.get(ctx.body_id)?;
    let window = windows.single().ok()?;
    let cursor = window.cursor_position()?;
    let (camera, cam_gt) = cameras.single().ok()?;
    let (_, body_gt) = bodies.iter().find(|(rsb, _)| rsb.body_id == ctx.body_id)?;
    let center_render = body_gt.translation();
    let ray = camera.viewport_to_world(cam_gt, cursor).ok()?;
    let dir_render = ray_vs_sphere_dir(
        ray.origin - center_render,
        *ray.direction,
        (ctx.pad_r * SHIP_SCALE) as f32,
    )?;
    let dir_body = (body_state.orientation.inverse() * dir_render.as_dvec3()).normalize();

    // Nearest hoverable structure to the click point on the pad.
    let mut best: Option<(StructureId, f64)> = None;
    for site in registry.sites_on(ctx.body_id) {
        let Some(bound) = selectable_bound(&site.kind) else {
            continue;
        };
        let ang = site
            .anchor_dir
            .normalize()
            .dot(dir_body)
            .clamp(-1.0, 1.0)
            .acos();
        let dist = ang * ctx.pad_r;
        if dist <= bound && best.is_none_or(|(_, d)| dist < d) {
            best = Some((site.id, dist));
        }
    }
    best.map(|(id, _)| id)
}

/// Outline the hovered structure with a coloured gizmo silhouette — warm for an
/// enterable facility, cool otherwise.
fn draw_hover(
    sc: Res<SpaceCenter>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    registry: Res<StructureRegistry>,
    bodies: Query<(&RealSpaceBody, &GlobalTransform)>,
    mut gizmos: Gizmos<GodViewGizmos>,
) {
    let Some(id) = sc.hovered else {
        return;
    };
    let Some(site) = registry.get(id) else {
        return;
    };
    let Some(frame) = structure_render_frame(site, &sim, &solar, &registry, &bodies) else {
        return;
    };
    let color = if site.facility.is_some() {
        ENTERABLE_COLOR
    } else {
        PLAIN_COLOR
    };
    draw_structure_outline(&mut gizmos, &site.kind, &frame, color);
}

/// The render-space frame under a structure: the ground-flush centre plus the
/// local up/heading/across unit axes. Everything the outline + label anchor need.
struct RenderFrame {
    /// Ground-flush centre of the structure, big_space render space.
    base_render: Vec3,
    up: Vec3,
    heading: Vec3,
    across: Vec3,
}

/// Resolve a structure's [`RenderFrame`]. Buildings drape on their parent site's
/// flattened pad, so the frame sits at the pad elevation.
fn structure_render_frame(
    site: &StructureSite,
    sim: &SimulationState,
    solar: &SolarSystemState,
    registry: &StructureRegistry,
    bodies: &Query<(&RealSpaceBody, &GlobalTransform)>,
) -> Option<RenderFrame> {
    let states = solar.states.as_deref()?;
    let body_state = states.get(site.body_id)?;
    let body = sim.system.bodies.get(site.body_id)?;
    let (_, body_gt) = bodies.iter().find(|(rsb, _)| rsb.body_id == site.body_id)?;

    let elevation_m = site
        .parent_site
        .and_then(|p| registry.get(p))
        .map(|p| match p.placement {
            StructurePlacement::FlattenTo { elevation_m, .. } => elevation_m,
            StructurePlacement::Drape => 0.0,
        })
        .unwrap_or(0.0);
    let pad_r = body.radius_m + elevation_m;
    let orientation = body_state.orientation.normalize();
    let up = (orientation * site.anchor_dir).as_vec3().normalize();
    let heading = (orientation * site.heading_tangent).as_vec3().normalize();
    let across = heading.cross(up).normalize();
    let base_render =
        body_gt.translation() + (orientation * (site.anchor_dir * pad_r)).as_vec3() * SHIP_SCALE as f32;
    Some(RenderFrame {
        base_render,
        up,
        heading,
        across,
    })
}

/// Draw the hover outline for `kind` in the structure's [`RenderFrame`]: a box
/// silhouette for a building, a footprint ring (or ringed cylinder) otherwise.
fn draw_structure_outline(
    gizmos: &mut Gizmos<GodViewGizmos>,
    kind: &StructureKind,
    f: &RenderFrame,
    color: Color,
) {
    let s = SHIP_SCALE as f32;
    match *kind {
        StructureKind::Building {
            half_x_m,
            half_z_m,
            height_m,
        } => {
            let center = f.base_render + f.up * (height_m * 0.5 * s);
            draw_box(
                gizmos,
                center,
                f.heading * (half_x_m * s),
                f.up * (height_m * 0.5 * s),
                f.across * (half_z_m * s),
                color,
            );
        }
        StructureKind::Launchpad { radius_m } => {
            draw_ground_ring(gizmos, f.base_render, f.up, (radius_m + 1.0) * s, color);
        }
        StructureKind::Tank { radius_m, height_m } => {
            let r = (radius_m + 0.5) * s;
            draw_ground_ring(gizmos, f.base_render, f.up, r, color);
            draw_ground_ring(gizmos, f.base_render + f.up * (height_m * s), f.up, r, color);
        }
        _ => {}
    }
}

/// Wireframe box: half-axis vectors `hx`/`hy`/`hz` about `center` (already in
/// render space, so no rotation needed).
fn draw_box(
    gizmos: &mut Gizmos<GodViewGizmos>,
    center: Vec3,
    hx: Vec3,
    hy: Vec3,
    hz: Vec3,
    color: Color,
) {
    let corner = |sx: f32, sy: f32, sz: f32| center + hx * sx + hy * sy + hz * sz;
    let c = [
        corner(-1.0, -1.0, -1.0),
        corner(1.0, -1.0, -1.0),
        corner(1.0, 1.0, -1.0),
        corner(-1.0, 1.0, -1.0),
        corner(-1.0, -1.0, 1.0),
        corner(1.0, -1.0, 1.0),
        corner(1.0, 1.0, 1.0),
        corner(-1.0, 1.0, 1.0),
    ];
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    for (a, b) in edges {
        gizmos.line(c[a], c[b], color);
    }
}

/// Draw a horizontal ring of `radius` centred at `center`, lying in the tangent
/// plane whose normal is `up`.
fn draw_ground_ring(
    gizmos: &mut Gizmos<GodViewGizmos>,
    center: Vec3,
    up: Vec3,
    radius: f32,
    color: Color,
) {
    const SEGS: usize = 48;
    let rot = Quat::from_rotation_arc(Vec3::Y, up);
    let point = |a: f32| center + rot * (Vec3::new(a.cos(), 0.0, a.sin()) * radius);
    let mut prev = point(0.0);
    for i in 1..=SEGS {
        let a = i as f32 / SEGS as f32 * std::f32::consts::TAU;
        let p = point(a);
        gizmos.line(prev, p, color);
        prev = p;
    }
}

/// Height (m) used to lift the hover callout above a structure.
fn structure_height_m(kind: &StructureKind) -> f32 {
    match *kind {
        StructureKind::Building { height_m, .. } | StructureKind::Tank { height_m, .. } => height_m,
        _ => 2.0,
    }
}

/// The floating hover callout chip (a single one, reused for whatever is hovered).
#[derive(Component)]
struct HoverLabel;

/// The callout's text child (retargeted to the hovered structure's name).
#[derive(Component)]
struct HoverLabelText;

/// Spawn the (hidden) hover callout once at startup.
fn setup_hover_label(mut commands: Commands, theme: Res<HudTheme>) {
    commands
        .spawn((
            HoverLabel,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(0.0),
                padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            GlobalZIndex(85),
            Visibility::Hidden,
            // Never a pointer sink — it must not trip `UiPointerGate` (it carries
            // no `Interaction`, so it doesn't) nor block the ground pick raycast.
            Pickable::IGNORE,
            Name::new("SpaceCenterHoverLabel"),
        ))
        .with_children(|c| {
            c.spawn((
                HoverLabelText,
                Text::new(""),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(theme.text_primary),
            ));
        });
}

/// Position + fill the hover callout from [`SpaceCenter::hovered`], or hide it
/// when nothing is hovered (or the hub is closed, or the anchor is off-screen).
#[allow(clippy::too_many_arguments)]
fn update_hover_label(
    sc: Res<SpaceCenter>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    registry: Res<StructureRegistry>,
    theme: Res<HudTheme>,
    ui_scale: Res<UiScale>,
    cameras: Query<(&Camera, &GlobalTransform), (With<ShipCamera>, With<ActiveCamera>)>,
    bodies: Query<(&RealSpaceBody, &GlobalTransform)>,
    mut label_root: Query<(&mut Node, &mut Visibility), With<HoverLabel>>,
    mut label_text: Query<(&mut Text, &mut TextColor), With<HoverLabelText>>,
) {
    let Ok((mut node, mut vis)) = label_root.single_mut() else {
        return;
    };
    let hide = |vis: &mut Visibility| {
        if *vis != Visibility::Hidden {
            *vis = Visibility::Hidden;
        }
    };

    // Resolve the hovered site (only while the hub is open).
    let Some(site) = sc
        .open
        .then_some(sc.hovered)
        .flatten()
        .and_then(|id| registry.get(id).copied())
    else {
        hide(&mut vis);
        return;
    };

    // Project the callout anchor (top of the structure) to the screen.
    let Some(frame) = structure_render_frame(&site, &sim, &solar, &registry, &bodies) else {
        hide(&mut vis);
        return;
    };
    let top_render = frame.base_render + frame.up * (structure_height_m(&site.kind) * SHIP_SCALE as f32);
    let Ok((camera, cam_gt)) = cameras.single() else {
        hide(&mut vis);
        return;
    };
    let Ok(screen) = camera.world_to_viewport(cam_gt, top_render) else {
        hide(&mut vis); // behind the camera / off-viewport
        return;
    };

    // `world_to_viewport` yields window-logical px; `Node` left/top are UI-logical
    // (Bevy multiplies them by `UiScale`), so divide it out — the star-flare fix.
    let inv_ui = 1.0 / ui_scale.0.max(1.0e-6);
    let pos = screen * inv_ui;
    node.left = Val::Px(pos.x + 12.0);
    node.top = Val::Px(pos.y - 26.0);
    if *vis != Visibility::Inherited {
        *vis = Visibility::Inherited;
    }

    // Text + colour: warm accent + a click hint for an enterable facility.
    let (name, enterable) = match site.facility {
        Some(f) => (f.label().to_string(), true),
        None => (kind_name(&site.kind).to_string(), false),
    };
    let display = if enterable {
        format!("{name}  ·  click to enter")
    } else {
        name
    };
    if let Ok((mut text, mut color)) = label_text.single_mut() {
        if text.0 != display {
            **text = display;
        }
        let target = if enterable {
            theme.text_accent
        } else {
            theme.text_primary
        };
        if color.0 != target {
            color.0 = target;
        }
    }
}
