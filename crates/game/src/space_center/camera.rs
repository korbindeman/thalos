//! God-view camera for the space-center hub.
//!
//! Identical treatment to the base editor: gather raw input, resolve the hub's
//! focus ([`super::hub_context`]), and hand both to the shared driver
//! ([`crate::god_view::drive_god_view`]), which repositions the ship camera
//! around the spaceport. Ungated by `SimStage` so it runs while the hub pauses
//! the sim; the flight camera systems are gated off (`space_center_closed` in
//! `main.rs`) so they don't fight it.

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_physics_local::HeightSourceRegistry;

use crate::camera::ShipCamera;
use crate::god_view::{self, GodViewFocus, GodViewInput, GodViewOrbit};
use crate::rendering::{SimulationState, SolarSystemState};
use crate::spawn::Homeworld;
use crate::structures::StructureRegistry;

use super::{SpaceCenter, hub_context, space_center_open};

pub(super) struct SpaceCenterCameraPlugin;

impl Plugin for SpaceCenterCameraPlugin {
    fn build(&self, app: &mut App) {
        // `GodViewOrbit` and the shadow-focus clear live in `god_view::GodViewPlugin`.
        // `reset_orbit_on_open` runs *every* frame (not gated on `space_center_open`)
        // so it can observe the closed state and fire on the closed→open edge;
        // `drive_camera` is chained after it and gated to run only while open.
        app.add_systems(
            Update,
            (reset_orbit_on_open, drive_camera.run_if(space_center_open)).chain(),
        );
    }
}

/// Fresh establishing view whenever the hub *opens* (the closed→open edge).
///
/// Edge-detected with a `Local` rather than `sc.is_changed()`: the hover picker
/// writes `SpaceCenter::hovered` every frame the cursor moves over the base
/// (`select.rs`), which also trips `is_changed()`. Keying the reset off that
/// would snap the camera back to the default pad-centred establishing view
/// constantly — discarding the user's orbit/zoom/pan — when hovering a building
/// should leave the camera untouched.
/// Initial god-view boom for the hub, metres. The kilometre-scale spaceport
/// (a 5 km runway basin) needs a wide establishing shot — the shared god-view
/// default (`DEFAULT_DISTANCE_M`, ~500 m) frames only a corner of it, deep inside
/// the tree-cleared basin, so PLAY looked bare. Matches the headless
/// spaceport-aerial screenshot framing.
const HUB_ESTABLISHING_DISTANCE_M: f32 = 4000.0;

fn reset_orbit_on_open(
    sc: Res<SpaceCenter>,
    mut orbit: ResMut<GodViewOrbit>,
    mut was_open: Local<bool>,
) {
    if sc.open && !*was_open {
        orbit.reset_framed(HUB_ESTABLISHING_DISTANCE_M);
    }
    *was_open = sc.open;
}

#[allow(clippy::too_many_arguments)]
fn drive_camera(
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    height_sources: Res<HeightSourceRegistry>,
    registry: Res<StructureRegistry>,
    homeworld: Res<Homeworld>,
    ui_gate: Res<crate::hud::UiPointerGate>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time<Real>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut orbit: ResMut<GodViewOrbit>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut camera: Query<(&mut Transform, &mut CellCoord), With<ShipCamera>>,
    mut shadow_focus: ResMut<crate::rendering::sun_shadow::ShadowFocusOverride>,
    mut diag_frames: Local<u32>,
) {
    // Always drain the input streams so they don't pile up across frames.
    let drag: Vec2 = motion.read().map(|m| m.delta).sum();
    let scroll: f32 = wheel.read().map(|w| w.y).sum();

    let Some(ctx) = hub_context(&sim, &solar, &height_sources, &registry, homeworld.0) else {
        warn!(target: "thalos::space_center", "hub camera: no focus (body state not ready)");
        return;
    };
    let Ok(root_grid) = root_grid.single() else {
        warn!(target: "thalos::space_center", "hub camera: no BigSpace root grid");
        return;
    };
    let Ok((mut transform, mut cell)) = camera.single_mut() else {
        warn!(target: "thalos::space_center", "hub camera: no ShipCamera");
        return;
    };

    god_view::drive_god_view(
        GodViewFocus {
            center_world: ctx.center_world,
            up_world: ctx.up_world,
        },
        &mut orbit,
        &keys,
        GodViewInput {
            over_ui: ui_gate.hovered,
            orbit_held: mouse_buttons.pressed(MouseButton::Right),
            drag,
            scroll,
            dt: time.delta_secs(),
        },
        root_grid,
        &mut transform,
        &mut cell,
        &mut shadow_focus,
    );

    // DIAGNOSTIC (remove once the hub view is verified): periodic snapshot of
    // where the god-view is pointed vs where the ship actually is.
    *diag_frames += 1;
    if *diag_frames % 90 == 1 {
        let body_id = homeworld.0;
        let has_base = super::home_base_site(&registry, body_id).is_some();
        let ship = sim.simulation.ship_state().position;
        let alt_km = solar
            .states
            .as_deref()
            .and_then(|s| s.get(body_id))
            .map(|b| ((ship - b.position).length() - ctx.pad_r) / 1000.0);
        info!(
            target: "thalos::space_center",
            "hub view: base_site={} pad_r={:.0}m cam_cell={:?} cam_local={:.0?} ship_alt_km={:?}",
            has_base, ctx.pad_r, *cell, transform.translation, alt_km
        );
    }
}
