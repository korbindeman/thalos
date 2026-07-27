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

use super::{SpaceCenter, hub_context};

pub(super) struct SpaceCenterCameraPlugin;

impl Plugin for SpaceCenterCameraPlugin {
    fn build(&self, app: &mut App) {
        // `GodViewOrbit` and the shadow-focus clear live in `god_view::GodViewPlugin`.
        // `reset_orbit_on_open` runs *every* frame (not gated on `space_center_open`)
        // so it can observe the closed state and fire on the closed→open edge;
        // `drive_camera` is chained after it and gated to run only while open.
        app.add_systems(
            Update,
            (
                reset_orbit_on_open,
                drive_camera
                    .run_if(hub_owns_camera)
                    .in_set(god_view::GodViewCameraSet),
            )
                .chain(),
        );
    }
}

/// Run condition for [`drive_camera`]: the hub is open **and** we are not in
/// headless-screenshot mode. When `ScreenshotConfig` is present the capture
/// driver ([`crate::screenshot`]) owns the camera pose — running the hub drive
/// too would race it (ambiguous Update ordering) for the same `ShipCamera`
/// transform.
fn hub_owns_camera(
    sc: Option<Res<SpaceCenter>>,
    screenshot: Option<Res<crate::screenshot::ScreenshotConfig>>,
) -> bool {
    screenshot.is_none() && sc.map(|s| s.open).unwrap_or(false)
}

/// Fresh establishing view whenever the hub *opens* (the closed→open edge).
///
/// Edge-detected with a `Local` rather than `sc.is_changed()`: the hover picker
/// writes `SpaceCenter::hovered` every frame the cursor moves over the base
/// (`select.rs`), which also trips `is_changed()`. Keying the reset off that
/// would snap the camera back to the default pad-centred establishing view
/// constantly — discarding the user's orbit/zoom/pan — when hovering a building
/// should leave the camera untouched.
fn reset_orbit_on_open(
    sc: Res<SpaceCenter>,
    mut orbit: ResMut<GodViewOrbit>,
    mut was_open: Local<bool>,
) {
    if sc.open && !*was_open {
        // The one canonical god-view framing over a base (`god_view`), shared
        // with the base editor and the launch-point picker.
        orbit.reset_over_base();
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
    ui_keyboard: Res<crate::hud::UiKeyboardGate>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time<Real>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut orbit: ResMut<GodViewOrbit>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut camera: Query<(&mut Transform, &mut CellCoord), With<ShipCamera>>,
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
            text_entry: ui_keyboard.text_entry(),
            orbit_held: mouse_buttons.pressed(MouseButton::Right),
            drag,
            scroll,
            dt: time.delta_secs(),
        },
        root_grid,
        &mut transform,
        &mut cell,
    );

    // Periodic JSONL snapshot of where the god-view is pointed vs where the
    // ship actually is.
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
            target: "thalos::diagnostic::space_center",
            event = "hub_camera",
            has_base,
            pad_radius_m = ctx.pad_r,
            camera_cell = ?*cell,
            camera_local = ?transform.translation,
            ?alt_km,
            "space-center camera gauge"
        );
    }
}
