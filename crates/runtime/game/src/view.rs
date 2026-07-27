//! View mode: map (far-scale orbit view) vs. ship (1:1 scale ship view).
//!
//! M toggles between the two (within [`GameContext::Flight`]). Each view has its
//! own camera entity with a fixed `RenderLayers` set. Which camera is active is
//! owned by the **single authority** [`apply_active_camera`] — a pure function of
//! `(GameContext, ViewMode)` that also selects the shipyard editor camera in
//! [`GameContext::Vab`](crate::game_context::GameContext) and holds the ship
//! camera for the hub / base editor (their god-view drives it). It sets
//! `Camera::is_active`, [`ActiveCamera`](crate::camera::ActiveCamera) (ship/map
//! only), and `IsDefaultUiCamera` (the active camera).
//!
//! Mesh-based overlays opt in by carrying [`HideInShipView`] or
//! [`HideInMapView`]. Observers in this module forward those tags onto
//! the appropriate `RenderLayers` so the inactive camera physically
//! cannot draw them — no per-frame visibility flipping needed.
//!
//! Gizmo systems (trajectories, orbit lines) can't be hidden by
//! visibility; configure their gizmo group's `render_layers` so they
//! draw to one camera only.

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use thalos_input::game::GameInputIntent;

use crate::camera::{ActiveCamera, MapCamera, ShipCamera};
use crate::coords::{MAP_LAYER, SHIP_LAYER};
use crate::game_context::GameContext;
use crate::rendering::sun_shadow::SHADOW_CASTER_LAYER;
use crate::shipyard_editor::scene::EditorCamera;

#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    Map,
    #[default]
    Ship,
}

/// Marker: entities with this component are hidden while the view is
/// [`ViewMode::Ship`]. Attach to overlays that only make sense in the
/// far-scale map view (planet icons, impostor billboards, maneuver arrows,
/// ghost bodies, the flat ship marker).
#[derive(Component)]
pub struct HideInShipView;

/// Marker: entities with this component are hidden while the view is
/// [`ViewMode::Map`]. Attach to the 3D ship mesh and the real-scale body
/// spheres that only make sense up close.
#[derive(Component)]
pub struct HideInMapView;

pub struct ViewPlugin;

impl Plugin for ViewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ViewMode>()
            .init_resource::<crate::coords::WorldScale>()
            .add_observer(attach_map_layer_for_hide_in_ship)
            .add_observer(attach_ship_layer_for_hide_in_map)
            .add_systems(
                Update,
                toggle_view_input.run_if(crate::pause_menu::not_game_paused),
            )
            .add_systems(
                Update,
                apply_active_camera
                    .after(toggle_view_input)
                    .after(crate::SimStage::Sync)
                    .before(crate::SimStage::Camera),
            )
            .add_systems(PostUpdate, propagate_view_render_layers);
    }
}

/// On insertion of [`HideInShipView`], attach `RenderLayers(MAP_LAYER)`
/// so only the map camera can see the entity.
fn attach_map_layer_for_hide_in_ship(
    trigger: On<Add, HideInShipView>,
    mut commands: Commands,
    existing: Query<&RenderLayers>,
) {
    let entity = trigger.entity;
    let layers = match existing.get(entity) {
        Ok(rl) => rl.clone().with(MAP_LAYER).without(SHIP_LAYER).without(0),
        Err(_) => RenderLayers::layer(MAP_LAYER),
    };
    commands.entity(entity).insert(layers);
}

/// On insertion of [`HideInMapView`], attach `RenderLayers(SHIP_LAYER)`
/// so only the ship camera can see the entity.
fn attach_ship_layer_for_hide_in_map(
    trigger: On<Add, HideInMapView>,
    mut commands: Commands,
    existing: Query<&RenderLayers>,
) {
    let entity = trigger.entity;
    let layers = match existing.get(entity) {
        Ok(rl) => rl.clone().with(SHIP_LAYER).without(MAP_LAYER).without(0),
        Err(_) => RenderLayers::layer(SHIP_LAYER),
    };
    commands.entity(entity).insert(layers);
}

/// Propagate the `RenderLayers` of any entity carrying [`HideInShipView`]
/// or [`HideInMapView`] down its full descendant tree.
///
/// `RenderLayers` does not propagate through Bevy's hierarchy on its
/// own, but the natural mental model for callers is "tag the root, the
/// whole vehicle disappears in the other view." This system reasserts
/// the layer on every descendant each frame so reparenting, late-spawned
/// children (ship parts that load after the root), and rebuilt mesh
/// children all stay tied to the right view.
///
/// Ship-view solid geometry ([`HideInMapView`] — the craft + the EVA player) is
/// *also* placed on [`SHADOW_CASTER_LAYER`] so it casts into the custom sun-shadow
/// cascades and grounds with a shadow on the terrain/grass, exactly like trees +
/// rocks (F6 of the graphics-fidelity unification — `docs/roadmap/graphics_fidelity.md`
/// §4.2). The cascade cameras read only depth, so this adds the mesh to the shadow
/// map without changing how the craft itself is shaded (receiving is separate:
/// `ship_part.wgsl` / `shadowed_standard.wgsl` sample the same rig).
fn propagate_view_render_layers(
    mut commands: Commands,
    roots: Query<
        (Entity, Has<HideInShipView>, Has<HideInMapView>),
        Or<(With<HideInShipView>, With<HideInMapView>)>,
    >,
    children_q: Query<&Children>,
    layers_q: Query<&RenderLayers>,
) {
    for (root, hide_ship, hide_map) in &roots {
        let target = if hide_ship {
            RenderLayers::layer(MAP_LAYER)
        } else if hide_map {
            RenderLayers::from_layers(&[SHIP_LAYER, SHADOW_CASTER_LAYER])
        } else {
            continue;
        };
        let mut stack: Vec<Entity> = Vec::new();
        if let Ok(c) = children_q.get(root) {
            stack.extend(c.iter());
        }
        while let Some(e) = stack.pop() {
            let needs = layers_q.get(e).map(|rl| rl != &target).unwrap_or(true);
            if needs {
                commands.entity(e).insert(target.clone());
            }
            if let Ok(c) = children_q.get(e) {
                stack.extend(c.iter());
            }
        }
    }
}

/// The **single authority** over which scene camera is active and carries the
/// UI / [`ActiveCamera`] markers — a pure function of `(GameContext, ViewMode)`.
/// The VAB owns its editor camera; the hub / base editor own the ship camera
/// (their god-view drives its transform); Flight picks ship or map per
/// [`ViewMode`]. This replaced the former split authority (`view` owned
/// ship/map, `shipyard_editor::apply_open_state` owned the editor camera).
///
/// Runs every frame with change-guarded writes, so a `GameContext` transition or
/// a `ViewMode` toggle is reflected without an explicit change-poke.
///
/// `IsDefaultUiCamera` follows whichever camera is active so the Bevy-UI HUD /
/// editor UI renders to it. `ActiveCamera` (the flight-camera / freecam marker)
/// is applied to the active **ship/map** camera only, never the editor camera.
fn apply_active_camera(
    view: Res<ViewMode>,
    ctx: Option<Res<State<GameContext>>>,
    mut commands: Commands,
    mut cameras: Query<
        (
            Entity,
            &mut Camera,
            Has<MapCamera>,
            Has<ShipCamera>,
            Has<EditorCamera>,
            Has<ActiveCamera>,
            Has<IsDefaultUiCamera>,
        ),
        Or<(With<MapCamera>, With<ShipCamera>, With<EditorCamera>)>,
    >,
) {
    let ctx = ctx.map(|c| *c.get()).unwrap_or(GameContext::Flight);
    let (want_ship, want_map, want_editor) = match ctx {
        GameContext::Vab => (false, false, true),
        // The hub / base editor render through the ship camera; their god-view
        // (`crate::god_view`) drives its transform.
        GameContext::SpaceCenter | GameContext::BaseEditor => (true, false, false),
        GameContext::Flight => match *view {
            ViewMode::Map => (false, true, false),
            ViewMode::Ship => (true, false, false),
        },
    };
    for (entity, mut camera, is_map, is_ship, is_editor, has_active, has_ui) in &mut cameras {
        let should = (is_ship && want_ship) || (is_map && want_map) || (is_editor && want_editor);
        if camera.is_active != should {
            camera.is_active = should;
        }
        // The active window camera is the default UI camera (bevy_ui renders to
        // it, and an inactive default means no UI at all).
        if should && !has_ui {
            commands.entity(entity).insert(IsDefaultUiCamera);
        } else if !should && has_ui {
            commands.entity(entity).remove::<IsDefaultUiCamera>();
        }
        // `ActiveCamera` is the flight-scene marker (freecam + camera drivers);
        // the editor camera is not a flight camera, so it never carries it.
        let want_active = should && !is_editor;
        if want_active && !has_active {
            commands.entity(entity).insert(ActiveCamera);
        } else if !want_active && has_active {
            commands.entity(entity).remove::<ActiveCamera>();
        }
    }
}

pub fn in_map_view(view: Res<ViewMode>) -> bool {
    *view == ViewMode::Map
}

fn toggle_view_input(
    input: Res<GameInputIntent>,
    ui_keyboard: Res<crate::hud::UiKeyboardGate>,
    mut view: ResMut<ViewMode>,
) {
    if !input.toggle_view {
        return;
    }
    if ui_keyboard.text_entry() {
        return;
    }
    *view = match *view {
        ViewMode::Map => ViewMode::Ship,
        ViewMode::Ship => ViewMode::Map,
    };
}
