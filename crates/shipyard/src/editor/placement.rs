//! Picking observers and surface-mount placement math: click→select,
//! click→place, hull-hit→`(station, angle)` resolution, and the symmetry
//! group helpers shared by placement and the inspector.

use bevy::picking::events::{Click, Pointer};
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use thalos_input::shipyard::ShipyardInputIntent;

use crate::{
    AttachNodes, CatalogEntry, EngineGeometry, Part, PartCatalog, PartParams, SurfaceMount,
    SurfaceMountKind, SymmetryGroup, SymmetryRole, Wing, wing_panel_frame,
};

use super::debug_log::{Field, SelectionLog};
use super::state::{
    AttachNodePin, BuildOrientation, CLICK_THRESHOLD_PX, DeselectTracker, EditorState,
    EditorUiGate, PartBody, TankResizeArrow,
};

/// Snap increment for body-skin mount azimuth — 15° (24 positions around the
/// cylinder). The belly (π), top (0), and sides (±π/2) are all exact steps.
pub const BODY_SKIN_SNAP_STEP: f32 = std::f32::consts::TAU / 24.0;

pub fn snap_body_skin_angle(angle: f32) -> f32 {
    (angle / BODY_SKIN_SNAP_STEP).round() * BODY_SKIN_SNAP_STEP
}

pub(super) fn on_body_click(
    click: On<Pointer<Click>>,
    bodies: Query<&PartBody>,
    wings: Query<(), With<Wing>>,
    catalog: Res<PartCatalog>,
    ui_gate: Res<EditorUiGate>,
    log: Res<SelectionLog>,
    mut state: ResMut<EditorState>,
) {
    if ui_gate.pointer_busy {
        log.event(&[
            ("event", "body_click".into()),
            ("clicked", click.entity.into()),
            ("action", "blocked_ui_busy".into()),
        ]);
        return;
    }
    let Ok(body) = bodies.get(click.entity) else {
        log.event(&[
            ("event", "body_click".into()),
            ("clicked", click.entity.into()),
            ("action", "no_part_body".into()),
        ]);
        return;
    };
    let pending_surface_kind = state.pending.as_ref().and_then(|pending| {
        // Wings and landing gear both body-skin-mount on a hull (not a wing).
        if matches!(
            pending.params,
            PartParams::Wing { .. } | PartParams::Gear { .. }
        ) && wings.get(body.0).is_err()
        {
            return Some(SurfaceMountKind::BodySkin);
        }
        let entry = catalog.resolve(&pending.catalog_id).ok()?;
        match entry {
            CatalogEntry::Engine(e)
                if e.geometry == EngineGeometry::JetNacelle && wings.get(body.0).is_ok() =>
            {
                Some(SurfaceMountKind::WingPylon)
            }
            _ => None,
        }
    });
    if let Some(kind) = pending_surface_kind {
        let has_hit = click.hit.position.is_some();
        if let Some(pos) = click.hit.position {
            state.place_surface_at = Some((body.0, pos, kind));
        }
        log.event(&[
            ("event", "body_click".into()),
            ("clicked", click.entity.into()),
            ("part", body.0.into()),
            ("action", "place_surface".into()),
            ("surface_kind", format!("{kind:?}").into()),
            ("has_hit_pos", has_hit.into()),
        ]);
        return;
    }
    if state.pending.is_some() {
        state.status = "Pick a compatible surface or attach node for the pending part".into();
        log.event(&[
            ("event", "body_click".into()),
            ("clicked", click.entity.into()),
            ("part", body.0.into()),
            ("action", "pending_incompatible".into()),
        ]);
        return;
    }
    log.event(&[
        ("event", "body_click".into()),
        ("clicked", click.entity.into()),
        ("part", body.0.into()),
        ("action", "select".into()),
        ("prev_selected", state.selected.into()),
    ]);
    state.selected = Some(body.0);
}

pub(super) fn on_pin_click(
    click: On<Pointer<Click>>,
    pins: Query<&AttachNodePin>,
    ui_gate: Res<EditorUiGate>,
    log: Res<SelectionLog>,
    mut state: ResMut<EditorState>,
) {
    if ui_gate.pointer_busy {
        log.event(&[
            ("event", "pin_click".into()),
            ("clicked", click.entity.into()),
            ("action", "blocked_ui_busy".into()),
        ]);
        return;
    }
    if let Ok(pin) = pins.get(click.entity) {
        if state.pending.is_some() {
            state.place_at = Some((pin.part, pin.node_id.clone()));
            log.event(&[
                ("event", "pin_click".into()),
                ("part", pin.part.into()),
                ("node", pin.node_id.as_str().into()),
                ("action", "place_at".into()),
            ]);
        } else {
            log.event(&[
                ("event", "pin_click".into()),
                ("part", pin.part.into()),
                ("node", pin.node_id.as_str().into()),
                ("action", "select".into()),
                ("prev_selected", state.selected.into()),
            ]);
            state.selected = Some(pin.part);
        }
    }
}

/// Clear selection when the user clicks on empty space. Tracks the
/// press cursor so a camera orbit (press → drag → release) doesn't
/// deselect at release.
pub(super) fn deselect_on_empty_click(
    mut tracker: ResMut<DeselectTracker>,
    input: Res<ShipyardInputIntent>,
    ui_gate: Res<EditorUiGate>,
    windows: Query<&Window, With<PrimaryWindow>>,
    hover_map: Res<HoverMap>,
    pickables: Query<(), Or<(With<PartBody>, With<AttachNodePin>, With<TankResizeArrow>)>>,
    log: Res<SelectionLog>,
    mut state: ResMut<EditorState>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let cursor = window.cursor_position();

    if input.primary_started {
        if ui_gate.pointer_busy {
            tracker.press_cursor = None;
        } else {
            let on_pickable = hover_map
                .0
                .values()
                .any(|hovers| hovers.keys().any(|e| pickables.get(*e).is_ok()));
            tracker.press_cursor = if on_pickable { None } else { cursor };
        }
        if log.enabled() {
            log.event(&[
                ("event", "deselect_press".into()),
                ("ui_busy", ui_gate.pointer_busy.into()),
                ("armed", tracker.press_cursor.is_some().into()),
                ("selected", state.selected.into()),
            ]);
        }
    }

    if input.primary_released {
        let press = tracker.press_cursor.take();
        let dist = match (press, cursor) {
            (Some(p), Some(c)) => Some((c - p).length()),
            _ => None,
        };
        let will_deselect = dist.is_some_and(|d| d < CLICK_THRESHOLD_PX);
        if log.enabled() && (press.is_some() || state.selected.is_some()) {
            log.event(&[
                ("event", "deselect_release".into()),
                ("armed", press.is_some().into()),
                ("drag_px", dist.map_or(Field::Null, Into::into)),
                ("deselect", will_deselect.into()),
                ("selected", state.selected.into()),
            ]);
        }
        if will_deselect {
            state.selected = None;
        }
    }
}

/// The symmetry-group members of `host`, primary first, or `[host]` if the
/// host isn't part of a group. Used to stamp a footprint part onto every
/// counterpart of a symmetric host (KSP nested symmetry — a nacelle on a
/// mirrored wing lands on both wings).
pub fn host_group_members(
    host: Entity,
    groups: &Query<(Entity, &SymmetryGroup), With<super::state::EditorPart>>,
) -> Vec<Entity> {
    let Ok((_, hg)) = groups.get(host) else {
        return vec![host];
    };
    let gid = hg.id;
    let mut primary = None;
    let mut mirrors = Vec::new();
    for (e, g) in groups.iter() {
        if g.id == gid {
            match g.role {
                SymmetryRole::Primary => primary = Some(e),
                SymmetryRole::Mirror => mirrors.push(e),
            }
        }
    }
    let mut out: Vec<Entity> = primary.into_iter().collect();
    out.extend(mirrors);
    if out.is_empty() { vec![host] } else { out }
}

/// The entity whose params the inspector should edit for a given selection:
/// the selection's symmetry-group **primary** if it belongs to a group, else
/// the selection itself. `sync_symmetry_groups` copies the primary onto its
/// mirror counterparts every frame, so editing a counterpart directly would be
/// reverted next frame — its inspector sliders would look dead. KSP-style, an
/// edit on any member is applied to the controlling (primary) part and the
/// mirrors follow.
pub fn symmetry_edit_target(
    sel: Entity,
    groups: &Query<(Entity, &SymmetryGroup), With<super::state::EditorPart>>,
) -> Entity {
    let Ok((_, sg)) = groups.get(sel) else {
        return sel;
    };
    match sg.role {
        SymmetryRole::Primary => sel,
        SymmetryRole::Mirror => groups
            .iter()
            .find(|(_, g)| g.id == sg.id && g.role == SymmetryRole::Primary)
            .map(|(e, _)| e)
            .unwrap_or(sel),
    }
}

/// Resolve a body-skin (cylinder) hit into a `(station, angle)` pair, with
/// optional magnetic angle snapping. Shared by the commit path
/// ([`surface_mount_from_hit`]) and the live placement preview so the ghost and
/// the placed part land at exactly the same spot.
pub fn body_skin_mount(
    parent: Entity,
    world_pos: Vec3,
    part_transforms: &Query<&Transform, With<Part>>,
    host_nodes: &Query<&AttachNodes>,
    orientation: &BuildOrientation,
    snap: bool,
) -> (f32, f32) {
    let parent_t = part_transforms
        .get(parent)
        .map(|t| t.translation)
        .unwrap_or(Vec3::ZERO);
    // Undo the build-layout rotation so the hit lands in the upright build
    // frame, where all persisted surface coordinates are defined.
    let local = orientation.rotation().inverse() * (world_pos - parent_t);
    let host = host_nodes.get(parent).ok();
    let radius = host
        .and_then(|n| n.get("top").map(|nd| nd.diameter * 0.5))
        .unwrap_or(1.0);
    let height = host
        .and_then(|n| n.get("bottom").map(|nd| -nd.offset.y))
        .unwrap_or(radius * 2.0);
    let station = if height > 0.0 {
        (-local.y / height).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let mut angle = local.x.atan2(local.z);
    if snap {
        angle = snap_body_skin_angle(angle);
    }
    (station, angle)
}

/// Resolve a hull/wing surface hit into the persisted `(station, angle)`
/// pair for `kind`. Symmetry is no longer decided here — the global
/// [`super::state::SymmetryMode`] + the host's own symmetry drive group
/// stamping at the call site. `snap` magnetically rounds the azimuth of
/// body-skin mounts.
pub fn surface_mount_from_hit(
    kind: SurfaceMountKind,
    parent: Entity,
    world_pos: Vec3,
    part_transforms: &Query<&Transform, With<Part>>,
    host_nodes: &Query<&AttachNodes>,
    surface_mounts: &Query<(Entity, &SurfaceMount), With<super::state::EditorPart>>,
    wings: &Query<&Wing>,
    orientation: &BuildOrientation,
    snap: bool,
) -> Option<(f32, f32, String)> {
    let parent_t = part_transforms
        .get(parent)
        .map(|t| t.translation)
        .unwrap_or(Vec3::ZERO);
    // Undo the build-layout rotation so the hit lands in the upright build
    // frame, where all persisted surface coordinates are defined.
    let local = orientation.rotation().inverse() * (world_pos - parent_t);

    match kind {
        SurfaceMountKind::BodySkin => {
            let (station, angle) = body_skin_mount(
                parent,
                world_pos,
                part_transforms,
                host_nodes,
                orientation,
                snap,
            );
            Some((station, angle, "Mounted wing".into()))
        }
        SurfaceMountKind::WingPylon => {
            let wing = wings.get(parent).ok()?;
            let (_, wing_mount) = surface_mounts.iter().find(|(e, _)| *e == parent)?;
            let parent_radius = host_nodes
                .get(wing_mount.parent)
                .ok()
                .and_then(|n| n.get("top").map(|nd| nd.diameter * 0.5))
                .unwrap_or(1.0);
            // The click is on a specific wing entity; project it onto that
            // wing's own panel frame.
            let frame = wing_panel_frame(wing, wing_mount.angle, parent_radius);
            let span_axis = frame.tip_center - frame.root_center;
            let span_len2 = span_axis.length_squared();
            let station = if span_len2 > f32::EPSILON {
                ((local - frame.root_center).dot(span_axis) / span_len2).clamp(0.08, 0.92)
            } else {
                0.5
            };
            let chord = frame.chord_at(wing, station).max(0.1);
            let chord_center = frame.center_at(station);
            let chord_fraction =
                ((local - chord_center).dot(frame.fore_dir) / chord).clamp(-0.4, 0.4);
            Some((
                station,
                chord_fraction,
                "Mounted jet nacelle with pylon".into(),
            ))
        }
    }
}
