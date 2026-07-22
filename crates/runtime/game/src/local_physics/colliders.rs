//! Aggregate craft collider construction from rendered ship parts.
//!
//! Split out of the former monolithic `local_physics.rs` (Phase B, `docs/simulation/regimes.md`).

#[allow(unused_imports)]
use super::*;
use std::collections::{HashMap, VecDeque};

use bevy::math::{DMat3, DQuat, DVec3};
use thalos_physics_local::{LocalPrimitiveCollider, LocalPrimitiveShape};
use thalos_shipyard::{
    Adapter, AirIntake, AttachNodes, Attachment, CommandPod, Decoupler, Engine, EngineGeometry,
    FuelTank, Part, SurfaceMount, SurfaceMountKind, Wing, wing_panel_frame,
};

pub(crate) type PartColliderQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AttachNodes,
        Option<&'static Attachment>,
        Option<&'static SurfaceMount>,
        Option<&'static CommandPod>,
        Option<&'static Decoupler>,
        Option<&'static Adapter>,
        Option<&'static FuelTank>,
        Option<&'static Engine>,
        Option<&'static AirIntake>,
        Option<&'static Wing>,
    ),
    // The in-game shipyard editor's build shares these components; it must
    // never contribute colliders, wheels, or clearance to the flight craft.
    (
        With<Part>,
        Without<crate::shipyard_editor::core::EditorPart>,
    ),
>;

pub(crate) fn build_ship_collider_primitives(
    parts: &PartColliderQuery,
) -> Vec<LocalPrimitiveCollider> {
    let part_positions = compute_part_collider_positions(parts);
    let nodes_by_entity: HashMap<Entity, &AttachNodes> =
        parts.iter().map(|(e, nodes, ..)| (e, nodes)).collect();
    let mut primitives = Vec::new();
    for (entity, nodes, _, surface_mount, pod, dec, adapter, tank, engine, intake, wing) in
        parts.iter()
    {
        let Some(part_position) = part_positions.get(&entity).copied() else {
            continue;
        };
        // Wings are thin lifting surfaces, not body-axis solids: give each a
        // thin angled slab matching its planform so a wingtip catches the
        // ground (e.g. on an over-banked landing) instead of passing through.
        if let (Some(wing), Some(mount)) = (wing, surface_mount) {
            let parent_radius = nodes_by_entity
                .get(&mount.parent)
                .and_then(|n| n.get("top"))
                .map(|node| node.diameter * 0.5)
                .unwrap_or(1.0);
            primitives.push(wing_collider_primitive(
                wing,
                mount,
                parent_radius,
                part_position,
            ));
            continue;
        }
        if matches!(
            (engine, surface_mount.map(|m| m.kind)),
            (Some(engine), Some(SurfaceMountKind::WingPylon))
                if engine.geometry == EngineGeometry::JetNacelle
        ) {
            continue;
        }
        let Some((shape, local_offset)) =
            part_collider_shape(nodes, pod, dec, adapter, tank, engine, intake)
        else {
            continue;
        };
        primitives.push(LocalPrimitiveCollider {
            offset_m: part_position + local_offset,
            rotation: DQuat::IDENTITY,
            shape,
        });
    }
    if primitives.is_empty() {
        primitives.push(fallback_collider());
    }
    primitives
}

/// A thin oriented-cuboid collider matching a wing panel's planform, in the
/// craft body frame. Reuses [`wing_panel_frame`] — the same geometry the wing
/// mesh draws — so the collider tracks the rendered wing. `host_axis_pos` is the
/// wing's mount point on the host axis (from [`compute_part_collider_positions`]).
pub(crate) fn wing_collider_primitive(
    wing: &Wing,
    mount: &SurfaceMount,
    parent_radius: f32,
    host_axis_pos: DVec3,
) -> LocalPrimitiveCollider {
    let frame = wing_panel_frame(wing, mount.angle, parent_radius);
    let center_local = (frame.root_center + frame.tip_center) * 0.5;
    let span_len = (frame.tip_center - frame.root_center).length().max(0.1);
    let chord = wing.root_chord.max(wing.tip_chord).max(0.1);
    let thickness = (wing.root_chord * wing.thickness).max(0.05);
    // Orthonormal slab axes: span (y) and thickness (z) are perpendicular by
    // construction (`thick = span × fore`); recover a clean chord axis as
    // `span × thick` so the basis is a valid rotation even though `fore_dir`
    // itself is tilted by incidence and not exactly ⊥ to span.
    let span_dir = frame.span_dir.as_dvec3().normalize_or(DVec3::X);
    let thick_dir = frame.thick_dir.as_dvec3().normalize_or(DVec3::Z);
    let chord_dir = span_dir.cross(thick_dir).normalize_or(DVec3::Y);
    let basis = DMat3::from_cols(chord_dir, span_dir, thick_dir);
    LocalPrimitiveCollider {
        offset_m: host_axis_pos + center_local.as_dvec3(),
        rotation: DQuat::from_mat3(&basis).normalize(),
        shape: LocalPrimitiveShape::Cuboid {
            x: chord as f64,
            y: span_len as f64,
            z: thickness as f64,
        },
    }
}

pub(crate) fn compute_part_collider_positions(parts: &PartColliderQuery) -> HashMap<Entity, DVec3> {
    let mut nodes_by_entity: HashMap<Entity, &AttachNodes> = HashMap::new();
    let mut children_by_parent: HashMap<Entity, Vec<(Entity, Attachment)>> = HashMap::new();
    let mut surface_children_by_parent: HashMap<Entity, Vec<(Entity, SurfaceMount)>> =
        HashMap::new();
    let mut roots = Vec::new();

    for (entity, nodes, attachment, surface_mount, ..) in parts.iter() {
        nodes_by_entity.insert(entity, nodes);
        if let Some(attachment) = attachment {
            children_by_parent
                .entry(attachment.parent)
                .or_default()
                .push((entity, attachment.clone()));
        } else if let Some(surface_mount) = surface_mount {
            surface_children_by_parent
                .entry(surface_mount.parent)
                .or_default()
                .push((entity, *surface_mount));
        } else {
            roots.push(entity);
        }
    }

    let mut positions = HashMap::new();
    let mut queue = VecDeque::new();
    for root in roots {
        positions.insert(root, DVec3::ZERO);
        queue.push_back(root);
    }

    while let Some(parent) = queue.pop_front() {
        let Some(parent_position) = positions.get(&parent).copied() else {
            continue;
        };
        let Some(parent_nodes) = nodes_by_entity.get(&parent).copied() else {
            continue;
        };
        if let Some(children) = children_by_parent.get(&parent) {
            for (child, attachment) in children {
                let Some(parent_node) = parent_nodes.get(&attachment.parent_node) else {
                    continue;
                };
                let child_offset = nodes_by_entity
                    .get(child)
                    .and_then(|nodes| nodes.get(&attachment.my_node))
                    .map(|node| node.offset)
                    .unwrap_or(Vec3::ZERO);
                let child_position =
                    parent_position + (parent_node.offset - child_offset).as_dvec3();
                positions.insert(*child, child_position);
                queue.push_back(*child);
            }
        }

        if let Some(children) = surface_children_by_parent.get(&parent) {
            for (child, mount) in children {
                let local_offset = match mount.kind {
                    SurfaceMountKind::BodySkin => {
                        let host_height = parent_nodes
                            .get("bottom")
                            .map(|node| -node.offset.y)
                            .unwrap_or(0.0);
                        DVec3::new(0.0, -(mount.station as f64) * host_height as f64, 0.0)
                    }
                    SurfaceMountKind::WingPylon => DVec3::ZERO,
                };
                positions.insert(*child, parent_position + local_offset);
                queue.push_back(*child);
            }
        }
    }

    positions
}

pub(crate) fn fallback_collider() -> LocalPrimitiveCollider {
    LocalPrimitiveCollider {
        offset_m: DVec3::ZERO,
        rotation: DQuat::IDENTITY,
        shape: LocalPrimitiveShape::Cuboid {
            x: 2.0,
            y: 6.0,
            z: 2.0,
        },
    }
}

pub(crate) fn part_collider_shape(
    nodes: &AttachNodes,
    pod: Option<&CommandPod>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    tank: Option<&FuelTank>,
    engine: Option<&Engine>,
    intake: Option<&AirIntake>,
) -> Option<(LocalPrimitiveShape, DVec3)> {
    if let Some(pod) = pod {
        let height = pod.diameter * pod.geometry.length_factor();
        Some((
            LocalPrimitiveShape::Cylinder {
                radius: (pod.diameter * 0.5) as f64,
                height: height as f64,
            },
            DVec3::Y * -(height as f64 * 0.5),
        ))
    } else if dec.is_some() {
        let diameter = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let height = 0.2;
        Some((
            LocalPrimitiveShape::Cylinder {
                radius: (diameter * 0.5) as f64,
                height,
            },
            DVec3::Y * -(height * 0.5),
        ))
    } else if let Some(adapter) = adapter {
        let top_d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let bot_d = adapter.target_diameter;
        let height = ((top_d + bot_d) * 0.5).max(0.4);
        Some((
            LocalPrimitiveShape::Cylinder {
                radius: (top_d.max(bot_d) * 0.5) as f64,
                height: height as f64,
            },
            DVec3::Y * -(height as f64 * 0.5),
        ))
    } else if let Some(tank) = tank {
        let diameter = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        Some((
            LocalPrimitiveShape::Cylinder {
                radius: (diameter * 0.5) as f64,
                height: tank.length as f64,
            },
            DVec3::Y * -(tank.length as f64 * 0.5),
        ))
    } else if let Some(engine) = engine {
        let height = match engine.geometry {
            EngineGeometry::RocketBell => engine.diameter * 0.9,
            EngineGeometry::JetNacelle => thalos_shipyard::jet_nacelle_length(engine),
        };
        Some((
            LocalPrimitiveShape::Cylinder {
                radius: (engine.diameter * 0.5) as f64,
                height: height as f64,
            },
            DVec3::Y * -(height as f64 * 0.5),
        ))
    } else if let Some(intake) = intake {
        Some((
            LocalPrimitiveShape::Cylinder {
                radius: (intake.diameter * 0.5) as f64,
                height: intake.length as f64,
            },
            DVec3::Y * -(intake.length as f64 * 0.5),
        ))
    } else {
        None
    }
}
