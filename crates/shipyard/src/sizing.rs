use crate::attach::{AttachNodes, Attachment, NodeId, Ship};
use crate::part::{Adapter, Decoupler, FuelTank};
use bevy::prelude::*;
use std::collections::{HashMap, VecDeque};

/// Target (diameter, offset) for a single node on a child part.
type NodeTarget = (NodeId, f32, Vec3);

/// Apply a batch of targets to a part's attach nodes, only triggering
/// `Changed<AttachNodes>` when at least one value actually differs.
fn apply_targets(cnodes: &mut Mut<AttachNodes>, targets: &[NodeTarget]) {
    // Check if any target differs from the current state — read via
    // Deref so we don't mark the component as changed.
    let needs_update = {
        let current = &**cnodes;
        targets
            .iter()
            .any(|(id, d, off)| match current.nodes.get(id) {
                Some(n) => {
                    (n.diameter - *d).abs() > f32::EPSILON
                        || n.offset.distance_squared(*off) > f32::EPSILON
                }
                None => false,
            })
    };
    if !needs_update {
        return;
    }
    // DerefMut only once we know a change is needed.
    for (id, d, off) in targets {
        if let Some(n) = cnodes.nodes.get_mut(id) {
            n.diameter = *d;
            n.offset = *off;
        }
    }
}

/// BFS from each `Ship` root, propagating parent attach-node diameter into
/// parametric children. Fixed parts (CommandPod, Engine) are untouched, so
/// the walk terminates sizing at them but still recurses for further
/// children attached on the far side.
pub fn propagate_node_sizes(
    ships: Query<&Ship>,
    attachments: Query<(Entity, &Attachment)>,
    mut nodes: Query<&mut AttachNodes>,
    decouplers: Query<(), With<Decoupler>>,
    tanks: Query<&FuelTank>,
    adapters: Query<&Adapter>,
) {
    let mut children: HashMap<Entity, Vec<(Entity, Attachment)>> = HashMap::new();
    for (child, att) in attachments.iter() {
        children
            .entry(att.parent)
            .or_default()
            .push((child, att.clone()));
    }

    for ship in ships.iter() {
        let mut queue: VecDeque<Entity> = VecDeque::from([ship.root]);
        while let Some(parent) = queue.pop_front() {
            let parent_sizes: HashMap<NodeId, f32> = match nodes.get(parent) {
                Ok(n) => n
                    .nodes
                    .iter()
                    .map(|(k, v)| (k.clone(), v.diameter))
                    .collect(),
                Err(_) => continue,
            };

            let Some(kids) = children.get(&parent) else {
                continue;
            };

            for (child, att) in kids {
                let Some(&input_d) = parent_sizes.get(&att.parent_node) else {
                    queue.push_back(*child);
                    continue;
                };

                if let Ok(mut cnodes) = nodes.get_mut(*child) {
                    let targets: Vec<NodeTarget> = if decouplers.contains(*child) {
                        vec![
                            ("top".into(), input_d, Vec3::ZERO),
                            ("bottom".into(), input_d, Vec3::new(0.0, -0.2, 0.0)),
                        ]
                    } else if let Ok(tank) = tanks.get(*child) {
                        vec![
                            ("top".into(), input_d, Vec3::ZERO),
                            ("bottom".into(), input_d, Vec3::new(0.0, -tank.length, 0.0)),
                        ]
                    } else if let Ok(adapter) = adapters.get(*child) {
                        let bot_d = adapter.target_diameter;
                        let h = ((input_d + bot_d) * 0.5).max(0.4);
                        vec![
                            ("top".into(), input_d, Vec3::ZERO),
                            ("bottom".into(), bot_d, Vec3::new(0.0, -h, 0.0)),
                        ]
                    } else {
                        Vec::new()
                    };
                    if !targets.is_empty() {
                        apply_targets(&mut cnodes, &targets);
                    }
                }
                queue.push_back(*child);
            }
        }
    }
}
