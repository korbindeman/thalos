//! Right panel: the parametric inspector for the current selection.
//!
//! Slider rows bind directly to part-component fields through
//! [`ParamBinding`]; dragging writes the field (value-guarded so symmetric
//! no-op writes don't re-trigger the core's `Changed`-driven mesh rebuilds),
//! and the core's recompute/rebuild systems do the rest. A refresh pass
//! pulls model → slider whenever the user isn't dragging, so values stay
//! honest under diameter propagation and symmetry sync.

use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;

use crate::shipyard_editor::core::{
    EditorPart, EditorState, inspector_params, symmetry_edit_target,
};
use thalos_shipyard::{
    Adapter, AirIntake, AttachNodes, CatalogRef, CommandPod, Decoupler, Engine, FuelTank, Fuselage,
    Gear, PartCatalog, PartResources, Resource, ResourcePool, SymmetryGroup, Wing,
    resource_capacity_for,
};

use thalos_ui::{
    self as ui, ButtonVariant, ScrollableColumn, SliderFormat, UiSlider, UiTheme, spawn_button,
    spawn_heading, spawn_slider_row,
};

/// Which part-component field a slider drives.
#[derive(Component, Clone, Copy, PartialEq)]
pub(super) enum ParamBinding {
    DecouplerDiameter,
    AdapterDiameter,
    AdapterTargetDiameter,
    TankDiameter,
    TankLength,
    FuselageLength,
    FuselageWidth,
    FuselageHeight,
    FuselageRoundness,
    FuselageNoseFraction,
    FuselageNoseBluntness,
    FuselageTailFraction,
    FuselageNoseDroop,
    FuselageTailUpsweep,
    FuselageTailTipDiameter,
    FuselageTailBluntness,
    WingSpan,
    WingRootChord,
    WingTipChord,
    WingSweepDeg,
    WingDihedralDeg,
    WingIncidenceDeg,
    WingThickness,
    GearStrutLength,
    GearWheelRadius,
    ResourceAmount(Resource),
}

#[derive(Component, Clone, Copy)]
pub(super) enum InspectorAction {
    SetRoot,
    Delete,
    AddResource(Resource),
    RemoveResource(Resource),
}

/// The content column the rebuild system repopulates per selection.
#[derive(Component)]
pub(super) struct InspectorContent;

/// Live read-only info block (kind, fixed specs, dry mass, attach nodes).
#[derive(Component)]
pub(super) struct InspectorInfoText;

type KindQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static CatalogRef,
        &'static AttachNodes,
        Option<&'static CommandPod>,
        Option<&'static Decoupler>,
        Option<&'static Adapter>,
        Option<&'static FuelTank>,
        Option<&'static Fuselage>,
        Option<&'static Engine>,
        Option<&'static AirIntake>,
        Option<&'static Wing>,
        Option<&'static Gear>,
        Option<&'static PartResources>,
    ),
    With<EditorPart>,
>;

type KindQueryMut<'w, 's> = Query<
    'w,
    's,
    (
        Option<&'static mut Decoupler>,
        Option<&'static mut Adapter>,
        Option<&'static mut FuelTank>,
        Option<&'static mut Fuselage>,
        Option<&'static mut Wing>,
        Option<&'static mut Gear>,
        Option<&'static mut PartResources>,
    ),
    With<EditorPart>,
>;

pub(super) fn spawn(root: &mut ChildSpawnerCommands<'_>, theme: &UiTheme) {
    root.spawn((
        Node {
            right: Val::Px(12.0),
            top: Val::Px(64.0),
            bottom: Val::Px(12.0),
            width: Val::Px(300.0),
            ..ui::floating_panel_node()
        },
        theme.glass(),
        Interaction::None,
        Name::new("ShipyardInspector"),
    ))
    .with_children(|panel| {
        spawn_heading(panel, theme, "INSPECTOR", false);
        panel.spawn((
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(5.0),
                overflow: Overflow::scroll_y(),
                flex_grow: 1.0,
                ..default()
            },
            ScrollPosition::default(),
            RelativeCursorPosition::default(),
            Interaction::None,
            ScrollableColumn,
            InspectorContent,
            Name::new("ShipyardInspectorContent"),
        ));
    });
}

/// The selection key the inspector content is built for. Pool set included
/// so add/remove-resource rebuilds the rows.
type InspectorKey = (Entity, bool, Vec<Resource>);

fn kind_label(
    pod: Option<&CommandPod>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    tank: Option<&FuelTank>,
    fuselage: Option<&Fuselage>,
    engine: Option<&Engine>,
    intake: Option<&AirIntake>,
    wing: Option<&Wing>,
    gear: Option<&Gear>,
) -> &'static str {
    if pod.is_some() {
        "COMMAND POD"
    } else if dec.is_some() {
        "DECOUPLER"
    } else if adapter.is_some() {
        "ADAPTER"
    } else if tank.is_some() {
        "FUEL TANK"
    } else if fuselage.is_some() {
        "FUSELAGE"
    } else if engine.is_some() {
        "ENGINE"
    } else if intake.is_some() {
        "AIR INTAKE"
    } else if wing.is_some() {
        "WING"
    } else if gear.is_some() {
        "LANDING GEAR"
    } else {
        "PART"
    }
}

/// Rebuild the inspector content when the (resolved) selection, its
/// root-ness, or its active resource pools change.
pub(super) fn rebuild_inspector(
    mut commands: Commands,
    state: Res<EditorState>,
    theme: Res<UiTheme>,
    catalog: Res<PartCatalog>,
    groups: Query<(Entity, &SymmetryGroup), With<EditorPart>>,
    parts: KindQuery,
    content: Query<(Entity, Option<&Children>), With<InspectorContent>>,
    mut shown: Local<Option<Option<InspectorKey>>>,
) {
    let key: Option<InspectorKey> = state.selected.map(|sel| {
        let target = symmetry_edit_target(sel, &groups);
        let is_root = Some(target) == state.ship_root;
        let pools: Vec<Resource> = parts
            .get(target)
            .ok()
            .and_then(|p| p.12.map(|r| r.pools.keys().copied().collect()))
            .unwrap_or_default();
        (target, is_root, pools)
    });
    if shown.as_ref() == Some(&key) {
        return;
    }
    *shown = Some(key.clone());

    let Ok((content_entity, children)) = content.single() else {
        return;
    };
    if let Some(children) = children {
        for child in children.iter() {
            commands.entity(child).despawn();
        }
    }

    let Some((target, is_root, _)) = key else {
        let placeholder = theme.faint("(no selection)");
        commands.entity(content_entity).with_children(|c| {
            c.spawn(placeholder);
        });
        return;
    };
    let Ok((
        _,
        catalog_ref,
        nodes,
        pod,
        dec,
        adapter,
        tank,
        fuselage,
        engine,
        intake,
        wing,
        gear,
        resources,
    )) = parts.get(target)
    else {
        return;
    };

    let theme = theme.clone();
    let entry = catalog.resolve(&catalog_ref.id).ok().cloned();
    let params = inspector_params(dec, adapter, tank, fuselage, wing, gear);
    let pools: Vec<(Resource, ResourcePool)> = resources
        .map(|r| r.pools.iter().map(|(res, p)| (*res, *p)).collect())
        .unwrap_or_default();
    let kind = kind_label(
        pod, dec, adapter, tank, fuselage, engine, intake, wing, gear,
    );
    let tank_max_len = nodes
        .get("top")
        .map(|n| n.diameter)
        .or(tank.map(|t| t.diameter))
        .map(|d| 8.0 * d)
        .unwrap_or(20.0);

    // Snapshot current values for slider seeds.
    let dec_v = dec.cloned();
    let adapter_v = adapter.cloned();
    let tank_v = tank.cloned();
    let fuselage_v = fuselage.cloned();
    let wing_v = wing.cloned();
    let gear_v = gear.cloned();

    commands.entity(content_entity).with_children(|c| {
        let mut kind_text = theme.body_strong(kind);
        kind_text.2 = TextColor(thalos_ui::tokens::ACCENT);
        c.spawn(kind_text);
        // Live info block, refreshed per frame by `update_info_text`.
        let mut info = theme.mono_dim("");
        info.1.font_size = FontSize::Px(9.5);
        c.spawn((info, InspectorInfoText));

        let slider = |c: &mut ChildSpawnerCommands<'_>,
                      label: &str,
                      value: f32,
                      min: f32,
                      max: f32,
                      format: SliderFormat,
                      binding: ParamBinding| {
            spawn_slider_row(
                c,
                &theme,
                label,
                UiSlider::new(min, max, value, format),
                binding,
            );
        };

        use ParamBinding as B;
        use SliderFormat as F;
        if let Some(d) = &dec_v {
            if is_root {
                slider(
                    c,
                    "DIAMETER",
                    d.diameter,
                    0.3,
                    6.0,
                    F::Meters,
                    B::DecouplerDiameter,
                );
            }
        } else if let Some(a) = &adapter_v {
            if is_root {
                slider(
                    c,
                    "DIAMETER",
                    a.diameter,
                    0.3,
                    6.0,
                    F::Meters,
                    B::AdapterDiameter,
                );
            }
            slider(
                c,
                "TARGET Ø",
                a.target_diameter,
                0.3,
                6.0,
                F::Meters,
                B::AdapterTargetDiameter,
            );
        } else if let Some(t) = &tank_v {
            if is_root {
                slider(
                    c,
                    "DIAMETER",
                    t.diameter,
                    0.3,
                    6.0,
                    F::Meters,
                    B::TankDiameter,
                );
            }
            slider(
                c,
                "LENGTH",
                t.length,
                0.5,
                tank_max_len,
                F::Meters,
                B::TankLength,
            );
        } else if let Some(f) = &fuselage_v {
            slider(
                c,
                "LENGTH",
                f.length,
                2.0,
                60.0,
                F::Meters,
                B::FuselageLength,
            );
            if is_root {
                slider(
                    c,
                    "WIDTH (Ø)",
                    f.max_width,
                    0.5,
                    8.0,
                    F::Meters,
                    B::FuselageWidth,
                );
            }
            slider(
                c,
                "HEIGHT",
                f.max_height,
                0.5,
                8.0,
                F::Meters,
                B::FuselageHeight,
            );
            slider(
                c,
                "ROUNDNESS",
                f.roundness,
                0.0,
                1.0,
                F::Plain2,
                B::FuselageRoundness,
            );
            slider(
                c,
                "NOSE FRAC",
                f.nose_fraction,
                0.0,
                0.45,
                F::Plain2,
                B::FuselageNoseFraction,
            );
            slider(
                c,
                "NOSE SHAPE",
                f.nose_bluntness,
                0.0,
                1.0,
                F::Plain2,
                B::FuselageNoseBluntness,
            );
            slider(
                c,
                "TAIL FRAC",
                f.tail_fraction,
                0.0,
                0.9,
                F::Plain2,
                B::FuselageTailFraction,
            );
            slider(
                c,
                "NOSE DROOP",
                f.nose_droop,
                0.0,
                2.0,
                F::Meters,
                B::FuselageNoseDroop,
            );
            slider(
                c,
                "TAIL UPSWEEP",
                f.tail_upsweep,
                0.0,
                3.0,
                F::Meters,
                B::FuselageTailUpsweep,
            );
            slider(
                c,
                "TAIL TIP Ø",
                f.tail_tip_diameter,
                0.0,
                3.0,
                F::Meters,
                B::FuselageTailTipDiameter,
            );
            slider(
                c,
                "TAIL SHAPE",
                f.tail_bluntness,
                0.0,
                1.0,
                F::Plain2,
                B::FuselageTailBluntness,
            );
        } else if let Some(w) = &wing_v {
            slider(c, "SPAN /SIDE", w.span, 0.5, 30.0, F::Meters, B::WingSpan);
            slider(
                c,
                "ROOT CHORD",
                w.root_chord,
                0.3,
                15.0,
                F::Meters,
                B::WingRootChord,
            );
            slider(
                c,
                "TIP CHORD",
                w.tip_chord,
                0.1,
                15.0,
                F::Meters,
                B::WingTipChord,
            );
            slider(
                c,
                "SWEEP",
                w.sweep.to_degrees(),
                -10.0,
                60.0,
                F::Degrees,
                B::WingSweepDeg,
            );
            slider(
                c,
                "DIHEDRAL",
                w.dihedral.to_degrees(),
                -15.0,
                15.0,
                F::Degrees,
                B::WingDihedralDeg,
            );
            slider(
                c,
                "INCIDENCE",
                w.incidence.to_degrees(),
                -5.0,
                10.0,
                F::Degrees,
                B::WingIncidenceDeg,
            );
            slider(
                c,
                "THICK t/c",
                w.thickness,
                0.04,
                0.25,
                F::Plain2,
                B::WingThickness,
            );
        } else if let Some(g) = &gear_v {
            slider(
                c,
                "STRUT LEN",
                g.strut_length,
                0.3,
                4.0,
                F::Meters,
                B::GearStrutLength,
            );
            slider(
                c,
                "WHEEL R",
                g.wheel_radius,
                0.1,
                1.2,
                F::Meters,
                B::GearWheelRadius,
            );
        }

        // ---- Resources -----------------------------------------------------
        if let Some(entry) = &entry {
            let storable: Vec<_> = entry
                .storage_options()
                .iter()
                .filter_map(|o| {
                    resource_capacity_for(entry, &params, o.resource).map(|cap| (o, cap))
                })
                .collect();
            if !storable.is_empty() {
                spawn_heading(c, &theme, "RESOURCES", true);
                for (option, capacity) in storable {
                    if let Some((_, pool)) = pools.iter().find(|(r, _)| *r == option.resource) {
                        slider(
                            c,
                            option.resource.display_name(),
                            pool.amount,
                            0.0,
                            pool.capacity.max(1.0e-3),
                            F::Amount(option.resource.unit_label()),
                            B::ResourceAmount(option.resource),
                        );
                        c.spawn(Node {
                            justify_content: JustifyContent::FlexEnd,
                            ..default()
                        })
                        .with_children(|row| {
                            spawn_button(
                                row,
                                &theme,
                                InspectorAction::RemoveResource(option.resource),
                                "REMOVE",
                                ButtonVariant::Danger,
                                18.0,
                            );
                        });
                    } else {
                        spawn_button(
                            c,
                            &theme,
                            InspectorAction::AddResource(option.resource),
                            &format!(
                                "ADD {} ({:.0} {})",
                                option.resource.display_name().to_ascii_uppercase(),
                                capacity,
                                option.resource.unit_label()
                            ),
                            ButtonVariant::Ghost,
                            22.0,
                        );
                    }
                }
            }
        }

        // ---- Actions --------------------------------------------------------
        c.spawn((
            Node {
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(6.0),
                margin: UiRect::top(Val::Px(8.0)),
                ..default()
            },
            Name::new("ShipyardInspectorActions"),
        ))
        .with_children(|row| {
            if !is_root {
                spawn_button(
                    row,
                    &theme,
                    InspectorAction::SetRoot,
                    "SET ROOT",
                    ButtonVariant::Ghost,
                    24.0,
                );
            }
            spawn_button(
                row,
                &theme,
                InspectorAction::Delete,
                "DELETE",
                ButtonVariant::Danger,
                24.0,
            );
        });
    });
}

/// Live read-only readout for the selection — fixed specs, dry mass,
/// attach nodes, symmetry note. Regenerated each frame (cheap), written
/// only on change.
pub(super) fn update_info_text(
    state: Res<EditorState>,
    groups: Query<(Entity, &SymmetryGroup), With<EditorPart>>,
    parts: KindQuery,
    mut info: Query<&mut Text, With<InspectorInfoText>>,
) {
    let Ok(mut text) = info.single_mut() else {
        return;
    };
    let Some(sel) = state.selected else {
        return;
    };
    let target = symmetry_edit_target(sel, &groups);
    let Ok((
        _,
        catalog_ref,
        nodes,
        pod,
        dec,
        adapter,
        tank,
        fuselage,
        engine,
        intake,
        wing,
        gear,
        _,
    )) = parts.get(target)
    else {
        return;
    };

    let mut lines: Vec<String> = vec![format!("catalog: {}", catalog_ref.id)];
    if let Some(p) = pod {
        lines.push(format!("{} · Ø{:.2} m (fixed)", p.model, p.diameter));
        lines.push(format!("dry mass {:.0} kg (fixed)", p.dry_mass));
    } else if let Some(d) = dec {
        if Some(target) != state.ship_root {
            lines.push(format!("Ø{:.2} m (from parent)", d.diameter));
        }
        lines.push(format!("ejection {:.0} N·s", d.ejection_impulse));
        lines.push(format!("dry mass {:.0} kg", d.dry_mass));
    } else if let Some(a) = adapter {
        if Some(target) != state.ship_root {
            lines.push(format!("Ø{:.2} m (from parent)", a.diameter));
        }
        lines.push(format!("dry mass {:.0} kg", a.dry_mass));
    } else if let Some(t) = tank {
        if Some(target) != state.ship_root {
            lines.push(format!("Ø{:.2} m (from parent)", t.diameter));
        }
        lines.push(format!("dry mass {:.0} kg", t.dry_mass));
    } else if let Some(f) = fuselage {
        if Some(target) != state.ship_root {
            lines.push(format!("Ø{:.2} m (from parent)", f.max_width));
        }
        lines.push(format!("dry mass {:.0} kg", f.dry_mass));
    } else if let Some(e) = engine {
        lines.push(format!("{} · {}", e.model, e.geometry.label()));
        lines.push(format!(
            "Ø{:.2} m · {:.0} kN · Isp {:.0} s (fixed)",
            e.diameter,
            e.thrust / 1000.0,
            e.isp
        ));
        lines.push(format!("dry mass {:.0} kg", e.dry_mass));
        for r in &e.reactants {
            lines.push(format!(
                "  {} {:.0}%",
                r.resource.display_name(),
                r.mass_fraction * 100.0
            ));
        }
        if let Some(req) = e.intake_requirement {
            lines.push(format!(
                "intake req {:.2} m² {}",
                req.area_m2,
                req.kind.label()
            ));
        }
        if let Some(cap) = e.builtin_intake {
            lines.push(format!(
                "built-in intake {:.2} m² ({:.0}%)",
                cap.area_m2,
                cap.efficiency * 100.0
            ));
        }
    } else if let Some(i) = intake {
        lines.push(format!("{} · Ø{:.2} m (fixed)", i.model, i.diameter));
        lines.push(format!(
            "capture {:.2} m² {} ({:.0}%)",
            i.capture.area_m2,
            i.capture.kind.label(),
            i.capture.efficiency * 100.0
        ));
    } else if let Some(w) = wing {
        lines.push(format!("dry mass {:.0} kg/panel", w.dry_mass));
        lines.push(if groups.get(target).is_ok() {
            "symmetry: mirrored pair (edits sync)".into()
        } else {
            "symmetry: single".into()
        });
    } else if let Some(g) = gear {
        lines.push(if g.track_fraction > 0.0 {
            "main gear (L/R legs)".into()
        } else {
            "nose gear".into()
        });
        lines.push(format!("dry mass {:.0} kg", g.dry_mass));
    }
    let mut node_line = String::from("nodes:");
    for (id, node) in &nodes.nodes {
        node_line.push_str(&format!(" {id} Ø{:.2}m", node.diameter));
    }
    lines.push(node_line);

    let joined = lines.join("\n");
    if **text != joined {
        **text = joined;
    }
}

/// Push slider drags into the bound part fields on the edit target.
pub(super) fn apply_param_bindings(
    state: Res<EditorState>,
    groups: Query<(Entity, &SymmetryGroup), With<EditorPart>>,
    changed: Query<(&UiSlider, &ParamBinding), Changed<UiSlider>>,
    mut parts: KindQueryMut,
) {
    if changed.is_empty() {
        return;
    }
    let Some(sel) = state.selected else {
        return;
    };
    let target = symmetry_edit_target(sel, &groups);
    let Ok((mut dec, mut adapter, mut tank, mut fuselage, mut wing, mut gear, mut resources)) =
        parts.get_mut(target)
    else {
        return;
    };

    // Value-guarded writes: only DerefMut (and so only trigger the core's
    // Changed-driven rebuilds) when the value actually moved.
    fn set(dst: &mut f32, v: f32) -> bool {
        if (*dst - v).abs() > 1.0e-5 {
            *dst = v;
            true
        } else {
            false
        }
    }
    macro_rules! write_field {
        ($opt:ident, $field:ident, $v:expr) => {
            if let Some(part) = $opt.as_mut()
                && (part.$field - $v).abs() > 1.0e-5
            {
                set(&mut part.$field, $v);
            }
        };
    }

    for (slider, binding) in &changed {
        let v = slider.value;
        match binding {
            ParamBinding::DecouplerDiameter => write_field!(dec, diameter, v),
            ParamBinding::AdapterDiameter => write_field!(adapter, diameter, v),
            ParamBinding::AdapterTargetDiameter => write_field!(adapter, target_diameter, v),
            ParamBinding::TankDiameter => write_field!(tank, diameter, v),
            ParamBinding::TankLength => write_field!(tank, length, v),
            ParamBinding::FuselageLength => write_field!(fuselage, length, v),
            ParamBinding::FuselageWidth => write_field!(fuselage, max_width, v),
            ParamBinding::FuselageHeight => write_field!(fuselage, max_height, v),
            ParamBinding::FuselageRoundness => write_field!(fuselage, roundness, v),
            ParamBinding::FuselageNoseFraction => write_field!(fuselage, nose_fraction, v),
            ParamBinding::FuselageNoseBluntness => write_field!(fuselage, nose_bluntness, v),
            ParamBinding::FuselageTailFraction => write_field!(fuselage, tail_fraction, v),
            ParamBinding::FuselageNoseDroop => write_field!(fuselage, nose_droop, v),
            ParamBinding::FuselageTailUpsweep => write_field!(fuselage, tail_upsweep, v),
            ParamBinding::FuselageTailTipDiameter => write_field!(fuselage, tail_tip_diameter, v),
            ParamBinding::FuselageTailBluntness => write_field!(fuselage, tail_bluntness, v),
            ParamBinding::WingSpan => write_field!(wing, span, v),
            ParamBinding::WingRootChord => write_field!(wing, root_chord, v),
            ParamBinding::WingTipChord => write_field!(wing, tip_chord, v),
            ParamBinding::WingSweepDeg => write_field!(wing, sweep, v.to_radians()),
            ParamBinding::WingDihedralDeg => write_field!(wing, dihedral, v.to_radians()),
            ParamBinding::WingIncidenceDeg => write_field!(wing, incidence, v.to_radians()),
            ParamBinding::WingThickness => write_field!(wing, thickness, v),
            ParamBinding::GearStrutLength => write_field!(gear, strut_length, v),
            ParamBinding::GearWheelRadius => write_field!(gear, wheel_radius, v),
            ParamBinding::ResourceAmount(res) => {
                if let Some(resources) = resources.as_mut()
                    && let Some(pool) = resources.pools.get(res)
                {
                    let clamped = v.clamp(0.0, pool.capacity);
                    if (pool.amount - clamped).abs() > 1.0e-3 {
                        resources.pools.get_mut(res).unwrap().amount = clamped;
                    }
                }
            }
        }
    }
}

/// Pull model → slider whenever the user isn't dragging, so propagated
/// diameters, recomputed capacities, and symmetry-synced params stay honest.
pub(super) fn refresh_sliders_from_model(
    state: Res<EditorState>,
    groups: Query<(Entity, &SymmetryGroup), With<EditorPart>>,
    parts: KindQuery,
    mut sliders: Query<(&Interaction, &mut UiSlider, &ParamBinding)>,
) {
    let Some(sel) = state.selected else {
        return;
    };
    let target = symmetry_edit_target(sel, &groups);
    let Ok((_, _, nodes, _, dec, adapter, tank, fuselage, _, _, wing, gear, resources)) =
        parts.get(target)
    else {
        return;
    };

    for (interaction, mut slider, binding) in &mut sliders {
        if matches!(interaction, Interaction::Pressed) {
            continue;
        }
        let model = match binding {
            ParamBinding::DecouplerDiameter => dec.map(|d| d.diameter),
            ParamBinding::AdapterDiameter => adapter.map(|a| a.diameter),
            ParamBinding::AdapterTargetDiameter => adapter.map(|a| a.target_diameter),
            ParamBinding::TankDiameter => tank.map(|t| t.diameter),
            ParamBinding::TankLength => {
                // Length cap tracks the effective (propagated) diameter.
                if let Some(t) = tank {
                    let effective_d = nodes.get("top").map(|n| n.diameter).unwrap_or(t.diameter);
                    let max = 8.0 * effective_d;
                    if (slider.max - max).abs() > 1.0e-4 {
                        slider.max = max;
                    }
                }
                tank.map(|t| t.length)
            }
            ParamBinding::FuselageLength => fuselage.map(|f| f.length),
            ParamBinding::FuselageWidth => fuselage.map(|f| f.max_width),
            ParamBinding::FuselageHeight => fuselage.map(|f| f.max_height),
            ParamBinding::FuselageRoundness => fuselage.map(|f| f.roundness),
            ParamBinding::FuselageNoseFraction => fuselage.map(|f| f.nose_fraction),
            ParamBinding::FuselageNoseBluntness => fuselage.map(|f| f.nose_bluntness),
            ParamBinding::FuselageTailFraction => fuselage.map(|f| f.tail_fraction),
            ParamBinding::FuselageNoseDroop => fuselage.map(|f| f.nose_droop),
            ParamBinding::FuselageTailUpsweep => fuselage.map(|f| f.tail_upsweep),
            ParamBinding::FuselageTailTipDiameter => fuselage.map(|f| f.tail_tip_diameter),
            ParamBinding::FuselageTailBluntness => fuselage.map(|f| f.tail_bluntness),
            ParamBinding::WingSpan => wing.map(|w| w.span),
            ParamBinding::WingRootChord => wing.map(|w| w.root_chord),
            ParamBinding::WingTipChord => wing.map(|w| w.tip_chord),
            ParamBinding::WingSweepDeg => wing.map(|w| w.sweep.to_degrees()),
            ParamBinding::WingDihedralDeg => wing.map(|w| w.dihedral.to_degrees()),
            ParamBinding::WingIncidenceDeg => wing.map(|w| w.incidence.to_degrees()),
            ParamBinding::WingThickness => wing.map(|w| w.thickness),
            ParamBinding::GearStrutLength => gear.map(|g| g.strut_length),
            ParamBinding::GearWheelRadius => gear.map(|g| g.wheel_radius),
            ParamBinding::ResourceAmount(res) => {
                let pool = resources.and_then(|r| r.pools.get(res));
                if let Some(pool) = pool {
                    let max = pool.capacity.max(1.0e-3);
                    if (slider.max - max).abs() > 1.0e-4 {
                        slider.max = max;
                    }
                }
                pool.map(|p| p.amount)
            }
        };
        if let Some(model) = model
            && (slider.value - model).abs() > 1.0e-4
        {
            slider.value = model;
        }
    }
}

pub(super) fn handle_actions(
    interactions: Query<(&Interaction, &InspectorAction), Changed<Interaction>>,
    mut state: ResMut<EditorState>,
    groups: Query<(Entity, &SymmetryGroup), With<EditorPart>>,
    catalog: Res<PartCatalog>,
    // KindQuery reads PartResources, the pool edit writes it — disjoint in
    // time but not in access, so they share a ParamSet.
    mut queries: ParamSet<(KindQuery, Query<&mut PartResources, With<EditorPart>>)>,
) {
    for (interaction, action) in &interactions {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        match action {
            InspectorAction::SetRoot => state.set_as_root = true,
            InspectorAction::Delete => state.delete_selected = true,
            InspectorAction::AddResource(res) => {
                let Some(sel) = state.selected else { continue };
                let target = symmetry_edit_target(sel, &groups);
                let pool = {
                    let info = queries.p0();
                    let Ok((
                        _,
                        catalog_ref,
                        _,
                        _,
                        dec,
                        adapter,
                        tank,
                        fuselage,
                        _,
                        _,
                        wing,
                        gear,
                        _,
                    )) = info.get(target)
                    else {
                        continue;
                    };
                    let Ok(entry) = catalog.resolve(&catalog_ref.id) else {
                        continue;
                    };
                    let params = inspector_params(dec, adapter, tank, fuselage, wing, gear);
                    let Some(capacity) = resource_capacity_for(entry, &params, *res) else {
                        continue;
                    };
                    let fill = entry
                        .storage_options()
                        .iter()
                        .find(|o| o.resource == *res)
                        .map(|o| o.default_fill_fraction.clamp(0.0, 1.0))
                        .unwrap_or(1.0);
                    ResourcePool {
                        capacity,
                        amount: capacity * fill,
                    }
                };
                if let Ok(mut resources) = queries.p1().get_mut(target) {
                    resources.pools.insert(*res, pool);
                }
            }
            InspectorAction::RemoveResource(res) => {
                let Some(sel) = state.selected else { continue };
                let target = symmetry_edit_target(sel, &groups);
                if let Ok(mut resources) = queries.p1().get_mut(target) {
                    resources.pools.remove(res);
                }
            }
        }
    }
}
