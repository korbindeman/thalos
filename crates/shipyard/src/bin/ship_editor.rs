//! Standalone interactive 3D ship editor — the **egui front-end** over the
//! UI-agnostic editor core in [`thalos_shipyard::editor`].
//!
//! All editing behaviour (placement, symmetry, visuals, blueprint I/O) lives
//! in the core's `ShipEditorCorePlugin`; this binary contributes the app
//! shell, the orbit camera, the celestial backdrop, and the egui panels that
//! read/write [`EditorState`]. The in-game editor
//! (`thalos_game::shipyard_editor`) is a second front-end over the same core
//! with native Bevy UI.
//!
//! Workflow:
//! - Left panel: parts palette + file I/O. Clicking a part arms it as
//!   "pending" — a popup then lists free attach nodes on the existing ship
//!   to place the pending part at.
//! - Right panel: inspector for the selected part (editable params,
//!   resource pools, delete).
//! - Viewport: orbit camera (drag + scroll), gizmo spheres at each
//!   attach node, parts rendered as cylinders/frustums sized from their
//!   attach-node diameters.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::gestures::PinchGesture;
use bevy::mesh::{Indices, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::picking::hover::HoverMap;
use bevy::picking::mesh_picking::ray_cast::RayCastVisibility;
use bevy::picking::mesh_picking::{MeshPickingPlugin, MeshPickingSettings};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;
use bevy::window::PrimaryWindow;
use bevy_egui::{EguiContextSettings, EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use std::collections::HashMap;

use thalos_celestial::Universe;
use thalos_celestial::generate::{DefaultGenParams, generate_default};
use thalos_input::enhanced::{ActionSources, EnhancedInputSystems};
use thalos_input::settings::InputSettings;
use thalos_input::shipyard::{ShipyardInputIntent, ShipyardInputPlugin};
use thalos_shipyard::blueprint::default_params_for;
use thalos_shipyard::editor::{
    CLICK_THRESHOLD_PX, CollectQuery, EditorState, EditorUiGate, EditorViewCamera, PendingPart,
    PlacementSnap, ShipEditorCorePlugin, SymmetryMode, TankResizeArrow, TankResizeDrag,
    collect_blueprint, format_delta_v, format_duration_s, format_mass_kg, format_thrust,
    inspector_params, kind_order, palette_category_label, palette_category_order,
    palette_part_summary, symmetry_edit_target,
};
use thalos_shipyard::editor::{BuildOrientation, EditorPart};
use thalos_shipyard::*;

const CATALOG_PATH: &str = "assets/parts.ron";

fn palette_part_button(ui: &mut egui::Ui, entry: &CatalogEntry) -> bool {
    let label = format!("{}\n{}", entry.display_name(), palette_part_summary(entry));
    ui.add(
        egui::Button::new(label)
            .wrap()
            .min_size(egui::vec2(ui.available_width(), 38.0)),
    )
    .on_hover_text(entry.kind_name())
    .clicked()
}

fn main() {
    let catalog = match PartCatalog::load_from_path(CATALOG_PATH) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load parts catalog from {CATALOG_PATH}: {e}");
            std::process::exit(1);
        }
    };

    App::new()
        .insert_resource(
            InputSettings::load_from_path("assets/input.ron")
                .expect("Failed to load input bindings from assets/input.ron"),
        )
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Thalos Shipyard".into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(bevy::asset::AssetPlugin {
                    // Resolve shaders from the workspace-root `assets/` dir,
                    // matching `thalos_game` and `thalos_body_editor`.
                    file_path: "../../assets".to_string(),
                    ..default()
                }),
        )
        .insert_resource(catalog)
        .add_plugins(EguiPlugin::default())
        .add_plugins(ShipyardInputPlugin)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(MeshPickingPlugin)
        .insert_resource(MeshPickingSettings {
            require_markers: false,
            // `VisibleInView` so hidden handles (resize arrow, non-pending
            // pins) don't absorb clicks from the body behind them.
            ray_cast_visibility: RayCastVisibility::VisibleInView,
        })
        .add_plugins(ShipyardPlugin)
        .add_plugins(ShipEditorCorePlugin)
        .add_plugins(SkyBackdropPlugin)
        .init_resource::<SkyBackdropEnabled>()
        .add_systems(Startup, setup)
        .add_systems(
            PreUpdate,
            gate_shipyard_input_sources.before(EnhancedInputSystems::Update),
        )
        .add_systems(
            Update,
            (
                orbit_camera,
                recenter_camera_on_orientation_change,
                disable_egui_pointer_capture,
            ),
        )
        .add_systems(EguiPrimaryContextPass, editor_ui)
        .run();
}

// ---------------------------------------------------------------------------
// Setup: camera + lights (the editor scene shell)
// ---------------------------------------------------------------------------

#[derive(Component)]
struct OrbitCamera {
    focus: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(8.0, 4.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitCamera {
            focus: Vec3::new(0.0, -2.0, 0.0),
            distance: 12.0,
            yaw: 0.8,
            pitch: 0.4,
        },
        EditorViewCamera,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        PointLight {
            intensity: 400_000.0,
            ..default()
        },
        Transform::from_xyz(-6.0, 4.0, -4.0),
    ));
}

/// Re-centre the orbit camera when the build layout flips, so the craft
/// stays framed (it moves from a tall upright stack to a level fuselage).
fn recenter_camera_on_orientation_change(
    orientation: Res<BuildOrientation>,
    mut cam: Query<&mut OrbitCamera>,
) {
    if !orientation.is_changed() {
        return;
    }
    for mut c in cam.iter_mut() {
        c.focus = if orientation.horizontal {
            Vec3::ZERO
        } else {
            Vec3::new(0.0, -2.0, 0.0)
        };
    }
}

/// bevy_egui's default `capture_pointer_input` writes a fake top-priority
/// PointerHits for the egui context entity whenever egui wants pointer
/// input, which redirects every click away from our meshes. Disable it and
/// filter picks manually via [`EditorUiGate`] (kept in sync from egui in
/// [`gate_shipyard_input_sources`]).
fn disable_egui_pointer_capture(mut q: Query<&mut EguiContextSettings>) {
    for mut s in q.iter_mut() {
        if s.capture_pointer_input {
            s.capture_pointer_input = false;
        }
    }
}

/// Gate raw input sources on egui focus, and mirror egui's pointer state
/// into the core's [`EditorUiGate`] so the core's picking observers and
/// preview systems stand down over panels.
fn gate_shipyard_input_sources(
    mut action_sources: ResMut<ActionSources>,
    mut contexts: EguiContexts,
    mut ui_gate: ResMut<EditorUiGate>,
) {
    let (pointer_busy, keyboard_busy) = contexts
        .ctx_mut()
        .map(|ctx| {
            (
                ctx.is_pointer_over_area() || ctx.wants_pointer_input(),
                ctx.wants_keyboard_input(),
            )
        })
        .unwrap_or((false, false));
    thalos_input::gating::set_mouse_sources(&mut action_sources, !pointer_busy);
    thalos_input::gating::set_keyboard_source(&mut action_sources, !keyboard_busy);
    if ui_gate.pointer_busy != pointer_busy {
        ui_gate.pointer_busy = pointer_busy;
    }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

fn orbit_camera(
    mut cam: Query<(&mut Transform, &mut OrbitCamera)>,
    input: Res<ShipyardInputIntent>,
    mut pinch: MessageReader<PinchGesture>,
    mut contexts: EguiContexts,
    state: Res<EditorState>,
    resize_drag: Res<TankResizeDrag>,
    hover_map: Res<HoverMap>,
    orientation: Res<BuildOrientation>,
    arrows: Query<(), With<TankResizeArrow>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut press_cursor: Local<Option<Vec2>>,
    mut orbit_active: Local<bool>,
) {
    let pointer_over_egui = contexts
        .ctx_mut()
        .map(|c| c.is_pointer_over_area() || c.wants_pointer_input())
        .unwrap_or(false);

    let pointer_on_arrow = hover_map
        .0
        .values()
        .any(|hovers| hovers.keys().any(|e| arrows.get(*e).is_ok()));

    let delta = input.camera_motion;
    let wheel = input.camera_wheel;
    let mut pinch_d: f32 = 0.0;
    for p in pinch.read() {
        pinch_d += p.0;
    }

    let shift = input.precision_slow;

    // Click/drag arbitration for LMB: we want a press→release on a pin to
    // fire `Pointer<Click>`, which Bevy's picking only emits when the same
    // entity is hovered at press and at release. Rotating the camera mid-
    // press moves the world under the cursor and breaks that. So while a
    // part is pending we hold orbit until the cursor has moved past
    // CLICK_THRESHOLD_PX from the press location; once over, we stay in
    // orbit mode for the remainder of the press. With no pending part
    // there's no click target to protect, so orbit is unconditional.
    let cursor = windows.single().ok().and_then(|w| w.cursor_position());
    if input.primary_started {
        *press_cursor = cursor;
        *orbit_active = false;
    }
    if input.primary_released {
        *press_cursor = None;
        *orbit_active = false;
    }
    if !*orbit_active
        && let (Some(press), Some(current)) = (*press_cursor, cursor)
        && (current - press).length() >= CLICK_THRESHOLD_PX
    {
        *orbit_active = true;
    }

    // Also suppress while the pointer is over a resize arrow (or actively
    // dragging one) so the camera doesn't twitch between mouse-down and
    // DragStart firing.
    let orbit_allowed = !pointer_over_egui
        && resize_drag.active.is_none()
        && !pointer_on_arrow
        && (state.pending.is_none() || *orbit_active);

    for (mut t, mut orbit) in cam.iter_mut() {
        if orbit_allowed && input.primary_pressed {
            orbit.yaw -= delta.x * 0.005;
            orbit.pitch = (orbit.pitch - delta.y * 0.005).clamp(-1.5, 1.5);
        }

        if !pointer_over_egui && (wheel.x.abs() > 0.0 || wheel.y.abs() > 0.0) {
            if shift {
                orbit.distance = (orbit.distance * (1.0 - wheel.y * 0.05)).clamp(2.0, 200.0);
            } else {
                // Vertical scroll: pan along the build's long axis (body +Y),
                // which the horizontal layout lays down to −Z — so scrolling
                // tracks the fuselage in either layout instead of always world-up.
                if wheel.y.abs() > 0.0 {
                    let pan = wheel.y * orbit.distance * 0.015;
                    orbit.focus += orientation.rotation() * Vec3::Y * pan;
                }
                // Horizontal scroll (trackpad two-finger): pan perpendicular to
                // the build axis using the camera's current azimuth so left/right
                // always matches what's on screen regardless of orbit angle.
                if wheel.x.abs() > 0.0 {
                    let cam_right = Quat::from_rotation_y(orbit.yaw) * Vec3::X;
                    let pan = wheel.x * orbit.distance * 0.015;
                    orbit.focus += cam_right * pan;
                }
            }
        }

        // Trackpad pinch zooms regardless of shift.
        if !pointer_over_egui && pinch_d.abs() > 0.0 {
            orbit.distance = (orbit.distance * (1.0 - pinch_d * 8.0)).clamp(2.0, 200.0);
        }

        let rot = Quat::from_euler(EulerRot::YXZ, orbit.yaw, -orbit.pitch, 0.0);
        let offset = rot * Vec3::new(0.0, 0.0, orbit.distance);
        t.translation = orbit.focus + offset;
        t.look_at(orbit.focus, Vec3::Y);
    }
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

type InspectorQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static CatalogRef,
        &'static AttachNodes,
        Option<&'static mut CommandPod>,
        Option<&'static mut Decoupler>,
        Option<&'static mut Adapter>,
        Option<&'static mut FuelTank>,
        Option<&'static mut Fuselage>,
        Option<&'static mut Engine>,
        Option<&'static mut AirIntake>,
        Option<&'static mut Wing>,
        Option<&'static mut Gear>,
        Option<&'static mut PartResources>,
    ),
    With<EditorPart>,
>;

fn draw_ship_stats(ui: &mut egui::Ui, stats: &ShipStats) {
    // Δv is reported per-stage in the Staging panel (the whole-ship rocket
    // equation is misleading for a multi-stage vessel), so it is not repeated
    // here. `vacuum` is still used for the whole-ship burn-time line.
    let vacuum = stats.vacuum_delta_v();
    ui.label(format!("Wet mass: {}", format_mass_kg(stats.wet_mass_kg())));
    ui.label(format!("Dry mass: {}", format_mass_kg(stats.dry_mass_kg)));
    ui.label(format!(
        "Propellant: {}",
        format_mass_kg(stats.propellant_mass_kg)
    ));
    ui.label(format!("Thrust: {}", format_thrust(stats.total_thrust_n)));
    if stats.wet_mass_kg() > 0.0 && stats.total_thrust_n > 0.0 {
        ui.label(format!("TWR: {:.2}", stats.current_acceleration() / G0));
    }
    if let Some(burn_s) = vacuum.burn_time_s {
        ui.label(format!("Full burn: {}", format_duration_s(burn_s)));
    }
    // Geometry-derived "will it fly" feedback. There is no flight model
    // yet (M6) — these are design references.
    if stats.wing_area_m2 > 0.0 {
        ui.label(format!("Wing area: {:.1} m²", stats.wing_area_m2));
        ui.label(format!("MAC: {:.2} m", stats.mean_aerodynamic_chord_m));
    }
}

/// Per-stage Δv / fuel breakdown, one card per stage in firing order. Stages
/// are derived from decoupler position (there is no authored stage list), so
/// this is a readout — you reorder staging by moving decouplers in the part
/// tree, not by dragging here. Tanks are previewed full.
fn draw_staging(ui: &mut egui::Ui, summaries: &[StageSummary]) {
    if summaries.is_empty() {
        ui.label("(no stages)");
        return;
    }

    let total_dv: f64 = summaries.iter().map(|s| s.delta_v_m_s).sum();
    ui.label(format!("Total Δv: {}", format_delta_v(total_dv)));
    ui.add_space(4.0);

    for s in summaries {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong(format!("Stage {}", s.number));
                ui.separator();
                if s.has_engine {
                    ui.label(format_delta_v(s.delta_v_m_s));
                } else {
                    ui.weak("drop only");
                }
            });
            if s.fuel_kg > 0.0 {
                ui.label(format!("Fuel: {}", format_mass_kg(s.fuel_kg)));
            }
            for res in thalos_shipyard::Resource::MASS_BEARING {
                let Some(totals) = s.resources.get(&res) else {
                    continue;
                };
                if totals.capacity <= 0.0 && totals.amount <= 0.0 {
                    continue;
                }
                let frac = if totals.capacity > 0.0 {
                    (totals.amount / totals.capacity).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };
                ui.add(
                    egui::ProgressBar::new(frac)
                        .desired_height(8.0)
                        .text(format!(
                            "{} {}",
                            res.display_name(),
                            format_mass_kg(totals.mass_kg)
                        )),
                );
            }
        });
    }
}

fn editor_ui(
    mut contexts: EguiContexts,
    mut state: ResMut<EditorState>,
    mut part_queries: ParamSet<(InspectorQuery, CollectQuery)>,
    mut ships: Query<&mut Ship, With<EditorPart>>,
    attachments: Query<(Entity, &Attachment), With<EditorPart>>,
    surface_mounts: Query<(Entity, &SurfaceMount), With<EditorPart>>,
    groups: Query<(Entity, &SymmetryGroup), With<EditorPart>>,
    catalog: Res<PartCatalog>,
    mut sky: ResMut<SkyBackdropEnabled>,
    mut clear_color: ResMut<ClearColor>,
    mut orientation: ResMut<BuildOrientation>,
    mut symmetry_mode: ResMut<SymmetryMode>,
    mut placement_snap: ResMut<PlacementSnap>,
    diagnostics: Res<DiagnosticsStore>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let ctx = ctx.clone();

    // Collect the blueprint once; both the aggregate stats and the per-stage
    // staging preview are projections of it.
    let (ship_stats, stage_summaries) = {
        let collect_parts = part_queries.p1();
        let blueprint = state.ship_root.and_then(|root| {
            let ship = Ship {
                name: String::new(),
                root,
            };
            collect_blueprint(&ship, &collect_parts, &attachments, &surface_mounts, &groups)
        });
        let stats = blueprint.as_ref().map(|bp| bp.stats(&catalog));
        let staging = blueprint.as_ref().map(|bp| bp.stage_summaries(&catalog));
        (stats, staging)
    };

    // -------- Left palette --------
    egui::SidePanel::left("palette")
        .default_width(180.0)
        .show(&ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
            let fps = diagnostics
                .get(&FrameTimeDiagnosticsPlugin::FPS)
                .and_then(|d| d.smoothed())
                .unwrap_or(0.0);
            ui.label(format!("FPS: {:.0}", fps));
            ui.separator();
            ui.heading("Parts");
            // Sort by category/kind/display name so palette ordering is
            // stable across runs (HashMap iteration is not).
            let mut entries: Vec<(&CatalogId, &CatalogEntry)> = catalog.parts.iter().collect();
            entries.sort_by_key(|(_, e)| {
                (
                    palette_category_order(e),
                    kind_order(e),
                    e.display_name().to_string(),
                )
            });
            let mut current_category = None;
            for (id, entry) in entries {
                let category = palette_category_label(entry);
                if current_category != Some(category) {
                    if current_category.is_some() {
                        ui.add_space(6.0);
                    }
                    ui.label(egui::RichText::new(category).strong());
                    current_category = Some(category);
                }

                if palette_part_button(ui, entry) {
                    state.pending = Some(PendingPart {
                        catalog_id: id.clone(),
                        params: default_params_for(entry),
                    });
                }
            }

            ui.separator();
            ui.heading("Ship");
            ui.horizontal(|ui| {
                ui.label("Name:");
                if let Some(se) = state.ship_entity {
                    if let Ok(mut ship) = ships.get_mut(se) {
                        ui.text_edit_singleline(&mut ship.name);
                    }
                } else {
                    ui.text_edit_singleline(&mut state.ship_name);
                }
            });
            ui.add_enabled_ui(state.ship_entity.is_some(), |ui| {
                if ui.button("Save").clicked() {
                    state.save_requested = true;
                }
            });
            if ui.button("Refresh list").clicked() {
                state.refresh_list = true;
            }

            ui.separator();
            ui.heading("Ship stats");
            match &ship_stats {
                Some(Ok(stats)) => draw_ship_stats(ui, stats),
                Some(Err(e)) => {
                    ui.colored_label(egui::Color32::from_rgb(220, 110, 60), format!("{e}"));
                }
                None => {
                    ui.label("(no ship)");
                }
            }

            ui.separator();
            ui.heading("Saved ships");
            let ship_list = state.ship_list.clone();
            if ship_list.is_empty() {
                ui.label("(none)");
            }
            for saved in ship_list {
                ui.horizontal(|ui| {
                    if ui.button("Load").clicked() {
                        state.load_target = Some(saved.slug.clone());
                    }
                    if ui.button("X").clicked() {
                        state.delete_file = Some(saved.slug.clone());
                    }
                    ui.label(&saved.name)
                        .on_hover_text(format!("{}.ron", saved.slug));
                });
            }

            ui.separator();
            ui.heading("Symmetry");
            ui.checkbox(&mut symmetry_mode.mirror, "Mirror (2×)")
                .on_hover_text(
                    "KSP-style: placing a wing/gear off-centre stamps a linked left/right pair. \
                     A part placed on a mirrored wing (e.g. a nacelle) auto-mirrors onto both.",
                );
            ui.checkbox(&mut placement_snap.enabled, "Angle snap (15°)")
                .on_hover_text(
                    "Magnetic snapping around the fuselage: a body-skin mount's azimuth rounds to \
                     15° steps as the cursor sweeps the hull, so gear/wings land dead-on the \
                     belly / sides. Off = free placement.",
                );

            ui.separator();
            ui.heading("View");
            {
                // Read through bypass_change_detection so the mere act of
                // rendering the checkbox doesn't mark BuildOrientation as
                // changed every frame (which would fire recenter_camera_on_
                // orientation_change and reset orbit.focus on every frame).
                let mut horiz = orientation.bypass_change_detection().horizontal;
                if ui
                    .checkbox(&mut horiz, "Horizontal layout (aircraft)")
                    .on_hover_text(
                        "Lay the build down like KSP's SPH — fuselage fore/aft, wings level, fin up.",
                    )
                    .changed()
                {
                    orientation.horizontal = horiz;
                }
            }
            if ui.checkbox(&mut sky.0, "Celestial backdrop").changed() {
                // Black clears behind the additively-blended stars so
                // they read as points of light; the default grey washes
                // them out.
                clear_color.0 = if sky.0 {
                    Color::BLACK
                } else {
                    ClearColor::default().0
                };
            }

            ui.separator();
            ui.label(format!("Status: {}", state.status));
            if state.pending.is_some() {
                let pending = state.pending.as_ref().unwrap();
                let surface_hint = matches!(
                    pending.params,
                    PartParams::Wing { .. } | PartParams::Gear { .. }
                ) || catalog.resolve(&pending.catalog_id).is_ok_and(|entry| {
                    matches!(
                        entry,
                        CatalogEntry::Engine(e) if e.geometry == EngineGeometry::JetNacelle
                    )
                });
                ui.colored_label(
                    egui::Color32::YELLOW,
                    if surface_hint {
                        "Pending part — click a compatible surface to place."
                    } else {
                        "Pending part — pick an attach node to place."
                    },
                );
                if ui.button("Cancel pending").clicked() {
                    state.pending = None;
                }
            }
            }); // scroll area
        });

    // -------- Right inspector --------
    egui::SidePanel::right("inspector")
        .default_width(260.0)
        .show(&ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Inspector");
            let Some(sel) = state.selected else {
                ui.label("(no selection)");
                return;
            };
            // KSP symmetry: edit the group's primary regardless of which member
            // was clicked. `sync_symmetry_groups` propagates the change to the
            // counterpart(s); editing a mirror counterpart directly would be
            // reverted next frame, leaving its sliders looking dead.
            let sel = symmetry_edit_target(sel, &groups);
            let mut parts = part_queries.p0();
            let Ok((
                entity,
                catalog_ref,
                nodes,
                mut pod,
                mut dec,
                mut adapter,
                mut tank,
                mut fuselage,
                mut engine,
                mut intake,
                mut wing,
                mut gear,
                mut res,
            )) = parts.get_mut(sel)
            else {
                ui.label("(invalid selection)");
                return;
            };
            ui.label(format!("Entity: {entity:?}"));
            let is_root = Some(sel) == state.ship_root;

            if let Some(p) = pod.as_deref_mut() {
                ui.label(format!("Kind: Command Pod ({})", p.geometry.label()));
                ui.label(format!("Model: {}", p.model));
                ui.label(format!("Diameter: {:.2}m (fixed)", p.diameter));
                ui.label(format!("Dry mass: {:.0} kg (fixed)", p.dry_mass));
            } else if let Some(d) = dec.as_deref_mut() {
                ui.label("Kind: Decoupler");
                if is_root {
                    ui.add(egui::Slider::new(&mut d.diameter, 0.3..=6.0).text("Diameter"));
                } else {
                    ui.label(format!("Diameter: {:.2}m (from parent)", d.diameter));
                }
                // Ejection impulse and dry mass are catalog-derived from
                // diameter (`ejection_impulse_per_diameter`, `mass_per_diameter`).
                // Editing them here would just be overwritten by
                // `recompute::recompute_decoupler_state`.
                ui.label(format!("Ejection impulse: {:.0} N·s", d.ejection_impulse));
                ui.label(format!("Dry mass: {:.0} kg", d.dry_mass));
            } else if let Some(a) = adapter.as_deref_mut() {
                ui.label("Kind: Adapter");
                if is_root {
                    ui.add(egui::Slider::new(&mut a.diameter, 0.3..=6.0).text("Diameter"));
                } else {
                    ui.label(format!("Diameter: {:.2}m (from parent)", a.diameter));
                }
                ui.add(
                    egui::Slider::new(&mut a.target_diameter, 0.3..=6.0).text("Target diameter"),
                );
                // dry_mass scales with frustum surface area via the
                // catalog's `wall_mass_per_m2`; recomputed by
                // `recompute::recompute_adapter_state` on every change.
                ui.label(format!("Dry mass: {:.0} kg", a.dry_mass));
            } else if let Some(t) = tank.as_deref_mut() {
                ui.label("Kind: Fuel Tank");
                if is_root {
                    ui.add(egui::Slider::new(&mut t.diameter, 0.3..=6.0).text("Diameter"));
                } else {
                    ui.label(format!("Diameter: {:.2}m (from parent)", t.diameter));
                }
                let effective_d = nodes.get("top").map(|n| n.diameter).unwrap_or(t.diameter);
                let max_length = 8.0 * effective_d;
                ui.add(egui::Slider::new(&mut t.length, 0.5..=max_length).text("Length"));
                // dry_mass and pool capacities scale with cylinder
                // geometry via the catalog; recomputed by
                // `recompute::recompute_tank_state` on every change.
                ui.label(format!("Dry mass: {:.0} kg", t.dry_mass));
            } else if let Some(f) = fuselage.as_deref_mut() {
                ui.label("Kind: Fuselage (stationed loft)");
                ui.add(egui::Slider::new(&mut f.length, 2.0..=60.0).text("Length"));
                if is_root {
                    ui.add(egui::Slider::new(&mut f.max_width, 0.5..=8.0).text("Width (Ø)"));
                } else {
                    ui.label(format!("Width: {:.2}m (from parent)", f.max_width));
                }
                ui.add(egui::Slider::new(&mut f.max_height, 0.5..=8.0).text("Height"));
                ui.add(egui::Slider::new(&mut f.roundness, 0.0..=1.0).text("Roundness"));
                ui.add(egui::Slider::new(&mut f.nose_fraction, 0.0..=0.45).text("Nose fraction"));
                ui.add(egui::Slider::new(&mut f.nose_bluntness, 0.0..=1.0).text("Nose shape (cone→radome)"));
                ui.add(egui::Slider::new(&mut f.tail_fraction, 0.0..=0.9).text("Tail fraction"));
                ui.add(egui::Slider::new(&mut f.nose_droop, 0.0..=2.0).text("Nose droop"));
                ui.add(egui::Slider::new(&mut f.tail_upsweep, 0.0..=3.0).text("Tail upsweep"));
                ui.add(
                    egui::Slider::new(&mut f.tail_tip_diameter, 0.0..=3.0).text("Tail tip Ø"),
                );
                ui.add(
                    egui::Slider::new(&mut f.tail_bluntness, 0.0..=1.0)
                        .text("Tail shape (cone→dome)"),
                );
                // dry_mass tracks lofted skin area via `recompute_fuselage_state`.
                ui.label(format!("Dry mass: {:.0} kg", f.dry_mass));
            } else if let Some(e) = engine.as_deref_mut() {
                let optimized_for = catalog
                    .resolve(&catalog_ref.id)
                    .ok()
                    .and_then(|entry| match entry {
                        CatalogEntry::Engine(spec) => Some(spec.optimized_for.label()),
                        _ => None,
                    })
                    .unwrap_or("Unknown");
                ui.label(format!("Kind: Engine ({optimized_for})"));
                ui.label(format!("Model: {}", e.model));
                ui.label(format!("Geometry: {}", e.geometry.label()));
                if e.requires_atmosphere {
                    ui.label("Requires atmosphere");
                }
                ui.label(format!("Diameter: {:.2}m (fixed)", e.diameter));
                ui.label(format!("Thrust: {:.1} kN (fixed)", e.thrust / 1000.0));
                ui.label(format!("Isp: {:.0} s (fixed)", e.isp));
                ui.label(format!("Dry mass: {:.0} kg (fixed)", e.dry_mass));
                if e.power_draw_kw > 0.0 {
                    ui.label(format!("Power draw: {:.1} kW (fixed)", e.power_draw_kw));
                }
                ui.label("Reactants:");
                for r in &e.reactants {
                    ui.label(format!(
                        "  {}: {:.1}%",
                        r.resource.display_name(),
                        r.mass_fraction * 100.0,
                    ));
                }
                if let Some(requirement) = e.intake_requirement {
                    ui.label(format!(
                        "Intake required: {:.2} m² {}",
                        requirement.area_m2,
                        requirement.kind.label()
                    ));
                }
                if let Some(capture) = e.builtin_intake {
                    ui.label(format!(
                        "Built-in intake: {:.2} m² {} (eff {:.0}%)",
                        capture.area_m2,
                        capture.kind.label(),
                        capture.efficiency * 100.0
                    ));
                }
            } else if let Some(i) = intake.as_deref_mut() {
                ui.label("Kind: Air Intake");
                ui.label(format!("Model: {}", i.model));
                ui.label(format!("Diameter: {:.2}m (fixed)", i.diameter));
                ui.label(format!("Length: {:.2}m (fixed)", i.length));
                ui.label(format!(
                    "Capture: {:.2} m² {} (eff {:.0}%)",
                    i.capture.area_m2,
                    i.capture.kind.label(),
                    i.capture.efficiency * 100.0
                ));
                ui.label(format!("Dry mass: {:.0} kg (fixed)", i.dry_mass));
            } else if let Some(w) = wing.as_deref_mut() {
                ui.label("Kind: Wing");
                ui.add(egui::Slider::new(&mut w.span, 0.5..=30.0).text("Span (per side)"));
                ui.add(egui::Slider::new(&mut w.root_chord, 0.3..=15.0).text("Root chord"));
                ui.add(egui::Slider::new(&mut w.tip_chord, 0.1..=15.0).text("Tip chord"));
                // Angles authored in degrees, stored in radians.
                let mut sweep_deg = w.sweep.to_degrees();
                if ui
                    .add(egui::Slider::new(&mut sweep_deg, -10.0..=60.0).text("Sweep °"))
                    .changed()
                {
                    w.sweep = sweep_deg.to_radians();
                }
                let mut dihedral_deg = w.dihedral.to_degrees();
                if ui
                    .add(egui::Slider::new(&mut dihedral_deg, -15.0..=15.0).text("Dihedral °"))
                    .changed()
                {
                    w.dihedral = dihedral_deg.to_radians();
                }
                let mut incidence_deg = w.incidence.to_degrees();
                if ui
                    .add(egui::Slider::new(&mut incidence_deg, -5.0..=10.0).text("Incidence °"))
                    .changed()
                {
                    w.incidence = incidence_deg.to_radians();
                }
                ui.add(egui::Slider::new(&mut w.thickness, 0.04..=0.25).text("Thickness t/c"));
                // dry_mass tracks planform area via `recompute_wing_state`.
                ui.label(format!("Dry mass: {:.0} kg/panel", w.dry_mass));
                // `sel` was resolved to the group primary above, so a grouped
                // wing always lands here as the primary — editing either side
                // updates both.
                match groups.get(sel).ok() {
                    Some(_) => {
                        ui.label("Symmetry: mirrored pair");
                        ui.label(
                            egui::RichText::new(
                                "Editing either side updates both; deleting either removes both.",
                            )
                            .small()
                            .weak(),
                        );
                    }
                    None => {
                        ui.label("Symmetry: single");
                    }
                }
            } else if let Some(g) = gear.as_deref_mut() {
                ui.label(if g.track_fraction > 0.0 {
                    "Kind: Landing Gear (main, L/R)"
                } else {
                    "Kind: Landing Gear (nose)"
                });
                ui.add(egui::Slider::new(&mut g.strut_length, 0.3..=4.0).text("Strut length"));
                ui.add(egui::Slider::new(&mut g.wheel_radius, 0.1..=1.2).text("Wheel radius"));
                if g.track_fraction > 0.0 {
                    ui.label(format!(
                        "Track: ±{:.0}% of host radius (fixed)",
                        g.track_fraction * 100.0
                    ));
                }
                // dry_mass tracks strut length × leg count via `recompute_gear_state`.
                ui.label(format!("Dry mass: {:.0} kg", g.dry_mass));
                ui.label(
                    egui::RichText::new(
                        "Self-contained gearbox — draws its own legs, not mirrored.",
                    )
                    .small()
                    .weak(),
                );
            }

            ui.separator();
            ui.label("Attach nodes:");
            for (id, node) in &nodes.nodes {
                ui.label(format!("  {id}: Ø{:.2}m", node.diameter));
            }

            ui.separator();
            ui.label("Resources:");
            if let Some(r) = res.as_deref_mut() {
                let params = inspector_params(
                    dec.as_deref(),
                    adapter.as_deref(),
                    tank.as_deref(),
                    fuselage.as_deref(),
                    wing.as_deref(),
                    gear.as_deref(),
                );
                let mut any_resource_row = false;
                let mut remove_resource = Vec::new();
                let mut add_resource = Vec::new();
                if let Ok(entry) = catalog.resolve(&catalog_ref.id) {
                    for option in entry.storage_options() {
                        let Some(capacity) = resource_capacity_for(entry, &params, option.resource)
                        else {
                            continue;
                        };
                        any_resource_row = true;
                        if let Some(pool) = r.pools.get_mut(&option.resource) {
                            ui.horizontal(|ui| {
                                if ui.small_button("Remove").clicked() {
                                    remove_resource.push(option.resource);
                                }
                                ui.label(format!(
                                    "{}: {:.0}/{:.0} {}",
                                    option.resource.display_name(),
                                    pool.amount,
                                    pool.capacity,
                                    option.resource.unit_label(),
                                ));
                            });
                            ui.add(
                                egui::Slider::new(&mut pool.amount, 0.0..=pool.capacity)
                                    .text("amount"),
                            );
                        } else if ui
                            .button(format!(
                                "Add {} ({:.0} {})",
                                option.resource.display_name(),
                                capacity,
                                option.resource.unit_label()
                            ))
                            .clicked()
                        {
                            add_resource.push((
                                option.resource,
                                ResourcePool {
                                    capacity,
                                    amount: capacity * option.default_fill_fraction.clamp(0.0, 1.0),
                                },
                            ));
                        }
                    }
                }
                for resource in remove_resource {
                    r.pools.remove(&resource);
                }
                for (resource, pool) in add_resource {
                    r.pools.insert(resource, pool);
                }
                if !any_resource_row {
                    ui.label("  (none)");
                }
            }

            ui.separator();
            ui.add_enabled_ui(!is_root, |ui| {
                if ui.button("Set as root").clicked() {
                    state.set_as_root = true;
                }
            });
            if ui.button("Delete part").clicked() {
                state.delete_selected = true;
            }
            }); // scroll area
        });

    // -------- Staging preview (right, left of the inspector) --------
    egui::SidePanel::right("staging")
        .resizable(true)
        .default_width(210.0)
        .show(&ctx, |ui| {
            ui.heading("Staging");
            ui.label(
                egui::RichText::new("Derived from decoupler position")
                    .small()
                    .weak(),
            );
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| match &stage_summaries {
                Some(Ok(summaries)) => draw_staging(ui, summaries),
                Some(Err(e)) => {
                    ui.colored_label(egui::Color32::from_rgb(220, 110, 60), format!("{e}"));
                }
                None => {
                    ui.label("(no ship)");
                }
            });
        });

    // -------- Bottom: ship hierarchy & placement picker --------
    egui::TopBottomPanel::bottom("hierarchy")
        .default_height(180.0)
        .show(&ctx, |ui| {
            ui.horizontal(|ui| {
                // Hierarchy list
                ui.vertical(|ui| {
                    ui.heading("Ship");
                    let Some(root) = state.ship_root else {
                        return;
                    };
                    let mut child_map: HashMap<Entity, Vec<Entity>> = HashMap::new();
                    for (e, att) in attachments.iter() {
                        child_map.entry(att.parent).or_default().push(e);
                    }
                    // Surface-mounted wings are part of the ship tree too.
                    for (e, sm) in surface_mounts.iter() {
                        child_map.entry(sm.parent).or_default().push(e);
                    }
                    draw_hierarchy(ui, root, &child_map, &mut state, 0);
                });

                ui.separator();

                // Placement picker. Surface parts click a compatible body;
                // stack parts use a free attach node listed here.
                if let Some(pending) = state.pending.clone() {
                    let pending_wing = matches!(pending.params, PartParams::Wing { .. });
                    let pending_gear = matches!(pending.params, PartParams::Gear { .. });
                    let pending_nacelle = catalog.resolve(&pending.catalog_id).is_ok_and(|entry| {
                        matches!(
                            entry,
                            CatalogEntry::Engine(e) if e.geometry == EngineGeometry::JetNacelle
                        )
                    });
                    ui.vertical(|ui| {
                        if pending_wing {
                            ui.heading("Mount wing");
                            ui.label("Click a hull body where the wing root should sit.");
                            ui.label(
                                egui::RichText::new(
                                    "Side hit → mirrored pair · top/bottom hit → single fin",
                                )
                                .small()
                                .weak(),
                            );
                            return;
                        }
                        if pending_gear {
                            ui.heading("Mount landing gear");
                            ui.label("Click the fuselage belly where the gear should sit.");
                            ui.label(
                                egui::RichText::new(
                                    "Self-contained gearbox — main draws both legs; never mirrored",
                                )
                                .small()
                                .weak(),
                            );
                            return;
                        }
                        if pending_nacelle {
                            ui.heading("Mount nacelle");
                            ui.label("Click a wing where the pylon should sit.");
                            ui.label(
                                egui::RichText::new(
                                    "A mirrored wing creates a mirrored nacelle pair",
                                )
                                .small()
                                .weak(),
                            );
                            return;
                        }
                        ui.heading("Place at…");
                        let occupied: std::collections::HashSet<(Entity, String)> = attachments
                            .iter()
                            .map(|(_, a)| (a.parent, a.parent_node.clone()))
                            .collect();
                        let mut rows: Vec<(Entity, String, f32)> = Vec::new();
                        let parts = part_queries.p0();
                        for (e, _, nodes, _, _, _, _, _, _, _, _, _, _) in parts.iter() {
                            for (nid, node) in &nodes.nodes {
                                if occupied.contains(&(e, nid.clone())) {
                                    continue;
                                }
                                rows.push((e, nid.clone(), node.diameter));
                            }
                        }
                        for (e, nid, d) in rows {
                            if ui.button(format!("{e:?} / {nid} (Ø{d:.2}m)")).clicked() {
                                state.place_at = Some((e, nid));
                            }
                        }
                    });
                }
            });
        });
}

fn draw_hierarchy(
    ui: &mut egui::Ui,
    entity: Entity,
    child_map: &HashMap<Entity, Vec<Entity>>,
    state: &mut EditorState,
    depth: usize,
) {
    let indent = "  ".repeat(depth);
    let selected = state.selected == Some(entity);
    let label = format!("{indent}{entity:?}");
    if ui.selectable_label(selected, label).clicked() {
        state.selected = Some(entity);
    }
    if let Some(kids) = child_map.get(&entity) {
        for c in kids {
            draw_hierarchy(ui, *c, child_map, state, depth + 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Celestial backdrop
// ---------------------------------------------------------------------------
//
// Duplicated from `thalos_game::sky_render` with the game-specific bits
// (CameraExposure, SimStage, OrbitCamera) stripped out. Keep until sky
// rendering is extracted into its own crate.

#[derive(Resource, Default)]
struct SkyBackdropEnabled(bool);

#[derive(Component)]
struct SkyBackdrop;

#[derive(Clone, Copy, ShaderType)]
struct StarsParams {
    pixel_radius: f32,
    brightness: f32,
    size_gamma: f32,
    _pad0: f32,
}

impl Default for StarsParams {
    fn default() -> Self {
        Self {
            pixel_radius: 4.0,
            brightness: 140.0,
            size_gamma: 0.50,
            _pad0: 0.0,
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct StarsMaterial {
    #[uniform(0)]
    params: StarsParams,
}

impl Material for StarsMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/stars.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/stars.wgsl".into()
    }
    fn prepass_vertex_shader() -> ShaderRef {
        "shaders/stars_prepass.wgsl".into()
    }
    fn prepass_fragment_shader() -> ShaderRef {
        "shaders/stars_prepass.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }

    fn specialize(
        _: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(1),
            Mesh::ATTRIBUTE_COLOR.at_shader_location(2),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = Some(false);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, ShaderType)]
struct GalaxyParams {
    pixel_radius_scale: f32,
    min_pixel_radius: f32,
    brightness: f32,
    _pad0: f32,
}

impl Default for GalaxyParams {
    fn default() -> Self {
        Self {
            pixel_radius_scale: 2000.0,
            min_pixel_radius: 1.2,
            brightness: 1_500.0,
            _pad0: 0.0,
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
struct GalaxyMaterial {
    #[uniform(0)]
    params: GalaxyParams,
}

impl Material for GalaxyMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/galaxy.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/galaxy.wgsl".into()
    }
    fn prepass_vertex_shader() -> ShaderRef {
        "shaders/galaxy_prepass.wgsl".into()
    }
    fn prepass_fragment_shader() -> ShaderRef {
        "shaders/galaxy_prepass.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }

    fn specialize(
        _: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(1),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(2),
            Mesh::ATTRIBUTE_TANGENT.at_shader_location(3),
            Mesh::ATTRIBUTE_COLOR.at_shader_location(4),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = Some(false);
        }
        Ok(())
    }
}

struct SkyBackdropPlugin;

impl Plugin for SkyBackdropPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<StarsMaterial>::default())
            .add_plugins(MaterialPlugin::<GalaxyMaterial>::default())
            .add_systems(Startup, spawn_sky_backdrop)
            .add_systems(Update, (update_sky_visibility, update_galaxy_uniform));
    }
}

fn spawn_sky_backdrop(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut stars_materials: ResMut<Assets<StarsMaterial>>,
    mut galaxy_materials: ResMut<Assets<GalaxyMaterial>>,
) {
    let universe = generate_default(&DefaultGenParams::default());

    commands.spawn((
        SkyBackdrop,
        Mesh3d(meshes.add(build_star_mesh(&universe))),
        MeshMaterial3d(stars_materials.add(StarsMaterial {
            params: StarsParams::default(),
        })),
        Transform::IDENTITY,
        Visibility::Hidden,
        NoFrustumCulling,
    ));

    commands.spawn((
        SkyBackdrop,
        Mesh3d(meshes.add(build_galaxy_mesh(&universe))),
        MeshMaterial3d(galaxy_materials.add(GalaxyMaterial {
            params: GalaxyParams::default(),
        })),
        Transform::IDENTITY,
        Visibility::Hidden,
        NoFrustumCulling,
    ));
}

fn update_sky_visibility(
    enabled: Res<SkyBackdropEnabled>,
    mut q: Query<&mut Visibility, With<SkyBackdrop>>,
) {
    let target = if enabled.0 {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut v in q.iter_mut() {
        if *v != target {
            *v = target;
        }
    }
}

fn update_galaxy_uniform(
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<&Projection, With<Camera3d>>,
    handles: Query<&MeshMaterial3d<GalaxyMaterial>>,
    mut materials: ResMut<Assets<GalaxyMaterial>>,
) {
    let Ok(window) = windows.single() else { return };
    let Ok(projection) = cameras.single() else {
        return;
    };
    let Projection::Perspective(p) = projection else {
        return;
    };
    let px_per_rad = window.resolution.physical_height() as f32 / p.fov;

    for handle in &handles {
        if let Some(mut mat) = materials.get_mut(&handle.0) {
            mat.params.pixel_radius_scale = px_per_rad;
        }
    }
}

fn build_star_mesh(universe: &Universe) -> Mesh {
    let n = universe.stars.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n * 4);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 6);

    const CORNERS: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

    for (i, star) in universe.stars.iter().enumerate() {
        let dir = star.position.normalize();
        let rgb = star.linear_srgb();
        let flux = star.magnitude_flux();
        for corner in CORNERS {
            positions.push([dir.x, dir.y, dir.z]);
            uvs.push(corner);
            colors.push([rgb[0], rgb[1], rgb[2], flux]);
        }
        let base = (i * 4) as u32;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn build_galaxy_mesh(universe: &Universe) -> Mesh {
    let n = universe.galaxies.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n * 4);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut tangents: Vec<[f32; 4]> = Vec::with_capacity(n * 4);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 6);

    const CORNERS: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

    for (i, galaxy) in universe.galaxies.iter().enumerate() {
        let dir = galaxy.position.normalize();
        let rgb = galaxy.linear_srgb();
        let flux = galaxy.magnitude_flux();
        let (sin_pa, cos_pa) = galaxy.position_angle_rad.sin_cos();
        for corner in CORNERS {
            positions.push([dir.x, dir.y, dir.z]);
            uvs.push(corner);
            normals.push([galaxy.effective_radius_rad, galaxy.sersic_n, 0.0]);
            tangents.push([galaxy.axis_ratio, cos_pa, sin_pa, 0.0]);
            colors.push([rgb[0], rgb[1], rgb[2], flux]);
        }
        let base = (i * 4) as u32;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
