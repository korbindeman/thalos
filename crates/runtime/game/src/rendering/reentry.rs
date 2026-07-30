//! Reentry shock layer — the shock-heated air standing off a vehicle's windward
//! side during atmospheric entry.
//!
//! The visual layer over [`FlowSignals`]; it owns *how shock-heated air looks*,
//! not what the vehicle is doing. Everything it needs comes from that one
//! boundary, so it can never disagree with the plume about the air they share.
//!
//! # What this is and is not
//!
//! People conflate three things under "reentry effects". This module is the
//! **first** one only:
//!
//! 1. **The shock layer** (here) — attached, zero-memory, a function of the
//!    *current* freestream. Present exactly while the vehicle is fast in air, and
//!    gone the instant it is not.
//! 2. **Hull glow** — not an effect at all, but an emissive term on the hull
//!    material driven by an integrated per-part heat state. That belongs in
//!    shading, next to the rest of the surface response.
//! 3. **The ablation wake** — a trail shed *into* the air, which needs memory and
//!    therefore the ribbon primitive, not this.
//!
//! Keeping (1) separate is most of the work: it is the part whose geometry is
//! pinned to the airflow rather than to the craft, and the part the existing
//! emission model already covers.
//!
//! # Two frames, and why the shell lives in the craft's
//!
//! **The layer is aimed at the airflow, but fitted to the hull.** An entering
//! vehicle flies at high angle of attack — a capsule blunt-end-first, a lifting
//! body belly-first — so a cap keyed to the craft's forward axis sits in
//! completely the wrong place through the whole entry. But the *body* the shock
//! stands off from is the hull, which is craft-aligned.
//!
//! So the shell entity is a plain child of the craft with an identity transform
//! (which also keeps it in the craft's BigSpace cell for free), and the freestream
//! direction arrives as a **uniform in craft-local axes**. The shader works
//! entirely in the craft's frame: the body is its bounding ellipsoid, and
//! `flow.xyz` says which way the wind comes from. Nothing has to be
//! counter-rotated, and no query touches the craft's `Transform`.
//!
//! The body is an **ellipsoid, not a sphere**. A bounding sphere on a 40 m rocket
//! is a 20 m ball, so a shell hugging it hangs metres out in empty space along
//! every axis — the first version did exactly that and rendered a saturated white
//! blob that filled the frame with the craft nowhere in sight.
//!
//! Verify with `just screenshot reentry`, which drives [`FlowDebugOverride`] to a
//! peak-heating state so the look is reproducible without flying an entry.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::NotShadowCaster;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};

use super::flow::{FlowProxyMesh, FlowSignals, axial_proxy_prism_mesh};
use super::types::PlayerShip;
use crate::SimStage;
use crate::shipyard_editor::core::EditorPart;

/// Proxy-hull resolution. It only bounds a smooth spherical segment, so it can be
/// coarse — the silhouette comes from the density integral, not these rings.
const PROXY_SIDES: usize = 16;
const PROXY_RINGS: usize = 12;

/// Mach range over which the shock layer fades in.
///
/// Below Mach 1 there is no shock at all, and just above it the layer is a weak,
/// invisible compression. Ramping from 1.5 keeps a supersonic aircraft clean
/// while an entering vehicle (Mach 20+) is fully lit.
const MACH_RAMP_LO: f32 = 1.5;
const MACH_RAMP_HI: f32 = 3.0;

/// Heat flux that reads as a "full" fireball, W/m². Peak stagnation-point heating
/// for a capsule entry is order 1e6 W/m², so this is where the effect saturates.
const HEAT_FLUX_REF_W_M2: f32 = 1.0e6;

/// Normalized intensity below which the shell is hidden outright rather than
/// drawn as an invisible smear.
const INTENSITY_FLOOR: f32 = 0.02;

/// Reference temperature the shader's Wien term normalises against, K.
///
/// Deliberately **equal to [`REAL_GAS_TEMP_CAP_K`]**: the hottest air physically
/// reachable is then exactly the reference, so normalized temperature lands in
/// `0..=1` and `band_emission` cannot exceed 1. With the reference below the cap
/// the emission term runs away above 1 and every entry blows out to flat white
/// before the colour ramp has a chance to say anything.
const TEMP_REF_K: f32 = REAL_GAS_TEMP_CAP_K;

/// Real-gas ceiling on the stagnation temperature, K.
///
/// **This cap is physics, not taste.** The ideal-gas relation
/// `T·(1 + (γ−1)/2·M²)` gives ~36 000 K at Mach 25, but real air does not get
/// there: above ~2 500 K the energy goes into vibrational excitation,
/// dissociation and then ionisation instead of into temperature, and shock-layer
/// gas measures out around 10 000–11 000 K across a wide entry range. Without
/// this cap every entry above about Mach 8 saturates to the same blue-white and
/// the colour stops carrying any information — the vehicle looks identical at
/// Mach 10 and Mach 30.
const REAL_GAS_TEMP_CAP_K: f32 = 11_000.0;

/// Density that reads as a fully opaque shell, kg/m³ — around the density at
/// peak heating. Optical depth scales as `sqrt(rho / this)`: in thin air the
/// layer is a tenuous halo, lower down it is a solid sheath.
const SHELL_DENSITY_REF: f32 = 3.0e-4;
const SHELL_KAPPA_REF: f32 = 1.2;
const SHELL_KAPPA_MIN: f32 = 0.05;
const SHELL_KAPPA_MAX: f32 = 4.0;

/// HDR radiance at full intensity. The post-stack bloom haloes this, same as the
/// plume — but much lower than the plume's core, because a shock layer is hot air
/// rather than incandescent exhaust and should read as a luminous sheath the
/// vehicle is still visible through.
const RADIANCE_GAIN: f32 = 2.0;

/// How much the standoff grows as the shock goes oblique (dimensionless, applied
/// to `1 − cos θ`). 3 sweeps the layer back into a teardrop instead of leaving it
/// a concentric bubble.
const STANDOFF_GROWTH: f32 = 3.0;

/// Stagnation standoff as a fraction of the body surface, in the shader's
/// normalized space. Bounds a *thin* layer: a real bow shock stands off a few
/// percent of the windward radius at hypersonic speed, and anything much thicker
/// stops reading as a shock and starts reading as a halo.
const STANDOFF_FRAC_MIN: f32 = 0.04;
const STANDOFF_FRAC_MAX: f32 = 0.18;

/// Plasma shimmer amplitude. Deliberately small: a shock layer is a smooth
/// continuum, and heavy noise reads as fire rather than as compressed air.
const SHIMMER_AMPLITUDE: f32 = 0.22;

/// Colour stops, indexed by normalized temperature in the shader.
const COOL_COLOR: Vec3 = Vec3::new(1.0, 0.30, 0.08);
const MID_COLOR: Vec3 = Vec3::new(1.0, 0.72, 0.42);
const HOT_COLOR: Vec3 = Vec3::new(0.72, 0.84, 1.0);

/// Shared proxy-hull handle.
#[derive(Resource)]
struct ReentryMesh(Handle<Mesh>);

/// Marker on the shell child entity.
#[derive(Component)]
pub struct ReentryShell {
    material: Handle<ReentryMaterial>,
    /// Last published visibility, so the diagnostic below fires on the
    /// *transition* rather than every frame.
    lit: bool,
}

pub struct ReentryPlugin;

impl Plugin for ReentryPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<ReentryMaterial>::default())
            .add_systems(Startup, insert_reentry_mesh)
            .add_systems(
                Update,
                (spawn_reentry_shell, update_reentry_shell)
                    .chain()
                    .in_set(SimStage::Sync)
                    .after(super::flow::update_flow_signals),
            );
    }
}

fn insert_reentry_mesh(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let handle = meshes.add(axial_proxy_prism_mesh(PROXY_SIDES, PROXY_RINGS));
    commands.insert_resource(ReentryMesh(handle));
}

/// Resolved shock-layer profile. Mirrors `ReentryParams` in `reentry.wgsl`.
#[derive(Clone, Copy, ShaderType, Debug)]
pub struct ReentryParams {
    /// rgb = hottest colour stop, a = HDR radiance gain.
    pub hot_color: Vec4,
    /// rgb = mid stop, a = shell opacity kappa.
    pub mid_color: Vec4,
    /// rgb = coolest stop, a = normalized stagnation temperature.
    pub cool_color: Vec4,
    /// xyz = craft-local body half-extents (m), w = stagnation standoff as a
    /// fraction of the body surface.
    pub body: Vec4,
    /// xyz = freestream arrival direction in craft-local axes, w = standoff growth.
    pub flow: Vec4,
    /// x = time (s), y = seed, z = shimmer amplitude, w = supersonic ramp 0..1.
    pub anim: Vec4,
}

impl Default for ReentryParams {
    fn default() -> Self {
        Self {
            hot_color: HOT_COLOR.extend(RADIANCE_GAIN),
            mid_color: MID_COLOR.extend(1.0),
            cool_color: COOL_COLOR.extend(0.0),
            body: Vec3::ONE.extend(STANDOFF_FRAC_MIN),
            flow: Vec3::Z.extend(STANDOFF_GROWTH),
            anim: Vec4::ZERO,
        }
    }
}

#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct ReentryMaterial {
    #[uniform(0)]
    pub params: ReentryParams,
}

impl Material for ReentryMaterial {
    fn vertex_shader() -> bevy::shader::ShaderRef {
        "shaders/reentry.wgsl".into()
    }

    fn fragment_shader() -> bevy::shader::ShaderRef {
        "shaders/reentry.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        // Premultiplied additive, as the plume — hot gas emits, it does not
        // occlude what is behind it except through its own optical depth.
        AlphaMode::Add
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // The proxy hull is closed, so a ray crosses it twice; culling one side
        // keeps the raymarch — and the additive blend — from running twice per
        // pixel. Back faces go, so the surviving fragment's depth is the front of
        // the shock bound and the layer depth-tests ahead of the hull it wraps.
        descriptor.primitive.cull_mode = Some(bevy::render::render_resource::Face::Back);
        Ok(())
    }
}

/// Give the flight craft its shock shell. One per vehicle; editor parts opt out.
fn spawn_reentry_shell(
    mut commands: Commands,
    mesh: Option<Res<ReentryMesh>>,
    mut materials: ResMut<Assets<ReentryMaterial>>,
    ships: Query<Entity, (Added<PlayerShip>, Without<EditorPart>)>,
) {
    let Some(mesh) = mesh else {
        return;
    };
    for root in ships.iter() {
        let material = materials.add(ReentryMaterial {
            params: ReentryParams::default(),
        });
        let shell = commands
            .spawn((
                Mesh3d(mesh.0.clone()),
                MeshMaterial3d(material.clone()),
                // Rotation is rewritten every frame to aim `+Z` upstream; the
                // translation stays at the craft origin, which is the centre the
                // shell's spherical geometry is measured about.
                Transform::IDENTITY,
                Visibility::Hidden,
                NoFrustumCulling,
                NotShadowCaster,
                // Excluded from the craft-bounds sweep that sizes this very shell.
                FlowProxyMesh,
                ReentryShell {
                    material,
                    lit: false,
                },
            ))
            .id();
        commands.entity(root).add_child(shell);
    }
}

/// Aim the shell upstream and resolve [`FlowSignals`] into the shader uniform.
fn update_reentry_shell(
    time: Res<Time>,
    flow: Res<FlowSignals>,
    mut materials: ResMut<Assets<ReentryMaterial>>,
    mut shells: Query<(&mut ReentryShell, &mut Visibility)>,
) {
    let profile = resolve_params(&flow, time.elapsed_secs());

    for (mut shell, mut visibility) in shells.iter_mut() {
        let Some(params) = profile else {
            if shell.lit {
                shell.lit = false;
                info!(
                    target: "thalos::diagnostic::reentry",
                    event = "reentry_shell_extinguished",
                    mach = flow.mach,
                    heat_flux_w_m2 = flow.heat_flux_w_m2,
                    "reentry shell off"
                );
            }
            *visibility = Visibility::Hidden;
            continue;
        };
        // No transform to touch: the shell is hull-fitted and identity-parented,
        // and the freestream direction rides in the uniform instead.
        if let Some(material) = materials.get_mut(&shell.material).as_mut() {
            material.params = params;
        }
        if !shell.lit {
            shell.lit = true;
            // Fires on the transition, not per frame: the resolved geometry is the
            // one thing a still cannot show, and every wrong-looking shell so far
            // has been a geometry-resolution bug (a bounding sphere standing in for
            // an elongated hull, then an optical depth normalised by the wrong
            // thickness). This is the record that separates "the physics is wrong"
            // from "the body we fitted is wrong".
            info!(
                target: "thalos::diagnostic::reentry",
                event = "reentry_shell_lit",
                mach = flow.mach,
                airspeed_m_s = flow.airspeed_m_s,
                density_kg_m3 = flow.density_kg_m3,
                heat_flux_w_m2 = flow.heat_flux_w_m2,
                stagnation_temp_k = flow.stagnation_temp_k,
                temp_norm = params.cool_color.w,
                kappa = params.mid_color.w,
                gain = params.hot_color.w,
                standoff_frac = params.body.w,
                body_half_x_m = params.body.x,
                body_half_y_m = params.body.y,
                body_half_z_m = params.body.z,
                craft_radius_m = flow.craft_radius_m,
                measured_mesh_count = flow.measured_mesh_count,
                flow_local_x = params.flow.x,
                flow_local_y = params.flow.y,
                flow_local_z = params.flow.z,
                "reentry shell lit"
            );
        }
        *visibility = Visibility::Inherited;
    }
}

/// Map the freestream onto the shell profile, or `None` when there is nothing to
/// draw.
fn resolve_params(flow: &FlowSignals, elapsed_s: f32) -> Option<ReentryParams> {
    if !flow.in_atmosphere || flow.flow_from_dir == Vec3::ZERO {
        return None;
    }
    let ramp = smoothstep(MACH_RAMP_LO, MACH_RAMP_HI, flow.mach);
    if ramp <= 0.0 {
        return None;
    }
    // Brightness rides on heat flux, which is the only quantity that knows the
    // difference between orbital speed in vacuum and orbital speed in air.
    let intensity = (flow.heat_flux_w_m2.max(0.0) / HEAT_FLUX_REF_W_M2)
        .sqrt()
        .min(4.0);
    if intensity < INTENSITY_FLOOR {
        return None;
    }

    // Standoff shrinks as Mach rises — a hypersonic shock hugs the body.
    let mach = flow.mach.max(1.0);
    let standoff_frac =
        (STANDOFF_FRAC_MIN + 0.35 / mach.powf(1.6)).clamp(STANDOFF_FRAC_MIN, STANDOFF_FRAC_MAX);

    let kappa = (SHELL_KAPPA_REF * (flow.density_kg_m3.max(0.0) / SHELL_DENSITY_REF).sqrt())
        .clamp(SHELL_KAPPA_MIN, SHELL_KAPPA_MAX);

    let temp_norm = (flow.stagnation_temp_k.min(REAL_GAS_TEMP_CAP_K) / TEMP_REF_K).clamp(0.0, 1.0);

    Some(ReentryParams {
        hot_color: HOT_COLOR.extend(RADIANCE_GAIN * intensity),
        mid_color: MID_COLOR.extend(kappa),
        cool_color: COOL_COLOR.extend(temp_norm),
        body: flow
            .craft_half_extents_m
            .max(Vec3::splat(0.05))
            .extend(standoff_frac),
        flow: flow.flow_from_local.normalize_or(Vec3::Z).extend(STANDOFF_GROWTH),
        anim: Vec4::new(elapsed_s, 0.37, SHIMMER_AMPLITUDE, ramp),
    })
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry_flow() -> FlowSignals {
        FlowSignals {
            in_atmosphere: true,
            mach: 24.0,
            density_kg_m3: 3.0e-4,
            stagnation_temp_k: 36_000.0,
            heat_flux_w_m2: 1.0e6,
            airspeed_m_s: 7_400.0,
            flow_from_dir: Vec3::NEG_Z,
            flow_from_local: Vec3::NEG_Z,
            // A stubby lifting body: long across the wind, shallow through it.
            craft_radius_m: 12.0,
            craft_half_extents_m: Vec3::new(4.0, 11.0, 2.0),
            nose_radius_m: 0.7,
            ..Default::default()
        }
    }

    /// A craft at orbital speed in vacuum must not glow. This is the failure the
    /// heat-flux drive exists to prevent: keying brightness to speed alone lights
    /// a fireball on every orbiting vehicle.
    #[test]
    fn vacuum_orbit_does_not_glow() {
        let flow = FlowSignals {
            in_atmosphere: false,
            mach: 0.0,
            airspeed_m_s: 7_800.0,
            flow_from_dir: Vec3::Z,
            ..Default::default()
        };
        assert!(resolve_params(&flow, 0.0).is_none());
    }

    /// Subsonic flight has no shock layer, however dense the air.
    #[test]
    fn subsonic_has_no_shock_layer() {
        let flow = FlowSignals {
            in_atmosphere: true,
            mach: 0.8,
            density_kg_m3: 1.225,
            heat_flux_w_m2: 1.0e5,
            airspeed_m_s: 270.0,
            flow_from_dir: Vec3::Z,
            ..Default::default()
        };
        assert!(resolve_params(&flow, 0.0).is_none());
    }

    /// A real entry lights up.
    #[test]
    fn peak_heating_lights_the_shell() {
        let params = resolve_params(&entry_flow(), 0.0).expect("entry should glow");
        assert_eq!(params.anim.w, 1.0, "Mach 24 must be fully ramped in");
        assert!(params.hot_color.w > 0.0);
    }

    /// The real-gas cap must actually bite, or colour stops carrying information
    /// above ~Mach 8 and every entry looks the same.
    #[test]
    fn stagnation_temperature_is_capped_by_real_gas_effects() {
        let params = resolve_params(&entry_flow(), 0.0).unwrap();
        assert!(
            (params.cool_color.w - 1.0).abs() < 1e-6,
            "ideal-gas 36000 K should have been capped to {REAL_GAS_TEMP_CAP_K} K, got {}",
            params.cool_color.w * TEMP_REF_K
        );
    }

    /// Thin air gives a tenuous halo, dense air a solid sheath. If opacity did
    /// not track density the whole entry would look like one altitude.
    #[test]
    fn opacity_tracks_density() {
        let thin = resolve_params(
            &FlowSignals {
                density_kg_m3: 1.0e-6,
                ..entry_flow()
            },
            0.0,
        )
        .unwrap();
        let thick = resolve_params(
            &FlowSignals {
                density_kg_m3: 1.0e-2,
                ..entry_flow()
            },
            0.0,
        )
        .unwrap();
        assert!(
            thin.mid_color.w < thick.mid_color.w,
            "thin {} should be more transparent than thick {}",
            thin.mid_color.w,
            thick.mid_color.w
        );
    }

    /// The emitting layer must stay strictly inside the proxy hull at every Mach
    /// number the ramp admits.
    ///
    /// The hull is a spherical segment of `R_b + standoff(HULL_AFT_COS)` closed at
    /// `HULL_AFT_COS`, while emission survives out to `WRAP_LO`. Because the
    /// standoff grows with obliqueness, the hull cutoff must sit *further* aft
    /// than the emission cutoff — otherwise the bound clips a still-glowing layer,
    /// which is the defect class of
    /// INC-20260724T235437Z-plume-ended-on-a-lit-rim.
    #[test]
    fn emitting_layer_stays_within_the_proxy_hull() {
        // MIRRORED from `reentry.wgsl`. These must stay equal to the shader's
        // constants; the bound is computed there and the emission cutoff is
        // applied there, so a drift between the two files is exactly the failure
        // this test exists to catch.
        const WRAP_LO: f32 = -0.15;
        // `hull_scale()` takes the standoff at the worst case, cos = -1.
        const HULL_WORST_COS: f32 = -1.0;
        assert!(
            HULL_WORST_COS < WRAP_LO,
            "the hull must be sized past where emission reaches"
        );

        for mach in [2.0f32, 3.0, 8.0, 25.0, 40.0] {
            let params = resolve_params(
                &FlowSignals {
                    mach,
                    ..entry_flow()
                },
                0.0,
            )
            .unwrap_or_else(|| panic!("mach {mach} should be past the ramp"));
            let standoff_frac = params.body.w;
            let growth = params.flow.w;
            let standoff = |cos_t: f32| standoff_frac * (1.0 + growth * (1.0 - cos_t));

            // Both in normalized units, where the body surface is 1.
            let hull = 1.0 + standoff(HULL_WORST_COS);
            let outermost_emitting = 1.0 + standoff(WRAP_LO);
            assert!(
                hull >= outermost_emitting,
                "mach {mach}: hull {hull} clips the emitting layer at {outermost_emitting}"
            );
            assert!(
                standoff_frac > 0.0 && standoff_frac < 0.5,
                "mach {mach}: standoff {standoff_frac} is not a thin layer on the body"
            );
        }
    }

    /// The standoff must *shrink* as Mach rises — a hypersonic shock hugs the
    /// body, and getting this backwards would put the layer further out the
    /// faster the vehicle goes.
    #[test]
    fn standoff_shrinks_with_mach() {
        let at = |mach: f32| {
            resolve_params(&FlowSignals { mach, ..entry_flow() }, 0.0)
                .unwrap()
                .body
                .w
        };
        assert!(at(25.0) < at(3.0));
    }

    /// The shell must be fitted to the craft's **box**, not its bounding sphere.
    /// The first version used the sphere, and on an elongated vehicle that put the
    /// layer metres off the hull in the narrow directions — a glowing ball with the
    /// craft lost inside it.
    #[test]
    fn shell_is_fitted_to_the_box_not_the_sphere() {
        let flow = entry_flow();
        let params = resolve_params(&flow, 0.0).unwrap();
        assert_eq!(params.body.truncate(), flow.craft_half_extents_m);
        assert!(
            params.body.truncate().min_element() < flow.craft_radius_m * 0.5,
            "a bounding sphere of {} would sit far outside the thin axis {}",
            flow.craft_radius_m,
            params.body.truncate().min_element()
        );
    }

    /// Emission must never exceed the Wien reference, or the colour ramp stops
    /// carrying temperature information and every entry blows out to flat white.
    #[test]
    fn normalized_temperature_never_exceeds_one() {
        for mach in [2.0f32, 10.0, 25.0, 60.0] {
            let params = resolve_params(&FlowSignals { mach, ..entry_flow() }, 0.0).unwrap();
            assert!(
                params.cool_color.w <= 1.0,
                "mach {mach}: normalized temperature {} exceeds the reference",
                params.cool_color.w
            );
        }
    }
}
