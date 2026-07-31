//! Transonic vapour cone — the condensation collar that forms around an airframe
//! near Mach 1.
//!
//! Third consumer of [`FlowSignals`], after the plume and the reentry shock layer.
//! It shares their geometry approach (a proxy hull, a ray-marched analytic shell,
//! compact support at both surfaces) and differs in the one way that matters:
//!
//! **This is a scattering medium, not an emitter.** The visible cloud is condensed
//! water droplets scattering *sunlight*. Shading it like the plume gives a self-lit
//! white cone that looks the same at midnight and reads as a decal. So the source
//! term is illumination with a forward-scattering phase function, and the material
//! blends rather than adds — a cloud is not transparent to what it covers.
//!
//! **It is also not a shock.** The shock is invisible; what you see is the
//! condensation in the low-pressure region behind the expansion. That distinction
//! is why the collar exists in a *window* around Mach 1 rather than everywhere
//! supersonic, and why it needs humidity at all.
//!
//! Verify with `just screenshot vapor-cone`, which boots an atmospheric cruise
//! and drives [`FlowDebugOverride`] to a humid transonic freestream.

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

const PROXY_SIDES: usize = 16;
const PROXY_RINGS: usize = 12;

/// The Mach window the collar lives in.
///
/// **A vapour cone is a transonic effect, not a supersonic one**, and this window
/// is the whole reason it reads as a specific moment rather than as permanent
/// decoration. Below ~0.75 nothing on the airframe reaches the local expansion
/// needed to condense; well above ~1.3 the low-pressure region has moved off the
/// body and the collar is gone. Real photographs cluster hard around Mach 0.95.
const MACH_LO: f32 = 0.75;
const MACH_PEAK_LO: f32 = 0.92;
const MACH_PEAK_HI: f32 = 1.05;
const MACH_HI: f32 = 1.35;

/// Relative humidity below which no collar forms at all.
///
/// Without a humidity gate the effect either never appears or appears on every
/// transonic pass, and both are wrong: the reason vapour cones are *photographed*
/// rather than routine is that they need moist air. This is also why the same
/// aircraft shows one over the sea and nothing over a desert.
const HUMIDITY_FLOOR: f32 = 0.35;

/// Dynamic pressure below which the air is too thin to carry visible moisture,
/// Pa. A transonic pass in the stratosphere makes no collar.
const Q_FLOOR_PA: f32 = 8_000.0;
const Q_FULL_PA: f32 = 25_000.0;

/// Extinction per metre at full condensation.
///
/// Sized so a collar a couple of metres thick is already optically thick. The
/// reference bells are opaque, and the diffusion-limit brightness the shared
/// scattering model produces only appears once the medium actually is — a thin
/// medium renders as the grey veil that model's notes warn about.
const EXTINCTION_PER_M: f32 = 1.6;

/// Sun colour reaching the collar. Near-white; **brightness comes from the
/// scattering model, not from here**, so this is a tint and not a gain.
const DROPLET_TINT: Vec3 = Vec3::new(0.96, 0.97, 1.0);

/// Where the bell's apex sits, as a fraction of the craft's along-flow
/// half-extent measured back from the leading end. The apex is *on* the airframe
/// but a little aft of the leading point: the tip is a stagnation point, the last
/// place air expands.
const COLLAR_APEX_FRAC: f32 = 0.35;

/// Collar length and maximum radius, as multiples of the craft's along-flow
/// half-extent.
///
/// **Sized from the reference photographs, not from the Mach angle.** `tan(mu)`
/// is the angle of the shock surface, and near Mach 1 it opens toward 90°, which
/// over a vehicle-length of run puts the surface tens of metres out. Measured:
/// with the angle governing extent, a 22 m craft grew a 60 m collar that filled
/// the frame and banded, because the march could no longer resolve it. In the
/// reference shots the bell's widest radius is roughly half the airframe's length
/// and it reaches about one length aft — a body-scale feature. Mach still shapes
/// the collar through [`bell_flare`]; it just does not set its size.
const COLLAR_LENGTH_FACTOR: f32 = 2.0;
const COLLAR_RADIUS_FACTOR: f32 = 1.0;

#[derive(Resource)]
struct VaporConeMesh(Handle<Mesh>);

#[derive(Component)]
pub struct VaporCone {
    material: Handle<VaporConeMaterial>,
    /// Last gate state written to the diagnostic lane. This is transition-only:
    /// ordinary flight spends almost all its time with the cone hidden, and
    /// logging that every frame would bury every other runtime signal.
    logged_state: Option<&'static str>,
}

pub struct VaporConePlugin;

impl Plugin for VaporConePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<VaporConeMaterial>::default())
            .add_systems(Startup, insert_vapor_cone_mesh)
            .add_systems(
                Update,
                (spawn_vapor_cone, update_vapor_cone)
                    .chain()
                    .in_set(SimStage::Sync)
                    .after(super::flow::update_flow_signals),
            );
    }
}

fn insert_vapor_cone_mesh(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let handle = meshes.add(axial_proxy_prism_mesh(PROXY_SIDES, PROXY_RINGS));
    commands.insert_resource(VaporConeMesh(handle));
}

/// Resolved collar profile. Mirrors `VaporConeParams` in `vapor_cone.wgsl`.
#[derive(Clone, Copy, ShaderType, Debug)]
pub struct VaporConeParams {
    /// rgb = droplet scattering albedo, a = opacity kappa across the shell.
    pub tint: Vec4,
    /// xyz = freestream arrival direction in craft-local axes, w = cone
    /// half-angle tangent.
    pub flow: Vec4,
    /// x = collar start station (m), y = collar length (m),
    /// z = shell thickness (m), w = proxy bound radius (m).
    pub shape: Vec4,
    /// xyz = sun direction in craft-local axes, w = opacity ramp 0..1.
    pub sun: Vec4,
}

impl Default for VaporConeParams {
    fn default() -> Self {
        Self {
            tint: DROPLET_TINT.extend(EXTINCTION_PER_M),
            flow: Vec3::Z.extend(1.0),
            shape: Vec4::new(0.0, 1.0, 0.5, 1.0),
            sun: Vec3::Y.extend(0.0),
        }
    }
}

#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct VaporConeMaterial {
    #[uniform(0)]
    pub params: VaporConeParams,
}

impl Material for VaporConeMaterial {
    fn vertex_shader() -> bevy::shader::ShaderRef {
        "shaders/vapor_cone.wgsl".into()
    }

    fn fragment_shader() -> bevy::shader::ShaderRef {
        "shaders/vapor_cone.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        // Premultiplied blend, NOT `Add`. The plume and the shock layer are hot
        // gas and only ever add light; a condensation cloud also *hides* what is
        // behind it in proportion to its opacity.
        AlphaMode::Premultiplied
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Closed proxy hull: cull one side so each ray shades exactly once.
        descriptor.primitive.cull_mode = Some(bevy::render::render_resource::Face::Back);
        Ok(())
    }
}

fn spawn_vapor_cone(
    mut commands: Commands,
    mesh: Option<Res<VaporConeMesh>>,
    mut materials: ResMut<Assets<VaporConeMaterial>>,
    ships: Query<Entity, (Added<PlayerShip>, Without<EditorPart>)>,
) {
    let Some(mesh) = mesh else {
        return;
    };
    for root in ships.iter() {
        let material = materials.add(VaporConeMaterial {
            params: VaporConeParams::default(),
        });
        let cone = commands
            .spawn((
                Mesh3d(mesh.0.clone()),
                MeshMaterial3d(material.clone()),
                Transform::IDENTITY,
                Visibility::Hidden,
                NoFrustumCulling,
                NotShadowCaster,
                // Not vehicle structure: keep it out of the craft-bounds sweep
                // that sizes this very cone.
                FlowProxyMesh,
                VaporCone {
                    material,
                    logged_state: None,
                },
            ))
            .id();
        commands.entity(root).add_child(cone);
    }
}

fn update_vapor_cone(
    flow: Res<FlowSignals>,
    // The existing sun authority — the same direction the shadow cascade block
    // publishes, so the collar cannot disagree with what lights everything else.
    sun: Res<super::contact_shadow::ContactShadowSun>,
    mut materials: ResMut<Assets<VaporConeMaterial>>,
    mut cones: Query<(&mut VaporCone, &mut Transform, &mut Visibility)>,
) {
    // Into craft axes through the rotation `FlowSignals` publishes, rather than a
    // second query on the craft's `Transform`.
    let sun_local = (flow.craft_rotation.inverse() * sun.dir).normalize_or(Vec3::Y);
    let profile = resolve_params(&flow, sun_local);
    let gate = vapor_cone_gate(&flow);

    for (mut cone, mut transform, mut visibility) in cones.iter_mut() {
        if cone.logged_state != Some(gate) {
            cone.logged_state = Some(gate);
            info!(
                target: "thalos::diagnostic::vapor_cone",
                event = "vapor_cone_state",
                state = gate,
                visible = profile.is_some(),
                in_atmosphere = flow.in_atmosphere,
                altitude_m = flow.altitude_m,
                mach = flow.mach,
                airspeed_m_s = flow.airspeed_m_s,
                dynamic_pressure_pa = flow.dynamic_pressure_pa,
                humidity_frac = flow.relative_humidity_frac,
                flow_local_x = flow.flow_from_local.x,
                flow_local_y = flow.flow_from_local.y,
                flow_local_z = flow.flow_from_local.z,
                measured_mesh_count = flow.measured_mesh_count,
                "vapour cone gate changed"
            );
        }
        let Some(params) = profile else {
            *visibility = Visibility::Hidden;
            continue;
        };
        // Sit on the hull's bounding-box centre, which is not the craft origin.
        transform.translation = flow.craft_bounds_centre_m;
        if let Some(material) = materials.get_mut(&cone.material).as_mut() {
            material.params = params;
        }
        *visibility = Visibility::Inherited;
    }
}

/// Stable diagnostic reason for the cone's current gate state.
///
/// Mirrors [`resolve_params`] by named gate so the capture reader can say why a
/// probe rendered no cone. This function is intentionally scalar-only and runs
/// only as part of the already-required resolver update.
fn vapor_cone_gate(flow: &FlowSignals) -> &'static str {
    if !flow.in_atmosphere {
        return "outside_atmosphere";
    }
    if flow.flow_from_local == Vec3::ZERO {
        return "no_flow_direction";
    }
    let mach = mach_window(flow.mach);
    if mach <= 0.0 {
        return "mach_window_closed";
    }
    let humidity =
        ((flow.relative_humidity_frac - HUMIDITY_FLOOR) / (1.0 - HUMIDITY_FLOOR)).clamp(0.0, 1.0);
    if humidity <= 0.0 {
        return "humidity_gate_closed";
    }
    let q = ((flow.dynamic_pressure_pa - Q_FLOOR_PA) / (Q_FULL_PA - Q_FLOOR_PA)).clamp(0.0, 1.0);
    if q <= 0.0 {
        return "dynamic_pressure_gate_closed";
    }
    if mach * humidity * q <= 0.01 {
        return "condensation_below_floor";
    }
    "visible"
}

/// Map the freestream onto the collar profile, or `None` when no collar forms.
fn resolve_params(flow: &FlowSignals, sun_local: Vec3) -> Option<VaporConeParams> {
    if !flow.in_atmosphere || flow.flow_from_local == Vec3::ZERO {
        return None;
    }

    // Three independent gates, all of which a real collar needs simultaneously.
    let mach_window = mach_window(flow.mach);
    let humidity =
        ((flow.relative_humidity_frac - HUMIDITY_FLOOR) / (1.0 - HUMIDITY_FLOOR)).clamp(0.0, 1.0);
    let q = ((flow.dynamic_pressure_pa - Q_FLOOR_PA) / (Q_FULL_PA - Q_FLOOR_PA)).clamp(0.0, 1.0);
    let condensation = mach_window * humidity * q;
    if condensation <= 0.01 {
        return None;
    }

    // Along-flow half-extent of the craft box: the scale everything is sized from.
    let extents = flow.craft_half_extents_m.max(Vec3::splat(0.05));
    let along = (extents * flow.flow_from_local).length().max(0.5);

    // Apex a little aft of the leading end, then the bell runs downstream.
    let apex = -along * (1.0 - COLLAR_APEX_FRAC);
    let collar_len = (along * COLLAR_LENGTH_FACTOR).max(0.5);
    let max_radius = (along * COLLAR_RADIUS_FACTOR).max(0.25);
    let bound = (apex.abs() + collar_len).hypot(max_radius) * 1.05;

    Some(VaporConeParams {
        tint: DROPLET_TINT.extend(EXTINCTION_PER_M),
        flow: flow
            .flow_from_local
            .normalize_or(Vec3::Z)
            .extend(max_radius),
        shape: Vec4::new(apex, collar_len, bell_flare(flow.mach), bound),
        sun: sun_local.normalize_or(Vec3::Y).extend(condensation),
    })
}

/// Bell flare exponent: `R(a) = R_max · a^flare`.
///
/// Lower = flares harder off the apex (a blunt bell); higher = a slimmer cone.
/// Below Mach 1 the low-pressure region sits close over the airframe and the
/// collar is blunt; as the flow goes supersonic the shock lies down and the shape
/// slims toward a true Mach cone. This is where Mach still shapes the collar now
/// that it no longer sets its extent.
fn bell_flare(mach: f32) -> f32 {
    let t = ((mach - MACH_PEAK_LO) / (MACH_HI - MACH_PEAK_LO)).clamp(0.0, 1.0);
    0.42 + 0.38 * t
}

/// How fully the transonic window is open at this Mach number.
fn mach_window(mach: f32) -> f32 {
    if mach <= MACH_LO || mach >= MACH_HI {
        return 0.0;
    }
    if mach < MACH_PEAK_LO {
        smoothstep(MACH_LO, MACH_PEAK_LO, mach)
    } else if mach <= MACH_PEAK_HI {
        1.0
    } else {
        1.0 - smoothstep(MACH_PEAK_HI, MACH_HI, mach)
    }
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

    fn transonic() -> FlowSignals {
        FlowSignals {
            in_atmosphere: true,
            mach: 0.98,
            density_kg_m3: 0.9,
            dynamic_pressure_pa: 40_000.0,
            relative_humidity_frac: 0.7,
            airspeed_m_s: 320.0,
            flow_from_dir: Vec3::NEG_Z,
            flow_from_local: Vec3::NEG_Z,
            craft_half_extents_m: Vec3::new(4.0, 11.0, 2.0),
            craft_radius_m: 12.0,
            ..Default::default()
        }
    }

    /// The collar is a *transonic* effect. A cruising airliner and a hypersonic
    /// vehicle must both be clean — if the window were simply "supersonic", every
    /// fast craft would wear one permanently.
    #[test]
    fn only_transonic_speeds_form_a_collar() {
        assert!(resolve_params(&transonic(), Vec3::Y).is_some(), "mach 0.98");
        for mach in [0.4f32, 0.7, 1.6, 5.0, 24.0] {
            let flow = FlowSignals {
                mach,
                ..transonic()
            };
            assert!(
                resolve_params(&flow, Vec3::Y).is_none(),
                "mach {mach} should not form a collar"
            );
        }
    }

    /// Dry air gives no collar however fast the vehicle. This is why the same
    /// aircraft shows one over the sea and nothing over a desert.
    #[test]
    fn dry_air_forms_no_collar() {
        let flow = FlowSignals {
            relative_humidity_frac: 0.1,
            ..transonic()
        };
        assert!(resolve_params(&flow, Vec3::Y).is_none());
    }

    /// Thin air gives no collar: a transonic pass in the stratosphere carries no
    /// visible moisture even at high humidity fraction.
    #[test]
    fn thin_air_forms_no_collar() {
        let flow = FlowSignals {
            dynamic_pressure_pa: 500.0,
            ..transonic()
        };
        assert!(resolve_params(&flow, Vec3::Y).is_none());
    }

    /// Vacuum is not a special case to remember — `in_atmosphere` gates it.
    #[test]
    fn vacuum_forms_no_collar() {
        let flow = FlowSignals {
            in_atmosphere: false,
            ..transonic()
        };
        assert!(resolve_params(&flow, Vec3::Y).is_none());
        assert_eq!(vapor_cone_gate(&flow), "outside_atmosphere");
    }

    #[test]
    fn diagnostic_gate_names_the_closed_input() {
        assert_eq!(vapor_cone_gate(&transonic()), "visible");
        assert_eq!(
            vapor_cone_gate(&FlowSignals {
                relative_humidity_frac: 0.1,
                ..transonic()
            }),
            "humidity_gate_closed"
        );
        assert_eq!(
            vapor_cone_gate(&FlowSignals {
                dynamic_pressure_pa: 500.0,
                ..transonic()
            }),
            "dynamic_pressure_gate_closed"
        );
    }

    /// Opacity must peak in the middle of the window and fall off on both sides,
    /// or the collar snaps on and off instead of blooming through the transition.
    #[test]
    fn opacity_peaks_mid_window() {
        let kappa = |mach: f32| {
            resolve_params(
                &FlowSignals {
                    mach,
                    ..transonic()
                },
                Vec3::Y,
            )
            .map(|p| p.sun.w)
        };
        let peak = kappa(0.98).unwrap();
        let low = kappa(0.85).unwrap();
        let high = kappa(1.2).unwrap();
        assert!(low < peak, "{low} should be below peak {peak}");
        assert!(high < peak, "{high} should be below peak {peak}");
    }

    /// The collar must stay a body-scale feature. Sized by the Mach angle alone it
    /// grew a 60 m radius on a 22 m craft, filled the frame, and banded because the
    /// march could no longer resolve the shell.
    #[test]
    fn collar_stays_body_scale() {
        for mach in [0.8f32, 0.98, 1.1, 1.3] {
            let flow = FlowSignals {
                mach,
                ..transonic()
            };
            let params = resolve_params(&flow, Vec3::Y).unwrap();
            let widest = params.flow.w;
            let craft = flow.craft_half_extents_m.length();
            assert!(
                widest <= craft,
                "mach {mach}: collar radius {widest} dwarfs the craft ({craft})"
            );
        }
    }

    /// The proxy bound must contain the collar's full downstream reach and widest
    /// radius. A bound that cut inside would clip a still-condensing shell — the
    /// defect class of INC-20260724T235437Z-plume-ended-on-a-lit-rim.
    #[test]
    fn bound_contains_the_collar() {
        for mach in [0.8f32, 0.98, 1.1, 1.3] {
            let params = resolve_params(
                &FlowSignals {
                    mach,
                    ..transonic()
                },
                Vec3::Y,
            )
            .unwrap();
            let start = params.shape.x;
            let len = params.shape.y;

            let bound = params.shape.w;
            let max_radius = params.flow.w;
            // Farthest point the shell can reach from the collar's origin.
            let reach = (start.abs() + len).hypot(max_radius);
            assert!(
                bound >= reach,
                "mach {mach}: bound {bound} does not contain reach {reach}"
            );
        }
    }
}

#[cfg(test)]
mod flare_tests {
    use super::*;

    /// The bell slims as the flow goes supersonic — the shock lies down. Backwards
    /// would make a Mach 1.3 pass blunter than a Mach 0.9 one.
    #[test]
    fn bell_slims_with_mach() {
        assert!(bell_flare(0.95) < bell_flare(1.3));
    }
}
