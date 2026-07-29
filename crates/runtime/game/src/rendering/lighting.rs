//! Per-frame lighting state: camera exposure, scene lighting struct
//! population (one star + eclipse occluders + planetshine), planet/solid
//! material light updates, sun-light direction.

use bevy::prelude::*;
use thalos_body_render::{
    AU_M, FilmGrain, LIGHT_AT_1AU, MAX_ECLIPSE_OCCLUDERS, SceneLighting, SolidPlanetHaloMaterial,
    SolidPlanetMaterial, StarLight,
};
use thalos_physics_canonical::types::BodyStates;

use super::types::{
    CameraExposure, CelestialBody, MoonLight, PlanetshineTints, SimulationState, SolarSystemState,
    SolidPlanetMaterials, SunLight,
};
use crate::camera::{CameraFocus, CameraFocusTarget};
use crate::coords::{MAP_SCALE, RenderOrigin, SHIP_SCALE};
use crate::view::ViewMode;

/// Ambient floor. Vacuum has no fill light — night sides are black.
const PLANET_AMBIENT: f32 = 0.0;

/// Exposure exponent. 2.0 = full compensation (distant bodies look identical
/// to focused Thalos — destroys distance cue). 0.0 = no compensation (Nyx is
/// black). 1.0 = linear-in-distance compensation: display flux at focus is
/// `LIGHT_AT_1AU / focus_d_AU`, so Nyx focus lands at ~0.24 — visibly dim,
/// leaves shadows dark, and doesn't collide with Bevy `AutoExposure` pulling
/// the scene up independently in the post stack.
const EXPOSURE_ALPHA: f64 = 1.0;

/// Maximum positive EV used to drive grain. Beyond this, grain saturates.
/// log2(42^1.0) ≈ 5.4 — Nyx is roughly here.
const EXPOSURE_EV_GRAIN_MAX: f32 = 6.0;

/// Update the `CameraExposure` resource from the current focus body. This is
/// the single source of truth for how much gain the "camera" applies to the
/// raw inverse-square solar flux each body sees. Runs once per frame after
/// `sync_solar_system_state`, before any consumer reads `CameraExposure`.
pub(super) fn update_camera_exposure(
    cache: Res<SolarSystemState>,
    focus: Res<CameraFocus>,
    bodies: Query<&CelestialBody>,
    sim: Res<SimulationState>,
    player: Option<Res<crate::player_controller::PlayerControllerState>>,
    mut exposure: ResMut<CameraExposure>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let star_pos = states.first().map(|s| s.position).unwrap_or_default();

    let focus_dist_m = match focus.target {
        CameraFocusTarget::Body(body_id) => bodies
            .iter()
            .find(|body| body.body_id == body_id && !body.is_star)
            .and_then(|body| states.get(body.body_id))
            .map(|s| (s.position - star_pos).length()),
        CameraFocusTarget::Ship => Some((sim.simulation.ship_state().position - star_pos).length()),
        CameraFocusTarget::PlayerController => player
            .as_deref()
            .and_then(|state| state.active_position_m())
            .map(|position| (position - star_pos).length())
            .or_else(|| Some((sim.simulation.ship_state().position - star_pos).length())),
        CameraFocusTarget::Ghost(ghost_focus) => states
            .get(ghost_focus.body_id)
            .map(|s| (s.position - star_pos).length()),
        CameraFocusTarget::None => None,
    }
    .unwrap_or(AU_M);

    let focus_d_au = (focus_dist_m / AU_M).max(1.0e-3);
    let gain = focus_d_au.powf(EXPOSURE_ALPHA) as f32;

    exposure.focus_dist_m = focus_dist_m;
    exposure.gain = gain;
    exposure.ev = gain.max(1.0e-6).log2();
}

/// Drive per-camera film grain strength from the current exposure push. When
/// the exposure system is lifting a dark outer-system scene by several EV,
/// that's equivalent to running a real sensor at high ISO: the visible result
/// is more grain. We add grain proportional to the positive EV push so Nyx
/// reads as "dim, sensor-limited" rather than "just another 1 AU body in
/// weird light."
pub(super) fn sync_film_grain_to_exposure(
    exposure: Res<CameraExposure>,
    mut grains: Query<&mut FilmGrain>,
) {
    // Only positive EV adds grain. Pulling bright scenes down (inner-system
    // focus) doesn't add noise in a real sensor.
    let push_ev = exposure.ev.max(0.0);
    let normalized = (push_ev / EXPOSURE_EV_GRAIN_MAX).clamp(0.0, 1.0);
    const BASE_INTENSITY: f32 = 0.006;
    const MAX_EXTRA: f32 = 0.010;
    let target = BASE_INTENSITY + normalized * MAX_EXTRA;
    for mut grain in &mut grains {
        grain.intensity = target;
    }
}

/// Build a `SceneLighting` snapshot for one body: one star (index 0),
/// eclipse occluders drawn from every other non-trivial body, shared
/// exposure gain, ambient floor. Planetshine is filled separately by
/// the caller because only terrestrial moons need it.
pub(super) fn build_scene_lighting(
    body_id: usize,
    states: &BodyStates,
    occluders: &[(usize, Vec3, f32)],
    gain: f32,
) -> SceneLighting {
    let mut scene = SceneLighting {
        ambient_intensity: PLANET_AMBIENT,
        ..Default::default()
    };

    let star_pos = states.first().map(|s| s.position).unwrap_or_default();
    let body_pos = states.get(body_id).map(|s| s.position).unwrap_or_default();
    let offset = star_pos - body_pos;
    let distance_m = offset.length();
    let to_star = if distance_m > 0.0 {
        (offset / distance_m).as_vec3()
    } else {
        Vec3::Y
    };
    let au_over_d = AU_M / distance_m.max(1.0);
    let flux = LIGHT_AT_1AU * (au_over_d * au_over_d) as f32 * gain;

    scene.star_count = 1;
    scene.stars[0] = StarLight {
        dir_flux: Vec4::new(to_star.x, to_star.y, to_star.z, flux),
        color: Vec4::new(1.0, 1.0, 1.0, 0.0),
    };

    let mut count = 0usize;
    for (other_id, pos, radius) in occluders {
        if *other_id == body_id {
            continue;
        }
        if count >= MAX_ECLIPSE_OCCLUDERS {
            break;
        }
        scene.occluders[count] = Vec4::new(pos.x, pos.y, pos.z, *radius);
        count += 1;
    }
    scene.occluder_count = count as u32;

    scene
}

/// Collect eclipse-occluder candidates from every visible non-star body
/// at the given metres → render-units scale. Used twice per frame: once
/// at [`MAP_SCALE`] for the map-layer impostor materials, once at
/// [`SHIP_SCALE`] for the ship-layer ones.
///
/// Caller owns the buffer (typically a `Local`) so the per-frame
/// allocation is paid once at startup and reused thereafter.
pub(super) fn collect_occluders<'a>(
    out: &mut Vec<(usize, Vec3, f32)>,
    states: &BodyStates,
    origin: &RenderOrigin,
    scale: f64,
    bodies: impl IntoIterator<Item = &'a CelestialBody>,
) {
    out.clear();
    for body in bodies {
        if body.is_star || body.radius_m < 1.0 {
            continue;
        }
        let Some(state) = states.get(body.body_id) else {
            continue;
        };
        let render_pos = ((state.position - origin.position) * scale).as_vec3();
        let render_radius = ((body.radius_m * scale) as f32).max(0.005);
        out.push((body.body_id, render_pos, render_radius));
    }
}

/// Push lighting state into every [`SolidPlanetMaterial`] each frame.
///
/// Mirrors the terrestrial planet light update for placeholder bodies (no terrain
/// pipeline yet): same scene-lighting build, same planetshine logic for
/// moons. The placeholder has no orientation, atmosphere, or cloud state,
/// so the work stops at `params.scene`.
pub(super) fn update_solid_planet_params(
    query: Query<(&CelestialBody, &SolidPlanetMaterials)>,
    mut materials: ResMut<Assets<SolidPlanetMaterial>>,
    mut halo_materials: ResMut<Assets<SolidPlanetHaloMaterial>>,
    cache: Res<SolarSystemState>,
    origin: Res<RenderOrigin>,
    sim: Res<SimulationState>,
    exposure: Res<CameraExposure>,
    view: Res<ViewMode>,
    planetshine: Res<PlanetshineTints>,
    mut map_occluders: Local<Vec<(usize, Vec3, f32)>>,
    mut ship_occluders: Local<Vec<(usize, Vec3, f32)>>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let body_defs = sim.simulation.bodies();
    let gain = exposure.gain;

    // See note on `update_planet_light_dirs` — only push uniforms into
    // the active view's material; the other one isn't being rendered.
    let force_both = view.is_changed();
    let do_map = force_both || matches!(*view, ViewMode::Map);
    let do_ship = force_both || matches!(*view, ViewMode::Ship);

    let body_iter = || query.iter().map(|(b, _)| b);
    collect_occluders(&mut map_occluders, states, &origin, MAP_SCALE, body_iter());
    collect_occluders(
        &mut ship_occluders,
        states,
        &origin,
        SHIP_SCALE,
        body_iter(),
    );
    let map_slice: &[(usize, Vec3, f32)] = &map_occluders;
    let ship_slice: &[(usize, Vec3, f32)] = &ship_occluders;

    for (body, mats) in &query {
        let body_def = &body_defs[body.body_id];
        // World→body-fixed rotation for sampling the impostor albedo cube, so it
        // co-rotates with the planet's spin. Identity for solid-colour bodies
        // (they ignore it). Scale-invariant, so one value serves both views.
        let orientation = states
            .get(body.body_id)
            .map(|s| {
                let q = s.orientation.inverse().as_quat().normalize();
                Vec4::new(q.x, q.y, q.z, q.w)
            })
            .unwrap_or(Vec4::new(0.0, 0.0, 0.0, 1.0));
        for (handle, occluders, scale, want, halo) in [
            (
                &mats.map,
                map_slice,
                MAP_SCALE,
                do_map,
                mats.map_halo.as_ref(),
            ),
            (&mats.ship, ship_slice, SHIP_SCALE, do_ship, None),
        ] {
            if !want {
                continue;
            }
            let radius = ((body.radius_m * scale) as f32).max(0.005);
            let mut scene = build_scene_lighting(body.body_id, states, occluders, gain);

            if let Some(parent_id) = body_def.parent {
                let parent_def = &body_defs[parent_id];
                if !matches!(parent_def.kind, thalos_world::BodyKind::Star)
                    && let Some(parent_state) = states.get(parent_id)
                    && let Some(tint) = planetshine.by_body.get(&parent_id)
                {
                    let parent_render_pos =
                        ((parent_state.position - origin.position) * scale).as_vec3();
                    let parent_radius = (parent_def.radius_m * scale) as f32;
                    scene.planetshine_pos_radius = Vec4::new(
                        parent_render_pos.x,
                        parent_render_pos.y,
                        parent_render_pos.z,
                        parent_radius,
                    );
                    scene.planetshine_tint_flag = Vec4::new(tint[0], tint[1], tint[2], 1.0);
                }
            }

            // Atmosphere rim companion (map only): same radius + scene as the
            // body disc; the static atmosphere block is left untouched. Updated
            // before the body material so `scene` is still available to move
            // into it below (`SceneLighting` is Clone, not Copy).
            if let Some(halo_handle) = halo {
                let halo_dirty = matches!(
                    halo_materials.get(halo_handle),
                    Some(m) if m.params.radius != radius || m.params.scene != scene
                );
                if halo_dirty && let Some(mut m) = halo_materials.get_mut(halo_handle) {
                    m.params.radius = radius;
                    m.params.scene = scene.clone();
                }
            }

            // Peek before mutating; `get_mut` re-uploads the uniform even
            // on identical writes.
            let dirty = matches!(
                materials.get(handle),
                Some(mat) if mat.params.radius != radius
                    || mat.params.scene != scene
                    || mat.params.orientation != orientation
            );
            if dirty && let Some(mut mat) = materials.get_mut(handle) {
                mat.params.radius = radius;
                mat.params.scene = scene;
                mat.params.orientation = orientation;
            }
        }
    }
}

/// Calibration: Bevy lux per unit of shading-spine flux. The Bevy-lit
/// `StandardMaterial` surfaces (hull, gear, structures, runway) take their sun
/// illuminance from the SAME heliocentric flux the spine gives every terrain /
/// vegetation surface (`build_scene_lighting`: `LIGHT_AT_1AU·(AU/d)²·gain`),
/// converted to lux through this one constant. At the homeworld (spine flux ≈ 10,
/// focus gain ≈ 1) it lands the daytime hull at ~10 000 lux — the value the old
/// flat `SUN_DAY_ILLUMINANCE` pinned — but now scaling with heliocentric distance
/// and exposure, so a ship at a far body dims like the ground beneath it instead
/// of staying noon-bright. Tune from a `just game runway` noon screenshot until the
/// hull reads at the same brightness as the terrain beside it.
/// Also the input to [`thalos_body_shading::spine_parity_exposure`], which
/// derives the camera exposure that makes a Bevy-lit surface render exactly as
/// bright as the spine's ground for this constant — retune it there and the two
/// universes stay matched (`camera::spawn_camera`).
pub(crate) const LUX_PER_SPINE_FLUX: f32 = 1_000.0;
/// **Space-regime** ambient (sky-fill) day + deep-night brightness, in Bevy lux,
/// for the Bevy-PBR surfaces (hull, gear, structures, runway). Since **F4** the
/// *surface* ambient is the physical sky irradiance from the sky-view LUT
/// ([`AMBIENT_SKY_LUX_GAIN`]); this flat pair is now only the **space/high-altitude
/// stand-in** the surface fades in over (`SkyAmbient::surface_blend`) — there is no
/// atmosphere out there, so it is a coarse fill for planetshine/zodiacal light that
/// env-map IBL at photometric intensity (W7/F7) will retire. The day value keeps a
/// ~7:1 lit:shadow ratio against the 10 000-lux sun. See `docs/roadmap/graphics_fidelity.md` §3.
/// (Both scaled ×2.77 with the NTR-X5 exposure fix — they were eyeball-tuned
/// against Bevy's uncalibrated default exposure, so carrying the exposure ratio
/// through keeps the space regime looking exactly as it did.)
const AMBIENT_DAY_BRIGHTNESS: f32 = 1_940.0;
/// The deep-night ambient floor (starlight / planetshine). Also read by the
/// tile-terrain driver, which needs its share of the resolved ambient to gate
/// the sky/space fill per fragment without touching this floor.
pub(super) const AMBIENT_NIGHT_BRIGHTNESS: f32 = 11.0;
/// Sky-blue tint for the **space-regime** ambient fill (see [`AMBIENT_DAY_BRIGHTNESS`]);
/// the surface tint now comes from the physical sky chroma. Tune alongside it.
const AMBIENT_DAY_TINT: Color = Color::srgb(0.62, 0.72, 0.95);
/// Calibration: scales the physical sky irradiance ([`crate::reflection_probe::SkyAmbient`])
/// into the **surface** `GlobalAmbientLight` brightness (F4), through the flux→lux
/// mapping shared with the sun ([`LUX_PER_SPINE_FLUX`]).
///
/// **1.0 would mean the flat ambient carries the sky's whole diffuse
/// irradiance**, i.e. the bridge is the one flux→lux constant and nothing
/// else. Measured against the spine ground it overfills — the spine's own sky
/// term carries an artistic `SURFACE_SKY_SCALE`, so 0.7 is where the two
/// universes' shadow fill actually meets (matched by capture on
/// `massif-valley`: shadowed-ground p05 luminance 0.125 both sides).
///
/// It was 0.2 on the theory that the env cubemap
/// (`GeneratedEnvironmentMapLight`, painted from the same sky-view LUT) already
/// delivered most of the sky's diffuse irradiance, so a full-strength flat
/// ambient would count the sky twice. That reasoning does not survive the
/// NTR-X5 calibration: **the cubemap is painted in scene-flux units** (radiance
/// ~0.1–1.3) while Bevy consumes it in the same photometric space as the
/// directional light's lux, so at `PROBE_INTENSITY = 1.0` its diffuse
/// contribution is three orders of magnitude short of the sky it depicts —
/// effectively zero. The 0.2 residual was therefore the *entire* sky fill, and
/// tuned against an uncalibrated exposure at that: with the exposure fixed it
/// left shadowed ground at a fifth of the spine's, crushing gullies to black
/// and reading asphalt as a hole in the ground.
///
/// **If the env map is ever put on physical units** (W7/F7), this must come
/// back down by whatever share the env then carries, or the sky *will* be
/// double-counted — the failure the 0.2 was guarding against.
///
/// **0.7 → 0.38 (2026-07-29).** The 0.7 calibration predates the shadow
/// direct/ambient split (BL-20260726T222119Z): it was matched against a shadow
/// model that multiplied the *whole* colour down, so the fill's own strength
/// barely reached the frame. Once shadows switched to "kill the sun, keep the
/// full sky fill", every shadowed fragment became a pure sample of this
/// ambient — and at 0.7 the tile ground's shadow fill measured 3.5× the spine
/// ground's beside it (`space-center-hill-view` A/B, shadowed-ground p05
/// luminance 0.0274 vs 0.0079), reading as a pale wash rather than a deep
/// sky-lit shadow. 0.38 puts shadowed ground at roughly twice the spine's
/// whole-multiply artifact — deep, still clearly sky-filled.
const AMBIENT_SKY_LUX_GAIN: f32 = 0.38;

/// Share of the flat surface ambient's *chroma* taken from the warm ground
/// bounce rather than the blue sky. The spine fills every fragment through
/// `sky_ambient_irradiance`: blue `sky_radiance` on up-facing normals blended
/// toward warm `ground_radiance` below the horizon plane. Bevy's flat
/// `GlobalAmbientLight` stands in for that whole surrounding fill with ONE
/// colour — and it was taking the sky's chroma alone, which is why every
/// shadow and diffuse-lit slope rendered blue-teal (the sky-view LUT's
/// cosine-weighted irradiance runs B/R ≈ 3.7, bluer than the sky dome the
/// renderer itself draws). Folding the bounce share back in is the flat-fill
/// equivalent of the spine's per-normal mix, not a new palette: the bounce
/// colour below mirrors `lighting.wgsl`'s `SURFACE_GROUND_ALBEDO`, lit by the
/// same reddened beam as the ground it bounces off.
const AMBIENT_GROUND_BOUNCE_SHARE: f32 = 0.25;

/// Mirror of `lighting.wgsl`'s `SURFACE_GROUND_ALBEDO` — the representative
/// sunlit-land albedo the spine's warm ground bounce reflects. Keep in
/// lockstep.
const SURFACE_GROUND_ALBEDO: Vec3 = Vec3::new(0.10, 0.085, 0.055);

/// Direct-beam airmass reddening for the Bevy sun, mirroring
/// `lighting.wgsl::compute_surface_sky` (`SURFACE_SUN_REDDEN_GAIN = 1.0`):
/// `exp(-τ_eff · (airmass − 1))`, airmass `= clamp(1/(sun_up + 0.10), 1, 8)`.
/// The spine reddens every surface's direct beam as the sun drops; the Bevy
/// `DirectionalLight` colour was never written, so tile ground / hull /
/// structures kept a pure-white noon sun at any elevation — at a 13° sun the
/// spine beam is (0.91, 0.80, 0.58) and the standard path's was (1, 1, 1),
/// which is most of why sunlit tile ground read cold beside udlod's
/// (`ground-photometry` A/B: sunlit B−R +0.02 vs −0.18). Energy loss rides the
/// colour (Bevy multiplies it into the light), exactly as `sun_color` does on
/// the spine.
fn surface_sun_tint(tau_v: Vec3, strength: f32, sun_elev: f32) -> Vec3 {
    let tau_eff = tau_v.max(Vec3::ZERO) * strength.max(0.0);
    let sun_up = sun_elev.clamp(0.0, 1.0);
    let airmass = (1.0 / (sun_up + 0.10)).clamp(1.0, 8.0);
    (-tau_eff * (airmass - 1.0)).exp()
}

/// `smoothstep`, matching the WGSL builtin the terrain shader uses so the
/// CPU-side day/night gate lines up with the ground's terminator exactly.
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The single day/night terminator for the Bevy-lit (`StandardMaterial`) surfaces
/// — hull, gear, structures, runway — so they fade across the terminator in
/// lockstep with the terrain shader AND with each other. One definition consumed
/// by both [`update_sun_light`] (the sun/ambient gate) and [`update_moon_light`]
/// (its night gate `= 1 - surface_daylight`), replacing the two hand-copied
/// smoothsteps that had drifted apart.
///
/// Mirrors `body_terrain.wgsl`'s `daylight = smoothstep(-0.06, 0.12, dot(up,
/// sun_dir))` at the surface (`altitude_ratio = R/r = 1` → threshold 0), and
/// slides the terminator to the geometric umbra entry as the craft climbs, so only
/// the true shadow cylinder behind the body darkens a high orbit (not the whole far
/// hemisphere). `sun_elevation = dot(local_up, to_sun)`, `altitude_ratio = R_body /
/// r_craft`. Returns 1 = full day, 0 = night.
fn surface_daylight(sun_elevation: f64, altitude_ratio: f64) -> f64 {
    let ratio = altitude_ratio.clamp(0.0, 1.0);
    let threshold = -((1.0 - ratio * ratio).max(0.0)).sqrt();
    smoothstep(threshold - 0.06, threshold + 0.12, sun_elevation)
}

/// The day/night gate [`update_sun_light`] actually applied this frame, published
/// so consumers can *undo* it rather than re-derive the terminator (which would
/// fork [`surface_daylight`] — the whole point of having one definition).
///
/// The tile-terrain shader is the consumer: `GlobalAmbientLight` is one value
/// baked at the craft, `floor + fill·daylight_craft`, and the shader re-spreads
/// it per fragment as `floor + fill·daylight_fragment`. It needs this divisor to
/// do that, and getting it exactly right is what makes the near field — where
/// every fragment shares the craft's horizon — an identity instead of a second
/// terminator ramp stacked on the first.
///
/// **Sole writer:** [`update_sun_light`].
#[derive(Resource, Debug, Clone, Copy)]
pub(crate) struct SunDaylight(pub f32);

impl Default for SunDaylight {
    fn default() -> Self {
        // No gate applied → dividing by it is a no-op.
        Self(1.0)
    }
}

/// Point the directional sun light from the star toward the camera's focus
/// body, and set its illuminance + the global ambient floor as a *projection of
/// the shading spine's lighting* (F1 of the graphics-fidelity unification —
/// `docs/roadmap/graphics_fidelity.md` §3).
///
/// The ship hull, gear, structures, and runway are Bevy `StandardMaterial`s lit by
/// this one directional light + `GlobalAmbientLight`, while the terrain / vegetation
/// are lit by the `thalos::lighting` spine. To keep the two universes in lockstep
/// this system drives the Bevy lights from the *same two quantities* the spine uses:
/// (1) the heliocentric flux `LIGHT_AT_1AU·(AU/d)²·gain` (`build_scene_lighting`),
/// so the hull dims with distance + exposure instead of the old flat 10 000 lux that
/// lit a ship at a far body identically to one at Thalos; and (2) one shared
/// terminator ([`surface_daylight`], mirroring `body_terrain.wgsl`), so direct sun +
/// sky-fill fade across the same day/night band the ground does. All surfaces dim
/// together as the sun sets or the body eclipses the craft. Deep space (dominant
/// body is the star) keeps full daylight — no body to block the sun. **Sole writer**
/// of the sun `DirectionalLight` + `GlobalAmbientLight`.
pub(super) fn update_sun_light(
    cache: Res<SolarSystemState>,
    focus: Res<CameraFocus>,
    sim: Res<SimulationState>,
    exposure: Res<CameraExposure>,
    player: Option<Res<crate::player_controller::PlayerControllerState>>,
    sky_ambient: Res<crate::reflection_probe::SkyAmbient>,
    time: Res<Time<Real>>,
    height_sources: Option<Res<thalos_physics_local::HeightSourceRegistry>>,
    mut ambient: ResMut<GlobalAmbientLight>,
    mut sun_daylight: ResMut<SunDaylight>,
    mut light_query: Query<(&mut Transform, &mut DirectionalLight), With<SunLight>>,
    mut last_logged_lux: Local<f32>,
) {
    let Some(ref states) = cache.states else {
        return;
    };

    // Find the focus body's physics-space position — or the ship's when
    // focus is on the player's ship (so sun direction tracks the ship in
    // ship view).
    let focus_pos = match focus.target {
        CameraFocusTarget::Body(body_id) => states.get(body_id).map(|s| s.position),
        CameraFocusTarget::Ship => Some(sim.simulation.ship_state().position),
        CameraFocusTarget::PlayerController => player
            .as_deref()
            .and_then(|state| state.active_position_m())
            .or_else(|| Some(sim.simulation.ship_state().position)),
        CameraFocusTarget::Ghost(ghost_focus) => {
            states.get(ghost_focus.body_id).map(|s| s.position)
        }
        CameraFocusTarget::None => None,
    }
    .unwrap_or(bevy::math::DVec3::ZERO);

    // Star is always at index 0.
    let star_pos = states
        .first()
        .map(|s| s.position)
        .unwrap_or(bevy::math::DVec3::ZERO);

    let offset = focus_pos - star_pos;
    if offset.length_squared() < 1.0e6 {
        return; // Focus is on (or very near) the star; direction undefined.
    }

    // Day/night gate, evaluated at the craft's position (not the camera focus,
    // which may be a remote body in map view) over the body it's gravitationally
    // bound to. `daylight` ∈ [0,1]: 1 with the sun high overhead, 0 on the deep
    // night side. Falls back to full daylight when the craft sits in the star's
    // SOI (interplanetary cruise) — there's no nearby horizon to occlude.
    let craft_pos = player
        .as_deref()
        .and_then(|state| state.active_position_m())
        .unwrap_or_else(|| sim.simulation.ship_state().position);
    let dominant = sim.simulation.dominant_body();
    // Kept for the diagnostics log below (NaN = star-SOI fallback branch).
    let mut logged_sun_elev = f64::NAN;
    // Terrain horizon-angle sun visibility at the craft (W12 object-side v1):
    // a mountain between the craft and the low sun pulls the direct term to 0,
    // so the parked ship / structures / EVA fall into the same relief shadow
    // the terrain shader's own self-shadow march darkens the valley with. The
    // ambient sky fill below deliberately does NOT take this factor — shadowed
    // ground still sees the sky.
    let mut horizon_vis = 1.0_f32;
    let daylight = match states.get(dominant) {
        Some(body) if dominant != 0 => {
            let to_center = body.position - craft_pos;
            let r = to_center.length();
            let radius = sim.simulation.bodies()[dominant].radius_m;
            let up = (-to_center).normalize_or_zero();
            let to_sun = (star_pos - craft_pos).normalize_or_zero();
            // Horizon march only near the surface (relief occlusion is
            // negligible from altitude) and only when the body has a live
            // height source (terrain resident).
            if r - radius < 15_000.0
                && let Some(source) = height_sources.as_deref().and_then(|hs| hs.get(dominant))
            {
                let inv_orient = body.orientation.normalize().inverse();
                let craft_bf = inv_orient * (craft_pos - body.position);
                let sun_bf = inv_orient * to_sun;
                horizon_vis = thalos_body_render::horizon_sun_visibility(
                    craft_bf,
                    sun_bf,
                    radius,
                    source.as_ref(),
                );
            }
            // One shared terminator (altitude-aware umbra entry) — see
            // `surface_daylight`. At the surface this reduces to the terrain
            // shader's `smoothstep(-0.06, 0.12, sun_elevation)`.
            let ratio = if r > 0.0 { radius / r } else { 1.0 };
            logged_sun_elev = up.dot(to_sun);
            surface_daylight(logged_sun_elev, ratio) as f32
        }
        _ => 1.0,
    };

    // Heliocentric flux the craft receives, in the SAME units the shading spine
    // gives every terrain/vegetation surface (`build_scene_lighting`):
    // LIGHT_AT_1AU · (AU/d)² · exposure_gain. Routing the Bevy sun through it ties
    // the StandardMaterial hull/structures to the spine's flux, so they dim with
    // heliocentric distance + exposure exactly like the ground.
    let helio_d_m = (craft_pos - star_pos).length();
    let au_over_d = (AU_M / helio_d_m.max(1.0)) as f32;
    let flux = LIGHT_AT_1AU * au_over_d * au_over_d * exposure.gain;

    // Surface-regime beam tint (see `surface_sun_tint`): the dominant body's
    // authored Rayleigh τ_v + strength — the same values every spine surface
    // recovers from its `AtmosphereBlock` — faded to white with altitude
    // (`surface_blend`), because the reddening is the beam's slant path through
    // *this* atmosphere and a hull in orbit is above it.
    let (tau_v, atm_strength) = sim
        .system
        .bodies
        .get(dominant)
        .and_then(|b| b.terrestrial_atmosphere.as_ref())
        .and_then(|a| a.scattering.as_ref())
        .map(|s| (Vec3::from_array(s.vertical_optical_depth), s.strength))
        .unwrap_or((Vec3::ZERO, 0.0));
    let sun_elev_f = if logged_sun_elev.is_finite() {
        logged_sun_elev as f32
    } else {
        1.0
    };
    let surface_blend = sky_ambient.surface_blend.clamp(0.0, 1.0);
    let sun_tint = Vec3::ONE.lerp(
        surface_sun_tint(tau_v, atm_strength, sun_elev_f),
        surface_blend,
    );

    let dir_f32 = offset.normalize().as_vec3();
    let illuminance = LUX_PER_SPINE_FLUX * flux * daylight * horizon_vis;
    for (mut transform, mut light) in &mut light_query {
        // DirectionalLight shines along its local -Z, so we look in the light's travel direction.
        transform.look_to(dir_f32, Vec3::Y);
        light.color = Color::linear_rgb(sun_tint.x, sun_tint.y, sun_tint.z);
        light.illuminance = illuminance;
    }

    // ── Ambient (sky-fill): physical on the surface (F4), stand-in in space ──
    //
    // On the surface the flat `GlobalAmbientLight` is now the PHYSICAL sky-fill
    // from the F3 sky-view LUT (`SkyAmbient`, published by the reflection probe):
    // its hemispherical irradiance (scene-flux) → lux through the SAME flux→lux
    // constant as the sun, so the sun and its sky fill share one calibration and
    // the fill tracks time-of-day, sun elevation, and the atmosphere instead of a
    // hand-tuned constant + fixed tint. Out in space (no atmosphere → no sky) it
    // fades by altitude to the unchanged flat stand-in, which env-map IBL at
    // photometric intensity will retire (W7/F7).
    sun_daylight.0 = daylight;

    let flux_norm = (flux / LIGHT_AT_1AU).clamp(0.0, 1.0);
    let space_day = AMBIENT_DAY_BRIGHTNESS * flux_norm;
    let space_ambient =
        AMBIENT_NIGHT_BRIGHTNESS + (space_day - AMBIENT_NIGHT_BRIGHTNESS).max(0.0) * daylight;

    let sky_irr = sky_ambient.surface_irradiance;
    let sky_lux = luminance(sky_irr) * LUX_PER_SPINE_FLUX * AMBIENT_SKY_LUX_GAIN;
    let surface_ambient = AMBIENT_NIGHT_BRIGHTNESS + sky_lux; // sky_lux → 0 at night

    let blend = surface_blend;
    let target_brightness = space_ambient * (1.0 - blend) + surface_ambient * blend;

    // Colour target: the surface fill chroma (luminance-normalised so brightness
    // owns the magnitude), fading to the space stand-in tint. The fill is the
    // flat stand-in for the spine's whole surrounding light — blue sky above,
    // warm sunlit-ground bounce below (`sky_ambient_irradiance`) — so its
    // chroma blends both (see `AMBIENT_GROUND_BOUNCE_SHARE`); sky-only was the
    // teal-shadow defect. The bounce is the representative ground albedo lit by
    // the same reddened beam, gated by daylight so it vanishes at night (each
    // term luminance-normalised first, so the blend weights are exact shares).
    let space_tint = AMBIENT_DAY_TINT.to_linear();
    let space_tint = Vec3::new(space_tint.red, space_tint.green, space_tint.blue);
    let bounce =
        SURFACE_GROUND_ALBEDO * surface_sun_tint(tau_v, atm_strength, sun_elev_f) * daylight;
    let surface_tint = match (normalized_chroma(sky_irr), normalized_chroma(bounce)) {
        (Some(sky), Some(gnd)) => {
            sky * (1.0 - AMBIENT_GROUND_BOUNCE_SHARE) + gnd * AMBIENT_GROUND_BOUNCE_SHARE
        }
        (Some(sky), None) => sky,
        _ => space_tint,
    };
    let target_tint = space_tint.lerp(surface_tint, blend);

    // Temporal smoothing (~0.7 s time constant, frame-rate independent): the
    // probe republishes `SkyAmbient` on a coarse real-time cadence, and under
    // warp consecutive publishes can jump across a whole day/night boundary —
    // smooth the flat ambient toward its target so those crossings fade instead
    // of hard-cutting the scene. (The env cubemap itself still cuts per repaint;
    // acceptable on reflective detail, jarring on the global fill.)
    let alpha = 1.0 - (-time.delta_secs() / 0.7).exp();
    ambient.brightness += (target_brightness - ambient.brightness) * alpha;
    let cur = ambient.color.to_linear();
    let cur = Vec3::new(cur.red, cur.green, cur.blue);
    let tint = cur.lerp(target_tint, alpha);
    ambient.color = Color::linear_rgb(tint.x, tint.y, tint.z);

    // Calibration signal (F4): record the resolved ambient when it moves > 5%,
    // so a "hull too dark/bright" screenshot comes with the number to retune
    // `AMBIENT_SKY_LUX_GAIN` against.
    if (ambient.brightness - *last_logged_lux).abs() > 0.05 * last_logged_lux.max(1.0) {
        info!(
            target: "thalos::diagnostic::sky",
            event = "ambient_light",
            ambient_lux = ambient.brightness,
            target_lux = target_brightness,
            sky_lux,
            surface_blend = blend,
            space_lux = space_ambient,
            sun_lux = illuminance,
            sun_elevation = logged_sun_elev,
            daylight,
            sim_time_s = sim.simulation.sim_time(),
            "resolved global ambient light"
        );
        *last_logged_lux = ambient.brightness;
    }
}

/// Rec. 709 relative luminance of a linear-RGB radiance.
fn luminance(c: Vec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// Luminance-normalised chroma (a unit-luminance tint), or `None` when the input
/// is too dark to carry a meaningful hue.
fn normalized_chroma(c: Vec3) -> Option<Vec3> {
    let l = luminance(c);
    (l > 1.0e-6).then(|| c / l)
}

/// Full-moon illuminance for the `StandardMaterial` hull + structures, in Bevy
/// lux. The terrain has its own moonlight term (`body_terrain.wgsl`); this is the
/// matching directional light so the craft and surface buildings catch the same
/// moonlight. Soft and night-gated; tune from a night screenshot.
const MOON_FULL_LUX: f32 = 60.0;
/// Reference "bright moon" reflectance shape (albedo_luminance × (R/d)²) mapping
/// to full brightness — mirrors `compute_moonlight` in `ground_terrain.rs` so the
/// hull and the ground agree on which moon is brightest and how bright it is.
const MOON_REF_SHAPE: f64 = 3.0e-6;

/// Drive the [`MoonLight`] directional light from the brightest child moon of the
/// body the craft is on. Mirrors the terrain's `compute_moonlight`: Lambert phase
/// × per-moon size/albedo/distance, night-gated over the craft's local horizon,
/// and faded in as the moon rises. Off (illuminance 0) by day or with no lit moon
/// up — so it only ever adds the night fill, never competes with the sun.
pub(super) fn update_moon_light(
    cache: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    player: Option<Res<crate::player_controller::PlayerControllerState>>,
    mut light_query: Query<(&mut Transform, &mut DirectionalLight), With<MoonLight>>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let Ok((mut transform, mut light)) = light_query.single_mut() else {
        return;
    };

    let star_pos = states
        .first()
        .map(|s| s.position)
        .unwrap_or(bevy::math::DVec3::ZERO);
    let craft_pos = player
        .as_deref()
        .and_then(|state| state.active_position_m())
        .unwrap_or_else(|| sim.simulation.ship_state().position);
    let dominant = sim.simulation.dominant_body();
    let dominant_pos = states.get(dominant).map(|b| b.position);

    // Night gate over the dominant body's horizon at the craft (mirrors
    // `update_sun_light`): moonlight fades in as the sun sets. Zero in the star's
    // own SOI (no nearby body to set night) — moonlight is a surface effect.
    let night = match dominant_pos {
        Some(dpos) if dominant != 0 => {
            // Same terminator the sun + ground use, so moonlight fades in exactly
            // as the sun's day-fill fades out (no overlap band, no gap).
            let to_center = dpos - craft_pos;
            let r = to_center.length();
            let radius = sim.simulation.bodies()[dominant].radius_m;
            let to_sun = (star_pos - craft_pos).normalize_or_zero();
            let up = (-to_center).normalize_or_zero();
            let ratio = if r > 0.0 { radius / r } else { 1.0 };
            1.0 - surface_daylight(up.dot(to_sun), ratio) as f32
        }
        _ => 0.0,
    };

    if night <= 0.0 || dominant_pos.is_none() {
        light.illuminance = 0.0;
        return;
    }
    let up = (craft_pos - dominant_pos.unwrap()).normalize_or_zero();

    let mut best_lux = 0.0f32;
    let mut best_dir = bevy::math::DVec3::Y;
    let mut best_color = Color::WHITE;
    for moon in sim.simulation.bodies() {
        if !matches!(moon.kind, thalos_world::BodyKind::Moon) || moon.parent != Some(dominant) {
            continue;
        }
        let Some(moon_state) = states.get(moon.id) else {
            continue;
        };
        let to_moon = moon_state.position - craft_pos;
        let d = to_moon.length();
        if d <= 0.0 {
            continue;
        }
        let moon_dir = to_moon / d;
        // Fade in as the moon clears the local horizon; skip when it's down.
        let horizon = smoothstep(0.0, 0.10, up.dot(moon_dir)) as f32;
        if horizon <= 0.0 {
            continue;
        }
        // Lambert phase as seen from the craft: angle AT the moon between the
        // star and the craft (full moon → 0 → phase 1).
        let to_star_from_moon = (star_pos - moon_state.position).normalize_or_zero();
        let to_craft_from_moon = (craft_pos - moon_state.position).normalize_or_zero();
        let cos_g = to_star_from_moon.dot(to_craft_from_moon).clamp(-1.0, 1.0);
        let g = cos_g.acos();
        let phase =
            ((g.sin() + (std::f64::consts::PI - g) * cos_g) / std::f64::consts::PI).clamp(0.0, 1.0);

        let color_lin = Color::srgb(moon.color[0], moon.color[1], moon.color[2]).to_linear();
        let albedo_lum =
            (0.2126 * color_lin.red + 0.7152 * color_lin.green + 0.0722 * color_lin.blue) as f64;
        let ang = moon.radius_m / d;
        let shape = albedo_lum * ang * ang;
        let rel = (shape / MOON_REF_SHAPE).clamp(0.0, 1.5);

        let lux = MOON_FULL_LUX * (phase * rel) as f32 * night * horizon;
        if lux > best_lux {
            best_lux = lux;
            best_dir = moon_dir;
            best_color = Color::srgb(moon.color[0], moon.color[1], moon.color[2]);
        }
    }

    light.illuminance = best_lux;
    if best_lux > 0.0 {
        light.color = best_color;
        // DirectionalLight travels along its local -Z: aim it from the moon down
        // toward the surface, i.e. opposite the craft→moon direction.
        transform.look_to((-best_dir).as_vec3(), Vec3::Y);
    }
}
