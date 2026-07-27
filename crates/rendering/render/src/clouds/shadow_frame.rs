//! Placement of the cloud sun-transmittance cascade — CLOUD-5 / W2's near tier
//! (`docs/rendering/clouds.md` §3.5), the one field that answers "how much
//! sunlight survives the deck to reach this point" for every surface receiver.
//!
//! **One frame, two consumers.** [`CloudShadowFrame`] is the single CPU truth
//! for where the cascade lives. The compute pass marches exactly this frame
//! (through [`CloudsUniform`](super::uniforms::CloudsUniform)); every receiving
//! material projects into exactly this frame (through [`CloudShadowBlock`], the
//! std140 mirror of `thalos::cloud_shadow`'s WGSL struct). Deriving either side
//! independently is how the near/far cloud tiers once ended up rendering
//! different skies (`ActiveCloudFrame`'s docs) — a shadow that disagrees with
//! the cloud casting it is the same failure with a shorter feedback loop.
//!
//! **Why a plane and not a volume.** The map is parameterised over the tangent
//! plane at the anchor's ground point: texel *(u,v)* stores the transmittance
//! of the sun beam that *passes through that plane point*. A receiver anywhere
//! below the deck therefore looks up by intersecting its own sun ray with the
//! plane — which is the parallax that makes a low sun throw shadows kilometres
//! downwind of the cloud, instead of the coverage-map projection the spec
//! explicitly rejects as a finished implementation. Over the cascade's extent
//! the sphere departs from its tangent plane by `d²/2R` (33 m at 20 km on a
//! Thalos-sized body), far inside one texel of vertical structure.

use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

/// Metres of half-extent at which the ladder starts: an eye-level view, whose
/// geometric horizon on a Thalos-sized body is ~7 km.
const MIN_HALF_EXTENT_M: f32 = 8_000.0;
/// Ceiling of the ladder. At 512² this is a 250 m texel — still several times
/// finer than the cumulus shadow it carries — and past it the far tail (a
/// coarse body-fixed cube integral) owns the planet-scale answer. That tail is
/// not built yet, so beyond the cascade receivers simply feather to lit.
const MAX_HALF_EXTENT_M: f32 = 64_000.0;
/// Half-extent as a multiple of the anchor's altitude above the reference
/// sphere.
///
/// Deliberately generous. The naive reading — "cover what a 60° FOV sees
/// straight down", ~1.5× altitude — is what a *nadir* view needs; every framing
/// that matters here is oblique, and an oblique camera at 1 km sees ground a
/// hundred kilometres out. Measured on `spaceport-aerial` at 1.5×: the cascade
/// covered only the basin under the camera and the entire rest of the frame
/// fell outside (`artifacts/visual/runs/cloud-shadow/spaceport_payload.png`).
/// Coverage is worth far more than resolution for a term whose features are
/// kilometres wide and whose edges are penumbra-soft.
const EXTENT_PER_ALTITUDE: f32 = 12.0;
/// Below this sun elevation cosine the plane projection degenerates (the sun
/// ray runs parallel to the reference plane and the lookup shoots to infinity).
/// Ground within a few degrees of its own terminator carries no direct sun
/// worth shadowing, so the cascade stands down instead.
const MIN_SUN_ELEVATION_COS: f32 = 0.06;

/// Where the cloud sun-transmittance cascade sits this frame, in the **active
/// cloud body's body-fixed frame** (the same frame `ActiveCloudFrame` publishes
/// — never a second world→body rotation).
#[derive(Clone, Copy, Debug, Reflect)]
pub struct CloudShadowFrame {
    /// Centre of the map: a point on the reference sphere directly under the
    /// view anchor, snapped to the texel lattice.
    pub center: Vec3,
    /// Reference-plane normal (radial up at [`center`](Self::center)).
    pub up: Vec3,
    /// Unit tangents spanning the map's +u / +v texel axes.
    pub axis_u: Vec3,
    pub axis_v: Vec3,
    /// Half the map's edge length, metres.
    pub half_extent_m: f32,
    /// Cosine of the sun's elevation above the reference plane at the centre.
    pub sun_elevation_cos: f32,
    /// False when there is nothing to march (no cloud body, clouds disabled,
    /// or the sun is at/below the anchor's horizon). Receivers read fully lit.
    pub active: bool,
}

impl Default for CloudShadowFrame {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            up: Vec3::Y,
            axis_u: Vec3::X,
            axis_v: Vec3::Z,
            half_extent_m: MIN_HALF_EXTENT_M,
            sun_elevation_cos: 0.0,
            active: false,
        }
    }
}

impl CloudShadowFrame {
    /// Resolve the cascade for a camera at `camera_body` (body-fixed, planet
    /// centred) over a body of `planet_radius_m`, lit from `sun_body`.
    ///
    /// The map rides the camera continuously — deliberately NOT texel-snapped
    /// like a depth cascade. A snap exists to stop sample points crawling
    /// through content finer than a texel, and this map has none: the extent
    /// ladder below keeps the texel at 23 m on the ground and 250 m at the top
    /// rung, while the field it integrates is band-limited to the authored
    /// erosion scale (≥ 450 m, and retired by the same footprint fade the view
    /// march uses once the texel outgrows it). Every rung oversamples, so
    /// there is nothing to alias — and no snap to misalign against the
    /// receivers, which project into this exact frame.
    pub fn resolve(camera_body: Vec3, sun_body: Vec3, planet_radius_m: f32) -> Self {
        let radius = planet_radius_m.max(1.0);
        let Some(up) = camera_body.try_normalize() else {
            return Self::default();
        };
        let sun = sun_body.normalize_or_zero();
        let sun_elevation_cos = up.dot(sun);

        // Altitude-keyed extent, QUANTISED TO OCTAVES. A continuously varying
        // extent resamples the field every frame at slightly different spacing,
        // which reads as a shimmer over the whole ground; doubling steps are
        // rare and land on a scale change the eye already expects.
        let altitude = (camera_body.length() - radius).max(0.0);
        let target = (EXTENT_PER_ALTITUDE * altitude).clamp(MIN_HALF_EXTENT_M, MAX_HALF_EXTENT_M);
        let octave = (target / MIN_HALF_EXTENT_M).log2().ceil().max(0.0);
        let half_extent_m = (MIN_HALF_EXTENT_M * octave.exp2()).min(MAX_HALF_EXTENT_M);

        // Stable tangent frame from the body's spin axis (+Y body-fixed), so
        // the map's axes depend only on WHERE the anchor is, never on how it
        // got there. Degenerate over the poles — fall back to +X there.
        let mut axis_u = Vec3::Y.cross(up);
        if axis_u.length_squared() < 1.0e-6 {
            axis_u = Vec3::X.cross(up);
        }
        let axis_u = axis_u.normalize();
        let axis_v = up.cross(axis_u);

        Self {
            center: up * radius,
            up,
            axis_u,
            axis_v,
            half_extent_m,
            sun_elevation_cos,
            active: sun_elevation_cos > MIN_SUN_ELEVATION_COS,
        }
    }

    /// Metres of ground per map texel.
    pub fn texel_m(&self, texels: u32) -> f32 {
        2.0 * self.half_extent_m / texels.max(1) as f32
    }
}

/// std140 mirror of `thalos::cloud_shadow`'s `CloudShadowBlock` — field order
/// is the contract. Embedded in each receiving material's existing params
/// uniform (never its own `#[uniform]` slot: the terrain pipeline is already at
/// the Metal 16-vertex-buffer ceiling, see `BodyTerrainExtras`).
///
/// `axis_v.w == 0` means "no cascade" and every sampler early-outs fully lit,
/// so an unwritten block can never darken anything.
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct CloudShadowBlock {
    /// World render space → body-fixed rotation (xyzw quaternion).
    pub world_to_body: Vec4,
    /// xyz = body centre in world render space, w = artistic strength (0 = off).
    pub body_center_ws: Vec4,
    /// xyz = map centre (body-fixed), w = half extent in metres.
    pub center: Vec4,
    /// xyz = +u tangent, w = metres per texel.
    pub axis_u: Vec4,
    /// xyz = +v tangent, w = live flag (0 ⇒ skip sampling entirely).
    pub axis_v: Vec4,
    /// xyz = reference-plane normal, w = sun elevation cosine at the centre.
    pub up_sun: Vec4,
    /// xyz = body-fixed unit direction toward the sun (the axis the cascade was
    /// marched along), w reserved.
    pub sun_body: Vec4,
}
