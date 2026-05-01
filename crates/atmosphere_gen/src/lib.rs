//! Atmosphere parameter schemas.
//!
//! This crate holds the *data* definition of a body's atmosphere — not
//! the renderer, not the shader, not the GPU uniforms. It is the
//! analogue of `thalos_terrain_gen` for the gaseous layer above a body:
//! a pure-Rust, Bevy-free definition of what a body's atmosphere looks
//! like and how its layers are configured, parsed straight from the RON
//! body file.
//!
//! Two sibling schemas live here:
//!
//! - [`AtmosphereParams`] — gas / ice giants. The cloud deck IS the
//!   visible disk; there is no solid surface. Rich schema: cloud palette,
//!   zonal banding, haze, rim halo, Rayleigh blue gaps, limb darkening.
//! - [`TerrestrialAtmosphere`] — terrestrial bodies with a thin gas
//!   shell over a baked solid surface. Much sparser schema: rim halo +
//!   limb shading + optional limb darkening. Built to composite over
//!   the impostor rather than replace it.
//!
//! The two schemas are sibling-exclusive at the body level: a body
//! carries either `atmosphere: Some(AtmosphereParams)` or
//! `terrestrial_atmosphere: Some(TerrestrialAtmosphere)`, never both.
//! (Sibling fields, not an enum, because the gas-giant schema is large
//! and already stable; an enum migration is a follow-up.)
//!
//! ## Layer model
//!
//! A gas giant's visible disk is composited from several optically
//! distinct layers. First-pass rendering supports the first three; the
//! remaining layers are wired as explicit stubs so fidelity can climb
//! without schema churn later.
//!
//! 1. **Cloud deck** — the optically thick layer. Defines the visible
//!    colour at each latitude/longitude and is what you actually see when
//!    looking at the disk. Parameterised by:
//!    - a latitude palette (`PaletteStop[]`) giving base colour vs.
//!      signed latitude,
//!    - zonal band frequency and warp amplitude (wind shear), and
//!    - per-body noise seed for reproducibility.
//!
//! 2. **Haze layer** — mid-altitude particulate layer above the cloud
//!    deck. Contributes a subtle chromatic shift, softens band edges, and
//!    modulates the terminator. First pass: uniform tint + thickness.
//!    Extended later to support altitude-varying opacity curves.
//!
//! 3. **Rim halo** — upper-atmosphere forward-scattered light visible
//!    just outside the cloud-deck disk. Approximates a Rayleigh-like
//!    limb glow via an exponential density falloff with altitude.
//!    Parameterised by colour and scale height.
//!
//! 4. **Storm features** *(future)* — discrete long-lived vortices like
//!    Jupiter's Great Red Spot. Will reuse the SSBO pattern from
//!    `thalos_terrain_gen::Crater` so GPU detail-layer code stays
//!    structurally consistent across body types.
//!
//! 5. **Aurora** *(future)* — polar emission ring. Separate layer so its
//!    additive blending and magnetic-field alignment can be tuned
//!    independently of cloud-deck shading.
//!
//! ## Integration
//!
//! `thalos_physics::BodyDefinition` carries an
//! `Option<AtmosphereParams>`. A body that has `atmosphere: Some(...)`
//! and no `generator` block is rendered as a gas/ice giant; the game's
//! rendering layer hands `AtmosphereParams` to `planet_rendering` which
//! builds GPU uniforms from it.

use serde::Deserialize;

/// Top-level atmosphere definition for a gas or ice giant.
///
/// Every field except the cloud deck has a sensible default so body files
/// can start minimal and get richer over time.
#[derive(Debug, Clone, Deserialize)]
pub struct AtmosphereParams {
    /// Per-body seed. Drives all procedural variation (band phases,
    /// turbulence field, storm placement). Changing the seed completely
    /// redraws the body without altering its palette.
    pub seed: u64,

    /// Cloud deck — mandatory. Defines the visible surface of the giant.
    pub cloud_deck: CloudDeck,

    /// Mid-altitude haze. Optional — omitted means no haze modulation.
    #[serde(default)]
    pub haze: Option<HazeLayer>,

    /// Upper-atmosphere rim halo. Optional — omitted means no limb glow.
    #[serde(default)]
    pub rim_halo: Option<RimHalo>,

    /// Optional limb-shading tweaks (terminator warmth, Fresnel rim).
    #[serde(default)]
    pub limb: Option<LimbShading>,

    /// Optional Rayleigh-scattering "blue gap" layer. Where the authored
    /// haze field thins, a scattered-blue contribution leaks through the
    /// cloud deck — this is what makes Cassini-era Saturn images show
    /// narrow blue clearings at mid-northern latitudes. None disables it.
    #[serde(default)]
    pub rayleigh: Option<RayleighLayer>,

    /// Optional per-channel limb darkening. Gas giants show strong
    /// wavelength-dependent darkening (Chandrasekhar / Minnaert laws):
    /// short wavelengths darken faster than long. Per-channel exponents
    /// round the disk and add chromatic limb colour for free.
    #[serde(default)]
    pub limb_darkening: Option<LimbDarkening>,
    // Storms and aurora come later. They are not listed here yet so the
    // schema is conservative: adding them will be an additive change,
    // with `#[serde(default)]` preserving backward compatibility.
}

/// Cloud deck — the optically thick "surface" of a gas giant.
#[derive(Debug, Clone, Deserialize)]
pub struct CloudDeck {
    pub palette: Vec<PaletteStop>,

    pub band_frequency: f32,

    #[serde(default = "default_band_warp")]
    pub band_warp: f32,

    #[serde(default = "default_turbulence")]
    pub turbulence: f32,

    /// Belt/zone luminance contrast. Authored as a multiplicative swing
    /// around the base palette colour: 0.22 = ±22% (muted), 0.55 = ±55%
    /// (Saturn-like). Pairs with `band_sharpness` to shape how crisp the
    /// transitions look.
    #[serde(default = "default_band_contrast")]
    pub band_contrast: f32,

    /// Width of the colour blend zone between adjacent bands, in units
    /// of a palette span. 1.0 = full smoothstep blend across the whole
    /// span (soft, looks blurred at body scale). Smaller values squeeze
    /// the blend into a narrower fraction of the span, producing crisp
    /// Saturn-style band edges. 0.15–0.25 is a good range.
    #[serde(default = "default_band_sharpness")]
    pub band_sharpness: f32,

    #[serde(default = "default_white")]
    pub tint: [f32; 3],

    /// Signed per-latitude scroll rates, sampled evenly from lat=-1 to
    /// lat=+1. Each entry is a retrograde(-)/prograde(+) offset applied
    /// on top of the body's bulk rotation. Empty = no differential
    /// rotation. Up to 16 entries are consumed (shader `PROFILE_N`).
    #[serde(default)]
    pub speed_profile: Vec<f32>,

    /// Scalar gain on `speed_profile`. Tune this to exaggerate belt
    /// retrograde motion without rewriting every profile entry.
    #[serde(default = "default_diff_rot")]
    pub differential_rotation_rate: f32,

    /// Per-latitude turbulence amplitude in [0, 1]. Drives warp, curl,
    /// and edge-wave amplitudes so poles can go Juno-chaotic while the
    /// equator stays laminar. Empty = uniform `turbulence` everywhere.
    #[serde(default)]
    pub turbulence_profile: Vec<f32>,

    /// Overall amplitude of the Kelvin–Helmholtz edge wave painted at
    /// band boundaries. 0 disables it.
    #[serde(default)]
    pub edge_wave_amp: f32,

    /// Curl noise amplitude. Warps local UV to fake fluid eddies. The
    /// sign is flipped across band edges so neighbouring eddies
    /// counter-rotate. 0 disables it.
    #[serde(default)]
    pub curl_amp: f32,

    /// Two-layer parallax offset: sample the cloud deck twice with a
    /// tiny view-space offset and blend. Tiny values (0.005–0.02) hint
    /// at cloud depth. 0 disables it.
    #[serde(default)]
    pub parallax_amp: f32,

    /// Named long-lived vortices (Great Red Spot and friends).
    #[serde(default)]
    pub named_vortices: Vec<NamedVortex>,

    /// Hashed edge vortex chain parameters. `None` disables it.
    #[serde(default)]
    pub edge_vortex_chain: Option<EdgeVortexChain>,
}

/// One analytic long-lived vortex in body-local coordinates.
#[derive(Debug, Clone, Deserialize)]
pub struct NamedVortex {
    /// Signed latitude in [-1, 1].
    pub lat: f32,
    /// Longitude in radians, in body-local frame (fixed w.r.t. rotation).
    pub lon: f32,
    /// Angular radius of the vortex (radians on the sphere).
    pub radius: f32,
    /// Peak swirl rotation at the centre, radians.
    pub strength: f32,
    /// Multiplicative tint blended into the band colour inside the
    /// vortex. Use `[1, 1, 1]` for a pure swirl with no recolour.
    #[serde(default = "default_white")]
    pub tint: [f32; 3],
}

/// Hashed edge vortex chain authoring.
#[derive(Debug, Clone, Deserialize)]
pub struct EdgeVortexChain {
    /// Angular radius of an individual chain vortex (radians).
    pub base_radius: f32,
    /// Peak swirl strength (radians).
    pub strength: f32,
    /// Lifetime of one vortex in seconds.
    pub lifetime_s: f32,
    /// Slots per band (number of potential spawn sites). Typical: 8–16.
    pub slots_per_band: u32,
}

/// One stop in the cloud-deck latitude palette.
#[derive(Debug, Clone, Deserialize)]
pub struct PaletteStop {
    /// Signed latitude in [-1, 1]. -1 = south pole, 0 = equator,
    /// +1 = north pole. Stops do not need to be evenly spaced.
    pub lat: f32,
    /// Linear-space RGB at this latitude.
    pub color: [f32; 3],
}

/// Optional mid-altitude haze layer.
///
/// First-pass implementation: a uniform multiplicative tint applied
/// across the disk, modulated by view angle so the terminator picks up a
/// chromatic shift. Future fidelity extends this to an altitude-varying
/// opacity curve and wavelength-dependent scattering.
#[derive(Debug, Clone, Deserialize)]
pub struct HazeLayer {
    /// Linear-space RGB tint. Multiplied into the cloud-deck colour.
    pub tint: [f32; 3],
    /// Overall opacity of the haze, 0 = invisible, 1 = fully replaces
    /// the cloud-deck colour with `tint`.
    pub thickness: f32,
    /// View-angle bias: 0 = uniform tint, 1 = tint only contributes near
    /// the terminator. Used to mimic Rayleigh-style oblique pathlength.
    #[serde(default = "default_half")]
    pub terminator_bias: f32,
}

/// Optional upper-atmosphere rim halo.
///
/// Models the bright glow visible just outside the cloud-deck disk when
/// sunlight scatters through the upper atmosphere at grazing angles.
#[derive(Debug, Clone, Deserialize)]
pub struct RimHalo {
    /// Linear-space RGB of the halo at peak intensity.
    pub color: [f32; 3],
    /// Peak intensity multiplier (applied to incoming light).
    pub intensity: f32,
    /// Atmospheric scale height, in meters. Controls how quickly the
    /// halo fades with altitude above the cloud deck. Gas giants have
    /// scale heights on the order of 20–60 km depending on temperature.
    pub scale_height_m: f32,
    /// Outer cutoff altitude in meters. The halo is effectively zero
    /// beyond this altitude; chosen so the rendered shell stays small
    /// relative to the cloud-deck radius.
    #[serde(default = "default_outer_cutoff")]
    pub outer_altitude_m: f32,
}

/// Terminator warmth + Fresnel rim for the cloud deck lighting stage.
#[derive(Debug, Clone, Deserialize)]
pub struct LimbShading {
    /// RGB tint added near the terminator (`NdotL ≈ 0`, lit side).
    #[serde(default)]
    pub terminator_warmth: [f32; 3],
    /// Strength of the terminator warmth contribution.
    #[serde(default)]
    pub terminator_strength: f32,
    /// RGB tint of the Fresnel rim on the lit limb (cold Rayleigh stand-in).
    #[serde(default)]
    pub fresnel_color: [f32; 3],
    /// Strength of the Fresnel rim contribution.
    #[serde(default)]
    pub fresnel_strength: f32,
}

/// Rayleigh "blue gap" layer.
///
/// Real Saturn shows narrow bluish clearings where the upper
/// photochemical haze thins enough for molecular scattering to dominate
/// the reflectance (Cassini's northern-hemisphere imaging, 2004–2006).
/// This layer is not a separate physical medium: it's a modulation on
/// the cloud-deck colour driven by an independent haze-density field.
#[derive(Debug, Clone, Deserialize)]
pub struct RayleighLayer {
    /// Linear-space RGB of the scattered-blue contribution at its
    /// brightest (typically a pale cyan).
    pub color: [f32; 3],
    /// Overall intensity of the scattered contribution.
    pub strength: f32,
    /// Independent noise scale of the haze-density field. Larger values
    /// make the gaps finer; smaller values make them continent-sized.
    #[serde(default = "default_rayleigh_scale")]
    pub haze_scale: f32,
    /// Density threshold below which the Rayleigh contribution turns on.
    /// Authored in [0, 1]; 0.5 means roughly half the disk has visible
    /// gaps, 0.2 keeps the gaps narrow and rare.
    #[serde(default = "default_rayleigh_threshold")]
    pub clearing_threshold: f32,
    /// Latitude bias: where on the disk the gaps concentrate. 0 =
    /// uniform, positive = north hemisphere favouring, negative = south.
    #[serde(default)]
    pub latitude_bias: f32,
}

/// Per-wavelength Rayleigh scattering for a terrestrial atmosphere.
///
/// Drives both the ground-illumination reddening at low sun (sunsets)
/// and the view-path in-scatter glow (the orange band seen from orbit
/// at the terminator, the blue haze at the sub-solar point). Unlike
/// [`RayleighLayer`] above — which is a gas-giant "blue gap" modulation
/// of the cloud-deck colour — this models the actual physical process:
/// short wavelengths accumulate more optical depth per unit path, so
/// long paths (low sun, grazing views) preferentially transmit red.
///
/// Parameters are per-channel vertical optical depth `τ_v`. Canonical
/// Earth values at sea level: `(0.046, 0.108, 0.264)` for R/G/B. Other
/// bodies scale with atmospheric column depth (thinner atmospheres
/// → smaller τ_v uniformly; dustier atmospheres → elevated red).
#[derive(Debug, Clone, Deserialize)]
pub struct RayleighAtmosphere {
    /// Vertical optical depth per RGB channel at zenith (unit airmass).
    /// Drives both transmission of sunlight to the ground and in-scatter
    /// intensity toward the viewer. Earth-like: `(0.046, 0.108, 0.264)`.
    pub vertical_optical_depth: [f32; 3],
    /// Overall multiplier on both ground transmission and view-path
    /// in-scatter. 0 disables Rayleigh entirely; 1 = physical scale.
    /// Values above 1 exaggerate the effect for artistic purposes
    /// (more saturated sunsets, deeper daylight haze) at the cost of
    /// physical accuracy.
    #[serde(default = "default_one")]
    pub strength: f32,
}

/// Per-channel Minnaert-style limb darkening.
///
/// Applied as a luminance-only multiplier `pow(n_dot_v, k_channel)`
/// BEFORE terminator warmth / Fresnel rim so the rim terms can still
/// tint against the darkened base. Red channels darken slower than blue
/// — matches Cassini limb-darkening curves.
#[derive(Debug, Clone, Deserialize)]
pub struct LimbDarkening {
    /// Exponent for the red channel. Typical 0.20–0.35.
    pub red: f32,
    /// Exponent for the green channel. Typical 0.25–0.40.
    pub green: f32,
    /// Exponent for the blue channel. Typical 0.30–0.45.
    pub blue: f32,
    /// Overall strength: 0 = no darkening, 1 = full darkening. Provides
    /// a single knob to dial the effect without rewriting the channels.
    #[serde(default = "default_one")]
    pub strength: f32,
}

/// Saturn-style ring system.
///
/// Rings are a body-level property (sibling of `atmosphere`,
/// `generator`, `terrestrial_atmosphere`) — any body can have them,
/// not just gas giants. Rendered as a flat annulus aligned with the
/// body's equatorial plane (axial tilt inherited from the physical
/// block). The radial profile is a mix of authored palette stops and
/// a procedural density field. Planet-shadow on rings is implemented
/// for every body; ring-shadow on the body itself is currently only
/// wired into the gas-giant cloud-deck shader (see
/// `gas_giant.wgsl` — terrain-impostor counterpart is a TODO).
#[derive(Debug, Clone, Deserialize)]
pub struct RingSystem {
    /// Inner edge radius, in meters from body center.
    pub inner_radius_m: f32,
    /// Outer edge radius, in meters from body center.
    pub outer_radius_m: f32,
    /// Per-ring authoring seed.
    pub seed: u64,
    /// Radial palette stops: signed linear position in [0, 1] where 0 =
    /// inner edge and 1 = outer edge. Up to 16 stops. Stops carry both
    /// colour and opacity so authors can sculpt the Cassini division,
    /// Encke gap, translucent C ring, etc.
    pub palette: Vec<RingStop>,
    /// Overall opacity scalar multiplied into every stop. Tune this to
    /// globally darken/lighten the ring without rewriting stops.
    #[serde(default = "default_one")]
    pub opacity: f32,
    /// Radial noise amplitude that breaks the pure palette interpolation
    /// into thousands of fine ringlets. 0 = smooth palette, 1 = very
    /// noisy. Saturn's ringlets justify ~0.4.
    #[serde(default = "default_ringlet_noise")]
    pub ringlet_noise: f32,
    /// Number of radial noise octaves. 5–7 gives visible detail from
    /// orbit to close-up without a per-pixel texture.
    #[serde(default = "default_ringlet_octaves")]
    pub ringlet_octaves: u32,
}

/// One stop on the radial ring palette.
#[derive(Debug, Clone, Deserialize)]
pub struct RingStop {
    /// Normalised radial position in [0, 1] — 0 = inner edge, 1 = outer.
    pub r: f32,
    /// Linear-space RGB reflectance at this radius.
    pub color: [f32; 3],
    /// Opacity in [0, 1]. 0 produces a gap (e.g. Cassini division).
    pub opacity: f32,
}

fn default_diff_rot() -> f32 {
    1.0
}

fn default_band_warp() -> f32 {
    0.3
}
fn default_turbulence() -> f32 {
    0.05
}
fn default_white() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
fn default_half() -> f32 {
    0.5
}
fn default_outer_cutoff() -> f32 {
    300_000.0
}
fn default_band_contrast() -> f32 {
    0.22
}
fn default_band_sharpness() -> f32 {
    0.20
}
fn default_one() -> f32 {
    1.0
}
fn default_rayleigh_scale() -> f32 {
    4.0
}
fn default_rayleigh_threshold() -> f32 {
    0.35
}
fn default_ringlet_noise() -> f32 {
    0.4
}
fn default_ringlet_octaves() -> u32 {
    6
}

// ---------------------------------------------------------------------------
// Terrestrial atmospheres
// ---------------------------------------------------------------------------

/// Thin atmosphere layered over a terrestrial (solid-surface) body.
///
/// Where `AtmosphereParams` describes the entirety of a gas giant's
/// visible disk, `TerrestrialAtmosphere` describes only what modifies
/// light passing through a thin gas shell above a baked planet impostor.
/// First-pass schema: the gas-giant `RimHalo` + `LimbShading` primitives
/// are reused directly — they already model the two dominant cues that
/// read as "planet with atmosphere" from orbit (blue halo outside the
/// silhouette, cool limb on the lit side, warm sunset band near the
/// terminator). A proper Rayleigh/Mie scattering model with per-fragment
/// in-scatter integration is a later pass.
///
/// Every field is optional. A body with `TerrestrialAtmosphere::default`
/// renders identically to a body with no atmosphere — the shader gates
/// each layer on its own intensity scalar.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TerrestrialAtmosphere {
    /// Rim glow visible just outside the body's silhouette. Uses the
    /// same exponential-density column integration the gas-giant
    /// shader uses. None disables the rim entirely.
    #[serde(default)]
    pub rim_halo: Option<RimHalo>,

    /// Terminator warmth (sunset band) plus Fresnel-style limb tint on
    /// the lit hemisphere. For a terrestrial impostor, the Fresnel tint
    /// is the dominant "blue atmosphere on the lit disk" cue. None
    /// disables limb shading.
    #[serde(default)]
    pub limb: Option<LimbShading>,

    /// Per-channel Minnaert limb darkening. Terrestrials typically show
    /// weaker chromatic limb darkening than gas giants (thinner
    /// scattering path), so this is usually left off for a first pass.
    /// None disables the effect.
    #[serde(default)]
    pub limb_darkening: Option<LimbDarkening>,

    /// Per-wavelength Rayleigh scattering: produces sunset-coloured
    /// terrain at the terminator, blue haze across the lit disk, and
    /// the red-orange in-scatter band seen from orbit. None disables
    /// the effect entirely and the impostor renders with unattenuated
    /// white sunlight.
    #[serde(default)]
    pub rayleigh: Option<RayleighAtmosphere>,

    /// Cloud cover layer. A shader-synthesized cloud field composited
    /// over the lit surface — bright white in sunlight, alpha-blended
    /// by density, drifting with differential rotation by latitude so
    /// weather systems evolve visibly as the impostor rotates. None
    /// disables the layer.
    ///
    /// The physical model is deliberately the cheapest thing that
    /// reads as "weather" at impostor distance: fractal noise on the
    /// sphere, no explicit storm list, no shadow-casting on the
    /// surface below. A real physical cloud volume (shadow casting,
    /// proper altitude parallax, named storms) belongs in the
    /// real-size volumetric renderer that comes after UDLOD lands;
    /// this layer is the stand-in so Thalos reads as a living planet
    /// from orbit today.
    #[serde(default)]
    pub clouds: Option<CloudCover>,
}

/// Shader-synthesized cloud layer for a terrestrial impostor.
///
/// Clouds are modelled as a fractal-noise density field over the unit
/// sphere, drifted by differential rotation (faster at the equator,
/// slower at the poles) and composited on top of the lit surface. The
/// parameters below are authored per body so different worlds can have
/// visibly distinct weather characters — Thalos's scattered mid-latitude
/// systems vs. an Exo-Venus's global overcast vs. a thin-atmosphere
/// Mars-analog with occasional dust haze.
#[derive(Debug, Clone, Deserialize)]
pub struct CloudCover {
    /// Total disk coverage fraction in [0, 1]. 0 = clear skies, 1 =
    /// fully overcast. Earth sits around 0.55–0.65; Thalos with a
    /// thinner atmosphere nominally a bit lower.
    pub coverage: f32,

    /// Soft-edge width of the cloud/no-cloud boundary in density units.
    /// Smaller = crisp cumulus edges, larger = hazy boundaries. Typical
    /// range 0.04–0.15.
    #[serde(default = "default_cloud_softness")]
    pub softness: f32,

    /// Base spatial frequency of the cloud fBm field, in cycles over
    /// the unit sphere. 3–5 → continent-sized cloud clusters; 8+ →
    /// fine-grained clutter suitable for tropical convection. Earth-
    /// analog starting point: 4.
    #[serde(default = "default_cloud_frequency")]
    pub frequency: f32,

    /// Number of fBm octaves. 3 is the minimum for recognisable shape;
    /// 5–6 gives storm-like substructure at the cost of per-fragment
    /// work.
    #[serde(default = "default_cloud_octaves")]
    pub octaves: u32,

    /// Linear-space RGB albedo of sunlit clouds. Typically very near
    /// `(1, 1, 1)` — water-vapour clouds are close to spectrally
    /// neutral. Tint here can model dust storms (warm ochre) or
    /// sulphate hazes (pale yellow).
    #[serde(default = "default_cloud_albedo")]
    pub albedo: [f32; 3],

    /// Scroll rate, in radians per second of sim time, at the equator.
    /// Positive = prograde drift. Typical Earth-analog weather moves
    /// at ~15 m/s zonal mean, which at Thalos's 3186 km radius is
    /// ~4.7e-6 rad/s relative to the surface.
    #[serde(default = "default_cloud_scroll_rate")]
    pub scroll_rate: f32,

    /// Differential rotation coefficient in [0, 1]. 0 = solid-body
    /// drift (all latitudes at `scroll_rate`), 1 = strongly latitude-
    /// banded (equator at `scroll_rate`, poles stationary). Typical
    /// terrestrial: 0.3–0.5.
    #[serde(default = "default_cloud_differential")]
    pub differential_rotation: f32,

    /// Per-body noise seed. Changes the cloud pattern without touching
    /// any other parameter.
    #[serde(default)]
    pub seed: u64,
}

fn default_cloud_softness() -> f32 {
    0.08
}
fn default_cloud_frequency() -> f32 {
    4.0
}
fn default_cloud_octaves() -> u32 {
    5
}
fn default_cloud_albedo() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}
fn default_cloud_scroll_rate() -> f32 {
    4.7e-6
}
fn default_cloud_differential() -> f32 {
    0.35
}
