//! Atmosphere parameter schemas.
//!
//! This crate holds the *data* definition of a body's atmosphere — not
//! the renderer, not the shader, not the GPU uniforms. It is the
//! analogue of `thalos_terrain` for the gaseous layer above a body:
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
//!    `thalos_terrain::Crater` so GPU detail-layer code stays
//!    structurally consistent across body types.
//!
//! 5. **Aurora** *(future)* — polar emission ring. Separate layer so its
//!    additive blending and magnetic-field alignment can be tuned
//!    independently of cloud-deck shading.
//!
//! ## Integration
//!
//! `thalos_physics_canonical::BodyDefinition` carries an
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

/// Single-scattering Rayleigh + Mie atmosphere for a terrestrial body.
///
/// This is the physical scattering model the impostor renders against:
/// per-fragment view raymarch through an exponential-density shell,
/// with per-channel Rayleigh (molecular scattering, blue-dominant) and
/// scalar Mie (aerosol scattering, near-spectrally-white, forward-
/// peaked phase function). The same integral produces the body's
/// surface aerial perspective, the daylight haze, the terminator
/// orange band, and the rim halo outside the silhouette — one path,
/// no parallel rim/Fresnel/warmth helpers.
///
/// Authored values use canonical photometric units:
///
/// - **Rayleigh `vertical_optical_depth`** is the per-channel vertical
///   τ_v at zenith. Earth at sea level: `(0.046, 0.108, 0.264)` for
///   R/G/B (β_R · H_R, with β_R from Bucholtz 1995). Thin atmospheres
///   scale uniformly down; dust-loaded atmospheres invert the slope
///   (red dominant, see Vaelen).
///
/// - **`mie_optical_depth`** is the scalar vertical Mie τ_v. Earth
///   clean conditions: ~0.02; hazy conditions: ~0.10; Mars dust
///   storm: 1.0+.
///
/// - **Scale heights** set the exponential density falloff and define
///   "where most of the column is" — Earth Rayleigh ~8 km, Mie ~1.2 km.
///   `atmosphere_top_m` truncates the integration; samples beyond
///   ~5 × scale_height are wasted, so keep this lean. The default
///   matches `5 × max(rayleigh_scale_height_m, mie_scale_height_m)`,
///   which clips at 1% of sea-level density.
///
/// - **`mie_asymmetry`** is the Henyey-Greenstein `g` parameter in
///   [-1, 1]. Earth aerosols: 0.76 (forward-peaked); Mars dust: ~0.5.
///   Drives the brightening of haze toward the sun.
///
/// - **`strength`** is an artistic overall multiplier (1.0 = physical).
///
/// `AtmosphericScattering::default()` produces a vacuum (no
/// scattering, no rim halo, no in-scatter) — bodies without an
/// atmosphere either omit `terrestrial_atmosphere` or set this layer
/// to None.
#[derive(Debug, Clone, Deserialize)]
pub struct AtmosphericScattering {
    /// Per-channel Rayleigh vertical optical depth at zenith. Earth at
    /// sea level: `(0.046, 0.108, 0.264)`.
    pub vertical_optical_depth: [f32; 3],

    /// Rayleigh scale height in meters. Density falls off as
    /// `exp(-h / H)`. Earth: 8000.
    #[serde(default = "default_rayleigh_scale_height")]
    pub rayleigh_scale_height_m: f32,

    /// Scalar Mie vertical optical depth at zenith. Mie is spectrally
    /// near-white in the visible, so a scalar instead of a per-channel
    /// vector. Earth clean: 0.02; hazy: 0.10; dust-loaded: 0.30+.
    #[serde(default = "default_mie_optical_depth")]
    pub mie_optical_depth: f32,

    /// Mie scale height in meters. Earth aerosols: 1200.
    #[serde(default = "default_mie_scale_height")]
    pub mie_scale_height_m: f32,

    /// Henyey-Greenstein asymmetry parameter `g` in [-1, 1]. Positive
    /// = forward-peaked (Earth aerosols ~0.76); 0 = isotropic; negative
    /// = back-peaked. Drives the "haze brightens near the sun" cue.
    #[serde(default = "default_mie_asymmetry")]
    pub mie_asymmetry: f32,

    /// Overall artistic multiplier on both in-scatter and surface
    /// transmittance. 0 disables the scattering model entirely (the
    /// impostor renders with unattenuated white sunlight); 1 = physical;
    /// > 1 exaggerates haze and sunsets at the cost of accuracy.
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
fn default_rayleigh_scale_height() -> f32 {
    8000.0
}
fn default_mie_optical_depth() -> f32 {
    0.021
}
fn default_mie_scale_height() -> f32 {
    1200.0
}
fn default_mie_asymmetry() -> f32 {
    0.76
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
/// light passing through a thin gas shell above a baked planet
/// impostor. The dominant layer is [`AtmosphericScattering`]: a
/// physically-based single-scattering Rayleigh + Mie raymarch that
/// produces the rim halo, the lit-disk haze, the terminator orange
/// band, the surface aerial perspective, and the silhouette glow from
/// one integral. The previous stand-in fields (a manual exponential
/// rim halo, a terminator-warmth tint, a Fresnel limb) were removed
/// when the raymarch landed — those signals now fall out of the
/// physics and don't need parallel parameters.
///
/// `karman_line_m` is the canonical atmosphere top: the altitude above
/// the body surface where atmospheric effects (rendering and gameplay)
/// cut off. It serves as the scattering raymarch integration cutoff,
/// the visibility shell for any in-atmosphere render passes, and the
/// gameplay boundary for drag / heating / "in atmosphere" state. A
/// `TerrestrialAtmosphere` with `karman_line_m == 0` is equivalent to
/// a vacuum body regardless of which sub-layers are populated.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TerrestrialAtmosphere {
    /// Kármán-line altitude in meters above the body's mean surface.
    /// Single source of truth for "where atmosphere ends" — read by
    /// rendering for shell intersection and integration, by gameplay
    /// for drag/heating gates, and by the LOD swap for sky-pass
    /// visibility. Authoring guidance: pick a value comfortably above
    /// `5 × max(rayleigh, mie)` scale height so the raymarch sees the
    /// full contribution; below that the integral truncates visibly.
    pub karman_line_m: f32,

    /// Per-channel Minnaert limb darkening on the lit surface. Pure
    /// artistic knob — most terrestrial bodies leave this None and
    /// rely on the scattering raymarch for limb shading.
    #[serde(default)]
    pub limb_darkening: Option<LimbDarkening>,

    /// Single-scattering Rayleigh + Mie atmospheric model. Drives the
    /// rim halo, in-scattered haze, sunset reddening, terminator band,
    /// and aerial perspective. None disables the model entirely and
    /// the impostor renders with unattenuated white sunlight + no
    /// rim glow (vacuum behaviour, identical to airless bodies).
    #[serde(default)]
    pub scattering: Option<AtmosphericScattering>,

    /// Cloud cover layer composited above the surface from reference
    /// cloud-cover overlays, with shader-side differential rotation.
    /// Independent of the scattering model — clouds reflect direct
    /// sunlight at cloud altitude and are darkened by the surface aerial
    /// perspective at composite time. None disables the layer.
    #[serde(default)]
    pub clouds: Option<CloudCover>,
}

/// Reference cloud overlay for a terrestrial impostor.
///
/// Clouds are read from a reference density cubemap, drifted by
/// differential rotation (faster at the equator, slower at the poles),
/// and composited on top of the lit surface. The procedural cloud
/// generator has been removed for now; bodies without a registered
/// reference overlay bind a blank cubemap.
#[derive(Debug, Clone, Deserialize)]
pub struct CloudCover {
    /// Total disk coverage fraction in [0, 1]. 0 = clear skies, 1 =
    /// fully overcast. Earth sits around 0.55–0.65; Thalos with a
    /// thinner atmosphere nominally a bit lower.
    pub coverage: f32,

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

    /// Base altitude of the cloud layer above the surface, in meters.
    /// The volumetric cloud raymarch in the terrain `BodySky` pass treats
    /// the layer as a slab spanning `[base_altitude_m, base_altitude_m +
    /// thickness_m]`. Earth's fair-weather cumulus base sits ~1–2 km. The
    /// orbital impostor keeps its own fixed reference shell and ignores
    /// this field.
    #[serde(default = "default_cloud_base_altitude")]
    pub base_altitude_m: f32,

    /// Vertical thickness of the cloud slab, in meters — the depth the
    /// volumetric raymarch integrates over. Thicker reads as puffier, more
    /// occluding cloud; ~4–6 km is a good fair-weather cumulus deck.
    #[serde(default = "default_cloud_thickness")]
    pub thickness_m: f32,

    /// Optical-density multiplier for the volumetric layer. Scales the
    /// per-meter extinction the raymarch accumulates; 1.0 is the tuned
    /// default, higher values give denser, more opaque cores.
    #[serde(default = "default_cloud_density")]
    pub density: f32,
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
fn default_cloud_base_altitude() -> f32 {
    1500.0
}
fn default_cloud_thickness() -> f32 {
    5000.0
}
fn default_cloud_density() -> f32 {
    1.0
}
