# Celestial sphere — design

Procedural sky model for Thalos. Serves both the realtime skybox and a
future astronomical imaging / research gameplay layer (player-launched
telescopes that point, integrate, and produce images). Must remain
open-ended enough that new source types and wavelength bands can be
added without reworking existing code.

## Goals

- HDR skybox generated procedurally at startup — stars, galaxies,
  nebulae. No hand-authored texture assets.
- Extensible to additional source types (variable stars, radio sources,
  X-ray sources, CMB maps) without touching the rendering layer.
- Extensible to additional wavelength bands (visible first; IR, UV,
  radio, microwave eventually).
- Telescope imaging pipeline reuses the same source data, integrating
  to a sensor frame with configurable FOV, PSF, and exposure. Deep
  images reveal faint sources that the display skybox clips.
- Deterministic: same seed → same universe.
- Pure Rust library. No Bevy dependency. Mirrors the
  `physics` / `terrain_gen` separation.

## Non-goals (initial scope)

- Physically accurate cosmological large-scale structure.
- Real star catalog integration (procedural for now; Hipparcos/Gaia
  import can drop in later behind the same trait).
- Dynamic sky (proper motion, variability) — static snapshot first,
  time evolution is a later extension.
- Gravitational lensing, relativistic effects.

## Crate layout

New workspace crate: `crates/celestial/` → `thalos_celestial`.

```
celestial/
  src/
    lib.rs
    coords.rs          // RA/Dec, unit vectors, rotations
    spectrum.rs        // SED representation, band filters
    sources/
      mod.rs           // Source trait
      star.rs
      galaxy.rs
      nebula.rs
    catalog.rs         // Universe = collection of source layers
    generate/
      mod.rs
      stars.rs         // IMF + galactic disk density
      galaxies.rs      // clustered distribution via GRF
      nebulae.rs       // 3D noise volumes along galactic plane
    render/
      mod.rs
      cubemap.rs       // bake Universe → HDR cubemap
      telescope.rs     // integrate Universe → sensor frame
      psf.rs           // point spread function kernels
  tests/
```

Game consumes `thalos_celestial` from a new `skybox` module in
`thalos_game`, plus (later) a `telescope` feature that reuses the same
`Universe`.

## Core data model

The central principle: **store physical quantities, not pixels**.
Sources carry flux, temperature, SEDs — never pre-baked RGB. This keeps
CMB, radio, and other bands open without a rewrite.

### Spectrum

```rust
/// Spectral energy distribution. Stored as either a parametric form
/// (blackbody, power law) or a tabulated curve. The renderer filters
/// through a passband to produce per-channel flux.
enum Spectrum {
    Blackbody { temperature_k: f32 },
    PowerLaw { alpha: f32, reference_flux: f32, reference_wavelength_nm: f32 },
    Tabulated { wavelengths_nm: Vec<f32>, flux: Vec<f32> },
}

impl Spectrum {
    fn integrate(&self, band: &Passband) -> f32 { ... }
}

struct Passband {
    wavelengths_nm: Vec<f32>,
    response: Vec<f32>,
}
```

Visible skybox uses three passbands approximating CIE RGB. Telescope
sensors pick arbitrary bands. CMB layer would use a microwave passband.

### Source trait

```rust
trait Source {
    fn position(&self) -> UnitVector3; // celestial unit vector
    fn flux_in_band(&self, band: &Passband) -> f32; // total flux
    fn angular_profile(&self) -> AngularProfile;    // point, sérsic, volumetric
}
```

Concrete sources:

- `Star { pos, spectrum: Spectrum::Blackbody, apparent_magnitude, proper_motion }`
- `Galaxy { pos, spectrum, sersic_n, effective_radius_arcsec, axis_ratio, pa, redshift }`
- `NebulaField { bounds, volume_noise: NoiseParams, emission: Spectrum }`

Future:

- `VariableStar { .. light curve .. }`
- `RadioSource { .. }`
- `CmbMap { healpix_data }` — implements `Source` by contributing a
  constant-ish background in its band.

### Universe

```rust
struct Universe {
    stars: Vec<Star>,
    galaxies: Vec<Galaxy>,
    nebulae: Vec<NebulaField>,
    // extensible: more layers later
    seed: u64,
}
```

Not a single `Vec<Box<dyn Source>>` — typed layers let each layer pick
its own spatial index (stars: HEALPix bucket, galaxies: BVH, nebulae:
bounding volumes) and its own fast path during rendering.

## Generation

Each generator is a pure function `(&mut Universe, seed)`. Analogous to
`terrain_gen` stages. Order-independent where possible.

### Stars

- Sample count from stellar density model: thin disk + thick disk +
  halo, projected to a density-per-steradian function of galactic
  latitude.
- Sample masses from an IMF (Salpeter or Kroupa).
- Assign spectral type from mass → main sequence relation → effective
  temperature → `Spectrum::Blackbody`.
- Assign distance from disk density along line of sight.
- Apparent magnitude from luminosity, distance, extinction.
- Add notable bright stars by hand-seeded list later if desired.

Target count for visible skybox: ~10^5 down to mag 8, extensible to
~10^7 for deep telescope images. Stars below display cutoff still live
in the catalog — the telescope integrator uses them.

### Galaxies

- Distribution clustered via Gaussian random field on the sphere
  (generate one 3D GRF, sample along radial shells). Produces cosmic
  web without simulating it.
- Local group and Milky Way satellites hand-placed.
- Each galaxy gets Sérsic profile parameters (n=1 disk, n=4 elliptical,
  mix for spirals), effective radius, axis ratio, position angle.
- Spectrum by type (elliptical = old population, spiral = mixed,
  irregular = young + emission lines).
- Redshift from distance → SED redshifted before band integration.

### Nebulae

- Milky Way band modeled as a 3D noise volume along the galactic plane
  with dust extinction modulating starlight behind it.
- Discrete emission nebulae placed along spiral arm density peaks,
  each a bounded 3D noise field with emission-line spectrum
  (Hα dominant).
- Reflection and dark nebulae as later extensions.

## Rendering

Two consumers of the same `Universe`.

### Skybox baker

One-shot at startup (or on seed change). Output: HDR cubemap, 6 faces
of configurable resolution (default 2048²), format `Rgba16Float`.

Pipeline:

1. For each face, for each pixel, compute the ray direction.
2. Rasterize stars: for each star, project to the appropriate face,
   splat a PSF kernel scaled by flux. Sub-pixel accumulation in f32.
   Bucket stars by HEALPix cell so each face only touches relevant
   cells.
3. Rasterize galaxies: evaluate Sérsic profile over a bounded patch,
   accumulate.
4. Rasterize nebulae: ray-march the volume noise for pixels whose
   direction intersects the nebula bounds.
5. Convert accumulated per-band flux to RGB via simple linear mapping
   (no tone mapping — leave that to Bevy's pipeline).
6. Hand cubemap to Bevy as an `Image` + `Skybox` component.

Bake cost budget: target < 2 s on release builds for default catalog.
Parallelize per face with `rayon`. Run async at startup like the
terrain bake.

### Telescope imager (later)

Same `Universe`, different renderer. Inputs:

- Pointing (RA, Dec, roll)
- FOV (arcminutes)
- Sensor resolution
- Passband (filter)
- Exposure time
- PSF (depends on optics)
- Noise model (dark current, read noise, shot noise)

Pipeline:

1. Gather sources within FOV + margin. Use per-layer spatial indices.
2. Project sources to sensor plane. Stars get PSF splat. Galaxies get
   Sérsic evaluation at sensor resolution.
3. Multiply by exposure time and throughput. Apply shot noise.
4. Add background (sky glow, zodiacal light, CMB if band applies).
5. Add sensor noise. Output 16-bit or 32-bit image for the player to
   process / stack / stretch.

Because sources are stored with full SED and faint objects are never
clipped, long exposures genuinely reveal dimmer sources than the
display skybox can show. This is the gameplay point.

## Wavelength extensibility

The model is already wavelength-agnostic. Adding a new band is:

1. Define a `Passband`.
2. Ensure relevant source types have sensible `Spectrum` coverage in
   that range (extend `Spectrum::Tabulated` tables as needed).
3. Add any band-specific source layers (e.g. `CmbMap` for microwave,
   `RadioSource` list for low frequency).

CMB specifically: a `CmbMap` layer carrying a low-resolution HEALPix
temperature map. Its `flux_in_band` only contributes in microwave
passbands. Visible-band skybox rendering ignores it automatically
because its flux integrates to zero there.

## Phasing

1. **Crate skeleton** — `thalos_celestial` crate, `Spectrum`, `Source`
   trait, `Universe`, coord helpers. Empty generators.
2. **Stars v1** — IMF + disk density generator, blackbody spectra,
   cubemap baker with PSF splatting. Integration with `thalos_game`:
   replace current skybox with baked HDR cubemap at startup.
3. **Galaxies v1** — GRF distribution, Sérsic rendering, local group
   hand placement.
4. **Nebulae v1** — Milky Way band via volumetric noise, a few named
   emission nebulae.
5. **Telescope imager v1** — offscreen render of a pointed FOV with
   fixed PSF and visible band. Gameplay hook comes later; this is the
   render path.
6. **Later** — proper motion, variability, additional bands, CMB,
   real catalog import behind the same `Source` trait.

Phase 1–2 is the minimum viable path to a better skybox. Everything
after is additive.

## Open questions

- Catalog persistence: regenerate every launch (fast enough?) or
  serialize to disk once per seed? Favors regenerate until proven
  slow.
- Coordinate frame for `Universe`: ICRS-equivalent is natural, but
  Thalos is not necessarily in our solar system — treat the celestial
  sphere as game-world absolute, decoupled from any specific body.
- Sky brightness for telescope imaging inside a planet's atmosphere —
  out of scope until atmospheric scattering exists in the planet
  renderer.
