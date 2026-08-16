# ADR-20260807T175108Z: Plate contacts author asymmetric relief provinces

- **Status:** Accepted
- **Date:** 2026-08-07
- **Decided by:** user, 2026-08-07 ("go for it")
- **Refines:** ADR-20260803T060232Z,
  ADR-20260803T072800Z

## Context

The grown plate graph removed the old Voronoi geometry and supplied coherent
convergent, divergent, and transform corridors. The global relief atlas still
did not read like terrestrial relief. Continents were broad green sheets with
similarly sized orange bumps, and the seafloor was generic cloudy noise.

The matched tectonic atlas ruled out missing or disconnected plate contacts.
Their area-weighted convergent influence was 0.047, while the preserved
mountain-building response averaged only 0.015. More importantly, the height
path represented collision only as narrow uplift plus peak texture. It had no
representation for the broad elevated hinterland and lower foreland that make
a terrestrial cordillera legible at planetary scale. In the ocean, 950 m of
generic seabed noise competed with a 520 m ridge core, so plate structure was
not the dominant bathymetric organizer.

## Decision

**A plate contact authors a whole asymmetric relief province, not only its
boundary core.**

For each convergent plate pair, one deterministic side carries a broad
hinterland-uplift lobe and the other a shallower foreland-basin lobe. The
existing preserved orogen remains the narrow rugged core between them. The side
choice is stable per plate pair, and both lobes reuse regional preservation so
old collisions survive as separated systems rather than complete rings.

Divergent ocean contacts gain a long-wavelength ridge swell around their
narrow core. Generic abyssal relief is reduced below the tectonic signal, and
trenches remain tied to submerged convergence. All contributions still pass
through `combine_macro_and_relief`: `continentalness` remains the sole
coastline authority.

The process field exposes `hinterland`, `foreland`, and `ridge_swell` weights.
`tectonic_preview` renders these independently in
`target/tectonic_provinces.png`, so future changes can distinguish a broken
plate signal from a broken height response.

## Alternatives

- **Change only the relief-map palette or hillshade.** Rejected because the
  map correctly exposed a structural absence in the height field.
- **Increase orogeny everywhere along convergence.** Rejected because it would
  recreate the uniform boundary rings closed by
  INC-20260804T020420Z and still would not create plateaus or forelands.
- **Add independent continent-scale uplift noise.** Rejected as the primary
  fix because it would make larger blobs without explaining their location or
  giving diffusion a causal structural prior.
- **Let tectonics move coastlines.** Rejected by the existing coastline
  authority decision; relief character and geography remain separate axes.

## Consequences

- Continental collisions read as nested lowland, plateau, and rugged-core
  systems at global scale rather than isolated peak patches.
- Ocean relief is organized primarily by ridge swells and trenches, with
  lower-amplitude abyssal texture between them.
- `GENERATOR_VERSION` advances to 32. Learned conditioning content generated
  before this version remains old content and must be regenerated before it can
  evidence these provinces.
- The 32,768-sample sign regression remains the coastline recurrence tell.
