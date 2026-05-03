# Vaelen Terrain Process Notes

Vaelen should read as a cold, formerly wet desert world: rusty dust, dark
volcanic/impact provinces, pale sedimentary and evaporite basins, degraded
craters, ancient dry channels, and active dune fields. The surface must not be
derived from visible province cells. Cells may exist later as hidden semantic
regions, but the rendered terrain is continuous.

## Visual Target

Vaelen is not just a red Mars analogue. It should read as a layered desert
planet whose present surface is dry, cold, and wind-worked, but whose largest
forms still expose an older wet history.

The strongest orbital read should be a few memorable compositional regions
rather than even planet-wide mottle:

- broad rusty highlands and dust plains
- dark basaltic or impact-melt provinces that break up the orange surface
- pale sedimentary lowlands and evaporite basins, with bone, tan, cream, and
  faint lavender-gray notes
- long scarps, basin rims, and canyon margins that organize the disk at a
  glance
- dune seas that pool against rougher old terrain instead of covering the
  whole world uniformly

The ground-level mood should be open and navigable: desert pavement, low relief
plains, distant mesas and scarps, local dune fields, and a dusty copper sky.
This is the first practical interplanetary surface world, so drama should often
sit on the horizon rather than under every landing site.

The present-day author is wind. Dunes, yardangs, streaks, mantled crater rims,
and polished plains should overprint older impact and hydrology features. The
world should feel useful and reachable, but also lonely: a frontier built on
borrowed time.

## Representation

The source of truth is a feature graph projected into a continuous
`SurfaceField`:

- height displacement in meters
- linear albedo
- weighted material mix
- roughness/detail spectra
- analytic feature buffers for craters, channels, scarps, dunes, and rocks

Each field is deterministic by feature seed stream. A global seed is only a
root; individual features own placement, shape, detail, and child seeds so an
authored region can be kept while another feature is rerolled.

## Macro Stack

1. Planet datum: low-order spherical fields and very low frequency warped fBm.
2. Crustal memory: highland swells, long rifts/scarps, broad volcanic plains.
3. Impact archive: crater and basin population with degradation and burial.
4. Former water: pale lowlands, dry channels, alluvial/delta fans, evaporites.
5. Present desert: dust mantle, dune seas, yardangs, wind streaks.
6. Detail spectrum: kilometer roughness down to shader-evaluated centimeter
   sand/ripple/rock fields.

The impostor bake should contain the macro bands the camera can see from orbit:
height, albedo, material, and later roughness/normal/provenance. Smaller
features should be represented by analytic buffers or deterministic shader
functions rather than globally baking centimeter data.

## Generation Priorities

Start with orbital composition before local detail. If Vaelen does not read
from the equirect bake, finer features will not save it.

1. **Anchor regions:** place 2-4 large, asymmetric visual anchors: at least one
   pale ancient basin, one dark volcanic or impact province, and one rusty
   highland/dust province with a strong boundary or scarp.
2. **Ancient wet story:** make pale basins structurally meaningful. They should
   sit low, collect channels and fans, carry evaporite floors, and sometimes
   reuse degraded impact basins instead of being free-floating color masks.
3. **Dark contrast:** strengthen dark basaltic and impact-melt terrain enough
   to prevent a one-note orange planet, but keep it patchy and geologically
   old. It should feel partly buried or dust-mantled.
4. **Wind overprint:** add dune seas, yardang belts, dust streaks, and mantled
   crater rims after the older features exist. Dunes should pool and spill
   around rough terrain, not appear as a procedural wallpaper.
5. **Surface usability:** preserve broad landing-friendly plains in basin
   interiors and dusty lowlands. Rough escarpments, crater clusters, and dune
   margins create local interest, but the baseline terrain remains forgiving.

This order keeps the generator aligned with the lore: crust and impacts create
old structure, ancient water modifies it, and the current desert reworks the
surface.

## First Vertical Slice

The first compiler slice intentionally avoids the old `TectonicSkeleton` and
`CoarseElevation` stages. It evaluates `VaelenColdDesertField` over cubemap
directions, then projects the samples into `BodyData` for the current impostor:

- continuous fractal macro height
- dark volcanic belt masks
- pale sediment/evaporite basin masks
- a long canyon/channel system
- one dune sea with wind-aligned ridge detail
- degraded crater population layered into the height field

This is not the final physical model, but it establishes the correct visual
contract: no visible polygon/province cells, no categorical debug paint, and a
field model that can grow toward authored terrain at every scale.

The current `material_cubemap` still stores one dominant material ID per texel
because the renderer expects that format. Vaelen's field now keeps a weighted
material mix before that projection step so the categorical ID map is no longer
the mental model for what the planet looks like.

## Bake Review Criteria

Use `just bake Vaelen` as the primary feedback loop. Review
`albedo-equirect.png` first, then `height-equirect.png` for whether the visual
anchors correspond to real terrain. Vaelen also writes `biome-equirect.png`;
use it to judge process-region layout before albedo and crater detail hide the
structure.

Vaelen is moving in the right direction when:

- the planet has an immediate pale-basin versus rust-highland versus dark
  province read from orbit
- the pale regions look like old basins or lakebeds, not arbitrary bleach
  stains
- channels and fans terminate into basins and margins instead of appearing as
  unrelated wrinkles
- dune fields are memorable local regions with wind-aligned structure
- craters are present but overprinted, softened, or partly buried often enough
  that Vaelen does not read as an airless impact moon
- there are large, plausible landing plains visible from orbit
- the color palette stays dry and cold: copper, ochre, umber, tan, cream, and
  dark basalt, with saturated gold reserved for active dunes and low-sun views
