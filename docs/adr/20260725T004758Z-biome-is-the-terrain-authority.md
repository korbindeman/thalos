# ADR-20260725T004758Z-biome-is-the-terrain-authority: the biome is the authority, carried as a blend over a per-tile palette

- **Status:** Accepted
- **Date:** 2026-07-25

## Context

Thalos needs planets with varied terrain — different generation character,
different materials, different surface detail — assigned by climate and rule. Four
mechanisms exist today and none of them is that: `BodyArchetype` varies whole
bodies, `biome_mask.rs` varies height fields regionally, `macro_band_ts` blends a
climate palette, and `MaterialBands` hands the tile shader two scalars. The biome
type we do have (`MacroBiome`) is documented in its own source as a read-only
argmax of the palette's band weights, consumed by `just map` and nothing else.

So the palette is the authority and the biome is derived from it. That ordering
caps what can ever be hung off a biome: the palette is a hardcoded chain of
smoothsteps over climate scalars, and a generator conditioning class, a material
layer set, or a scatter profile cannot be fields of a smoothstep. Meanwhile the
tile renderer's material layers (`NTR-X4`) hardcode slope thresholds globally,
tuned against one massif, and the landcover seam's one free channel (moisture in
the albedo attachment's alpha) is spent.

Three properties made this a real decision rather than an obvious refactor. The
current model is **fully continuous**, which is why the planet has no visible biome
seams; it obeys the **one-world rule** (map, impostor, ground, and scatter all read
one evaluation, so the class map provably cannot drift from the render); and the
biome map is wanted as an **input** to diffusion generation as well as an output of
it, which is circular unless the ordering is pinned.

## Decision

**The biome becomes the authority.** It is computed first, from climate and
geomorphology, and the palette, material layers, scatter profile, and generator
conditioning become fields of an authored `BiomeDef`. Design of record:
`docs/world/biomes.md` (`bio §N`).

Four parts are what this ADR commits:

1. **A blend, never a class.** The seam carries `BiomeBlend { ids: [BiomeId; 4],
   weights: [f32; 4] }`, normalized. The dominant id is a projection of the blend,
   inverting today's relationship with `MacroBiome`.
2. **Rules over signals, reusing `biome_mask.rs`.** A per-archetype `BiomeRuleSet`
   scores candidate biomes from named climate *and geomorphic* signals (slope,
   relief, coast distance, eco-altitude), evaluated by the existing
   `BiomeMaskExpr` machinery rather than a second rule evaluator.
3. **Transport = weights over a per-tile palette.** Each tile declares ≤4 biome
   ids; texels store weights over that tile's palette. Weights mip correctly;
   indices never would. The shader indexes a biome material table and blends layer
   stacks by weight — one shader, biomes as data. This is the "4-channel material
   splat" `query.rs`'s module doc reserved for P2.
4. **A one-way ordering invariant.** Coarse biome from the coarse band → detail
   generation conditioned by it → local biome refinement from the final tile
   height/slope. Climate biomes may not read fine height; geomorphic refinements
   may not feed back into generation.

Authoring: `BiomeDef` payloads in RON, rule sets in Rust.

## Alternatives

- **Keep the palette authoritative and bolt more scalars onto `MaterialBands`** —
  rejected. It is the status quo extended, and it caps out immediately: every new
  consumer needs a new scalar, a new channel, and a new mirror to keep in lockstep
  with `thalos::landcover`. The existing eco-altitude/forest pair is already two
  mirrors deep.
- **Discrete biome classification (one id per sample, hard boundaries)** —
  rejected. Simpler to author and to debug, and it is what most terrain systems
  do, but it would reintroduce exactly the seams the continuous band model
  eliminated, on a planet where the camera routinely crosses a climate gradient at
  cruise speed. Blending at the consumer instead of at the source also multiplies
  the blend logic by the number of consumers.
- **Global biome ids in the splat texture (index textures + weight textures)** —
  rejected. Mipping an index channel averages unrelated ids into a third biome, so
  every coarse tile would render a biome that exists nowhere on it. The known
  workarounds (nearest-only sampling, id-preserving custom mip chains) forfeit the
  automatic anti-aliasing the tile pyramid gives us for free.
- **Per-vertex biome only, no texture** — rejected as the *endpoint*, accepted as
  the first step (`BIO-3a`). Vertex interpolation cannot express a rock face
  narrower than a tile's vertex spacing, which is most of the detail this exists
  to produce. It does get the material table and blend maths landed early.
- **Derive biomes entirely from the diffusion model's learned climate channels** —
  rejected as the starting point, kept as a fork. The learned channels
  (BIO1/4/12/15) are real and already exported, but the biome map must also be
  *authorable* against lore constraints (35 % land, geologically old continents),
  and it must exist for airless bodies that have no climate at all. Learned climate
  replaces the `Climate` struct's contents later without touching the rule layer —
  which is the reason L1 is a struct.
- **A second, separate rule evaluator for landcover** — rejected. `biome_mask.rs`
  already does signals → expression tree → normalized weights with deterministic
  seed streams. A parallel evaluator would be two places to fix the next
  `smoothstep`-guard inversion (INC-0005).

## Consequences

- **One evaluation feeds everything.** The one-world rule extends from the palette
  to material, scatter, and conditioning: the ground, the orbital view, the
  impostor, the scatter, and the generator cannot disagree about what a place is,
  because they read the same `BiomeBlend`.
- **The tile attachment layout changes** at `BIO-3b` — cache namespace and
  `GENERATOR_VERSION` both move, and every baked package is invalidated.
- **A four-biome per-tile budget** is now a real constraint on rule sets: a rule
  set that puts five biomes in one tile forces a split. Expected to be benign
  (biome count correlates with climate gradient, which already earns tiles), but it
  is a number to watch at `BIO-3b`.
- **Slope-driven material behaviour moves out of the shader**, so `NTR-X4`'s
  constants (`ROCK_COS`, `SCREE_COS`, `ALPINE_ROCK_COS`, `SNOW_SHED_COS`) and the
  palette anchors become per-biome data. The showcase-patch tuning is preserved as
  the alpine biome's authored values, not lost.
- **The ordering invariant constrains future generation work**: anything that wants
  a geomorphic feature to change what the model generates must either move that
  signal into the coarse band or accept it as a post-generation refinement.
- Retires on contact: `woody_biome_gate`, moisture-in-albedo-alpha, the
  `macro_band_ts` palette ladder, `MacroBiome`-as-debug-view.
