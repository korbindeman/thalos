# Biomes — the terrain authority (`bio`)

**Status:** design / plan of record · **Written:** 2026-07-25
**Decision:** [ADR-20260725T004758Z-biome-is-the-terrain-authority](../adr/20260725T004758Z-biome-is-the-terrain-authority.md)
**Cross-ref prefix:** `bio §N` · **Rows:** `BIO-1` … `BIO-7` in [backlog.jsonl](../backlog.jsonl)

Companion docs: [terrain_macro.md](terrain_macro.md) (the climate/landcover fields
this consumes — its §4 "Phase 2 remainder" is superseded by this doc),
[terrain.md](terrain.md) (tile contract), [vegetation.md](vegetation.md) (scatter
cascade), [neural_terrain_renderer.md](../roadmap/neural_terrain_renderer.md)
(`ntr §4.3`/`§4.4` conditioning).

## §1 Thesis

Planets should have **varied terrain** — different generation character, different
materials, different surface detail — assigned by climate and rule rather than by a
global palette ladder. One record per biome carries what the generator, the
renderer, and the scatter systems each need, so "which terrain, which material,
which vegetation" stop being three unrelated questions with three answers.

The change is an **inversion**. Today the palette is the authority and the biome is
a debug view derived from it (`MacroBiome` is documented in `procedural.rs` as the
read-only argmax of the band weights `albedo_from_bands` blends, consumed only by
`just map`). The palette ladder is a hardcoded chain of smoothsteps over climate
scalars; nothing else can be hung off it. Under this plan the **biome is computed
first** and the palette, the material layers, the scatter profile, and the
generator's conditioning are all *fields of the biome*.

Two properties must survive the inversion, because they are what currently works:

- **Continuity.** Everything blends — the reason the planet has no visible biome
  seams. A hard classification would regress that immediately and visibly. The
  biome seam therefore carries a **weight simplex, never a class.**
- **The one-world rule.** Map, impostor, ground shader, and scatter all read *one*
  evaluation (`sample_biome_d` guarantees the class map cannot drift from the
  render). One biome evaluation must feed every consumer, for the same reason.

## §2 What exists today

Four biome-ish mechanisms that don't know about each other:

| Mechanism | Where | Scope | Drives |
|---|---|---|---|
| `BodyArchetype` (5 kinds) | `feature_compiler.rs:150` | whole body | field family, feature manifest, `TerrainShadingStyle` |
| `BiomeMaskPlan` / `BiomeMaskExpr` | `biome_mask.rs` | regional, generation-side | height-field variation on the feature-path bodies |
| `macro_band_ts` → `MacroBandTs` → `albedo_from_bands` | `procedural.rs` | regional, climate | the macro palette, continuously blended |
| `MaterialBands { eco_altitude_m, forest }` | `query.rs` → `tile_terrain.wgsl` | per-vertex → shader | the NTR-X4 rock/scree/snow/forest/meadow stack |

`biome_mask.rs` is the important find: it is already *exactly* the right shape —
named scalar signals + deterministic seed streams → an expression tree → weighted
scores → **normalized weights**, with a fallback. It has simply never been pointed
at landcover. `BIO-2` reuses it rather than inventing a second rule evaluator.

The gap is not "we have no biomes". It is that the biome is derived instead of
authoritative, and that the seam reaching the renderer carries **two scalars**
where it should carry a biome.

## §3 The layered model

### L0 — the archetype picks a biome *set*

`BodyArchetype` gains two things: which climate model applies, and which biomes can
exist here. Airless bodies get mare / highland / ejecta-ray / polar-cold-trap with
**no climate model at all**; earth-like bodies get the Whittaker-ish set. This is
what keeps the system from being Earth-only, and it is why `BIO-7` (an airless
biome set) is part of the plan rather than a nice-to-have — it is the falsification
test for the abstraction.

### L1 — one `Climate` struct

The scalars currently inlined in `macro_band_ts` — cold lift, warmth, moisture,
continentality, orogeny — become one struct with one writer, evaluated once per
sample. Two reasons beyond tidiness:

- It is the swap point for `ntr §4.4`: the diffusion pipeline already co-generates
  WorldClim BIO channels (BIO1/4/12/15 — see `NTR-FT-0`), and replacing our
  hand-built fields with learned ones is only cheap if there is a single named
  thing to replace.
- Biome rules need to read climate by name. A rule set written against a struct of
  named fields is authorable; one written against a smoothstep chain is not.

`BIO-1` is a pure output-identical refactor gated by `just map`.

### L2 — rules, not a palette chain

Each archetype owns a `BiomeRuleSet`: authored rules scoring each candidate biome
from **climate and geomorphology**. The geomorphic half matters as much as climate:

| Signal class | Examples |
|---|---|
| climate | temperature / cold lift, precipitation, aridity, seasonality, continentality |
| geomorphology | eco-altitude, slope, local relief (ruggedness), coast distance, orogeny |
| stochastic | the existing fBm seed streams, for ecotone mosaic and province variation |

Slope belongs *here*, not in the shader. "Scree collects on the bench below a rock
face" is a biome rule; today it is `SCREE_COS` and `ALPINE_ROCK_COS` hardcoded
globally in `tile_terrain.wgsl`, which is why those constants had to be tuned
against one massif and cannot vary per planet.

Output:

```rust
pub struct BiomeBlend {
    pub ids: [BiomeId; 4],
    pub weights: [f32; 4], // normalized, descending
}
```

Top-4 rather than the full simplex: four is what the transport (§4) can carry per
texel, and beyond three contributors the blend is visually indistinguishable from
its top three. `weights[0]`'s id is the dominant class — the `MacroBiome` role,
now a projection of the real thing rather than its source.

### L3 — `BiomeDef` is the payload

One authored record per biome. This is the piece that makes the generator, the
renderer, and the scatter the same problem:

| Group | Fields |
|---|---|
| **generation** | diffusion conditioning class, relief amplitude / spectral character, erosion style, sub-model-scale analytic octave set |
| **material** | material layer set (each keyed by slope/altitude *within* the biome), palette anchors, roughness |
| **scatter** | tree species mix, grass profile, rock/pebble density, clearing rules |
| **surface detail** | detail-normal amplitude and character, near-field micro texture |
| *(reserved)* | footstep / ambience slots — cheap to leave, expensive to retrofit |

### L4 — transport

See §4. Three consumers, one evaluation.

### L5 — conditioning

See §5. The biome map is an input to generation, not only an output of it.

## §4 Transport: the per-tile biome palette

Moisture currently rides the **alpha channel of the albedo attachment** (`Rgba8UnormSrgb`
— alpha stays linear, mips average correctly; `terrain_macro.md` §3). That was a
good trick with exactly one channel of headroom, and it is spent.

The replacement is the thing `query.rs`'s own module doc reserved for P2 and never
landed: **"the full Tile contract (4-channel material splat + macro-albedo
modulation)"**.

**Design — weights over a per-tile palette.** Each tile declares up to four
`BiomeId`s in its uniform / instance data; the splat texel stores **weights over
that tile's palette**, not global ids.

This is the load-bearing detail: **weights mip correctly, indices never would.**
Averaging two biome *ids* produces a third unrelated biome; averaging their weights
produces the blend that is actually there. It is the same discipline as
per-chunk material lists in conventional terrain splatting, and it is why the
palette is per-tile rather than global.

Tiles whose region needs more than four biomes split — which the LOD tree already
does for relief, and which is self-limiting because biome count correlates with
climate gradient, i.e. with the same transitions that already get more tiles.

Consumers:

| Consumer | Reads |
|---|---|
| CPU (scatter, colliders, map, `world_map`, HUD) | `SurfaceQuery::biome_at(dir) -> BiomeBlend`, extending the seam exactly as `landcover_moisture` did |
| tile shader | the splat + a **biome material table** (storage buffer of per-biome material params), blending layer stacks by weight — one shader, biomes as data |
| tile bake / provider | writes the splat + the per-tile palette alongside the albedo attachment |

**Staging.** The tile path already carries `MaterialBands` **per-vertex** (the
shader comment: "canonical forest band weight (vertex-carried)"). A per-vertex
`BiomeBlend` is therefore a strictly smaller first step than the attachment, and
gets the material table and the blend maths landed before the bake/mip work. `BIO-3`
is split accordingly (3a per-vertex, 3b per-texel).

## §5 The ordering invariant

Biomes need altitude. Biome-conditioned generation produces altitude. Left
unstated this is circular, and it is the failure mode most likely to be discovered
late, so it is an invariant:

> **Coarse biome from the coarse band → detail generation conditioned by it →
> local biome refinement from the final tile height and slope.**

Concretely: the coarse chart (23 km/px, already exported) plus climate yields the
**climatic** biome — forest vs steppe vs tundra — which conditions the detail band.
The **geomorphic** refinement — rock face, scree bench, snowfield, riparian strip —
is then computed from the height the model actually produced. Climate biomes may
never read fine height; geomorphic refinements may never feed back into generation.

`export_thalos_macro.rs` already conditions the diffusion export on Thalos's own
`ProceduralSurface` macro terrain, so the biome/climate channels join that same
conditioning raster stack rather than needing a new mechanism (`BIO-5`).

## §6 Authoring surface

**`BiomeDef` payloads in RON; rule sets in Rust** (user decision, 2026-07-25).

The defs are data — palettes, layer sets, scatter profiles, conditioning classes —
and belong alongside `thalos.ron` / `parts.ron` where the rest of the world is
authored. The rule sets stay Rust while the model is still moving, because a rule
set is where the *thinking* is and it will be rewritten often.

`BiomeMaskExpr` is already a serializable enum tree, so promoting rules to RON
later is a deserializer plus a validation pass — deliberately left as a cheap
future step rather than paid for now.

## §7 What this retires

- `MacroBiome`-as-debug-view → the real `Biome` type, with the argmax as a
  projection.
- The `macro_band_ts` smoothstep ladder → per-biome palette data.
- `woody_biome_gate` (`body_render::ground::scatter`) → biome scatter data; closes
  `TM-P2r`'s remaining half.
- Moisture-in-albedo-alpha → the splat attachment.
- `tile_terrain.wgsl`'s global layer constants (`ROCK_COS`, `SCREE_COS`,
  `ALPINE_ROCK_COS`, `SNOW_SHED_COS`, the palette anchors) → per-biome layer sets.

Every one of these is a delete-on-contact per the standing quality bar, not a
parallel path left "for reference".

## §8 Sequencing

| Row | Slice |
|---|---|
| `BIO-1` | `Climate` struct — one authority, output-identical, `just map` as the gate |
| `BIO-2` | `BiomeDef` (RON) + `BiomeRuleSet` (Rust, on `biome_mask.rs`) + `SurfaceQuery::biome_at`; palette reimplemented on top, identity-gated |
| `BIO-3a/3b` | per-vertex `BiomeBlend` + shader biome material table (3a); per-texel splat attachment + per-tile palette + mips (3b) |
| `BIO-4` | scatter reads `BiomeBlend` — grass profiles and tree mixes per biome (absorbs `TM-P2r`) |
| `BIO-5` | biome as diffusion conditioning; the §5 ordering invariant made real |
| `BIO-6` | biome identities content pass (absorbs `TM-P3b`): erg/reg, savanna, taiga, sea ice |
| `BIO-7` | airless biome set on Mira — the falsification test for L0 |

Every slice that changes generation output bumps `GENERATOR_VERSION`; `BIO-3b`
changes the tile attachment layout, so it also changes the cache namespace.

## §9 Open forks

| Fork | Gates | Notes |
|---|---|---|
| Rules in RON vs Rust | after `BIO-2` has been rewritten a few times | §6 — deliberately deferred, cheap to take later |
| Learned climate channels (BIO1/4/12/15) replacing the authored `Climate` | `BIO-1` landed; `ntr §4.4` | the whole point of L1 being a struct |
| Palette width — 4 per tile vs 8 | `BIO-3b` measurement | 4 assumed; raise only if tiles at climate transitions split pathologically |
| Whether geomorphic refinement can ever condition generation | `BIO-5` | §5 says no. Revisit only with evidence, and with an ADR |
