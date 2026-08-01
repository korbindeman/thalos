# INC-20260731T223731Z — Bevy silently overwrote a hand-authored `Aabb`, so a correct fix read as a no-op

**Symptom.** Terrain caster twins flooded the shadow cascades: 122 tile casters
inside a cascade covering **64 m** of ground, 1,639 across the set. Stamping an
explicit tight `Aabb` on the caster child changed the post-cull counts by
**exactly zero** — 145 / 208 / 980 / 1677 before and after, not one entity
different. The obvious reading ("the box is not the bottleneck; the frustum is")
was wrong, and would have sent the next round of work at the wrong target.

## Mechanism

Two independent defects compounded, and only the second is non-obvious.

**1. The skirt curtain inflated every tile's derived bounds.** Since the
floor-sphere skirt landed (`build_tile_mesh`), every border vertex hangs to
`radius − relief` — ~10 km below the datum on Thalos. Bevy derives a mesh's
culling `Aabb` from *all* of its positions, so a ~300 m level-14 tile gets a
~10 km-tall box: a 33:1 slab that intersects nearly any frustum aimed at the
body. The effect scales inversely with tile size and is therefore invisible where
you would look first — at level 6 the tile is ~78 km across and the two boxes
agree to 0.4 % — while the near cascades, which are packed with the finest tiles,
are exactly where it is pathological.

**2. `calculate_bounds` does not only fill in a MISSING box.** It is natural to
read its first query —

```rust
new_aabb: Query<(Entity, &Mesh3d), (Without<Aabb>, Without<NoFrustumCulling>, Without<NoAutoAabb>)>
```

— and conclude that a hand-authored `Aabb` is respected. It is not. The same
system carries a second query:

```rust
update_aabb: Query<(&Mesh3d, &mut Aabb), (Or<(AssetChanged<Mesh3d>, Changed<Mesh3d>)>, ...)>
```

which takes `&mut Aabb` and **overwrites** it from `mesh.compute_aabb()`.
`Changed<Mesh3d>` is true on the frame the component is inserted, so a box
authored at spawn is replaced by the derived one on the first
`VisibilitySystems::CalculateBounds` pass after it. The component was present the
whole time; by the time culling read it, it was Bevy's, not ours.

## Fix

`NoAutoAabb` on the caster child alongside the explicit `Aabb` — it excludes the
entity from *both* queries. Result at the same vantage: cascade 0 **122 → 4**
tile casters, cascade 1 **173 → 20**, total caster draws **3,010 → 2,362**, with
the before/after captures visually identical (no caster that should cast stopped
casting).

## Tell

A hand-authored `Aabb` that produces **no change whatsoever** in post-cull
counts — not a small change, an exactly-identical one. A partial effect means
something else; a zero effect means the component is being replaced rather than
ignored. Confirm by reading the live `Aabb` back off the entity rather than
trusting the spawn bundle.

`stability_gauge`'s `cascadeN_tiles` (added in the same change) is the standing
instrument: terrain twins inside a cascade far exceeding what its ground box can
geometrically hold is this defect's signature.

## Rules

- **When you author an `Aabb` by hand, author `NoAutoAabb` with it.** Bevy owns
  that component otherwise, and it takes it back on any `Mesh3d` change — not
  just at spawn, so this also bites anything that swaps a mesh handle at runtime.
- **A mesh whose geometry is far larger than its visible surface needs explicit
  bounds.** Skirts, curtains, billboard expansion and displacement-shader margins
  all put geometry in the vertex buffer that the culler should not be reasoning
  about. The tile skirt is the case here; it will not be the last.
- **A fix that changes a measured number by exactly zero has not been tested —
  it has been prevented from running.** Treat byte-identical output as evidence
  the change did not take effect, not as evidence the hypothesis was wrong.
