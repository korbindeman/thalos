# INC-20260726T233436Z — terrain-caster acne blurred into straight-edged veils

**Symptom.** The same session terrain tiles became sun-shadow cascade casters
(BL-20260726T222119Z-shadow-round), open plains grew *giant straight-edged
light/dark regions* — a bright trapezoid around the view anchor with dead-straight
diagonal edges — plus smooth grey smears and a fine directional ripple across the
ground, worst at low sun (user screenshots at the hub spaceport, dawn). Nothing
looked like classic shadow acne: no stipple, just large soft veils with hard
polygon boundaries.

**Mechanism — two effects composed.**

1. The cascade **bias model was calibrated for casters that are never their own
   receivers** (`sun_shadow.rs`: "terrain never renders into the maps, so it
   cannot self-shadow-acne"). Terrain-as-caster broke that assumption: at the
   outer cascades' multi-metre texels (×the altitude footprint scale) the
   hard-capped depth bias (`BIAS_MAX_M` = 2.5 m, capped to protect tree-height
   casters) cannot cover the per-texel depth error of even gently rolling
   ground, so the whole plain self-shadowed partially.
2. **PCSS then blurred the acne smooth.** The blocker search averages depth
   deltas over ±6 texels (tens of metres out there) and the Vogel filter spreads
   the result — so instead of texel stipple, the acne rendered as a *uniform
   partial-shadow veil*. Its inner boundary is exactly where a finer cascade
   (adequate bias) takes over: the "mysterious bright trapezoid" is the finer
   cascade's ortho box, traced by the veil around it.

**The tell for a recurrence:** large straight-edged brightness regions on open
ground whose edges are sun/anchor-aligned and move with the camera, plus a fine
directional ripple in the darker zone; disappears inside the near cascade.
Discriminate from cloud shadows with `THALOS_CLOUD_SHADOW=off` (the veil
survives; a cloud box would not).

**Fix.** Terrain caster proxies render **back faces only**
(`cull_mode: Some(Face::Front)` on the caster material,
`rendering/tile_terrain.rs`): the stored depth is the terrain's down-sun face,
so a sunward receiver always compares against depth *beyond* itself — acne is
impossible by construction at any texel size, no bias tuning involved. Valley
floors behind a ridge still fall deeper than the ridge's leeward face and keep
their cast shadow; the only surface that loses one is the leeward face itself,
which `n·l` already darkens. Verified same-conditions A/B:
`artifacts/visual/runs/shadow-round/artifact_repro.png` vs `artifact_fixed.png`.

**Rule.** A heightfield that both casts and receives the same shadow map must
cast its back faces (or carry a bias ∝ its own texel size, which conflicts with
small-caster bias caps). If a future caster class reintroduces front-face
terrain depth, this veil comes back.
