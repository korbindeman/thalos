# ADR-20260722T084154Z-airless-material-provinces-authored: airless material provinces stay authored/procedural over learned height

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

Mira's visual target is set by three reference framings (see
`roadmap/mira_learned_terrain.md` §"Visual targets"): a full disc from orbit, an
oblique low-sun approach, and a close crater rim. The **full-disc frame is
dominated by albedo, not relief** — at ~3 km/px every learned height band sits
below one pixel, and what makes the image read as a moon is mare basalt
darkness, highland brightness, fresh-crater bright ejecta, and ray systems.

Today that albedo exists, but it comes from the **compatibility producer**: the
retained deterministic airless compiler, authored in `assets/bodies/mira.ron`
(`highland_mature_albedo`, `mare_mature_albedo`, `mare_fresh_albedo`,
`mare_tint`, the `procellarum` mask). It reaches the runtime as the
`PackageBlobKind::StaticSurfaceV1` blob, whose own doc comment records that it is
a *"Compatibility substrate. Replaced by macro/residual node kinds in the
diffusion producer."*

The diffusion producer (`mira_airless_mvp.md` §6 Stage B) emits **height only**.
`PackageBlobKind` carries `HeightBase`, `HeightResidual`, and `Conditioning` —
there is no material kind. MIRA-2's scope names "H0 occupancy/material channels"
but nothing has landed.

So the moment the diffusion producer replaces the compatibility substrate, Mira
loses the single largest contributor to its most-visible framing. Where albedo
comes from after that point is a real fork, and it gates MIRA-2's package schema.

## Decision

**Material provinces remain an authored/procedural, seed-conditioned field
layered over learned height.** Mare masks, crater-age → fresh-ejecta brightness,
and ray systems are computed from the same 64-bit seed and normalized parameter
vector that condition the diffusion cascade, not emitted by the model.

Consequently:

- The learned cascade's contract stays **height-only**. Its channels, corpus,
  and metrics do not grow a material head.
- The package grows an explicit **material/province node kind** carrying the
  procedural field's output (or its compact parameters), replacing the
  `StaticSurfaceV1` substrate's implicit carriage rather than inheriting it.
- The province field is evaluated in pure Rust and is reproducible through the
  CPU `SurfaceQuery` path, matching the frequency/authority contract in
  `mira_airless_mvp.md` §3.
- Albedo remains **appearance-only** — it never moves a collider, landing leg,
  or EVA contact, so it carries no parity obligation beyond determinism.

## Alternatives

- **Learn material channels alongside height** (train albedo heads on LROC WAC
  co-registered with SLDEM2015) — rejected *for now*, not on principle. It is the
  most faithful route to the full-disc frame and is the only option where height
  and albedo agree by construction. But it multiplies the corpus work
  (co-registration, photometric normalization to a common illumination geometry,
  a second distribution-rights review) at exactly the moment MIRA-1 has just
  reached its first passing architecture, and it would make every future
  campaign more expensive. Revisit if the authored field visibly disagrees with
  learned relief at the oblique framing.
- **Derive albedo analytically from learned height** (slope + curvature + a
  crater-age proxy; fresh excavation bright, mature dark) — rejected as the
  primary mechanism. It always agrees with the terrain and needs no extra data,
  but the two features that carry the full-disc frame are *not* functions of
  local relief: mare province boundaries are volcanic flooding history, and ray
  systems are ballistic ejecta patterns extending far beyond their crater. A
  purely relief-derived albedo produces a uniformly grey cratered ball. Retained
  as a *modifier* — slope-conditioned talus/downslope brightening layered on the
  authored provinces — not as the province source.
- **Defer to MIRA-2** — rejected because the fork gates MIRA-2's package schema.
  Discovering it mid-bake means either a schema revision or a whole-sphere
  campaign that renders grey.

## Consequences

- MIRA-2's package schema must add a material/province node kind before the
  first whole-sphere campaign; it can no longer treat `StaticSurfaceV1` as the
  thing it silently replaces.
- The authored province field and the learned height field can **disagree** —
  a learned basin rim may cut across an authored mare boundary. This is the
  accepted risk of this option. The mitigation is that both are conditioned on
  the same seed and parameter vector, and Stage C (physical/profile correction)
  is the natural place to measure the disagreement. If the oblique framing shows
  visible mismatch, that is the trigger to revisit the learned-channel option.
- The authored albedo parameters in `assets/bodies/mira.ron` become **load-bearing
  production inputs**, not compatibility-producer tuning knobs. They survive the
  producer swap and need to be reviewed as such.
- Ray systems are currently *not* modelled by the compatibility producer at all
  (the authored params cover mare/highland and fresh/mature, but no ballistic
  ray pattern). Delivering the full-disc reference framing requires adding them.
  That is now explicit scoped work rather than an assumed freebie.
- Mira's layout stays fictional. Real lunar data continues to teach structure and
  process statistics only; this decision does not introduce a path where a
  recognisable real albedo map is cropped or remapped onto Mira.
