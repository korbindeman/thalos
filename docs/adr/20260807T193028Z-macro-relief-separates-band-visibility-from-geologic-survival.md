# ADR-20260807T193028Z: Macro relief separates band visibility from geologic survival

- **Status:** Accepted
- **Date:** 2026-08-07
- **Decided by:** user, 2026-08-07 (confirmed the reference-comparison scope)
- **Refines:** ADR-20260807T175108Z

## Context

The first asymmetric collision provinces gave plate contacts a hinterland,
foreland, range core, ridge swell, and trench response. Against the terrestrial
reference, Thalos still showed similarly scaled roughness across most land and
isolated high blobs instead of quiet interiors joined by coherent mountain
systems.

Two independent mechanisms produced the same image. First,
`octaves_for_lod` reduced the octave count but the normalized fractal evaluators
kept the remaining base octave at full amplitude even below Nyquist. Hills and
ridges therefore aliased into the planet map. Second, one preservation mask
controlled both whether a collision had any range at all and where its highest
massifs survived. Raising the response strengthened inherited interior blobs;
lowering it disconnected the ranges.

## Decision

**Band visibility, inherited relief, range continuity, and peak survival are
separate controls.**

- Every relief band has an explicit footprint gate: zero at two or fewer
  samples per base wavelength and full at four samples. Octave normalization
  never decides whether the whole band is resolved.
- Inherited terranes remain a low-amplitude interior signal. They cannot reach
  active-range strength merely because the range-core uplift increases.
- Convergent contacts keep a low continuous orogenic spine. The regional
  preservation field controls where tall massifs survive above that spine, not
  whether the mountain system exists at all.
- Learned chart sidecars record the procedural generator version that authored
  their conditioning. A mismatch is reported explicitly. Stale learned content
  remains loadable, but cannot be presented as evidence for the current
  tectonic field.

The learned producer must elaborate the new conditioning. We do not paste the
current tectonic response over an old learned chart after generation.

## Alternatives

- **Raise all montane relief together.** Tested and rejected: it strengthened
  real contacts but amplified inherited terranes into the same isolated blobs.
- **Remove inherited relief.** Rejected: old continents need subdued interior
  structure; it simply belongs at a lower amplitude.
- **Make every preserved collision continuous at full strength.** Rejected:
  it recreates uniform boundary rings. Only the low connecting spine is
  continuous; peak survival remains regional.
- **Post-process the installed learned chart with analytic ridges.** Rejected:
  valleys and drainage would not organize around the mountains, contradicting
  ADR-20260803T060232Z.

## Consequences

- Coarse views contain only relief they can resolve; quiet plate interiors stop
  turning into continent-wide noise.
- A range can read as one system while retaining separated old massifs and
  varied peak height.
- `tectonic_preview` reports the belt/interior slope contrast so future tuning
  can distinguish stronger geology from more noise.
- Generator version 33 invalidates procedural terrain caches. The checked-in
  learned chart records generator 28 and requires an external producer rebake.
