# `propagator-astro-4` — golden-section replica (**source lost, results preserved**)

> **The replica source was not preserved.** It was built by the refuter in a
> scratch crate outside the repo and reported only its results; the agent was
> stopped before it could write the source back, and the source never passed
> through a transcript that survives. What follows is everything that *is* known
> — enough to rebuild it, not enough to re-run it. Rebuilding is ~1 session.
>
> This is a harness defect, now fixed: `.claude/skills/expert-review/SKILL.md`
> requires evidence artifacts to be written to `docs/reviews/repro/` **in the same
> step that produces them**, never held in an agent's reply.

## What it demonstrated

`refine_collision` (`crates/simulation/physics_canonical/src/ship_propagator.rs:1111-1118`)
locates a detected terrain impact with `golden_section_extremum` (`:966-994`),
which is a textbook bracket-shrinker —

```rust
let pick_left = if seek_min { fc < fd } else { fc > fd };
```

— with no restarts and no sampling pass, so it **requires a unimodal objective**.
Its objective embeds `terrain.surface_elevation_m`, a procedural height field,
which is not unimodal over a step. When it converges on the wrong local minimum
and finds `f > 0` there, it returns `None`, the caller falls through to
`samples.push(...)` at `:472-475`, and a **correctly detected** collision is
discarded as a "Hermite false positive".

## Results (verbatim)

720 scenarios: 60 ridge phases × 12 periapsis altitudes (rp = R+0.5 … R+4.9 km,
ra = R+400 km).

```
detect_fired            = 192
  refine -> Some (kept) = 182
  refine -> None        =  10
    of which a REAL sub-surface root existed = 10
```

Diagnostic lines pinning the cause to wrong-local-minimum convergence:

```
DIAG golden_s=0.7528 f_ext=+814.8m   true_min_s=0.2567 f_min=-1241.6m
DIAG golden_s=0.2831 f_ext=+885.4m   true_min_s=0.4902 f_min= -740.2m
DIAG golden_s=0.2592 f_ext= +39.6m   true_min_s=0.7568 f_min= -123.4m
```

The 0.75/0.25 mirroring is the signature of the first two probes at 0.382/0.618
steering the bracket the wrong way.

Across terrain configurations (λ / amplitude → rejected-but-real / detections):

| wavelength | amplitude | rejected-but-real | detections |
|---|---|---|---|
| 200 km | 2 km | 2 | 153 |
| 400 km | 3 km | 4 | 287 |
| 90 km | 2 km | 7 | 60 |
| 45 km | 4.9 km | 8 | 324 |

**In every configuration, 100 % of `None` returns were real crossings** — the
escape hatch this branch exists for never once fired legitimately in 720 scenarios.

Consequence, narrower than the finding was filed:

```
rejections later re-caught in the same orbit = 9
orbits where the rejected impact was LOST    = 1     (of 720)
```

So the dominant observable defect is a **mislocated impact** — up to one step
late, ~40 s and ~300 km downrange at LEO speeds — not a silent fly-through.

**Calibration against `propagator-astro-2`:** the same sweep recorded **477**
steps where a real sub-surface root existed and `detect_step_crossings` never
fired at all. Detection density is the larger defect by an order of magnitude;
this one is the residual behind it.

## How to rebuild

A **replica**, not a harness over the live code: copy these functions verbatim
from `crates/simulation/physics_canonical/src/ship_propagator.rs` into a scratch
crate, and drive them with the real public
`thalos_physics_canonical::orbital_math::propagate_kepler`.

- `hermite_cubic`
- `swept_dist_sq_extremes`
- `altitude_at_q`
- `interior_min_altitude`
- `golden_section_extremum`
- `bisect_signs`
- `refine_collision`
- the collision branch of `detect_step_crossings`
- the coast loop's subdivision cap

Terrain: a ridged field at the established scale (90 km wavelength, 4.9 km peaks;
see `crates/domain/terrain/src/procedural.rs:534-590` for `MASSIF_SITES` /
`MASSIF_PEAK_M`). Ground truth: a 6000-point dense scan of the exact Kepler
altitude per step, so "a real sub-surface root existed" is decided independently
of the code under test.

**Anyone acting on this must re-check those functions against current source
first** — a replica silently drifts.

## Caveats that must survive into any fix

1. **The proposed fix needs a fallback.** "Bisect on the bracket
   `[s_prev, s_probe]` that `interior_min_altitude` already found" is right in
   shape, but `interior_min_altitude` probes the **Hermite** curve while
   `refine_collision` roots the **Kepler** curve. That bracket is a strong hint,
   not a guaranteed sign change.
2. **`:929-941` `refine_crossing` is NOT defective.** The finding listed it as
   "same pattern"; unimodality genuinely holds for the smooth distance-to-sphere
   case. Only the `refine_collision` site needs changing.
