# ADR-20260722T084155Z-mira-hero-visual-slice-parallel: run a Mira hero-visual slice in parallel with the L2 gate

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

`mira_learned_terrain_roadmap.md` defines a strict milestone ladder L1 → L6,
where styling (L5) sits *after* the whole-sphere bake (L3) and close
reconstruction (L4). Its definition of done states the principle plainly: *"An
attractive hillshade is a milestone, not completion."* That ordering is right for
guarding against advancing on training loss alone.

But the ladder implicitly assumes the visual risk lives in the height cascade.
Decomposing the three reference framings against what actually drives each one
shows otherwise:

| Framing | Dominant factor | Where it sits on the ladder |
|---|---|---|
| Full disc from orbit (~3 km/px) | albedo province structure; all learned bands below one pixel | not on the ladder at all — see ADR-20260722T084154Z |
| Oblique low-sun approach | macro height (S0–S2) × grazing light × Hapke/shadow | L2/L3 height + L5 rendering |
| Close crater rim | S3/S4 + `Rclient` close band | L4 |

Two of the three targets are gated on rendering and materials, not on the
learned cascade. Under strict ladder order the first frame that resembles the
references arrives several milestones out, and every rendering assumption
between here and there stays unvalidated — including whether the shared Hapke
path, exposure, and shadow rig can even produce that look on a body we already
have.

## Decision

**Run a bounded hero-visual slice in parallel with L2 gate closure**, against the
*existing* compatibility-producer package, rather than waiting for L3.

The slice is explicitly scoped to what does not depend on the learned bake:

1. **Reference-matched capture framings** — deterministic presets for the three
   reference geometries (orbital disc, oblique low-sun approach, close rim), so
   visual claims are made against fixed framings instead of ad-hoc cameras.
2. **Airless render calibration** — Hapke parameters, opposition surge,
   mare/highland albedo split, exposure and terminator response, evaluated at
   those framings.
3. **A single-face hero bake** at the campaign's 4096 face resolution using the
   v5b checkpoint — real learned terrain in-game at production resolution on one
   face, before committing to whole-sphere seam/consensus machinery.

L2 gate evidence (galleries, memorisation check, residual stipple) proceeds
concurrently and is **not** relaxed. The ladder's principle is retained: this
slice cannot advance L2, L3, or L4 — it can only retire visual risk and inform
their tuning.

## Alternatives

- **Strict ladder order** — rejected for sequencing only, not for rigor. It has
  the cleanest evidence chain and zero rework, but it validates the renderer last,
  after the most expensive campaigns are already spent against it. If the shared
  Hapke/exposure path cannot produce the reference look, the ladder discovers
  that at L5 — after L3 and L4 have been tuned to a target that was never checked.
- **Materials-first** (treat the full disc as the sole near-term target) —
  rejected as too narrow. It reaches a convincing orbital disc fastest, but the
  oblique framing is the one that actually exercises the learned height work, and
  dropping it would leave MIRA-1's output visually unvalidated for longer.
- **Fold styling into L3** — rejected because it re-couples the two and loses the
  independent signal. The point of the parallel slice is that it runs on a body
  we already have, so a bad result indicts the renderer rather than the bake.

## Consequences

- Some calibration will be **redone** once the real whole-sphere bake lands.
  Accepted: re-tuning against a known-good target is cheap; discovering the
  target was unreachable is not.
- The hero-bake face is a **throwaway integration artifact**, not a package
  milestone. It must not be mistaken for MIRA-2 progress or checked in as a
  shippable package.
- The roadmap's definition of done is unchanged. This slice adds no exit
  criterion and removes none; L5 still owns final styling acceptance.
- Capture work is blocked until the stale Mira package key is rebuilt
  (`just bake Mira`) — that becomes a prerequisite of the slice rather than an
  unrelated chore.
