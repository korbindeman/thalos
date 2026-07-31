# ADR-20260730T225312Z — A near cascade, and cascade boxes square on the ground

**Status:** accepted · 2026-07-30
**Relates to:** ADR-20260722T111848Z-shadows-three-tier-not-virtual-shadow-maps
(refines its mid-field tier; does **not** supersede it)

## Context

A user screenshot from the runway chase camera showed the aircraft's cast shadow
with hard, chunky, stair-stepped edges — blocks roughly a metre across, visibly
**elongated along the shadow's length** rather than square.

The rig was not obviously under-built: 3 cascades, 4096², PCSS with a blocker
search and a Vogel disk, a texel-snapped stable-CSM centre, a capped
texel-proportional bias model. Cascade 0 was a ±400 m box, i.e. 0.22 m per texel.

Two things made that inadequate, and only the second is non-obvious.

1. **0.22 m is coarser than the subject.** Landing gear, a wing edge, a hull
   panel, a strut — everything a surface camera actually looks at is finer than
   one texel of the cascade that covers it.

2. **The map was square in the LIGHT plane, so a low sun smeared it on the
   ground.** Projecting ground → light plane compresses the sun-azimuth direction
   by `sin(elevation)` and leaves the cross-azimuth direction alone. A box that is
   square in the light plane therefore spans `half / sin(elev)` of ground along
   the azimuth while spending the same 4096 texels on it. At the ~15° sun in the
   screenshot that is a **3.9× coarser ground texel in one direction than the
   other** — ~0.85 m — which is exactly the elongated staircase that was reported.

PCSS could not rescue it and was not at fault. The sun's true penumbra at ~30 m
of caster-to-receiver separation is ~0.14 m — *smaller than one texel* — so the
filter correctly took its sharp path. A hard edge on a 0.85 m grid is not
sharpness; it is aliasing. The contact tier (`rendering::contact_shadow`) could
not help either: its march is 0.6 m by design, a true contact band only.

## Decision

**1. `CASCADE_COUNT` 3 → 4, with the new cascade at the NEAR end (±64 m,
~0.03 m/texel).** The other three shift out one slot, unchanged.

**2. Cascade boxes are square on the GROUND, not in the light plane.** The V
(along-sun) half-extent is scaled by `sin(elevation)` so the ground footprint is
a square and the ground texel is isotropic.

**3. Casters standing up-sun of the covered ground get an explicit height
budget** (`CASCADE_MAX_CASTER_M`), not an implicit one. A caster `h` tall throws
its shadow `h / tan(elev)` down-sun, which is only `h · cos(elev)` of extra
**light-space** margin — bounded by caster height, never by the cascade's own
reach. That margin is added to V and the eye slides up-sun by half of it, keeping
the frustum symmetric. The shift is quantized to whole V texels, or it would
re-introduce the sub-texel crawl the snap exists to prevent.

**4. The light basis is azimuth-aligned**, not parallel-transported. Anisotropy
is only expressible if the box axes line up with the directions whose scales
differ. Transport survives solely as the degenerate-case fallback for a sun
within a hair of the local zenith — which is exactly where the anisotropy
vanishes anyway, so the frame is free to spin there without changing coverage.

**5. Craft-local (orbital) mode opts out of all of it** and keeps a light-plane
square box with two active cascades. There is no ground plane in orbit; the
receiver is the hull, and compressing V would clip the craft out of its own
shadow map. Two cascades rather than one because ±64 m can be overflowed by a
tall launch stack and, with stock Bevy CSM off, this is the only shadow there is.

Measured consequence at a 15° sun, near field: ground texel **0.85 m → 0.086 m**.

## Why this is not the fourth cascade ADR-20260722T111848Z rejected

That ADR rejected *"grow the cascades / add a fourth cascade **for the far
field**"*, on the grounds that at 100 km a 4096² cascade is 24 m/texel — below
the size of the casters that matter — and that each added cascade widens the
bias-cap conflict.

Both arguments are about **range extension**, and both still hold. This change
extends nothing: it subdivides the near end, where the reasoning inverts. The new
cascade's texel is ~0.03 m, so its bias sits at `BIAS_MIN_M` — the safe end of
the very conflict the rejection was protecting, not a new instance of it. The
far-field split (heightfield horizon term, W12) and the contact tier are
untouched, and the three-tier decision stands.

What this *does* revise is that ADR's assessment of its own mid-field tier
("50 m – 5 km · What cascades are for. **Adequate today.**"). It was not
adequate, for the projective reason above, which nine rounds of cascade tuning
had never named.

## Consequences

- One more depth pass per frame and ~67 MB more VRAM (4096² `Depth32Float`).
  Draw cost is near-free: the new cascade's box is 64 m, so almost nothing is
  inside it.
- The ortho depth range **shrinks** at a low sun. The old up-sun slack bounded a
  ground reach of `half / sin(elev)`; the box no longer has it, so the slack is
  `half · cos(elev)` — ~3.9× shorter at 15°, tightening `Depth32Float` precision
  and every bias derived from `1/(far − near)`.
- `ShadowCascadeBlock::params` gains a second texel size (`.z`, the V axis).
  `.y` deliberately stays the **U** texel so the calibrated bias model keeps the
  exact meaning it was tuned against — U is the axis with no compression. The
  PCSS kernel uses the pair to stay a circle in light space, which is an ellipse
  in map space once the box is anisotropic.
- **Binding index ≠ field name.** The new map is the *nearest* cascade, so it
  would naturally take the near→far slot and push every other map — plus AO and
  the contact tier — up by one in ten shaders and nine Rust derives. Only the
  ARGUMENT order at the `thalos::shadow` call site is ordering-significant, so
  the existing maps keep their binding numbers and cascade 0 takes each
  material's next free index. Field name still equals cascade index, so fan-out
  stays `map_N = images[N]`.

## Alternatives considered

- **Raise `SHADOW_MAP_SIZE` to 8192 instead.** Buys 2×, costs 4× the memory
  (268 MB per cascade), and does nothing about the projective smear, which was
  the larger of the two factors at a low sun.
- **Floor the PCSS filter radius hard (2–3 texels) to smear the staircase away.**
  Rejected as a placebo that trades the symptom for blur: with cascade 0 at
  0.03 m/texel the near field now resolves the sun's true penumbra, so PCSS
  produces the correct soft edge unaided. The floor kept is 1.0 texel — the
  honest "never claim an edge sharper than the map can represent" bound, which
  the bilinear contact branch already applied in practice.
- **Warped / trapezoidal shadow maps for the low-sun case.** Strictly more
  general, considerably more machinery, and the square-on-ground box captures
  most of the available win because the anisotropy here is a known analytic
  function of one angle.
