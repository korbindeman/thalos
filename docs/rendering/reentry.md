# Reentry shock layer

The shock-heated air standing off a vehicle's windward side during atmospheric
entry. One of the two built consumers of the `FlowSignals` boundary
(`flow_effects.md`); the plume is the other.

## Status

**Built 2026-07-29** — `rendering::reentry` + `assets/shaders/reentry.wgsl`.
Screenshot-verified via `just screenshot reentry`. Unverified: the live entry
sweep (a real trajectory from thin air through peak heating and out), which is a
user play session, and the `plume-skyline` composite-ordering framing.

## What this is, and what it is not

Three things get conflated under "reentry effects". This is the first only:

1. **The shock layer** (here) — attached, zero-memory, a function of the *current*
   freestream. Present exactly while the vehicle is fast in air.
2. **Hull glow** — not an effect but an emissive term on the hull material, driven
   by an integrated per-part heat state. Belongs in shading.
3. **The ablation wake** — shed *into* the air, so it needs memory and therefore
   the ribbon primitive.

Separating them is most of the work: (1) is the part whose geometry is pinned to
the airflow, and the part the existing emission model already covers.

## The model

The plume's emission chain applied to a different geometry — deliberately, because
both are hot gas whose brightness must *follow from* the physics rather than from
an authored fade:

```
cos_t      how windward a point is (1 = stagnation point, 0 = the flank)
delta      standoff, smallest at the stagnation point, sweeping back as the
           shock goes oblique:  delta0·(1 + growth·(1 − cos_t))
u          position across the shell, 0 at the wall, 1 at the shock front
rho  = 4u(1−u)                compact support at BOTH ends
T    = T_stag · aft · front   hottest at the stagnation point, near the shock
S    = exp(−W·(1/T − 1))      visible-band emission, Wien side
L    = S · (1 − exp(−tau))    emission through an absorbing shell
```

Driven entirely from `FlowSignals`: **heat flux** sets radiance, **stagnation
temperature** sets colour and the Wien falloff, **density** sets optical depth, and
**Mach** sets both the fade-in and the standoff.

### Two frames, and why the shell lives in the craft's

**The layer is aimed at the airflow but fitted to the hull.** An entering vehicle
flies at high angle of attack — a capsule blunt-end-first, a lifting body
belly-first — so a cap keyed to the craft's forward axis sits in the wrong place
for the whole entry. But the *body* the shock stands off from is the hull, which
is craft-aligned.

So the shell entity is a plain child of the craft with an **identity transform**
(which also keeps it in the craft's BigSpace cell for free), and the freestream
arrives as a **uniform in craft-local axes**. The shader works entirely in the
craft's frame: the body is its bounding ellipsoid, `flow.xyz` says where the wind
comes from. Nothing is counter-rotated and no query touches the craft's
`Transform` — which also sidesteps the B0001 query-conflict class that
`reflection_probe` hit.

## Four things that are load-bearing

Each replaced a version with a visible failure mode:

- **The body is an ellipsoid, not a sphere.** A bounding *sphere* on a 40 m rocket
  is a 20 m ball, so a shell hugging it hangs metres out in empty space along every
  axis. The first version did exactly that and rendered a saturated white blob
  filling the frame with the craft nowhere in sight. `craft_half_extents_m`
  collapses to the sphere for a capsule and stays tight for a rocket.
- **The Wien reference equals the real-gas temperature cap.** Ideal gas gives
  ~36 000 K at Mach 25; real air does not get there, because above ~2 500 K the
  energy goes into vibrational excitation, dissociation and ionisation instead of
  temperature, and shock-layer gas measures out around 10 000–11 000 K across a
  wide entry range. Capping at 11 000 K **and** normalising by the same number
  keeps normalized temperature in `0..=1`, so emission cannot exceed 1. Without
  both halves every entry above about Mach 8 blows out to the same flat white and
  the colour ramp stops carrying any information — Mach 10 and Mach 30 look
  identical.
- **Brightness rides on heat flux, never on speed.** `q = K·sqrt(ρ/R_n)·v³` is the
  only quantity that knows the difference between orbital speed in vacuum and
  orbital speed in air. Keying brightness to speed lights a fireball on every
  orbiting vehicle; there is a test pinning this, and another pinning the *sign* of
  the `R_n` dependence (a blunter nose is **cooler** — the reason heat shields are
  round).
- **The march stops at the body.** The shell is an annulus, so a ray aimed at the
  craft crosses it twice; integrating the far crossing paints the leeward shock
  over the hull that should hide it. Clamping at the body surface both fixes that
  and makes the marched interval tight, because the annulus crossing *is* the
  interval.

Also: compact support at **both** ends of the shell (`4u(1−u)`), so the layer
feathers into the hull and into the freestream with no mask — a top-hat shows a
hard bright line on each surface. And the windward weight tops out at `cos = 0.75`
rather than near 1, so full brightness concentrates near the stagnation point; a
wider setting lights the whole windward hemisphere evenly and reads as a glowing
ball rather than a shock.

## Constraint on the proxy hull

The hull is the craft's bounding ellipsoid grown by the standoff at its **worst
case** (`cos_t = −1`). It must stay an over-estimate: a bound that cut inside the
layer would clip a still-emitting shell, which is the defect class of
INC-20260724T235437Z-plume-ended-on-a-lit-rim. A test mirrors the shader's
`WRAP_LO` and asserts the hull is sized past where emission reaches — the two
constants live in different files, so drift between them is exactly what it
catches.

## Known gaps

- **The nose radius is a stand-in.** A real `R_n` is a property of the windward
  geometry — a capsule's heat shield, a wing leading edge — and the shipyard does
  not publish one. `NOSE_RADIUS_FRACTION` takes a fraction of the smallest
  cross-section instead, which at least scales with the vehicle and keeps blunt
  bodies cooler for the right reason. That constant is what an authored value
  replaces.
- **No wake.** The trailing ablation glow needs the ribbon primitive.
- **No hull heating.** The vehicle is not lit by its own shock layer, and its
  surface does not glow. Both want the analytic-light seam BL-40 is opening for the
  plume.
- **A camera inside the proxy hull** loses the part of the layer behind the near
  plane, same as the plume. Clamping the march against scene depth fixes both.

## Screenshot preset

`just screenshot reentry` boots the atmospheric landing approach and drives
`FlowDebugOverride` to a peak-heating freestream — actually flying an entry is
neither deterministic nor reachable headlessly. The real atmospheric placement
is load-bearing: overrides cannot manufacture air in orbit.
`THALOS_REENTRY_DENSITY` (kg/m³) and `THALOS_REENTRY_SPEED` (m/s) scrub the entry
point.

The wind is put on the craft's **belly** (`−Z` local, the dorsal convention the
gear uses), which is the attitude a lifting body actually enters in — and the
framing that would expose a shell wrongly keyed to the craft's nose axis, rather
than the degenerate nose-on case that hides it.
