# INC-0020 — Repurposing a uniform slot left a stale reader that erased the plume in vacuum

**Severity:** visual · **Date:** 2026-07-22 · **Area:** `rendering/plume`, `assets/shaders/plume.wgsl`

## Symptom

While iterating on the plume fidelity model, the vacuum plume disappeared
**entirely** — a firing engine rendered nothing at all — while the sea-level
plume still drew. The change that "caused" it was raising the propellant
radiance constant, i.e. making the plume *brighter* made it vanish.

Secondary symptom, present for several iterations before anyone noticed: the
sea-level plume was far dimmer than the arithmetic predicted, so the radiance
constant kept getting pushed up (3 → 7 → …) chasing a target it could not reach.

## Evidence

- `just screenshot-cold plume` with `THALOS_PLUME_PRESSURE=0` → craft renders,
  plume absent. Deterministic across reruns and across `THALOS_SCREENSHOT_WARMUP`
  200, so not a spool-up/timing race.
- Same binary, `THALOS_PLUME_PRESSURE=101325` → plume renders. So the material,
  mesh, visibility gate, and shader pipeline were all fine.
- Capture log showed `plume probe: throttle=1.0 ambient_pressure=0 Pa` — the
  engine *was* commanded to fire.
- No shader/pipeline validation errors in the log (ruling out the BL-20 class,
  where a fatal pipeline error still exits zero).

## Hypotheses considered

1. **WGSL compile failure** → rejected: the same shader rendered the sea-level
   capture in the same binary.
2. **Ignition transient not spooled at capture time** → rejected: raising warmup
   to 200 frames changed nothing, and the failure was pressure-dependent, not
   time-dependent.
3. **Geometry degenerate in vacuum** (zero-length plume, inverted cone) →
   rejected by hand-evaluating the resolver: vacuum gives `lip = 1.0`,
   `tan_theta = 0.19`, `length = 36.8·R0` — all finite and larger than the
   sea-level case.
4. **A term that is zero exactly when ambient pressure is zero multiplies the
   whole result** → confirmed by reading the fragment's final `gain` expression.

Hypothesis 4 is the only one that explains *both* the vacuum blackout and the
long-running sea-level dimness with a single mechanism.

## Root cause

`PlumeParams.anim.w` originally carried a "radiance trim" that was always `1.0`,
and the fragment folded it into the final brightness:

```wgsl
let gain = plume.core_color.a * plume.anim.w * plume.anim.z * mix(...);
```

Adding entrainment cooling needed a new scalar. `anim.w` was unused in practice,
so it was **repurposed as the entrainment rate** — and the new consumer was
added without auditing the slot's existing readers. The `gain` multiply survived.

Entrainment rate is `lerp(0.0, 0.016, atmo)`:

- **vacuum** → `atmo = 0` → rate `0.0` → `gain × 0` → plume mathematically erased.
- **sea level** → rate `0.016` → `gain × 0.016` → ~60× under-bright, which is
  what the radiance constant was fruitlessly chasing.

The value was semantically valid in both places, so nothing failed loudly: no
NaN, no validation error, no panic. It was silently multiplied into a slot that
had no business reading it.

## Fix

Remove the stale multiply and pin the slot's meaning at both ends of the
contract:

- `assets/shaders/plume.wgsl` — `gain` no longer reads `anim.w`; a comment states
  that `anim.w` is the entrainment rate and must not appear there.
- The WGSL `PlumeParams` struct comment for `anim` said `w = density scale`
  (already stale before this change) — corrected to the real meaning.
- Rust-side doc comment on `PlumeParams::anim` names the field's units.

Radiance constants were then re-tuned against the now-correct gain
(methalox 7.0 → 5.5).

## Prevention

**Repurposing a packed-uniform slot is a rename, not an edit.** These structs
pack unrelated scalars into `vec4` lanes addressed positionally (`anim.w`), so
the compiler cannot catch a reader that still expects the old meaning, and a
plausible-looking float will silently flow into unrelated math.

When changing what a packed lane means:

1. `rg` for every read of that lane in the shader **and** any CPU mirror before
   writing the new producer.
2. Update the field's doc comment on **both** sides in the same change — the
   comment is the only type system these lanes have.
3. Prefer adding a new lane to reusing one whose old meaning was "unused
   constant"; an always-`1.0` trim is indistinguishable from a live multiplier
   until the day its replacement is legitimately `0.0`.

**Recurrence tell:** an effect that vanishes or goes wildly out of scale in
*one regime only*, where that regime is exactly the one that drives some
newly-added parameter to `0.0` or `1.0`. Suspect a shared lane before suspecting
geometry, and hand-evaluate the final composite expression rather than the
intermediate terms.

A related trap: this bug made every brightness estimate wrong for ~5 iterations,
and the estimates were "corrected" by tuning rather than by re-deriving. When a
tuned constant has to move by an order of magnitude to hit an analytically
predicted target, the model is wrong, not the constant.
