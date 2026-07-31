# INC-20260730T223451Z — the shadow rig raced big_space's cell recentring

**Symptom.** Shadows flickered in and out while the camera moved — foliage
worst. Static camera: stable. Nothing in the console; the frame rate was fine.

**Tell.** `origin_frame_error_m` on the shadow lane's `stability_gauge`
(`thalos::diagnostic::shadow`) reading an *exact* multiple of the 1 km
`REAL_SPACE_CELL_SIZE_M` — 1000.0, or 1414.2 for a two-axis crossing. `just diag`
reports this as `shadow_frame_desync`. It was 0.0 on ~99 % of samples overall,
but on roughly half of them during sustained camera motion.

## Mechanism

`rendering::sun_shadow`'s `PostUpdate` chain is

```
update_real_space_origin → update_sun_shadow_camera → sync_craft_shadow
```

The first reads the floating origin's `CellCoord` to publish `RealSpaceOrigin`;
the second reads *the same component on the same entity* to place the cascade
cameras. Chained and adjacent, they cannot disagree — unless something rewrites
that `CellCoord` in the gap between them.

Something could. `CellCoord::recenter_large_transforms` — the system that moves
an entity to a new grid cell once its local `Transform` outgrows the old one —
is registered **plain in `PostUpdate`** by `BigSpaceCorePlugin`
(`crates/vendor/big_space/src/plugin.rs`). It is *not* inside
`TransformSystems::Propagate`; only `BigSpacePropagationPlugin`'s `configs()`
tuple is. So the chain's `.before(TransformSystems::Propagate)` placed **no
ordering constraint on it whatsoever**, and the multithreaded executor was free
to slot it between the two systems. It took `&mut CellCoord` while both readers
took `&CellCoord`, so it could not run *concurrently* with either — it simply
landed in the gap on some frames and not others, which is why the error
alternated under motion and vanished when parked.

On an affected frame `RealSpaceOrigin` described the *previous* cell, so the
whole cascade rig — which lives outside big_space and measures from exactly that
point — was placed one full kilometre from the world it was meant to cover.
Cascade 0's half-extent is ~450 m, so every near-field receiver projected clean
outside the crisp cascade and fell through to cascade 1, whose texel is ~6×
coarser and whose foliage depth bias (`NO_NORMAL_BIAS_SCALE`, saturating near
6 m) erases tree- and shrub-scale casters outright. Next frame the race went the
other way and they came back.

**Why foliage worst:** terrain and hull shadows are large, low-frequency, and
mostly survive the demotion to a coarser cascade. Tree and shrub shadows are
exactly the caster class the coarse cascade's bias deletes, so they were the
ones that visibly blinked.

## What made this expensive to find

The doc comment on `update_real_space_origin` asserted the opposite of the truth
— "uses the exact cell origin the current frame will render against, **including
on a cell crossing**". It read as a settled invariant, so the ordering was not
where anyone looked. The constraint it described was real and required; it had
simply never been expressed as a schedule edge.

The instrument, by contrast, worked perfectly. `origin_frame_error_m` and the
`shadow_frame_desync` check both predate this incident and were built for exactly
this failure. The data was sitting in `runtime.jsonl` the whole time.

## Fix

Order the chain `.after(CellCoord::recenter_large_transforms)`.

Order against the **system**, never the enclosing
`BigSpaceSystems::RecenterLargeTransforms` set: that set also contains
`BigSpace::find_floating_origin`, which *is* inside `TransformSystems::Propagate`,
so `.after(set)` would form a cycle with the chain's existing `.before`.

## Recurrence tells

- `origin_frame_error_m` non-zero at all. It is an exact cell multiple when the
  frame is incoherent and 0.0 when it is not; there is no benign middle ground.
  `just diag` → `shadow_frame_desync`.
- The same class of bug is available to **any** consumer that lives outside
  big_space and measures from `RealSpaceOrigin`. If one is added, it inherits
  this ordering requirement — the edge is declared once, where the chain is
  registered in `rendering::sun_shadow`'s plugin.

## Also fixed in the same change (a second, slower artifact)

Same complaint, different mechanism: the cascade footprint *breathed*.

- **`craft_local` had no hysteresis.** A bare `altitude > 50 km` test gates a
  mode that parks cascades 1–2 and turns every ground shadow off in one frame. A
  camera loitering near the threshold strobed the entire ground shadow world.
  The gauge showed `active_cascades` alternating 3 → 1 → 3 between one-second
  samples. Now latched across a 40–50 km band
  (`SHADOW_CRAFT_LOCAL_EXIT_M`), with a `shadow_mode_strobe` diag check that
  counts mode *reversals* — an honest climb reads 3 → 1 and stays, contributing
  none.
- **The footprint smoother's snap bypass was set at 2×**, which ordinary flight
  trips on every doubling of camera AGL, stepping every cascade's texel — and the
  texel-proportional foliage bias with it — by 2× in one frame. That is the exact
  scale pop the smoother was introduced to remove, left reachable by a threshold
  set too low. Now 8× (`SHADOW_FOOTPRINT_SNAP_RATIO`), i.e. genuine
  discontinuities only.
- **A missed terrain-height sample fell back to the datum (0 m).** That is a step
  equal to the whole site elevation, and `terrain_h` feeds both the cascade
  centre and — through `cam_agl` — the footprint, where the demanded box moves by
  up to 6× that step. An intermittently-cold height source therefore shoved every
  cascade extent and bias around between neighbouring frames. Now holds the last
  resolved sample per body; the datum remains the floor only before any sample
  has ever landed.

The smooth resolution ramp that remains during a zoom or climb is **by design**
(`SHADOW_FOOTPRINT_SMOOTH_TAU_S`) — footprint-scaled cascades trade texel size
for coverage, and a caster too small to be represented at the resulting texel is
correctly dropped. Only the discontinuities were defects.
