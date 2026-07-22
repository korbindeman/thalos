# INC-0020 — Bevy point lights are inert in the flight view (cluster grid spans 0.5 m → 1e11 m)

**Severity:** visual (capability limit) · **Date:** 2026-07-22 · **Area:** `camera`, craft lighting, `rendering/plume`

## Symptom

An attempt to give firing engines a `PointLight` that illuminates their own
craft (the plume "lighting the vehicle" cue from the reference imagery)
produced **exactly zero pixel change** — not a subtle effect, not a dim one:
byte-identical craft pixels with the light present and absent.

## Evidence

Measured, not eyeballed — the first read of the two captures *looked* like the
lower stage was lit, and that impression was wrong:

```
lower stage  OFF mean 92.3   ON mean 92.3   delta +0.0 (+0%)
upper stage  OFF mean 98.3   ON mean 98.3   delta -0.0 (control)
```

- Raising the light to an absurd **1.257e10 lumens** (range 66 m, 18 m from the
  hull) still gave `delta +0.0`. A tuning problem cannot survive four orders of
  magnitude; this is binary.
- Runtime log confirmed the CPU side every frame:
  `PLUME LIGHT set intensity=1.257e10 range=66.3 pos_y=-18.4`, query-miss count 0.
  The light entity exists, is found, and is written.
- Craft materials do support point lights: both `ship_part.wgsl` and
  `shadowed_standard.wgsl` call `pbr_functions::apply_pbr_lighting`, which
  includes Bevy's clustered point-light loop.

## Hypotheses considered

1. **Light never spawned / query mismatch** → rejected by the runtime log
   (0 misses, intensity written every frame).
2. **Craft shaders ignore point lights** (custom lighting path) → rejected:
   both call `apply_pbr_lighting`.
3. **Visibility gate** — light parented under a plume that starts
   `Visibility::Hidden` → rejected: the sibling plume billboard on the same
   parent renders correctly in the same frames.
4. **`RenderLayers` mismatch** — `propagate_view_render_layers` force-overwrites
   `RenderLayers` on *every descendant of the craft root* each frame with
   `[SHIP_LAYER, SHADOW_CASTER_LAYER]`, clobbering the light's explicit
   `[0, SHIP_LAYER]`. **Real, and a genuine trap for anything placed in the craft
   subtree** — but not sufficient here: `SHIP_LAYER` survives the overwrite and
   the ship camera renders `[0, SHIP_LAYER]`.
5. **Clustered-lighting depth range** → confirmed, see below.

## Root cause

The ship camera (`camera.rs`) is configured `near: 0.5`, `far: 1.0e11` — a
**2×10¹¹ depth ratio**, needed because one view spans a cockpit interior and a
heliocentric orbit.

Bevy's clustered forward lighting subdivides the view frustum into a fixed
Z-slice grid across `[near, far]`. At this ratio the slicing is degenerate for
anything at human scale: the entire 0.5 m – few-km band where craft geometry
lives collapses into the low end of the grid, and a point light with a 66 m
range cannot be assigned to clusters that the shaded fragments actually sample.

The light is correctly specified and simply never contributes. This is a
**structural property of the flight view's depth range**, not a bug in the plume
code — no intensity, range, or layer configuration fixes it.

It is a concrete instance of the "two lighting universes" debt in
`docs/graphics_fidelity.md` §3: terrain/vegetation/water shade through the
`thalos::lighting` spine with analytic light terms, while crafts and structures
go through Bevy's stock PBR — and stock PBR's *clustered* machinery is the part
that this project's camera cannot satisfy.

## Fix

None applied — the non-functional `PointLight` was **backed out** rather than
left in the tree as dead code (CLAUDE.md: delete dead code on contact).

The correct fix is an **analytic plume light term in the lighting spine**,
alongside the existing analytic sun/moonlight terms, rather than a Bevy
clustered light: publish per-craft plume light state (position, colour,
luminous power) into `SceneLighting`, and evaluate it in `ship_part.wgsl` /
`shadowed_standard.wgsl`. That is F-series unification work and wants an ADR;
filed as its own backlog row.

## Prevention

- **Do not add `PointLight` / `SpotLight` to anything rendered by the flight
  camera** and expect it to work. Bevy's clustered lighting is unusable at this
  view's depth ratio. Local light sources (engines, running lights, explosions,
  cockpit fill) must be analytic terms in the spine. The shipyard editor is the
  exception — it renders on its own camera with a sane near/far, which is why
  its `PointLight` key/fill lights work and can mislead.
- **Anything parented into the craft subtree has its `RenderLayers` overwritten
  every frame** by `view::propagate_view_render_layers`. Entities needing their
  own layers (notably lights, which is why `SunLight` and the moonlight light are
  separate root entities) must not live under the craft root.
- **Recurrence tell:** a light whose effect is *exactly* zero rather than weak.
  Measure the delta numerically before tuning — a 0.0 delta means "not wired",
  and no amount of intensity will move it. Two screenshots viewed minutes apart
  are not a comparison; crop the same region and diff the means.
