# INC-20260729T053218Z-craft-lighting-crossed-unit-scope-and-frame-boundaries

- **Status:** Fixed in code; runtime verification intentionally deferred
- **Date:** 2026-07-29
- **Severity:** visual
- **Surface:** metallic flight-craft hulls in every ship-view scene

## Summary

The spacecraft could read as though it belonged to a different lighting setup
even though its direct sun, exposure, and shadows shared the standard Bevy path
with tile terrain. Two independent inputs crossed the wrong boundary:

1. The hull is mirror-metal and therefore reads mostly from its environment
   map, but that map stayed in Thalos scene-flux units while Bevy's environment
   intensity is a cd/m² multiplier. Direct light used the canonical
   `LUX_PER_SPINE_FLUX = 1000` bridge; reflected light used `1`.
2. The procedural panel/rivet normal frame treated local part `+Y` as world
   `+Y`, so its cap mask and perturbed normals did not rotate with a flying
   craft.

The first error was noted but left open in
`INC-20260724T204059Z-standard-path-never-set-camera-exposure`; the second had
been present since the material first shipped and was even described by its
stale “parts are not rotated” comment.

## Root cause

`reflection_probe.rs` painted radiance around `0.1..30` in the same scene-flux
family used by the custom lighting spine, then attached
`GeneratedEnvironmentMapLight { intensity: 1.0 }` to the camera. Bevy 0.19
defines that intensity as the scale that brings cubemap samples into cd/m².
The directional path converted the same source scale through
`LUX_PER_SPINE_FLUX`; the hull's dominant specular source did not.

The camera attachment was a second latent bug. The cubemap is authored from the
selected craft's sun and planet directions. Making it photometric while leaving
it camera-wide would apply that craft-relative planet disc to every terrain and
structure fragment in the view, including terrain reflecting the planet it is
part of. The old near-zero intensity merely hid the wrong ownership.

Separately, `ship_part.wgsl` rebuilt its procedural TBN from a literal
`vec3(0, 1, 0)`. Flight attitude rotates the mesh geometry but not that literal,
so the shader classified the wrong faces as caps and bent panel normals in a
world-fixed frame.

## Fix

- `PROBE_INTENSITY` is derived directly from `LUX_PER_SPINE_FLUX`; there is no
  independently tunable unit bridge left.
- The realtime environment filter moved to a detached producer that cannot
  light scene meshes. A `LightProbe` child of each selected `PlayerShip` shares
  only the producer's filtered specular handle, keeping ship-authored radiance
  at the craft boundary.
- The local consumer uses a separate black diffuse cubemap. The existing
  `GlobalAmbientLight` projection remains the one diffuse-sky authority, while
  Bevy's producer retains its real generated diffuse storage target. This both
  prevents double-counted sky and avoids redirecting the compute filter into an
  invalid 1×1/non-storage texture.
- The local probe's child rotation cancels the craft root rotation before
  transform propagation. Bevy composes a local probe's `GlobalTransform` with
  `EnvironmentMapLight::rotation` into its sampling frame, so an identity map
  rotation plus an identity child transform would make the painted sun,
  horizon, and planet rotate with craft attitude.
- The hull shader transforms local `+Y` through Bevy's mesh-instance normal
  matrix via `mesh_normal_local_to_world`, so cap classification and the
  procedural TBN follow the actual part transform.

## Recurrence signals

- A scene-flux cubemap must never use a free-standing Bevy environment
  intensity. Its conversion derives from `LUX_PER_SPINE_FLUX`.
- A cubemap authored at a craft position is a local probe, not a view-global
  environment.
- A world-authored cubemap remains world-aligned when its probe volume follows
  a rotating craft.
- Enabling diffuse irradiance on the craft probe requires retiring or reducing
  the matching `GlobalAmbientLight` sky term in the same change.
- A material shader may not construct a world-space detail frame from a literal
  local axis. Transform the axis through the mesh instance or consume a real
  world tangent.

No capture was run at the user's request. The backlog remains at `verify` until
the user judges a flight scene; code verification is recorded in the change
report.
