# INC-20260810T211819Z: Kòrsou terrain shadowed itself in stripes

## Symptom

At a roughly 10 m AGL, 24 mm Caracasbaai view, the sandy foreground broke into
broad parallel bands whose screen-space spacing widened toward the camera. The
deterministic `reference-caracasbaai-close-coast` capture reproduced the same
pattern.

## Mechanism

Kòrsou overrode Bevy's directional-light shadow normal bias from the engine
default of `1.8` to `0.55`. That offset was too small for the terrain mesh and
the 22 km cascaded-shadow range, so adjacent terrain samples repeatedly failed
their own shadow-map depth test. Perspective turned that regular self-shadowing
into the widening horizontal bands.

An unlit material removed the bands. Removing both mesh and detail normals did
not; disabling shadow reception did. Restoring real normals and shadow reception
while changing only the normal bias to `1.8` removed them. Those probes ruled
out vertex colour, the detail normal, DEM terraces, and ordinary diffuse
lighting as the direct cause.

## Fix and recurrence tell

The sun now uses `DirectionalLight::DEFAULT_SHADOW_NORMAL_BIAS`. Terrain still
receives and casts real shadows; only the receiver offset returned to Bevy's
scene-scaled default.

The recurrence checks are:

- `sun_uses_directional_shadow_normal_bias` protects the light configuration;
- the matched `reference-caracasbaai-close-coast` headless capture checks the
  grazing-angle appearance;
- `reference-boka-tabla-cliffs` checks that steep relief and foliage shadows
  remain after the bias change.
