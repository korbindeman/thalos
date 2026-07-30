# INC-20260730T015528Z: a flat `GlobalAmbientLight` erases *specular* reflections on metals

- **Date:** 2026-07-30
- **Status:** diagnosed and quantified; the reflection-side unit bug is fixed, the ambient calibration is a scope decision (F7/W7)

## Symptom

A stainless hull (`metallic: 1.0`, `perceptual_roughness: 0.08`) read as a
uniform pale blue-grey in every orbital view: over the lit planet, in deep space
with no planet in frame, and over green cloudy terrain. The probe's sun-disc
glint was visible as a sharp streak *on top* of that flat tone, so the hull was
plainly receiving the reflection cubemap — it just showed nothing of the
environment's *content*.

The tell that made it confusing: the flat tone is almost exactly the material's
own `base_color` (`srgb(0.82, 0.84, 0.87)`), which at `metallic: 1.0` is the F0
tint. That reads like "the reflection is working and the environment is bright
and neutral", when in fact the environment was being bypassed.

## Mechanism

Two independent causes, both quantified in lux against the same bridge
(`LUX_PER_SPINE_FLUX = 1000`).

**1. Bevy's ambient light applies to specular, not just diffuse.**
`bevy_pbr::ambient::ambient_light()` returns

```wgsl
(diffuse_ambient + specular_ambient * specular_occlusion) * lights.ambient_color.rgb
```

with `specular_ambient = EnvBRDFApprox(specular_color, F_AB(roughness, NdotV))`
and `specular_occlusion = saturate(dot(specular_color, vec3(50.0 * 0.33)))`.

For a metal, `specular_color` **is** the base colour, so at F0 ≈ 0.85 the
occlusion term is `saturate(42) = 1.0` — the specular ambient passes at full
strength. `GlobalAmbientLight` is therefore *not* a diffuse-only fill: it lands
on mirrors as a **direction-independent** term equal to `F0 × ambient_color`.

The space-regime fill is `AMBIENT_DAY_BRIGHTNESS = 1940` lux, tinted
`srgb(0.62, 0.72, 0.95)` — blue. That is the pale blue-grey, and it is why the
hull looked identical with a planet below it and with nothing below it.

**2. The probe's orbital planet disc carried no incident flux.** Every other
painted term is in scene-flux units — sun disc `flux × SUN_DISC_GAIN`, ground
bounce `flux × SCENE_FLUX_SCALE × …`, sky from the `SkyViewLut` raymarch — but
`orbital_sample`'s planet was bare reflectance: `albedo × (lit·0.85 + 0.15)`.

The old hand-tuned `planet_color = Vec3(0.25, 0.35, 0.55)` hid this. Read as a
radiance it is 0.343; the physical form `albedo × flux / π` with a real ~0.09
albedo and `flux ≈ 10` gives 0.288. The tuned constant had been standing in for
`albedo × flux / π` all along, so replacing it with true albedo — correct in hue
— made the reflected planet **3.8× dimmer** and pushed it further under (1).

Measured, at a 200 km orbit over Thalos:

| Term | lux | vs reflection |
|---|---|---|
| Flat space ambient (applied, blended) | 975 | — |
| Planet reflection, old constant | 343 | ambient 2.8× |
| Planet reflection, true albedo, **no flux** | 91 | ambient **10.7×** |
| Planet reflection, true albedo, **with flux** | 293 | ambient 3.3× |

A third, smaller factor: at 200 km `surface_blend = 0.50` (Thalos's Kármán line
is 80 km and the probe's space fade runs to 4× it), so half the cube is the
surface-sky model — whose `sky_lux` is 0 at that altitude. That halves the
planet's weight in the cube *and* halves the flat ambient, so it hurts and helps
roughly equally; it is not the main term.

## Fix

The flux factor is fixed: `EnvParams::planet_irradiance = flux / π`, applied as
`albedo × planet_irradiance × N·L`. This is nearly brightness-neutral against
the pre-change look while being physically derived, and it makes the reflection
react to heliocentric distance and camera exposure, which a constant could not.

The ambient half is **not** fixed here. Reducing a 1940 lux flat fill changes
every surface in space, not just hulls — and `update_sun_light`'s own comment
already names the intended resolution: *"out in space it fades to the unchanged
flat stand-in, which env-map IBL at photometric intensity will retire (W7/F7)"*.
The probe now carries real photometric intensity, so that retirement is
unblocked, but it is a lighting-calibration change to make on one axis with
matched captures — not a constant to nudge while chasing a reflection.

## Recurrence tell

`just diag` reads the `planet_reflection` event
(`thalos::diagnostic::sky`). Compare its `planet_lux` against `ambient_lux` from
the `ambient_lux` event in the same session: **if the flat ambient outweighs
`planet_lux`, no cubemap content can show on a metal**, however correct the
cubemap is. `albedo_spread` separately confirms the cube itself is varying, so
the two failure modes — "cube is wrong" and "cube is correct but overpowered" —
are distinguishable from the lane alone, without a screenshot.

The generalisation worth remembering: **on this renderer a flat ambient is never
"just fill"** — it is a floor under every specular reflection in the scene, and
the shinier and more metallic the surface, the more completely it wins.
