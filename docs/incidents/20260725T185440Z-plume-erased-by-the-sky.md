# INC-20260725T185440Z-plume-erased-by-the-sky: a fullscreen composite sorted itself in front of the world

- **Date:** 2026-07-25 · **Surface:** any camera pitched **above** the local horizontal at a firing engine — `just game launch` under ascent, reproduced headlessly by `just screenshot plume-skyline`

## Symptom

An ascending rocket's exhaust column was **missing wherever the sky was behind
it, and present wherever terrain was**. The cut fell exactly on the skyline. The
craft itself was unaffected: hull sharp, correctly hazed, in the same pixels the
plume had vanished from.

Two things made it read as a shading bug rather than an ordering one. The plume
is an additive HDR emitter, so "invisible against a bright background, visible
against a dark one" is a perfectly good description of a *contrast* failure —
and the boundary between the two backgrounds is the same skyline either way.
And the effect was intermittent across sessions in a way nobody had tied to a
variable: it depends on **camera pitch**, not on the plume, the craft, or the
scene.

## Root cause

`BodySkyMaterial` (and its ocean/cloud siblings) is a fullscreen composite —
a 2×2 quad whose vertex shader writes NDC directly, `depth_compare = Always`,
premultiplied, clipping its raymarch against a *copy* of the main pass's depth
attachment. On a pixel with opaque geometry it lays down thin aerial
perspective; on a pixel with none it lays down opaque sky. That is correct, and
it is why the hull survived and the plume did not: the plume is transparent, so
it never enters the depth buffer, so the atmosphere pass has no idea it is
there and treats those pixels as empty sky.

Which leaves the only question that matters: **why was the atmosphere painted
after the plume at all?** It rides `Transparent3d` like any other see-through
mesh, and that phase is sorted by the view-space depth of each mesh's centre.
The composites are parented to the body, so their sort point is the planet
centre, and `clouds/composite.rs` justified its ordering on exactly that:

> A kilometre is negligible beside the planet-centre distance, so ordinary
> surface transparents remain in front of both passes.

The sort key is not the *distance to* the planet centre. It is the **projection
of that offset onto the view axis** (`ViewRangefinder3d::distance`, a dot with
row 2 of the view matrix). Standing on the surface, the planet centre is
straight down — perpendicular to a level view — so the key collapses through
zero, and for any camera pitched **above** the horizontal it goes *positive*:
the centre is behind the eye. Bevy sorts that phase ascending (most negative =
farthest = drawn first), so a positive key makes the atmosphere the last thing
drawn in the frame, over every transparent already there.

The magnitudes are not marginal. Pitch up 5° on a 6 371 km body and the key is
+5.5 × 10⁵ against the plume's −3 × 10², so the flip is total and stable, not a
flicker. Pitch back down and the key returns to −R·sin(pitch), the composites
sort to the back again, and the plume returns — which is precisely the "works on
my machine" pattern this had.

Two adjacent notes:

- Fullscreen composites were not the only pass on a geometric key. The stars and
  galaxy meshes sit at the render origin, so *their* key is ≈ 0 — meaning the
  celestial backdrop and the atmosphere were being ordered against each other by
  camera orientation as well. The per-pixel star crush in `body_sky.wgsl` only
  works when the air is drawn over the stars, so it was silently a coin flip.
- Same field, opposite lesson to INC-20260725T184654Z: there, `depth_bias` did
  nothing because sort order is irrelevant among opaque geometry. Here, sort
  order is the *only* thing that decides the result.

## Fix

`thalos_body_render::composite_order` pins the stack instead of deriving it.
Each fullscreen pass claims a slot — celestial backdrop, atmosphere, ocean,
clouds — biased far enough (−2 × 10⁹, spaced 65 536) that no camera orientation
can reorder them or lift one past ordinary world transparency, with the slot
spacing chosen to stay exactly representable in `f32` at that magnitude. Their
order *relative to each other* is unchanged. The standing rule lives in that
module: **a fullscreen composite must claim a slot — never bias 0.**

`ScreenshotPreset::PlumeSkyline` (`just screenshot plume-skyline`) is the
regression probe: a firing engine at 1.5 km AGL, camera 18° *below* the craft so
the view pitches above the horizontal, with the far ground still in the bottom of
the frame. Its focus context poles on the local radial rather than the ship's
nose, so the pitch that matters is the one the preset controls.

## Recurrence signal

A transparent effect that **disappears against sky and survives against
geometry**, with the cut on the skyline — especially if it comes back when the
camera pitches down. Do not start from the shading; check what the composites
sorted to. The fast discriminator is pitch, not brightness: a contrast failure
tracks the background, an ordering failure tracks the camera's elevation angle
and flips completely within a degree of the horizontal.

More generally: **a fullscreen pass has no position, so any ordering rule stated
in terms of where it "is" is a rule about where the camera is pointed.** If a
pass covers the frame, its place in the stack must be declared, not measured.
