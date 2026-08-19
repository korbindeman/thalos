# INC-20260816T065328Z: scaled 3D vanished under a magenta HUD blit

## Symptom

Laptop render scale (3D at 0.5×, HUD at native HiDPI) showed a solid magenta
window with working freecam and waypoint panels. Kòrsou still knew the camera
was at Caracasbaai, 2 m AGL.

## Mechanism

The overlay UI camera used `ClearColorConfig::None` on its main target, which
does not clear. Metal leaves that texture uninitialized magenta with alpha 1.
Bevy then upscale-blits the full 2D target over the 3D swapchain write. HUD
nodes looked fine because they were actually drawn; every undrawn pixel
replaced the scene.

The first read of the screenshot was "the 3D upscale never ran." That is
indistinguishable until the overlay target is transparent: a missing 3D blit
and an opaque magenta overlay are the same picture.

## Fix and recurrence tell

The overlay camera clears to `Color::NONE` and writes the swapchain with
alpha blending, without clearing the 3D blit. `overlay_ui_camera_clears_transparent_and_blends`
locks that contract.

The tell is a magenta world with a correct HUD while render scale is below 1.
A black world would mean the 3D camera never presented; magenta with HUD is
the uninitialized overlay blit.
