# INC-20260817T050612Z: Kòrsou cloud resize kept a stale material view

## Symptom

Kòrsou briefly showed the real island scene at startup, then became uniformly
black. The deterministic 1600×900 headless capture reproduced the black frame,
while a 1920×1080 capture rendered the scene and clouds correctly.

## Mechanism

The shared cloud images start at 1280×720. A 1920×1080 viewport uses that exact
2/3-scale cloud extent, but a 1600×900 viewport resizes the cloud images to
1072×600 after the first frame.

The compute pass rebuilds its bind group every frame, so it wrote into the new
GPU image view. Kòrsou's `CloudLayerMaterial` did not change when the image asset
resized, so Bevy kept its cached bind group pointed at the old zero-filled GPU
view. Zero in the cloud texture's alpha channel means zero transmittance; the
premultiplied compositor therefore covered the scene with opaque black. The
startup flash ended when the cloud material pipeline became ready and began
drawing that stale texture.

## Evidence and ruled-out candidates

- Headless capture uses identity render scale, so Laptop's 0.5× render scale was
  not required for the failure.
- Making only the cloud composite discard restored the complete scene.
- Both cloud compute pipelines reached `Update`, and all extracted bind groups
  and cloud resources were present.
- The compute texture's metadata row remained zero at 1600×900, but the same
  code rendered clouds at the initial 1280×720 target extent.

## Fix and recurrence tell

Kòrsou now tracks the asset identity and extent of scene depth, cloud colour,
and cloud distance. When any sampled image is replaced or resized, it marks the
cloud material modified before Bevy's asset-event extraction, forcing the
material bind group to use the current GPU views.

`resized_cloud_target_invalidates_the_composite_material` locks the invalidation
contract. The headless recurrence check is a non-native capture size such as
1600×900: if 1920×1080 works but another size turns black after a valid startup
frame, inspect cached material image views before changing cloud density or
camera composition.
