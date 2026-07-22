# INC-0018: mip-mapped weather cube crashed the capture host via `Image::new`'s level-0-only size assert

- **Status:** Fixed
- **Date:** 2026-07-22 (observed and fixed)
- **Severity:** startup crash (headless capture unusable; game would crash the
  same way on any cloudy-body boot in a dev build)
- **Surface:** every renderer boot after the CLOUD-6 weather-cube mip chain —
  the app panicked ~20 s in, during asset preparation

## Summary

CLOUD-6 gave the 256² weather cubemap a 6-level box mip chain so far
projections can footprint-filter (single-level fetches aliased the mesoscale
coverage into ring/speckle moiré at disc scale). The image was built with
`Image::new(size, dim, data, format, usage)` and `mip_level_count` set on the
descriptor afterwards — but `Image::new`'s debug assert compares `data.len()`
against `pixel_count(size) × pixel_size`, the **level-0 volume only**, before
any descriptor edit can happen. The 2,096,640-byte mip chain against the
1,572,864-byte level-0 expectation tripped
`assertion 'left == right' failed: Pixel data, size and format have to match`
in `bevy_image-0.19.0/src/image.rs:1110` on a Compute Task Pool thread, killing
the app.

## Symptoms

- Capture host exits ~20 s after boot with exit status 101; the panic names
  `bevy_image .../image.rs:1110` with `left`/`right` byte counts where
  `right / left ≈ 4/3` (the ratio of a full mip chain to its level 0).
- `visual_capture_server.json` left with `ready: false`, later `just
  screenshot` calls time out.

## Root cause

Bevy 0.19's `Image::new` size validation is mip-unaware; it is only correct
for single-level images. Constructing a mip-mapped image through it is
impossible with the mip data attached at construction time.

## Fix

`cloud_weather_image` (`crates/rendering/render/src/clouds/images.rs`) now
builds via `Image::new_uninit`, sets `texture_descriptor.mip_level_count`,
then assigns `image.data`. Our own byte-count assert (mip-aware,
`cube_layer_mip_bytes`) stays as the real contract check. Upload order is
Bevy's default `TextureDataOrder::LayerMajor` — face0[mip0..mipN],
face1[mip0..], … — which is what `CloudWeatherField::rgba8_mip_chain`
produces.

## Prevention & recurrence signals

- The tell is the exact assert message above with a 4:3 (2-D) byte ratio —
  any future mip-chained `Image` built through `Image::new` reproduces it.
- Gotcha promoted to CLAUDE.md's Bevy 0.19 section (`Image::new` cannot carry
  mip chains).
