# bevy_erosion_filter

A GPU-friendly per-fragment erosion filter for Bevy. Adds branched gully
detail to any heightfield analytically — no neighbour reads, no simulation,
trivially parallel, evaluable in chunks. Suitable as a sub-pixel detail
layer on baked terrain or as the high-frequency pass of a per-fragment
procedural surface.

This is a Rust/WGSL port of Rune Skovbo Johansen's *Advanced Terrain
Erosion Filter*
([Shadertoy](https://www.shadertoy.com/view/wXcfWn) ·
[blog](https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html)),
itself an evolution of work by Fewes, Clay John, and Inigo Quilez. See
[`THIRD-PARTY-NOTICES.md`](./THIRD-PARTY-NOTICES.md).

## What it does

Given a height function and its analytical gradient, the filter overlays
multi-octave gully patterns aligned to the steepest-descent direction.
Each octave's slope feeds the next, so smaller gullies branch off larger
ones. The result *looks* like erosion at any zoom level — but it's per-point
fakery, not a fluid solve, so neighbouring gullies don't connect into real
drainage networks. It's a detail layer, not a hydrology system.

## Install

```toml
[dependencies]
bevy = "0.18"
bevy_erosion_filter = "0.1"
```

Add the plugin to your `App`:

```rust
use bevy::prelude::*;
use bevy_erosion_filter::ErosionFilterPlugin;

App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(ErosionFilterPlugin)
    .run();
```

## Use it from your own shader

```wgsl
#import bevy_erosion_filter::erosion::{
    apply_erosion, erosion, fbm, gullies,
    ErosionParams, erosion_params_default,
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Your own height function returning vec3(height, d/dx, d/dy):
    let base = fbm(in.uv, 3.0, 4, 2.0, 0.5);
    let eroded = apply_erosion(in.uv, base, erosion_params_default());

    let height = eroded.x;
    let normal = normalize(vec3<f32>(-eroded.y, -eroded.z, 1.0));
    // … shade with `height` and `normal`.
    return vec4<f32>(height, normal.xy * 0.5 + 0.5, 1.0);
}
```

The library exposes:

| Symbol | What it does |
|---|---|
| `gullies(p, slope) -> vec3` | Single octave of gully noise. `vec3(h, d/dx, d/dy)`, amplitude bounded by 1. |
| `erosion(p, base, params) -> vec3` | Multi-octave delta to add to your base height+slope. |
| `apply_erosion(p, base, params) -> vec3` | Convenience wrapper: applies delta + height-offset bias. |
| `fbm(p, freq, oct, lac, gain) -> vec3` | IQ analytical-derivative fBm, suitable as the base height. |
| `noised(p) -> vec3` | IQ gradient noise with derivatives (the building block of `fbm`). |
| `ErosionParams` | Struct of all eight tunable parameters. |
| `erosion_params_default()` | Defaults matching the Shadertoy reference. |

## Use it from Rust (CPU)

The same algorithm is mirrored in pure Rust under `bevy_erosion_filter::cpu`,
useful for offline baking and parity tests:

```rust
use bevy_erosion_filter::cpu;
use glam::Vec2;

let p = Vec2::new(0.42, 0.31);
let base = cpu::fbm(p, 3.0, 4, 2.0, 0.5);
let eroded = cpu::apply_erosion(p, base, &cpu::ErosionParams::default());
println!("h = {}, ∂h/∂x = {}, ∂h/∂y = {}", eroded.x, eroded.y, eroded.z);
```

## Demo

```sh
cargo run --example plane_demo -p bevy_erosion_filter --release
```

Renders a full-screen 2D quad with the eroded heightmap, shaded with a
simple altitude colormap (water → sand → grass → rock → snow). egui
sliders on the right edit every parameter live.

## Parameters

The defaults match the Shadertoy reference and are a good starting point
for any heightmap. The most important parameter to dial in first is
**`scale`** — try `mountain_width / 5..10` in the same units as your
heightmap input.

| Parameter | Default | Range | Notes |
|---|---|---|---|
| `scale` | `0.0833` | — | Overall horizontal scale. |
| `strength` | `0.16` | `0..0.5` | Gully depth, relative to scale. |
| `slope_power` | `0.6` | `0.1..2.0` | `1.0` = smooth peaks, `0.5` = sharper. |
| `cell_scale` | `1.0` | `0.25..4.0` | Smaller = grainier; larger = curvier. |
| `octaves` | `5` | `1..8` | Per-fragment cost is roughly linear in octaves. |
| `gain` | `0.5` | `0.1..0.9` | Amplitude multiplier per octave. |
| `lacunarity` | `2.0` | `1.5..3.0` | Frequency multiplier per octave. |
| `height_offset` | `-0.5` | `-1..1` | `-1` = pure carve, `+1` = pure raise. |

## Tips

- **Suppress erosion in flat regions you don't want eroded** (e.g. below
  sea level) by multiplying `strength` by your own mask before calling
  `apply_erosion`. The Shadertoy reference does this with a smoothstep
  around the waterline.
- **The base height function's gradient must be analytical**, not
  finite-difference. The filter relies on the gradient being consistent
  with the height for the gullies to actually run downhill. If you're
  passing in a baked heightmap, finite-differencing the texture is fine
  in practice as long as the texture resolution is high enough.
- **Per-fragment cost** scales as `octaves × 16 cell evaluations × ~1 cos + 1 sin + a few muls`. On modest GPUs the default 5 octaves cost a few hundred ALU per fragment. Drop octaves if you're shadow-mapping the surface — shadow passes re-evaluate.
- **Sphere/cubemap use** is supported in principle (the filter is purely
  local) but seam handling at cube-face boundaries is on you. A small
  guard band on each face's UV input is the easy fix.

## License

MIT. See [`LICENSE-MIT`](./LICENSE-MIT) and
[`THIRD-PARTY-NOTICES.md`](./THIRD-PARTY-NOTICES.md).
