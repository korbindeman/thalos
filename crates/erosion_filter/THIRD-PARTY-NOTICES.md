# Third-Party Notices

`bevy_erosion_filter` is a Rust/WGSL port of work originally published as
GLSL on Shadertoy. The algorithm, mathematical structure, and parameter
defaults are upstream; the WGSL/Rust translation, Bevy integration, plane
demo, and tests are this crate's contribution.

## Upstream attribution

- **Rune Skovbo Johansen** — *Advanced Terrain Erosion Filter* (2025).
  Shadertoy: <https://www.shadertoy.com/view/wXcfWn>. Blog post:
  <https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html>.
  Source of the consolidated `EROSION_SCALE` parameterization, the corrected
  derivatives, the `slope_power` knob, the cell-scale separation, and the
  default parameter values used in this crate.

- **Fewes** — *Terrain Erosion Noise* (2023).
  Shadertoy: <https://www.shadertoy.com/view/7ljcRW>. Source of the polished
  rendering scaffolding that Rune's revision was forked from.

- **Clay John** — *Eroded Terrain Noise* (2020).
  Shadertoy: <https://www.shadertoy.com/view/MtGcWh>. Source of the original
  per-fragment erosion noise idea (gradient-aligned cell-blended sine
  stripes feeding back into the next octave's gradient).

- **Inigo Quilez** — *Noise - Gradient - 2D - Deriv*.
  Shadertoy: <https://www.shadertoy.com/view/XdXBRH>. Source of the
  analytical-derivative gradient noise (`noised`) and the 2D hash bundled
  in `assets/shaders/erosion.wgsl` and `src/cpu.rs`.

All upstream work is published under the MIT license; the relevant
copyright notice is preserved verbatim in `LICENSE-MIT` and at the head of
`assets/shaders/erosion.wgsl`.

## Why these are bundled

The erosion algorithm requires a noise function whose analytical gradient is
consistent with its height (otherwise the gully directions go wrong, which
was the motivating bug for Rune's revision). Bundling IQ's `noised` keeps
that consistency guarantee intact and lets the crate be drop-in usable
without forcing downstream users to provide a matching noise primitive.
