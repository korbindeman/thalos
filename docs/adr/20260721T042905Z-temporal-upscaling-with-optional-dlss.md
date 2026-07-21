# ADR-20260721T042905Z: Temporal upscaling is one renderer substrate; DLSS is an optional backend

- **Status:** Accepted
- **Date:** 2026-07-21

## Context

Thalos targets photographic planet-scale rendering and is expensive at a 4K
output: terrain, vegetation, atmosphere, clouds, AO, shadows, and post effects
all spend work per pixel. Rendering fewer scene pixels and temporally
reconstructing the 4K output can therefore buy more fidelity inside the same GPU
budget, but only when the frame is actually GPU/pixel-bound.

The current main view uses SMAA by default (or optional MSAA) and has no
whole-scene temporal contract. Bevy supplies a TAA path, but Thalos's custom
terrain/material passes, `big_space` floating origin, ship/map cameras, camera
teleports, and body-fixed view anchoring make correct motion vectors and history
resets an engine problem rather than a camera-component toggle. NVIDIA likewise
requires accurate motion vectors, depth, projection jitter, exposure, mip bias,
and reset signals for DLSS Super Resolution.

The integration surface is now credible: Bevy maintains
[`dlss_wgpu`](https://github.com/bevyengine/dlss_wgpu), whose v4 line targets the
same `wgpu 29` used by Thalos. It calls the DLSS SDK through Vulkan directly, but
requires the separately downloaded NVIDIA and Vulkan SDKs at build time and the
DLSS runtime library plus license text for distribution. It does not make DLSS
portable to non-RTX hardware or non-Vulkan backends.

## Decision

Thalos will pursue whole-scene temporal reconstruction and 4K upscaling, with
DLSS Super Resolution / DLAA as an **optional projection of one canonical
renderer substrate**, not as a separate NVIDIA renderer path.

1. W13 first establishes one per-view temporal input contract in render
   mechanism code: HDR scene color, depth, body-fixed motion vectors for every
   participating surface/pass, projection jitter, exposure, render/output
   extents, and a shared history-reset epoch for cuts, teleports, origin/view
   changes, scenario loads, and resolution changes.
2. A native-resolution temporal resolve validates motion, rejection, and reset
   behavior before upscaling is trusted. Native SMAA remains the universal,
   non-temporal fallback.
3. The main 3-D ship view renders scene content at the selected internal extent;
   HUD and UI composite at output resolution after temporal reconstruction. The
   map view stays on its native path initially, and switching active views resets
   temporal history.
4. W13-DLSS then integrates `dlss_wgpu` directly, pinned to the Thalos `wgpu`
   version. It is compile-feature-gated, checks Vulkan/RTX support at runtime,
   exposes supported Quality/Balanced/Performance/DLAA choices, and falls back
   automatically when unavailable.
5. DLSS is not enabled by default until a controlled 4K comparison demonstrates
   both a material GPU-frame-time win and acceptable temporal image quality in
   orbit, low flight, vegetation, thin geometry, water/coasts, clouds, and rapid
   camera motion.
6. Frame Generation and Ray Reconstruction are outside the first integration.
   They have different latency, presentation, UI-buffer, optical-flow, and
   denoising contracts and need separate evidence and decisions.

## Alternatives

- **Wire DLSS straight into the existing post stack now** — rejected because the
  missing body-fixed motion-vector and reset contract would turn integration
  defects into ghosting, shimmer, and view-switch history leaks.
- **Stay SMAA/MSAA-only** — rejected because it leaves the highest-value 4K
  performance lever unused and does not unlock temporally stable dithered LOD,
  foliage, stochastic AO, or future soft-shadow sampling.
- **Make DLSS the renderer default** — rejected because it is restricted to
  supported RTX/Vulkan systems and carries external SDK/runtime packaging and
  licensing obligations. Thalos must remain correct without it.
- **Adopt NVIDIA Streamline first** — rejected for the initial slice because the
  direct `dlss_wgpu` wrapper matches the current wgpu/Vulkan stack and keeps the
  scope to Super Resolution. Streamline can be reconsidered only if several
  vendor plug-ins or presentation-level features justify its broader hooking
  layer.
- **Include Frame Generation in the same slice** — rejected because generated
  frames do not reduce simulation latency and require a separate HUD-less/UI and
  presentation contract. Super Resolution should first improve the real rendered
  frames.

## Consequences

The expensive work is W13, not the DLSS call itself: custom passes must emit
correct motion, camera/origin changes must invalidate history centrally, and
screen-space effects must distinguish internal from output resolution. That work
also benefits non-NVIDIA rendering through native temporal stability and leaves a
single seam for a future vendor-neutral temporal upscaler.

The DLSS backend adds feature/build matrix and release-packaging work, and the
Vulkan path must be exercised explicitly on Windows and Linux. In return, RTX
users with a GPU-bound 4K workload can trade internal resolution for frame time
or reinvest the saved budget in clouds, shadows, vegetation, and atmosphere
without creating a second renderer architecture.
