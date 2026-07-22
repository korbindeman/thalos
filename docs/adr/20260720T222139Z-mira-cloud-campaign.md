# ADR-20260720T222139Z-mira-cloud-campaign: Benchmark Mira cloud training before an A6000-first campaign

- **Status:** Superseded by ADR-20260721T020849Z-local-cuda-first-mira-campaigns
- **Date:** 2026-07-21

## Context

MIRA-1 has a Rust/Burn training tracer and prepared lunar data, but the local
CPU smoke output is not visually useful. The user authorised rented compute
under a preferred $50 total budget. Thunder Compute currently offers 48 GB RTX
A6000 at $0.35/GPU-hour, 80 GB A100 at $1.09, and 80 GB H100 at $2.19. Choosing
hardware before the real-data loader, native CUDA backend, and visual overfit
gate exist would spend money measuring unfinished plumbing.

## Decision

Keep one backend-generic Burn model and add native CUDA as a third training-tool
backend beside CPU/Flex and WGPU. Pass the local one-to-four-patch visual
overfit gate before renting hardware.

Then benchmark one RTX A6000 hour and at most one A100 hour using the identical
commit, dataset, batch shape, and metric sink. Select by examples per dollar
subject to VRAM headroom, not headline throughput. Default to A6000 unless the
A100 materially wins that measurement or the model does not fit in 48 GB.

Cap the documented campaign ledger at $49.44. Any spend above $50 requires a
new user decision. Normal gameplay remains package-first and independent of
CUDA or cloud infrastructure.

## Alternatives

- **Start immediately on H100** — rejected because its 22.8-hour budget ceiling
  gives less room for research iteration before the architecture is proven.
- **Use only local WGPU** — rejected as the final campaign strategy because it
  hides the performance and tooling advantages of native CUDA and delays the
  visual loop; it remains the free smoke/fallback path.
- **A100-only campaign** — deferred until a measured throughput-per-dollar or
  memory result justifies its 3.1× A6000 hourly rate.
- **Port training to PyTorch for cloud convenience** — rejected by ADR-20260721T033713Z-rust-native-learned-terrain's
  single authored Rust/Burn model rule.

## Consequences

- Cloud spend is gated by visual and reproducibility evidence.
- The training binary gains an explicitly tested native CUDA feature.
- Every rented run records billed time and hardware evidence in the roadmap.
- A6000 is a starting hypothesis, not an unmeasured permanent choice.
