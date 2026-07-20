# ADR-0010: Mira learned terrain is authored once in Rust with Burn

- **Status:** Accepted
- **Date:** 2026-07-20

## Context

ADR-0008 deliberately kept the offline model implementation behind the terrain
package boundary and allowed Python/PyTorch. MIRA-1 now needs a concrete model
toolchain. The same small models may later be useful in an optional bundled
authoring tool or close-detail reconstruction, while normal play must remain
package-first and independent of planetary diffusion.

Candle and Burn both provide Rust tensors, autodiff, GPU execution, and
SafeTensors interoperability. Candle has strong lightweight inference and
existing Stable Diffusion implementations, but its first-party device path is
centred on CPU, CUDA, and Metal. Burn provides one backend-generic model API
across training and inference, including WGPU, CUDA, ROCm, CPU, and a Candle
backend.

## Decision

Author MIRA learned models, diffusion schedules, samplers, and tensor contracts
once in Rust using Burn 0.21. Keep them in a Bevy-independent library. The
offline training tool selects an appropriate Burn backend; Candle is an allowed
Burn backend and implementation reference, not a second model definition.

Store portable model records in SafeTensors-compatible form and keep the terrain
package as the durable gameplay boundary. Do not make normal gameplay run the
planetary diffusion model. A future bundled generator is an optional authoring
capability; MIRA-3's bounded client detail path remains separately budgeted.

## Alternatives

- **Python/PyTorch training plus a separately ported Rust inference graph** —
  rejected because two model definitions and tensor conventions will drift.
- **Candle as the only framework** — rejected because its deployment backends
  do not provide the general WGPU path wanted for Thalos's cross-platform game
  stack. Burn can still execute through Candle where advantageous.
- **Embed planetary diffusion in normal gameplay now** — rejected because it
  would make latency, model availability, and device support gameplay
  requirements, contrary to ADR-0008.

## Consequences

- Model code is reusable by offline training, baking, validation, and a future
  optional in-game authoring feature without a language port.
- The training tool may choose WGPU for local smoke runs and CUDA/ROCm/Candle
  for campaigns without changing the model source.
- Burn is pinned because its API is still evolving; upgrades are explicit.
- The learned crates stay out of `thalos_game` until a measured runtime feature
  needs them, so the game does not inherit ML compile or binary cost today.
