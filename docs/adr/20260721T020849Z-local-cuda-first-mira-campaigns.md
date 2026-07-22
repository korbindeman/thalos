# ADR-20260721T020849Z-local-cuda-first-mira-campaigns: Mira pilots default to persistent local CUDA

- **Status:** Accepted
- **Date:** 2026-07-21
- **Supersedes:** ADR-20260720T222139Z-mira-cloud-campaign

## Context

The controlled A6000 velocity campaign measured 552 MiB peak VRAM, 62.65
examples/s, and 291.15 seconds for 2,280 batches. The equivalent Mac CPU smoke
was about 32 times slower per batch, so CUDA materially accelerates research,
but this small model does not need the A6000's 48 GB capacity. Cold Thunder
provisioning, Rust installation, release codegen, source/data transfer, and
artifact recovery took longer than the five-minute training run.

Command-mediated archive transfer also conflicts with the tenant data-export
guard. Standard `tnr scp` works, but the Thunder integration requires the user
to execute SSH/SCP commands. The user has an RTX 4070 Ti whose 12 GB frame
buffer is far above the measured MIRA-1 footprint and explicitly chose manual
cloud transfers for future runs.

## Decision

Use the persistent local RTX 4070 Ti as the default MIRA-1 pilot backend once a
native CUDA preflight passes. CPU remains the deterministic unit/smoke fallback.
Cloud hardware is selected only when a measured run exceeds local VRAM, needs
cloud-specific reproducibility evidence, or several frozen campaigns can be
batched in one provisioned session so setup is amortized.

For Thunder campaigns, the agent owns the control plane: provision, preflight,
commands, telemetry, hash verification, and deletion. The user owns the data
plane and runs exactly one `tnr scp` upload and one `tnr scp` evidence download.
Private source and large artifacts are never transported as shell-command
payloads or authenticated file-API chunks. The agent deletes the instance only
after the downloaded evidence archive and its internal manifest verify locally.

The single Rust/Burn model, evidence requirements, and $49.44 hard campaign cap
from the superseded ADR remain in force.

## Alternatives considered

- **One cold A6000 per experiment:** rejected because setup and artifact
  transport dominate this model's five-minute training time.
- **Command/base64 or file-API chunk transfer:** rejected because it abuses the
  control plane, triggers export safeguards, and is dramatically slower than
  standard SCP.
- **Mac CPU-only iteration:** retained for tiny diagnostics but rejected for
  full pilots because the measured per-batch rate is about 32 times slower.
- **Cloud remains default despite local CUDA:** deferred until model footprint,
  throughput per dollar, or batched campaign evidence justifies it.

## Consequences

- Most MIRA-1 iterations reuse local compiler, dataset, and CUDA caches with no
  cloud spend or transfer ceremony.
- A 12 GB local ceiling is explicit; larger later-stage models may still require
  rented hardware.
- Thunder runs require two short user terminal actions, but the boundary is
  simple, auditable, and fast.
- Cloud comparisons are batched and evidence-driven instead of paying cold-start
  cost for every small ablation.
