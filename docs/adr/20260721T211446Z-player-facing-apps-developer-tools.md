# ADR-20260721T211446Z: Reserve `apps/` for player-facing applications

**Status:** accepted
**Date:** 2026-07-21
**Supersedes:** the placement of `thalos_capture_host` under `apps/` in
ADR-20260721T194628Z-role-based-agent-first-workspace

## Context

The role-based workspace initially put both the interactive game and the
headless capture host under `apps/`. That grouped processes by the fact that
they are runnable, but made `apps/` ambiguous to a human reader: the capture
host is not a program a player launches or a separately shipped product. It is
automation infrastructure behind the capture CLI.

The offline baker, trainer, texture generator, world-map exporter, capture
controller, and capture host share an operational role even though some are
production-critical: they are developer or pipeline executables.

## Decision

Reserve `apps/` for player-facing applications. It currently contains only
`apps/game`.

Place all developer, automation, and offline pipeline executables under
`tools/`, including `tools/capture_host`. Their importance does not determine
their folder; their audience does.

Reusable code remains under `crates/`. In particular, the capture protocol and
runtime stay in `crates/capture`, while the thin host and controller live under
`tools/`.

## Consequences

- A human can read `apps/` as the set of player-launched products.
- Headless capture remains first-class without masquerading as a player app.
- `tools/` is not disposable scratch space; it contains maintained workspace
  programs. Generated outputs must live elsewhere.
- If Thalos later gains another player-facing executable, it may join `apps/`;
  developer-only viewers, editors, exporters, and automation hosts do not.
