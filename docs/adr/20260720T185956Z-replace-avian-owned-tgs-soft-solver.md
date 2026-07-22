# ADR-20260720T185956Z-replace-avian-owned-tgs-soft-solver: Replace Avian with an owned TGS-Soft solver crate (`thalos_physics`)

- **Status:** Accepted (Phase 0 seam landed; Phase 1 specced, deferred until asked)
- **Date:** 2026-06-28 (recorded 2026-07-18; full design + roadmap in `docs/simulation/physics.md`)

## Context

Avian is the local-physics backend behind `thalos_physics_local`. Two forces make it the wrong
long-term substrate: (1) the endgame is BeamNG-style node-beam soft-body damage — a vehicle that
is *entirely* a mass-spring net with plastic yield, which no rigid-body engine provides (PhysX/
Jolt soft bodies are elastic FEM that spring back, not structural crumple); (2) the surface
regime needs jitter-free rest and stable landings in the SLF at f64, where owning the substep
loop matters more than a general-purpose feature set.

## Decision

Write our own solver: new pure-Rust crate **`thalos_physics`** depending only on `parry3d-f64`
(keep parry for collision detection — heightfield/compound/manifolds/casts/TOI). Solver =
**TGS-Soft** (substepped soft-constraint, the Catto/Box2D/Rapier lineage), fixed-dt +
accumulator + render interpolation, built so rigid (now) and node-beam (later) share one solver.
`physics_local` dissolves once Avian is gone. Determinism target is single-player stability, not
lockstep; future multiplayer is client-authority + state replication (remote craft kinematic —
the BeamMP model), which only requires kinematic bodies be designed in from day one.

Phased: **Phase 0** tighten the backend seam (landed — executor allowlist CI-guarded, see
`physics.md` §7.1) → **Phase 1** stand up `thalos_physics` in shadow mode vs Avian (fully
specced in `physics.md` §7.2–7.3, test-first; **do not start until Korbin asks**) → **Phase 2**
cut over and delete Avian → **Phase 3** node-beam deformation.

## Alternatives

- **XPBD** — rejected: NVIDIA-patented (the stated reason Avian itself left XPBD); TGS-Soft
  gives the same substepping stability, is more battle-tested for one-body-vs-terrain, and is
  patent-clean.
- **Keep Avian** — rejected: wrong substrate for node-beam damage regardless of cleanup, and its
  Bevy coupling forces the `physics_local` quarantine crate to exist at all.
- **Adopt an existing standalone solver** — rejected after a survey: no adoptable pure-Rust
  XPBD/TGS library exists (Avian = Bevy-coupled; Rapier = black box for our purposes;
  OxiPhysics = unproven single-author risk).

## Consequences

- Avian is scheduled-for-removal, not load-bearing: new physics work follows `physics.md`, keeps
  solver code out of Bevy, and treats the executor seam as the boundary.
- This supersedes the earlier regimes-doc posture of "keep Avian, re-evaluate parry-direct at
  Phase C" — the go/no-go is decided (replace).
- We own integrator + contact solver correctness ourselves; the mitigations are the phased
  shadow-mode rollout and the analytic/parity test layers specced in `physics.md` §7.3.
