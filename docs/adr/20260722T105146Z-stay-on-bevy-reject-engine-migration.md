# ADR-20260722T105146Z-stay-on-bevy-reject-engine-migration: Thalos stays on Bevy; Unreal Engine 5 is rejected

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

Thalos's two stated goals are a sense of **scale** and **visual fidelity** —
full planetary terrain, good detail in orbit, good surface views, over a real
simulation. Development against Bevy is slow: every fidelity feature is
hand-built (see `graphics_fidelity.md` F1–F9, nine rounds of shadow
unification, the SSAO node, the sky-view LUT, the ocean spectral tracer), and
the perceived bottleneck was iteration speed.

Unreal Engine 5 was evaluated as an alternative. The question is whether an
engine that ships VSM, Lumen, Nanite, PCG, Sky Atmosphere, and a mature editor
would delete enough of that work to justify migrating ~163k lines of Rust and
~17k lines of WGSL.

## Decision

**Stay on Bevy.** Do not migrate to Unreal Engine 5 or any other general-purpose
engine. Reduce cost instead by adopting more of Bevy's renderer rather than
building parallel replacements (`ExtendedMaterial` over `StandardMaterial`,
0.19's contact shadows / SSR / parallax-corrected cubemaps where they obey the
one-world rule), and by treating iteration cost as a hardware and
build-configuration problem rather than an engine problem.

## Alternatives

**Migrate to Unreal Engine 5.** Rejected on four grounds:

1. **UE has no planetary terrain.** Landscape is a flat rectangular heightfield;
   World Partition and Large World Coordinates do not change that. A planet
   requires a custom `PrimitiveSceneProxy`, which is the same work as
   `thalos_udlod` (~8k LOC) rebuilt in a harder language and build environment.
2. **UE's headline features degrade on exactly that surface.** Runtime-generated
   streaming geometry cannot use Nanite (which needs build-time cluster
   hierarchies) and gets only screen-traces from Lumen (no mesh distance
   fields). For the majority of on-screen pixels — the ground — UE would deliver
   a custom mesh with VSM shadows, which is approximately what exists today.
3. **It recreates the two-lighting-universes debt, inverted and unfixable.** The
   custom planet would not match Lumen-lit crafts, and the half we do not
   control becomes closed C++ modifiable only via a full engine-source build.
4. **It destroys the agent-first workflow.** Materials, Blueprints, levels, PCG
   graphs, and Niagara are binary `.uasset` files an agent cannot read, diff, or
   edit. Thalos's development model is text edits → `just check` →
   `just screenshot` → read the PNG. That leverage does not survive the move.

**Unigine / Outerra / other simulation engines.** Genuinely double-precision and
planet-capable, but small ecosystems, weaker art pipelines, and the same
rewrite cost. Not evaluated in depth; recorded as not pursued.

**Diagnosis of the stated problem.** "Slow iteration" was found to be largely
misattributed. `build_speed.md` already lands rust-lld, dynamic linking,
sccache, Subsecond hotpatching on the capture lane, and Defender exclusions. The
real cost is that we are *writing a renderer*, which is slow because it is hard,
not because Cargo is slow. UE C++ full-module rebuilds on a project this size
are worse, and Live Coding breaks on header/`UCLASS` changes.

## Consequences

- Every fidelity feature remains ours to build or to adopt from Bevy upstream.
  The buy-vs-build lever is *inside* Bevy: prefer 0.19 features over new custom
  passes wherever the one-world principle survives.
- Craft/interior/character fidelity — where UE genuinely wins — stays our
  weakest area. Accepted; it is not what a scale-and-planets game is judged on.
- Iteration-speed work is hardware and build configuration (cores, RAM, NVMe,
  extending hotpatching to the interactive lane, leaning on small probe
  binaries), not engine selection. See `build_speed.md`.
- This ADR does not settle *how* terrain is rendered within Bevy. That is
  ADR-20260722T105147Z-tile-native-surface-seam.
- Reopening requires new evidence, not new frustration: a shipped UE project
  doing streaming planetary terrain with Nanite/Lumen intact, or a decision to
  abandon agent-first development.
