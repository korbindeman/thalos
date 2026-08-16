# ADR-20260809T201216Z-light-runtime-capability-bundles: the shared runtime has a lightweight base and opt-in capability bundles

- **Status:** Accepted
- **Date:** 2026-08-09
- **Supersedes:** ADR-20260808T205119Z-korsou-second-application-render-kit

## Context

Kòrsou proved that Thalos's rendering mechanisms can serve a second spatial
adapter without importing the planetary simulation. It also reproduced the
application machinery around those mechanisms: a 465-line free camera, a
1,203-line saved-viewpoint implementation and manager, a separate HUD, separate
capture orchestration, and no access to the normal settings surface. Continuing
to extract only rendering leaves would preserve two implementations of the
player-facing shell and let their controls, persistence, and UI drift.

Making Kòrsou use the current `thalos_runtime` unchanged is not acceptable
either. That crate still composes simulation, local physics, gameplay, planetary
rendering, editors, HUD, and capture behavior. Product identity and dependency
weight are separate concerns: Kòrsou should be a distinct lightweight
application while sharing the normal application machinery.

The existing crate-granularity decision rejects using Cargo features as a
substitute for ownership boundaries. A feature-gated monolith would retain the
same edit cost and add conditional-compilation combinations. Features can still
be useful at a narrow composition facade after implementations have real crate
boundaries.

## Decision

`thalos_runtime` becomes a thin, capability-selected application facade with an
empty default feature set. Its lightweight interactive bundle owns the common
application shell: semantic camera input, the canonical freecam and its UI,
camera optics, saved viewpoints and their F8/F9 UI, window and graphics-settings
surfaces, shared UI assets, and process diagnostics. Kòrsou remains
`apps/korsou`, but consumes this facade with default features disabled and only
the lightweight capabilities it needs.

Simulation, gameplay, planetary composition, and headless capture are coarse,
additive opt-in features. They select optional dependencies and Bevy plugin
bundles whose implementations live in separate crates. The canonical game
explicitly enables the complete game bundle; Kòrsou does not enable simulation
or gameplay. Disabled capabilities must be absent from Kòrsou's normal
dependency graph and release binary, not merely dormant at runtime.

Cargo features are selectors at the facade, not implementation boundaries:

- feature implementations live in crates or already-proven lower modules;
- `#[cfg(feature = ...)]` is concentrated in the facade and small compatibility
  exports, never spread through feature systems;
- features are coarse and additive, with a small tested matrix rather than a
  flag for every subsystem;
- runtime quality choices such as MSAA, foliage, or cloud rendering remain
  persisted settings, not Cargo features;
- platform or dependency availability may remain Cargo-gated where it already
  changes the compiled graph.

The shared camera and viewpoint mechanisms operate on an application-supplied
stable spatial frame. Thalos supplies its rotating body-fixed/floating-origin
adapter; Kòrsou supplies its projected UTM/DEM adapter. Ground queries, map
bounds, simulation time, scripted framing, and spatial projection stay in those
adapters. This is not a universal renderer or spatial-world trait.

## Alternatives

- **Keep Kòrsou directly composed from rendering leaves** — rejected because it
  has already produced parallel camera, viewpoint, HUD, capture, and settings
  paths. Rendering reuse alone does not produce one application experience.
- **Enable the complete existing runtime in Kòrsou** — rejected because unused
  simulation, physics, gameplay, planetary, and editor dependencies would make
  the explorer heavy and couple it to systems outside its product.
- **Put `#[cfg]` throughout the current runtime crate** — rejected because it
  preserves the monolith, creates a combinatorial type-check surface, and
  conflicts with ADR-20260731T024003Z's crate-boundary rule.
- **One Cargo feature per plugin or graphics setting** — rejected because fine
  feature granularity produces invalid combinations and repeated build graphs.
  Player-selectable quality belongs in persisted runtime settings.
- **A dynamic plugin ABI or service container** — rejected because both
  applications are built together from source. Static Rust composition is
  smaller, easier to test, and sufficient.

## Consequences

- Kòrsou remains a separate product and spatial adapter, but no longer a
  separate implementation of ordinary application behavior.
- The earlier ADR's reusable-rendering-leaf and explicit-spatial-adapter
  decisions remain valid; only its prohibition on a Kòrsou runtime dependency
  is superseded.
- `thalos_runtime` must shed implementation mass before its light feature set is
  credible. The facade may not hide unconditional simulation/gameplay
  dependencies behind an API.
- CI must check the supported feature matrix and use `cargo tree` assertions to
  prove Kòrsou does not pull simulation or gameplay crates.
- The single dev-renderer fingerprint remains load-bearing. Capability features
  select product code; they do not create alternative Bevy/wgpu feature mixes.
