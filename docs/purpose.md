# Project purpose

**Status:** durable project north star

The name **Thalos** currently refers to both this wider project and its primary
application. A separate project or engine name may come later. Until then, the
documentation uses these terms when the distinction matters:

- **Thalos project** — this workspace and the continuing exploration of natural
  worlds;
- **Thalos game** — the primary application: a spaceflight simulator intended
  for release;
- **world foundation** — the project's internal architecture for representing,
  emulating, and rendering those worlds.

This document defines why all three exist together. [Gameplay](gameplay.md)
owns the game design; [architecture](architecture.md) owns the code boundaries.

## 1. The game is the primary product

The Thalos game is a real game, not a renderer showcase. It is a physically
grounded spaceflight simulator about building a space programme, flying through
a full-scale fictional solar system, and learning about that system through
exploration.

It is the project's product anchor. Architecture should help it become
coherent, playable, and releasable. Building the foundation is valuable because
it makes this game and the worlds inside it possible; it must not become an
open-ended substitute for finishing the game.

## 2. The project explores natural worlds

The wider project is a long-running personal exploration of how natural worlds
can be represented and rendered. Terrain is central, but a world is more than a
height field: water, coasts, atmosphere, weather, vegetation, geology, lighting,
and their relationships determine whether a place feels coherent.

The aim is usually better described as **emulation** than complete physical
simulation. Systems should preserve the structures, scales, responses, and
interactions that make nature legible. First-principles simulation is welcome
where it pays for that result, but it is not a purity test. A convincing,
internally coherent approximation can be the right model.

The intended range is deliberately broad:

- synthetic, real-world, and hybrid authored data;
- a small local or projected map;
- spherical and ellipsoidal bodies;
- several bodies present in one application;
- complete solar systems, from orbital scale to the ground beneath the camera.

Supporting this range does not mean pretending every world has the same
topology. Shared data and mechanisms describe what a world means and how it
looks; explicit spatial adapters describe how a planar map, cube sphere,
analytic body, far-body projection, or geodetic ellipsoid is placed, streamed,
and rendered.

## 3. The world foundation is an internal personal engine

The world foundation is intended to grow into the author's own engine for
natural-world applications. It is an internal architectural identity, not a
separate public engine or SDK product today.

Its job is to make several real application compositions possible without
making any one of them inherit the whole Thalos game. It includes only seams
earned by actual needs: authored world models, terrain and environment
contracts, reusable appearance mechanisms, spatial adapters, the lightweight
application shell, diagnostics, and deterministic visual evaluation.

That gives the foundation two complementary obligations:

1. The Thalos game must be able to compose full planetary simulation,
   spaceflight gameplay, and several simultaneous celestial bodies.
2. Smaller applications must be able to select only the world, rendering, and
   interaction capabilities they need.

Flexibility is therefore measured by working compositions, not by the number of
traits or configuration flags. A universal renderer that hides real spatial
differences is less useful than several explicit adapters over genuinely shared
mechanisms.

## 4. Focused applications are both projects and laboratories

Kòrsou is the first secondary application. It is a small passion project in its
own right: a focused real-world Curaçao explorer that can continue developing
tangentially to the spaceflight game and may eventually target the web.

Its constrained environment also makes it a productive laboratory. Ocean
rendering, coastlines, foliage, real terrain data, camera interaction, and
diagnostics can be explored without the full orbital simulation and gameplay
stack. That fresh context exposes assumptions and makes individual systems
easier to understand.

The feedback loop is:

1. Explore a system in the application where the question is clearest.
2. Measure and understand the result in that application's real constraints.
3. Promote the general mechanism or contract into the world foundation when
   its meaning genuinely matches another consumer.
4. Bring the improvement into the Thalos game where applicable and verify it in
   the game's full planetary composition.
5. Keep topology, content, and product behavior specific to the application
   that owns them.

Kòrsou is therefore neither a disposable demo nor a disguised Thalos game
scenario. Its own experience matters, and so does what the shared project learns
from it.

Future applications, such as an RTS, can exercise the foundation in different
ways. They are directional possibilities, not current product commitments and
not justification for speculative infrastructure before their requirements are
real.

## 5. Decision rules

When product and architecture pressures compete, use these rules:

1. **Keep the game primary.** The releasable spaceflight simulator remains the
   main product and integration target.
2. **Share meaning, not accidental shape.** Promote a mechanism when its inputs,
   outputs, and semantics match; keep spatial topology and product policy in
   explicit adapters.
3. **Let each application be itself.** A smaller application selects the shared
   capabilities it needs and owns the rest of its experience.
4. **Use focused exploration to improve the whole.** A constrained application
   is a legitimate place to deepen a system, but shared improvements complete
   the loop only when they return to the foundation and relevant applications.
5. **Build abstractions from evidence.** Two real callers, a measured boundary,
   or a compiler-enforced dependency payoff justify generalization. A possible
   future caller alone does not.
6. **Preserve the range of worlds.** Do not let a convenient assumption about
   one planet, one map, one coordinate frame, or one active application become
   the architecture's hidden limit.

The resulting project is game-led but not game-limited: one primary
spaceflight game, a growing personal world engine beneath it, and focused
applications that deepen both through real use.
