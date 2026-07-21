# ADR-20260721T210000Z-authored-parts-player-integration: Parts are authored canon; player agency lives at the integration layer

- **Status:** Accepted
- **Date:** 2026-07-21

## Context

Thalos differentiates from KSP on **achievement** and **scale** (`gameplay.md` *Pillars*), and
a recurring instinct is to push player creativity *down* into the parts themselves —
procedurally generated engines, player-designed command pods, a tech tree where
"projects" produce bespoke named engines with derived stats.

That instinct has real appeal: it makes progression feel like a space programme rather
than an unlock ladder, and it generates ownership ("*my* engine") cheaply.

It also has two problems that only show up when you try to build it:

1. **Procedural generation of good-looking hero parts is very hard.** Engines, command
   pods, and instrument packages are exactly the objects where silhouette, greebling,
   and proportion carry the entire read. A generator that produces *unique* engines
   almost certainly produces *bad* engines. The procedural work that has succeeded in
   this project is connective and organic (terrain, vegetation, fuselages, wings), not
   hero hardware.
2. **Player-named shared parts are hollow.** If every player has the same engine and
   merely renames it, the name is a skin over a common object. Naming does not create
   identity; distinctness does.

Meanwhile the interesting real-world job is not engine design. Arianespace buys Vulcain,
Atlas V bought RD-180, ULA has never built an engine. **The integrator is the interesting
job.**

## Decision

**Parts are authored by us and are shared canon.** Engines, command pods, instruments,
and structural end caps are hand-authored, arrive from in-world contractors as finished
named objects, and are identical for every player. They are unlocked by researched
**techniques** plus **acquired data** (some parts gate on knowing a target environment,
not only on theory).

**Player agency lives at the integration layer**: launch vehicles, spacecraft, and
mission architectures. Those are player-designed, player-named, and genuinely distinct.
Procedural geometry continues to carry the *connective* structure — tanks, fuselages,
interstages, fairings, wings — so proportions are the player's, not a fixed diameter set.

Two layers of canon follow:

- **Parts** — shared canon. A feature, not a limitation: shared vocabulary between
  players, an authored-feeling world, a community wiki that can exist.
- **Vehicles, missions, programmes** — personal canon, named by the player, and backed
  by a **record** (flights, successes, firsts) which is what actually makes a name mean
  something.

Reuse is rewarded through **flight heritage** (a configuration's flown record converges
its reliability, and human-rating requires N consecutive successes), **cheap derivation**
(stretch/booster/upper-stage variants inherit most heritage; a new core starts over), and
**shared infrastructure** (a family shares pad configuration and tooling). Full model in
`gameplay.md` *Where player agency lives: the integration layer*.

## Alternatives

- **Procedurally generated engines and pods** — rejected: the visual quality bar for hero
  hardware is not reachable by generation, and shipping mediocre-looking engines would
  undercut the "more realistic-looking rockets than KSP" goal that motivated procedural
  parts in the first place. Procedural stays where it demonstrably works: connective and
  organic geometry.
- **"Projects produce parts" — the player charters a specific engine, which enters their
  catalogue as a uniquely named, uniquely statted item** — rejected: it is the same
  generation problem wearing a design-vocabulary hat (a named engine still needs a mesh),
  and it fragments the shared vocabulary without adding real distinctness. The *project*
  abstraction is kept, but raised one level: a project charters a **vehicle**, not a part.
- **Player-authored parts (in-game part editor)** — rejected: a much larger scope than the
  differentiators need, and it moves the game's identity from "space programme" toward
  "CAD sandbox". Authored parts also let us guarantee the catalogue reads as a coherent
  industrial design language.
- **Let players rename shared parts for ownership** — rejected as hollow (see Context).
  Naming attaches to vehicles and programmes, which are actually distinct.

## Consequences

- **Part authoring is a permanent, ongoing content cost** on us, and the catalogue's
  breadth directly bounds the design space players can express. This is accepted
  deliberately: it is the cost of the visual quality bar.
- **Vehicle-family mechanics become load-bearing.** Heritage, derivation lineage, and the
  capability-envelope problem (`gameplay.md` *A family covers an envelope*) are now the primary agency and reward
  systems, not garnish. If they are weak, the game has no creative layer, because the
  parts layer was deliberately closed.
- **Derivation tiers must trade, never strictly upgrade** — a monotone ladder collapses a
  family into one best vehicle and destroys the envelope problem.
- Balance risk moves to heritage carry-over: how much confidence a shared engine or shared
  core transfers between vehicles is unspecified and is the crux (`gameplay.md` *Open
  questions*).
- Reliability must not be raw dice. Failures should be survivable anomalies with real
  abort modes, and odds must be visible before commit; *unproven* vehicles hide their
  odds, and that concealment is the penalty
  (ADR-20260721T210002Z-setback-not-failure-economy).
- `thalos_shipyard`'s existing model — authored parts catalogue + parametric procedural
  structure — is already the right shape. This ADR ratifies it rather than changing it.
