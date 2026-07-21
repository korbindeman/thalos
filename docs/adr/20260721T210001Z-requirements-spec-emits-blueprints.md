# ADR-20260721T210001Z-requirements-spec-emits-blueprints: The probe/mission requirements spec is a front end that emits a `ShipBlueprint`

- **Status:** Accepted
- **Date:** 2026-07-21

## Context

Probes are a differentiating pillar (`gameplay.md` *Probes and mission specification*): rather than assembling a probe
part-by-part, the player writes a **mission requirements document** — class, instrument
payload, mass/power/duration/comms/thermal constraints — and the programme delivers a bus
solved around those constraints.

That is a genuinely different *verb* from the shipyard's place-and-attach editing, which
raises an architectural fork: does probe construction get its own system?

It should not. The consolidation sprint's first rule (`CLAUDE.md`, `architecture_cleanup.md`)
is **one canonical path per operation**; a second assembler would be exactly the kind of
parallel near-copy that sprint exists to delete — two blueprint formats, two sizing/stats
paths, two staging derivations, two save/load flows, drifting apart from the first month.

## Decision

The requirements spec is a **front end over the existing construction model**. It solves
the bus and **emits a normal `ShipBlueprint`** composed of authored catalogue parts
(ADR-20260721T210000Z-authored-parts-player-integration). That blueprint is
indistinguishable downstream from a hand-built one: it can be opened in the shipyard
editor, hand-edited, saved, launched, staged, and flown through the paths that already
exist.

One construction model, one blueprint format, **two levels of zoom**: specify-and-solve,
or place-and-attach — with a one-way door from the former into the latter.

The generator owns **layout**; the player owns **the spec and the tradeoffs**, and still
designs the launch vehicle, plans the trajectory, and flies the mission.

## Alternatives

- **A dedicated probe construction system with its own data model** — rejected: a parallel
  mechanism for the same job. It would duplicate sizing, mass/CoM/MOI, staging, resource
  whitelisting, and (de)serialisation, and the two would drift. It also permanently forbids
  hand-editing a generated probe, which is the cheapest way to make delegation feel like
  agency rather than automation.
- **Auto-layout as a mode *inside* the shipyard editor rather than a separate screen** —
  not rejected outright, deferred: it may well be the right UI, and this ADR does not
  prejudge the presentation. What it fixes is the **data** boundary — whatever the UI, the
  artifact produced is a `ShipBlueprint`.
- **Generate probes as opaque single entities (a "probe" part)** — rejected: it collapses
  the instrument-selection tradeoffs that are the entire point of the spec, and it would need its
  own mass/power/staging model anyway, arriving back at a parallel system by another route.

## Consequences

- The auto-layout solver must produce blueprints that are **valid and legible by hand** —
  sane attach trees, sensible symmetry, real parts — not an internally-consistent soup. It
  is held to the same output contract as the editor.
- Instrument and bus parts must exist in the authored catalogue with real storage/power/mass
  declarations; the solver cannot invent capability that no part provides.
- The conflicting-constraints design (`gameplay.md` *Probes and mission specification*) lives in the solver, and needs the
  parts catalogue to expose enough dimensions (power draw, data rate, thermal envelope,
  design life) for the conflicts to be real rather than cosmetic.
- If a crewed-craft requirements front end is later wanted, it composes the same way — this
  decision does not have to be revisited.
