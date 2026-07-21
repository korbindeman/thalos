# ADR-20260721T210002Z-setback-not-failure-economy: The programme cannot fail; consequences are denominated in time

- **Status:** Accepted
- **Date:** 2026-07-21

## Context

The target feel for programme mode (`gameplay.md`) is a specific and awkward combination:
**hard but doable, perseverance rewarded, failure never grueling or insurmountable, no
true loss state — and yet decisions must genuinely matter.**

Those pull against each other. The genre's standard answers all break at least one of
them:

- KSP's career mode can strand a player: funds are a *balance*, so a run of bad launches
  produces the death spiral where you cannot afford a rocket to earn the money to afford a
  rocket. Recovery is grinding, and the failure state is real but unacknowledged.
- Hard-loss states (bankruptcy, cancellation, mission-critical permadeath) make the
  correct play *reload*, which converts drama into savescumming and teaches players to
  route around the systems rather than engage them.
- Removing consequences entirely makes the whole programme layer — budget, heritage, crew
  — decorative, and collapses the game into sandbox with extra clicking.

The setting also constrains the answer: the player *is* the Thalos global space
organisation, a civilisational project with broad public backing
(`lore/civilization.md`). Cancelling it is not a thing that happens. Whether it moves fast
or slow is the entire political question in the fiction.

## Decision

**The programme cannot be lost. Consequences are denominated in programme time.**

Three mechanisms, each chosen because it cannot produce a stuck state:

1. **Time is the penalty currency.** Failure sets you back; it never blocks you. A mistake
   costs a window; a catastrophe costs a stand-down. Time cannot be driven to zero, is
   genuinely the scarcest resource in a real space programme, and scales from a lost month
   to a lost two years without changing kind.
2. **Budget is a rate, not a balance.** Funding is a recurring appropriation. Overspending
   creates a deficit that eats future appropriations and heals with time; there is no
   reachable state in which the player cannot fly anything. **Mandate** (public support)
   modulates the size of the appropriation and gates progression phases — so contracts buy
   budget, but only milestones buy ambition.
3. **Crew-loss penalties scale with demonstrable negligence, not with outcome.** Losing
   crew to genuine bad luck on a well-prepared mission is absorbed by the organisation.
   Losing crew on an unproven vehicle rushed to hit a window, with no abort mode and no
   uncrewed test flights, is a scandal. The game can compute this because it already knows
   what was skipped: vehicle heritage, uncrewed flight history, presence of abort modes,
   whether target data was returned, and whether development was rushed.

Supporting rules that give failure the right texture:

- **Most failures are survivable anomalies the player responds to**, not binary loss. An
  engine out at T+80 s is a salvage problem and a story; an unrecoverable fireball is a
  quit-to-desktop.
- **Crewed vehicles carry real abort modes**, turning most launch failures into "we lost
  the vehicle, the crew came home" — which also makes the escape system a genuine design
  decision rather than a mass penalty players learn to skip.
- **The post-loss stand-down** (crewed flight paused pending review — Challenger grounded
  the shuttle 32 months, Columbia 29) is the model consequence: a pure time cost, dramatic,
  fully recoverable, and it does not halt the uncrewed programme, so the player keeps
  playing.

## Alternatives

- **Funds as a spendable balance (KSP career model)** — rejected: it is the direct cause of
  the death spiral, and every mitigation for it (rescue contracts, bailouts, grind loops) is
  a patch over a structural mistake. A rate has no such failure mode and is a better fit for
  a state-funded civilisational programme.
- **A hard loss state — cancellation, bankruptcy, game over** — rejected on both design and
  fiction grounds. It makes reloading the optimal response to any bad outcome, and the
  organisation being unconditionally supported is load-bearing in `lore/civilization.md`.
- **Penalising crew loss by outcome alone** — rejected: it punishes bad luck identically to
  negligence, which reads as arbitrary, teaches savescumming, and makes caution a mood
  rather than a strategy. Negligence-scaled penalties are only possible because heritage and
  mission data already exist as records — this is a reason to keep those systems honest.
- **No crew-loss penalty (deaths are narrative colour)** — rejected: it makes "lives are
  worth a lot" a claim the mechanics contradict, and removes the entire reason uncrewed
  proving flights are worth their cost.
- **Preventing reloads / ironman enforcement** — rejected as the wrong lever. Savescumming
  is addressed by making failure interesting and recoverable, not by locking the door. A
  stand-down the player has to fly out of deters reloading better than a lockout does.

## Consequences

- **Every consequence system must be auditable for "can this strand a player?"** That check
  is now a standing design constraint, not a per-feature judgement call. Anything
  denominated in a depletable resource needs justification.
- **Programme time must be a first-class simulated quantity** — development schedules,
  windows, stand-downs, appropriation cadence. This implies a programme/calendar layer that
  does not exist yet; the Space Center hub is its natural home.
- **Heritage, flight history and returned-data records become load-bearing for fairness**,
  not just for flavour. If they are inaccurate, the negligence computation is unjust, and
  the player will correctly perceive it as arbitrary.
- **Diligence must be legible before commit.** A penalty derived from skipped precautions is
  only fair if the player could see what they were skipping — without it degenerating into a
  nag screen. Unresolved (`gameplay.md` *Open questions*).
- **Balance risk concentrates in appropriation cadence and deficit depth.** The model
  guarantees no stuck state structurally, but "never stuck" and "genuinely constrained" still
  have to be tuned against each other with real numbers.
- Abort modes, escape systems and anomaly-response gameplay are now required content rather
  than optional realism, because the failure texture depends on them.
