# ADR-20260721T210003Z-diegetic-control-authority: The player always flies; guidance technology gates what they may command

- **Status:** Accepted
- **Date:** 2026-07-21

## Context

Two things need to be true at once in programme mode (`gameplay.md`):

1. Early rocketry should feel like early rocketry — crude, gyro-referenced, flown on fins
   and vanes, without the orbital-mechanics conveniences that make KSP launches routine.
2. The player should be *flying*, because that is the game. A space game whose opening
   hours are spent watching an autopilot execute a plan is not the game we are building.

There is also a problem every space game hand-waves: **you cannot pilot a vehicle in real
time across light-minutes.** KSP simply ignores it. Doing the same would waste the one
place where a fictional system, a real tech tree, and the narrative climax (piloting a
submersible under Pelagos's ocean) all want to meet.

The naive resolutions each fail one of the above. Restricting the early game to
pre-programmed, non-interactive flight satisfies (1) and breaks (2). Unrestricted piloting
from the first launch satisfies (2), breaks (1), and leaves the light-time problem
unanswered.

## Decision

**The player always flies, mid-flight, hands on — from the first sounding shot onward. The
tech tree gates *what they are allowed to command*, and the constraint is diegetic:**

> You may do what the onboard guidance could plausibly have been programmed to do.

The fiction is premeditated flight: the mission was planned and programmed before launch,
and the player is standing in for the computer executing it. Early limits are therefore
not a designer withholding control — they are a vehicle that genuinely cannot do more.

A four-rung **control-authority ladder** (`gameplay.md` *Control authority*):

1. **Unguided / spin-stabilised** — fins and spin; minimal authority.
2. **Open-loop programmed** — gyro attitude reference, fin/vane actuation: hold an
   attitude, command a tilt, cut off on a timer. Nothing requiring knowledge of position:
   no velocity-vector holds, no target states, no node following, and a poor state
   estimate.
3. **Closed-loop guidance** — the vehicle knows its state and can steer toward one. Target
   states, measured cutoff, velocity-vector holds, node following, conditional aborts.
4. **Autonomous flight computers** — reactive operation where the player cannot supervise;
   the **autonomy tier** sets how far from supervision a vehicle can be and still do
   something interesting.

Mechanically this is a **permissions-and-fidelity layer over the existing control stack**,
not a new one. `thalos_control` already models every attitude command as a tagged
`ControlDemand` arbitrated into one `AttitudeController` and allocated to reaction wheels
and aero surfaces (`control.md`). A tier gates: which `AttitudeDemand` modes exist, which
`DemandSource`s exist, actuation fidelity (rate limits, deadbands, lag), and
**state-estimate fidelity** — what the instruments actually tell the player.

## Alternatives

- **Early flights are pre-programmed and non-interactive; the player authors a pitch
  schedule and watches telemetry** — rejected: it removes the flying from a flying game for
  the entire opening, precisely when the player is learning to care. The pre-programming
  survives as *fiction and constraint*, not as the mechanic. (This was proposed and
  discarded during design; recorded so it is not re-proposed.)
- **Unrestricted piloting from the first launch (the KSP model)** — rejected: it makes
  guidance technology a cosmetic node, erases the felt difference between rungs 2 and 3,
  and leaves the light-time question unanswered, which in turn makes the advanced-computing
  line meaningless.
- **Model true light-time delay on player input at distance** — rejected as the primary
  mechanism: multi-minute input lag is not playable, and it would make the game's narrative
  climax unplayable exactly where it matters most. Autonomy tier is the playable expression
  of the same physical fact; light-time and bandwidth remain relevant to the deferred
  communications system as constraints on *data return and supervision*, not on stick feel.
- **Gate guidance behind difficulty settings rather than the tech tree** — rejected: it
  makes the capability an opt-out chore instead of a reward, and severs the link between the
  advanced-computing line and the discoveries it exists to enable.

## Consequences

- **The advanced-computing tech line becomes narratively load-bearing.** It is what permits
  piloting a submersible under Pelagos (`lore/civilization.md` Phase 3) and under the ice of
  Glacis (Phase 5) — the endpoint of the tech tree is the endpoint of the narrative, rather
  than a generic stat node.
- **A guidance tier must never become a second control stack.** The seam is
  `thalos_control`'s demand vocabulary; anything that forks it violates the one-canonical-path
  rule (`CLAUDE.md`) and will drift.
- **State-estimate fidelity becomes a real, gated quantity**, which the HUD/MFD layer must be
  able to express (an early vehicle has no solved orbit to display). This touches
  `hud_widgets.md` and is not currently modelled.
- **Tuning risk is concentrated and real.** The ladder is structurally right, but making rung
  2 read as *characterful and honest* rather than *sluggish and annoying* is a hands-on
  iteration problem, not something a spec can settle.
- Post-flight telemetry retains a purpose even though the player was at the controls: at low
  state-estimate fidelity, learning what actually happened versus what you thought was
  happening is a genuine part of the early loop.
- Sandbox mode grants all tiers, consistent with `gameplay.md` *Modes* — a constraint switched
  off, not a code fork.
