# Gameplay

**Status: design capture (2026-07-21).** The horizon vision — the shape the game is
aiming at, recorded so decisions downstream of it stay consistent. It is **not**
scheduled work; the active sprints are the architecture consolidation pass (`clean`)
and graphics fidelity (`gfx`). No backlog rows yet.

Lore and setting live in [lore/civilization.md](lore/civilization.md) (progression
phases, resource economy, the Pelagos question) and
[lore/solar_system.md](lore/solar_system.md). This doc is the *mechanical* layer: what
the player does, and why it is rewarding.

**Scope discipline.** This doc settles the **core loop**, the **consequence economy**,
and the **control-authority model**. Several large subsystems it touches — deep-space
observation, the communications network, resource extraction and colony logistics — are
named **referentially only** and deliberately left unspecified (§13). Do not treat a
passing mention here as a design for them.

---

## 1. Pillars

Thalos is a space-programme sandbox differentiating from KSP on two axes:

- **Achievement** — the payoff for a mission scales with what it cost you to be
  confident it would work, not just with the difficulty of the manoeuvre.
- **Scale** — a fictional system with real distances, real surfaces, and a genuine
  reason to go look: Pelagos.

You play the Thalos global space organisation. It is a civilisational project with
broad public backing (`lore/civilization.md`), and it **will not be shut down**. What
varies is how fast it moves, and that is entirely downstream of your decisions.

---

## 2. Modes

- **Programme mode** (main, the designed experience) — budget, mandate, techniques,
  flight heritage, crew, a calendar, contracts. Everything below describes this mode.
- **Sandbox** — everything unlocked, no budget, no consequence. For building and flying.

**Invariant: sandbox is programme mode with the constraints switched off, not a second
game.** Same construction model, same physics, same world, same vehicles. Anything that
forks the two into separate code paths is a bug (`CLAUDE.md`: one canonical path per
operation).

---

## 3. The core loop

```
  mandate + budget
        ↓
  charter a project ──── a launch vehicle, a probe, a crewed spacecraft, a facility
        ↓
  develop it ─────────── costs programme time and money; longer if it's a clean sheet
        ↓
  fly it ─────────────── heritage accrues, or it fails and you learn why
        ↓
  returns ────────────── data · capability · contract revenue · milestones
        ↓
  data qualifies techniques · milestones raise mandate · revenue funds the next cycle
        ↓
  the next thing is harder, further, and now possible
```

Inside a single mission the loop is the game you already have: **design → integrate →
launch → fly → return**. The programme layer is what makes the hundredth launch mean
something different from the first.

The two halves reinforce each other rather than compete. Flying payloads and probes is
how a vehicle earns the flight record that lets you put people on it (§7.1); scouting is
how a crewed mission stops being a gamble (§5). There is no branch of the game that is
"the boring one you have to do first".

---

## 4. The consequence economy: hard, but you cannot lose

The design constraint: **hard but doable, perseverance rewarded, failure never grueling
or insurmountable, no true loss state — yet decisions must genuinely matter.**

Those are only compatible if you are precise about what a bad decision *costs*. The model
below is recorded as
[ADR-20260721T210002Z-setback-not-failure-economy](adr/20260721T210002Z-setback-not-failure-economy.md).

### 4.1 Time is the penalty currency

**Failure sets you back; it never blocks you.** Almost every consequence in the game is
denominated in programme time, because time is the one currency that:

- cannot be driven to zero and strand you — the clock always advances, appropriations
  always arrive;
- is genuinely the scarcest resource in a real space programme (windows, decades,
  careers);
- scales smoothly from a lost month to a lost two years without changing kind.

A mistake costs you a window. A catastrophe costs you a programme stand-down. Neither
ends anything.

**A setback gates the shiny new thing behind overcoming it.** This is the framing that
keeps time penalties motivating rather than punitive: the cost of a failure is not
primarily "wait", it is *the next tier of capability is on the far side of fixing this*.
The player always has something to do, and the thing they most want is visible and
reachable. Dead time is a design failure; a blocked door with a known key is not.

### 4.2 Budget is a rate, not a balance

**Funding is a recurring appropriation, not a bank account you can bankrupt.** This is
the specific mechanism that prevents the career-mode death spiral where you cannot afford
a rocket to earn the money to afford a rocket.

- You can overspend into a deficit that eats future appropriations. That is a debt, and
  it heals with time.
- You can never reach a state where you are unable to fly anything.
- Contracts (§11) raise the rate; they do not rescue you from insolvency, because
  insolvency is not reachable.

This is also the lore's own framing: the world government funds the programme against
agriculture and ocean infrastructure, and the tension is *how fast do you push*, not
*can you make payroll*. A rate tension, not a solvency one.

### 4.3 Mandate modulates the rate

The second currency is **mandate** — public support and political capital. Milestones,
firsts, and discoveries raise it; negligence lowers it. Mandate sets the size of the
appropriation and gates the progression phases.

**Money pays for missions; mandate decides how ambitious you are allowed to be.** If a
player could win without leaving low orbit, the pillars would be broken — so contracts
buy budget, but only milestones buy mandate.

### 4.4 Crew: the penalty scales with negligence, not with outcome

Lives are the most valuable thing in the programme. Crew are named individuals with
years of training and their own histories, recorded in the archive (§12).

The word doing the work is **unnecessarily**. Losing crew on a well-prepared mission that
hit genuine bad luck is a tragedy the organisation absorbs and publicly backs you through.
Losing crew on an unproven vehicle you rushed to hit a window, with no abort mode and no
uncrewed test flights, is a scandal.

**That distinction is mechanically computable, because the game already knows what you
skipped:**

| Diligence | Known from |
|---|---|
| Was the vehicle flight-proven? | heritage record (§7.1) |
| Did you fly the uncrewed tests? | flight history |
| Were there abort modes? | the vehicle's parts |
| Did you have the target data? | mission data products (§5) |
| Did you rush development? | project schedule |

The mandate penalty is derived from **demonstrable skipped diligence**, not from the
death itself. That makes it feel like accountability rather than an arbitrary tax, and it
makes caution a strategy rather than a mood.

### 4.5 The stand-down, and why failure should be survivable

After a crew loss, crewed flight pauses pending review. Historically exactly right:
Challenger grounded the shuttle 32 months, Columbia 29. Mechanically it is close to
ideal — a pure **time** cost, dramatic, fully recoverable, and it does not stop the
uncrewed programme, so the player keeps playing; they just cannot do the thing they most
wanted for a while. That is precisely "hard but never stuck".

**Recovery from a setback is a campaign, not a timer.** A stand-down lifts when you have
*demonstrated the fix* — a qualifying uncrewed flight of the corrected vehicle, the way
STS-26 flew the redesigned SRB joint. That turns the penalty window into playable content
with a clear objective, rather than months of clicking past. Every setback should have a
return-to-flight shape like this: something to build, something to prove, and the door
opens.

Two rules give failure the right texture:

- **Most failures should be survivable anomalies you respond to, not binary loss.** An
  engine out at T+80 s means a lower orbit and a salvage problem, which is a story. An
  unrecoverable fireball with no warning is a quit-to-desktop.
- **Crewed vehicles have real abort modes.** A launch escape system turns the majority of
  launch failures into "we lost the vehicle, the crew came home" — the best failure in the
  game. This also makes the escape tower a genuine design decision rather than a mass
  penalty players learn to skip.

Savescumming is mitigated by making failure *interesting and recoverable*, not by trying
to prevent reloads. A stand-down you have to fly your way out of is a better deterrent
than a lockout.

---

## 5. Information as the scarce resource

**In KSP the player already knows everything.** Terrain is fixed and wiki'd, Δv maps
exist, every atmosphere is documented. The only unknown is the player's own competence.
So scouting buys nothing, probes are strictly worse crewed missions, and "science" is
currency-farming detached from what science *is*.

Thalos is a fictional system. **What you know is earned**, and that is the spine of both
the progression and the achievement pillar. The near-term expressions are concrete and
small:

- **Your own atmosphere** is an unknown you measure with sounding rockets (§9) — density
  profile, winds aloft, where the useful ceiling is. This is the entire core loop in
  miniature on turn one, and it needs no new subsystems.
- **Target environments** are what probes buy: atmosphere profiles that let you size an
  entry, terrain data that turns a landing site from a coin flip into a choice, gravity
  fields that tighten an insertion burn, composition that justifies an extraction
  programme.

**Science is data you returned, not points you farmed.** Data products qualify
techniques (§6); they are not a currency spent at a shop.

**Invariant: scouting is optional and expensive to skip.** The moment probe data is a
checkbox that unlocks crewed flight, it is a chore gate and the pillar inverts. It is
*risk you are permitted to accept* — you may send crew to an unscouted body, and it will
probably kill them, and the game lets you. That permission is where the achievement comes
from when it works. It also feeds §4.4 directly: flying blind is exactly the kind of
skipped diligence the consequence model can see.

> **Deferred (§13):** how an unscouted body *presents* — withheld maps, coarse terrain,
> progressive survey resolution — is a large feature entangled with deep-space
> observation and communications. Not designed here. The principle above stands
> independently of how it is eventually shown.

---

## 6. Progression: techniques → contractors → parts

Research is not a linear unlock ladder; it is a **vocabulary of techniques** (fin
stabilisation, gyroscopic attitude reference, inertial guidance, staged combustion,
cryogenic turbopumps, regenerative cooling, radioisotope power, autonomous flight
computers, …). Techniques plus **returned data** qualify the programme for parts — some
parts gate on knowing a target environment, not only on theory.

**Parts are authored by us and are shared canon.** Engines, command pods, instruments and
structural end caps arrive from in-world contractors as finished, named objects — the way
a KSP part does. They are not procedurally generated and not player-designed. See
[ADR-20260721T210000Z-authored-parts-player-integration](adr/20260721T210000Z-authored-parts-player-integration.md).

This is the authentic model. Arianespace buys Vulcain, Atlas V bought RD-180, ULA has
never built an engine. **The integrator is the interesting job**, and it is the job the
game gives the player.

Procedural geometry still carries the *connective* structure — tanks, fuselages,
interstages, fairings, wings ([construction.md](construction.md)) — so vehicles are
proportioned by the player rather than snapped from a fixed diameter set.

> **Realistic-looking rockets are a constraint outcome, not a modelling-tool outcome.**
> KSP vehicles read as toys because fixed diameters and no penalty for bad aspect ratios
> make stubby designs viable. What produces slender staged vehicles is aero, structural
> loads, fairing volume and tank stretch making them *the right answer*. If the
> constraints are soft, players build KSP rockets out of good procedural parts.

---

## 7. Where player agency lives: the integration layer

The player's creative identity is the **vehicle**, not the part. A vehicle is a
configuration they designed — stage count, engine selection, propellant choice, tank
proportions, fairing envelope, staging sequence — named by them, and unlike a part name
it refers to a genuinely distinct object.

The problem to solve is that KSP gives **zero reward for reusing a design**: every mission
you open the VAB and build a bespoke rocket perfectly sized for that payload, at no cost.
So nobody builds a family, and nobody ever feels like a space programme. Four converging
pressures make a robust, reused launch system the correct play.

### 7.1 Flight heritage — reliability *is* information

A vehicle configuration accumulates a flight record, and its observed failure rate
converges as you fly it. This closes the loop back to §5: **you know your proven vehicle's
odds because you flew it.** A clean sheet's odds are *unknown*, and that uncertainty is
the penalty — not a rigged die.

Human-rating is N consecutive successes. Real: Saturn V flew uncrewed twice before Apollo
8; Falcon 9 flew ~80 times before carrying crew. So **payloads and probes are how you earn
the right to put people on your rocket** — the scouting layer and the vehicle layer feed
each other, and §4.4's diligence check reads directly off this record.

### 7.2 Derivation is cheap; clean sheets are expensive

Stretching a tank, adding boosters, swapping an upper stage, uprating an engine: a short,
cheap programme inheriting most of the family's heritage. A new core is long, expensive,
and starts its record over.

That makes the **lineage/tier mechanic economically correct rather than cosmetic** — you
derive because it is the right call, not because a UI offers a "+1".

- Heritage attaches to **configurations** and to **components**. Reusing a proven engine
  on a new vehicle carries partial confidence (RD-180 → Atlas V); full confidence requires
  the full stack flown.
- Changing the **core** (first-stage structure + primary engine) starts a new family with
  partial carry-over.

> **Tiers must trade, not strictly upgrade.** A stretched tank costs margin somewhere;
> more boosters cost pad turnaround. A monotone ladder collapses the family into one best
> vehicle and destroys §7.4.

### 7.3 Launch windows bite

A new heavy-lift design needs 14 months; the window is in 8. Fly what you have and descope
the payload, or miss by two years. This is the best decision in the game, and it exists
only because development costs programme time (§4.1) and windows are real.

### 7.4 A family covers an envelope

"Efficient launch systems" is really: **cover a capability envelope with as few distinct
vehicles as possible.**

- Over-specialise → nine vehicles, each individually unproven, each needing its own
  tooling, pad configuration and integration flow.
- Under-specialise → flying a heavy-lift to lob a small comsat, at a loss.

This is the real EELV/NSSL procurement problem and the reason Ariane 6 has two
configurations. A player-owned optimisation that never resolves into a single right
answer, and stays interesting as the envelope grows.

Infrastructure is the fourth pressure, and it hooks into what exists: a family shares pad
configuration, tooling and integration facilities — the spaceport and its launchpads are
real structures in the world ([base_building.md](base_building.md)), not menu entries.

---

## 8. Control authority: the player is the flight computer

**The player flies. Always, mid-flight, hands on.** That is the game, from the first
sounding shot to the last outer-system insertion, and it is never taken away.

What the tech tree gates is **what you are allowed to do**, and the constraint is
diegetic:

> **You may do what the onboard guidance could plausibly have been programmed to do.**

The fiction is premeditated: the flight was planned and programmed before launch, and the
player is standing in for the computer executing it. That framing does a lot of work at
once. It keeps the game a flying game. It makes early limits *honest* rather than
arbitrary — you are not being denied control by a designer, you are flying a vehicle that
genuinely cannot do more than that. And it turns guidance technology into a capability
line the player feels in their hands rather than reads on a node.

Recorded as
[ADR-20260721T210003Z-diegetic-control-authority](adr/20260721T210003Z-diegetic-control-authority.md),
which also records the rejected alternatives — non-interactive pre-programmed early
flight, unrestricted KSP-style piloting, and modelling true light-time input lag.

### 8.1 The control-authority ladder

1. **Unguided / spin-stabilised.** Fins and spin do the work. You set the rail and ride
   it; authority is minimal and mostly consists of not making things worse.
2. **Open-loop programmed.** Gyro attitude reference, fin and vane actuation: you can hold
   an attitude and command a tilt. You *cannot* express anything that requires knowing
   where you are — no prograde/retrograde hold, no target state, no node following, and
   engine cutoff is on a timer, not on a measured velocity. Your instruments are poor, so
   you are partly flying blind even while flying by hand.
3. **Closed-loop guidance.** The vehicle knows its state and can steer toward one. Target
   states become expressible: burn to a measured velocity, hold a velocity vector, follow
   a manoeuvre node, abort on a condition. Orbit becomes repeatable rather than lucky.
4. **Autonomous flight computers.** The vehicle can operate reactively somewhere you
   cannot supervise it (§8.3).

### 8.2 How the ladder is expressed mechanically

`thalos_control` already models every attitude command as a tagged `ControlDemand`,
arbitrated by source priority into one `AttitudeController` and allocated to reaction
wheels and aero surfaces ([control.md](control.md)). The ladder is a **permissions and
fidelity layer over that**, not a new control path:

| Gated axis | What the tier changes |
|---|---|
| Available `AttitudeDemand` modes | `Hold` early; velocity-vector and `PointNose` holds at closed-loop |
| Available `DemandSource`s | the burn autopilot and node-following are closed-loop features |
| Actuation fidelity | rate limits, deadbands and lag on early gyro platforms |
| State-estimate fidelity | what the instruments actually tell you — early vehicles have no solved orbit |

The last row is the one that ties back to §5: **early flight is hard partly because you do
not know your own state precisely.** Post-flight telemetry is how you learn what actually
happened versus what you thought was happening — which is why the early loop of
*fly → read the trace → adjust → fly again* has teeth even though you were at the controls
the whole time.

### 8.3 Autonomy and distance — what the late tech tree is *for*

Every space game hand-waves the fact that you cannot pilot a probe in real time across
light-minutes. Thalos has an answer, and the answer is a progression system: **the
autonomy tier sets how far from supervision a vehicle can be and still do something
interesting.** Below the tier, distant operations are coarse and pre-sequenced; above it,
the vehicle is genuinely reactive and the player flies it.

This makes the advanced-computing line the thing that unlocks the game's two biggest
discoveries:

- **Piloting a submersible under Pelagos's ocean** — the narrative climax
  (`lore/civilization.md` Phase 3: the descent through blue haze into the shallows, and
  the first sight of a living reef).
- **Under the ice of Glacis** — the second, quieter discovery (Phase 5).

**The endpoint of the tech tree is the endpoint of the narrative.** The AI line is not a
stat bump; it is the thing that lets you go look. That is a much better shape than a
generic "electronics +15%" node, and it means the late game's most-wanted technology is
the one the whole civilisation founded the programme to use.

The exact coupling to light-time and bandwidth depends on the deferred communications
system (§13); the principle stands independently of it.

---

## 9. The opening arc: spaceport to first crew

You start with a **basic spaceport** — a pad, an integration building, a small authored
parts catalogue — and no orbital capability. The early game is uncrewed by necessity, and
it is where the core loop is taught with flights lasting minutes rather than hours.

### 9.1 Sounding rockets, and why they are not KSP's opening

Real programmes begin with instrumented suborbital shots: fin-stabilised solid vehicles
that measure the atmosphere and validate propulsion. Two properties make this a much
better opening than "here is a pad, go to orbit".

**You measure your own atmosphere before you can exploit it.** The density profile, useful
ceiling and winds aloft are unknowns you resolve with telemetry — the information pillar
(§5) running on turn one, on the homeworld, needing no observation or comms subsystem. It
teaches the whole game in miniature: *you don't know something → you build something to
measure it → the data makes the next mission possible.*

**You fly it, but with almost nothing to fly it with** (§8.1, rungs 1–2). A fin-stabilised
sounding vehicle gives you attitude and a tilt and little else, on instruments that barely
tell you where you are. The gravity turn stops being gamepad muscle memory and becomes
something you have to actually understand, because you cannot ask the vehicle to find it
for you. Each shot is short and cheap, so the early loop iterates fast and every flight
returns data whether or not it succeeded.

### 9.2 The uncrewed-first ladder

Each rung is a real milestone (mandate) and a real prerequisite (capability), not an
arbitrary gate:

1. **Sounding flights** — altitude and range milestones, atmospheric data, propulsion
   validation.
2. **Programmed flight and staging** — gyro attitude reference and vane actuation (§8.1
   rung 2) on a staged vehicle.
3. **First orbit, uncrewed.** Wants closed-loop guidance (§8.1 rung 3). Historically the
   hard one; expect several attempts, and design for those attempts to be interesting
   rather than punishing.
4. **First payload delivered** — the contract loop (§11) opens, and the budget rate stops
   being purely appropriated.
5. **Recovery and re-entry** — heat shields, chutes, a survivable descent. The hard
   prerequisite for crew, and the thing KSP lets you hand-wave.
6. **First crew** — requires vehicle heritage (§7.1), an abort mode (§4.5), and life
   support. By now your launcher has a flight record you earned doing useful work.

From there the phases in `lore/civilization.md` take over: Mira, the inner system,
Pelagos. This arc **refines the front end of that doc's Phase 1** rather than replacing
it.

---

## 10. Probes and mission specification

A probe is specified, not assembled part-by-part: the player writes a **mission
requirements document** and the programme delivers a bus.

- **Class** — flyby / orbiter / atmospheric probe / lander / rover / relay.
- **Payload** — which instruments, each an authored, technique-gated part with a real data
  product.
- **Constraints** — mass budget, power, mission duration, comms range, thermal
  environment, design life, and **autonomy tier** (§8.3), which bounds what the vehicle
  can be asked to do once it is out of supervision.

The generator solves the bus around that: array area vs. radioisotope power (a function of
distance from Pyros — in a fictional system the player must actually reason about it),
battery for eclipse, antenna gain for the link budget at range, propellant for insertion,
thermal management, redundancy for the duration.

**The design rule that keeps this from being a form: the constraints must conflict.** More
instruments → more mass and power → bigger arrays or RTG → more mass → a bigger launcher;
higher-energy target → longer cruise → longer design life → more redundancy → more mass
again. The player should feel like they are descoping a real mission. Nobody has shipped
that feeling in a game.

### 10.1 The spec screen is a front end, not a second construction system

The requirements spec **emits a normal `ShipBlueprint`** composed of authored parts, which
can then be opened in the shipyard and hand-edited. One construction model, one blueprint
format, two levels of zoom. See
[ADR-20260721T210001Z-requirements-spec-emits-blueprints](adr/20260721T210001Z-requirements-spec-emits-blueprints.md).

### 10.2 Delegation vs. agency

The player owns the **spec and the tradeoffs**; the generator owns the **layout**. The
player still designs the launch vehicle, plans the trajectory, and flies it — and at high
autonomy tiers, flies the probe itself at its destination (§8.3). Building is delegated.
*Getting it there, and operating it, is not.*

---

## 11. Contracts

Launch contracts sit alongside the player's own exploration goals, and their elegant
property is that **a contract is paid flight heritage** (§7.1). A commercial payload is
someone else funding the twelfth flight of your Meridian — which is what human-rates it
for the crewed programme. Exactly how Falcon 9 earned crew certification, and it means
contracts *serve* the exploration narrative rather than distracting from it.

A contract demands a specific point in the envelope: *4.2 t to a 700 km sun-synchronous
orbit, window in five months*. Your family covers it → straightforward money. It does not
→ decline, over-fly at a loss, or start a development programme.

Three rules keep contracts from becoming career-mode fetch-quest spam:

1. **Payload delivery only.** Never "plant a flag on Mira" — that is the player's own job,
   and mixing the two puts contracts in direct competition with the exploration narrative.
2. **Always declinable.** Being able to say no is what makes accepting mean something.
3. **Bounded cadence.** Contracts fund and prove; they must never *direct*.

The customers write themselves from `lore/civilization.md`: a maritime species whose
defining scarcities are arable land and fresh water funds ocean survey, weather, crop
monitoring and communications constellations before anything else. The regional
federations and the ocean/agricultural agencies are the natural clients.

---

## 12. Identity and the record

If parts are shared canon, every player has the same engine, and renaming it would be a
skin over a common object. The resolution is **two layers of canon**:

- **Parts are shared canon.** Everyone's engine is the same named engine. A *feature*:
  shared vocabulary between players, an authored-feeling world, a community and a wiki
  that can exist.
- **Vehicles, missions, crew and programmes are personal canon.** Genuinely distinct
  objects, so naming them is not cosmetic.

And what makes a name mean something is **the record attached to it**:

> *Meridian II — 14 flights, 13 successes. First crew to Mira. Retired after the Pelagos
> window.*

A résumé, not a label. A **mission archive** recording firsts with the actual vehicle,
date, crew, site and returned data is cheap to build and carries most of the emotional
payload of the achievement pillar — and it is what makes §4.4's crew losses land, because
the archive is where those people already were. In a fictional system, the player's
history *is* the canon.

---

## 13. Open questions and deferred systems

**Deferred — named referentially above, deliberately not designed here:**

- **Deep-space observation and the survey model** — how unscouted bodies present, how
  survey resolution progresses, what telescopes do. Large feature; §5's principle does not
  depend on its shape.
- **The communications network** — relay constellations, link budgets, light-time,
  bandwidth as a constraint on what a mission can return. It is the natural partner to the
  autonomy tier (§8.3): autonomy sets what a vehicle can do unsupervised, comms sets how
  supervised it is. Neither is specified here.
- **Resource extraction, ISRU and colony logistics** — `lore/civilization.md` has the
  economy; the mechanics are untouched here.

**Open within the core loop:**

- **Where exactly the control-authority limits bite** (§8.2). The ladder is right; making
  early flight feel *characterful and honest* rather than *sluggish and annoying* is a
  tuning problem with real risk, and it needs hands-on iteration rather than a spec.
- **Heritage carry-over numbers** — how much confidence a shared engine or shared core
  transfers. The balance crux of §7 and entirely unspecified.
- **Appropriation cadence and deficit depth** — §4.2 needs real numbers to prove it cannot
  strand a player while still biting.
- **How diligence is surfaced *before* a crewed flight** — §4.4 is only fair if the player
  can see what they are skipping at commit time, without it becoming a nag screen.
- **Whether crewed craft also get a requirements front end** (§10), or hand-assembly is the
  point for anything carrying people.
- **Time compression at the programme layer** — cruise time is when *other things happen*
  (several missions in flight, projects developing, a window closing). That implies a
  programme/calendar view. The Space Center hub is the natural home and is currently a
  menu.

---

## 14. Implications for systems already built

Design capture only; no work is scheduled. Recorded so current decisions do not foreclose
it:

- **The control-authority ladder is a permissions layer over `thalos_control`, not a new
  control path** (§8.2). The demand vocabulary, priority arbitration, one
  `AttitudeController` and one allocator already exist ([control.md](control.md)); the
  ladder gates *which demands and sources are available and how accurate the state estimate
  is*. Keep that seam clean — a guidance tier must never become a second control stack.
- **Fin and vane actuation is already modelled.** `Wing::control_surfaces` meshes hinged
  sub-surfaces about a real hinge axis, animates them from the fly-by-wire command, and
  derives per-axis authority from the authored window geometry and its CoM moment arm
  ([construction.md](construction.md), [aerodynamics.md](aerodynamics.md)). Early
  fin-stabilised vehicles are a parts-catalogue problem, not an engine problem.
- **`thalos_shipyard` stays the one construction model.** Any mission-spec UI is a front
  end over `ShipBlueprint` (§10.1), never a parallel assembler.
- **Sandbox must not fork the code path** (§2) — it is programme mode with constraints
  disabled.
- **The Space Center hub is the seam for the programme layer** (§13). Worth not hard-wiring
  it as "a menu with buttons".
- **Structures and pads are already real** (`base_building.md`), which is what makes §7.4's
  infrastructure pressure implementable rather than abstract, and what makes §9's "you start
  with a basic spaceport" a content decision rather than a new system.
- **Nothing here requires the terrain/observation work to move.** The opening arc (§9) runs
  entirely on the homeworld with systems that exist today.
