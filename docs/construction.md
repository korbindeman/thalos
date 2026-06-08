# Construction system (airframes, fuselages, modules)

Design spec for the next-generation shipyard: a single construction
model that builds **planes, rockets, ships, and space stations** from
one set of primitives, instead of the rocket-only stack the shipyard
ships today.

This is mostly a **design document**. It captures the shape agreed during
design so the next agent inherits the model rather than re-deriving it. The
current `thalos_shipyard` crate (see `crates/shipyard/`, summarised in §2)
is the foundation this **generalises** — we extend the existing attach-node
system, we do not fork it.

**Slices 1–2 (parametric wing, wing-pylon jet nacelle, and scalar intake
flow) have landed** — the rest of this spec is still forward-looking design.
See **§0 Implementation status** for exactly what exists in code today and
what is still on paper.

## 0. Implementation status

### Landed — Slice 1: the parametric wing

The first aircraft slice is in code. It adds a wing and the **surface /
footprint placement** capability (§4.3) without forking the rocket path:

- **`Wing` part** (`part.rs`, `catalog.rs::WingSpec`, `PartParams::Wing`):
  a tapered / swept / dihedral lifting surface parameterised by span,
  root/tip chord, sweep, dihedral, thickness (t/c), and incidence. One
  catalog entry (`wing_std`) serves main wing, tailplane, and fin —
  they differ only by parameters and mount. Dry mass = `mass_per_m2` ×
  planform area.
- **Surface mounting** (`attach.rs::SurfaceMount`): a placement component
  *parallel to* `Attachment`, carrying `(parent, kind, station, angle)`.
  It sits a part on a host's skin at a `(station, angle)` point and
  **opts out of diameter propagation**. Kept distinct from `Attachment`
  so sizing / shrouds / staging-topology are untouched; traversals that
  need the whole part graph union both. Serialized as a separate
  `surface_mounts` list on `ShipBlueprint` (`#[serde(default)]`, so every
  pre-wing save still loads).
- **Symmetry — KSP linked groups** (`attach.rs::SymmetryGroup`, §4.5).
  Symmetry is **not** a per-part flag: placing a footprint part under
  mirror mode stamps a **real counterpart entity** (a separate left and
  right wing), linked by a `SymmetryGroup { id, role }`. `sync_symmetry_groups`
  keeps each mirror in lockstep with its group's primary — params copied
  (handed fields like `Wing.incidence` negated), mount reflected across the
  host X = 0 plane; editing or deleting one affects the whole group.
  **Nesting** is first-class: a part placed on a wing that is itself a
  mirrored pair (a nacelle) is stamped onto *both* wings. The mirror is
  thus a normal part everywhere downstream — meshes are single-panel and
  stats / staging count each entity once (no ×2 multiplier). The blueprint
  persists a `symmetry_group` id per surface mount so a loaded craft
  re-links its groups in the editor; the game ignores it (flat parts).
- **Editor**: an *Aerodynamics* palette category; a **Mirror (2×)** toggle
  (Symmetry panel); arm a wing then click a hull body to mount it — the hit
  point becomes the `(station, angle)`, and with mirror on, an off-centre
  hit stamps a linked left/right pair while a top/bottom hit stays single.
  Inspector sliders for all wing parameters; the group status shows whether
  a part is a mirror primary / counterpart. Wing geometry rebuilds live. A
  **Horizontal layout** toggle (View panel) lays the whole build down
  KSP-SPH-style — a rigid display rotation in `update_part_transforms`;
  pointer-driven placement / tank resize convert hits back through its
  inverse so building stays correct either way.
- **Geometry feedback** (§8 "free now"): `ShipStats` reports total
  **wing area** and area-weighted **mean aerodynamic chord**; wing mass
  feeds dry mass, CoM, and a crude rod-model MOI.
- **In-game**: `crates/game/src/ship_view.rs` renders and positions wings
  with the same `wing_mesh` builder, so a saved aircraft looks right when
  spawned (as static mass — there is no flight model).

Deliberate slice-1 simplifications, to revisit:

- The wing cross-section is an **extruded box**, not a true airfoil loft
  (`wing_mesh.rs`). Recognizable at editor/orbit distance; upgrade to an
  airfoil section (and the control-surface fields below) later.
- Wing meshes rebuild on wing / mount change but **not** when the host
  diameter changes — re-touch a wing after resizing its fuselage to
  refit. Centreline-station mounting only (no fore/aft re-drag post-place;
  re-place to move).
- MOI for wings is a thin-rod approximation; CoM uses the on-axis mount
  point.

### Landed — Slice 2: wing-pylon jet nacelle

The first wing-hosted module is in code. It adds an atmospheric jet engine
catalog entry plus the **wing-host + auto-pylon** connector path from §4.1 /
§4.5:

- **`Mistral Jet Nacelle`** (`assets/parts.ron::mistral_jet`): an
  atmosphere-optimized kerosene air-breather (`EngineGeometry::JetNacelle`,
  `requires_atmosphere = true`). It consumes stored kerosene and declares an
  `intake_requirement`; its nacelle housing also declares a `builtin_intake`.
  Vacuum/air-breathing performance curves are still future work.
- **Surface mount kind** (`attach.rs::SurfaceMountKind`): old wing mounts
  default to `BodySkin`; nacelles use `WingPylon`, where `station` means
  span fraction and `angle` means chord fraction. Serialized
  `SurfaceConnection.kind` is defaulted, so pre-nacelle saves continue to
  load as body-skin mounts.
- **Editor**: arming a jet nacelle and clicking a wing mounts it under that
  wing. A mirrored wing automatically yields a mirrored nacelle pair from one
  engine part; a single wing/fin yields one nacelle. The generated connector is
  a rectangular pylon from the wing underside to the nacelle.
- **Geometry**: `engine_mesh.rs` owns the shared nacelle/pylon mesh builder,
  and both the shipyard editor and `crates/game/src/ship_view.rs` render from
  it. Rocket engines keep the old bell/frustum body.
- **Runtime graph**: live fuel crossfeed and staging now union
  `Attachment` + `SurfaceMount`, so wing-mounted engines can receive fuel and
  drop with their host subtree. Mirrored nacelles count double for dry mass,
  thrust, mass flow, and stage summaries.
- **Atmosphere gate**: `crates/game/src/fuel.rs` ignores
  `requires_atmosphere` engines unless the craft is inside the dominant
  body's `terrestrial_atmosphere.karman_line_m`. There is still no aerodynamic
  lift/drag model; the jet is construction-ready and scalar-thrust-ready, not
  a full flight model.
- **Ambient intake model**: air is not a stored resource and does not appear in
  `PartResources`. Engines can require an `AmbientIntakeKind` at full rated
  thrust; nacelles can provide `builtin_intake`, and separate catalog
  `Intake` parts (for example `intake_cone`) provide the same capture for
  later inlet/core layouts. At runtime, available capture and active demand
  are summed by ambient kind while the ship is in atmosphere. If capture is
  short, all engines of that kind are scalar-throttled by the same ratio; if it
  is absent, they produce no thrust and burn no fuel.
- **Resource storage whitelist**: the previous methalox-only tank schema
  (`methane_l_per_m3` / `lox_l_per_m3`) has been replaced by per-part
  `storage: [(resource, units, units_per_m3, ...)]` entries in the catalog.
  Any part kind can now opt into fixed or volume-scaled resource capacity, but
  a blueprint may only activate resources whitelisted by that part. Omitted
  blueprint resources mean "use catalog defaults"; explicit resource maps are
  the selected active pools. The editor inspector exposes this as Add/Remove
  controls plus amount sliders for the selected part.

Deliberate slice-2 simplifications, to revisit:

- Pylon placement is span/chord only; there is no fore/aft drag handle after
  placement. Re-place to move it.
- Mirrored nacelles are one logical part with doubled aggregate stats. The
  renderer draws both nacelles, but part-level torque, asymmetric failures,
  and per-nacelle collider/shadow primitives are deferred.
- Intake flow is whole-craft scalar plumbing, not a physical duct network.
  Capture has a constant efficiency and ignores speed, angle of attack,
  pressure, inlet shock losses, and engine-local starvation.
- Design-time Δv/stage estimates are still nominal and not environment-aware;
  the runtime gate enforces atmosphere availability.

### Landed — wet wings + structural fuselage

A small follow-on to the storage whitelist (§Slice 2) splits "carries fuel"
from "is structure", exercising the volume-scaled storage path on a
non-cylindrical part:

- **`wing_wet`** (`assets/parts.ron`): a `Wing` whose integral box is
  whitelisted for kerosene. Capacity is volume-scaled like a tank, but the
  volume comes from `catalog::wing_volume` — `planform_area × (t/c · MAC) ×
  WING_BOX_FILL` — so a bigger panel holds proportionally more fuel.
  `blueprint::storage_volume_for` and `recompute::recompute_wing_state` both
  route wings through it, so spawn-time and editor-resize capacity agree.
  Dry wings stay `wing_std` (empty storage). Because surface-mounted parts
  with no `FuelCrossfeed` component default to crossfeed-enabled, a wet wing
  is already in the same crossfeed component as the nacelle on its pylon —
  wing fuel feeds the engine with no runtime change.
- **`fuselage_structural`** (`assets/parts.ron`): a `Tank` with an empty
  `storage` whitelist — same stainless-steel skin, diameter propagation, and
  node stacking as a propellant tank, but load-bearing structure that holds
  no fuel.
- **Aircraft loadout**: `ships/jet.ron` and `ships/a220.ron` now stack a
  `fuselage_structural` body and carry their kerosene in `wing_wet` main
  wings (tailplanes/fins stay dry). This is the airliner pattern — fuel in
  the wings, dry structural fuselage — and a step toward the §5.3 internal
  fuel-fill layer.

### Next (the stated goal: gear + control surfaces)

- **Control surfaces become parameters of the `Wing`**, not separate
  parts: trailing-edge chord fraction + spanwise window + a
  hinge/deflection descriptor, so a flap / aileron / elevator / rudder is
  authored as part of the wing it lives on. The `Wing` struct documents
  this extension point.
- **Landing gear** is a *separate* footprint part kind reusing
  `SurfaceMount` + `MountSymmetry::Mirrored` for the L/R pair.

The remaining sections describe the full target model these slices build
toward; only the pieces named above exist in code.

Target home: M6 ("advanced ships"). The
geometry/editor work can start before the aero *simulation* exists —
the editor computes and displays geometry-derived references (mass,
CoM, wing area, MAC, volumes) long before there is a flight model to
fly them in.

## 1. Motivation

The shipyard today is an attach-tree of discrete parametric primitives
(pod / tank / engine / decoupler / adapter), each a body-of-revolution
mesh with named `top`/`bottom` nodes, diameter propagated parent→child,
staging derived from decoupler topology. That is a good rocket builder
and a poor airframe builder: a fuselage is not a chain of frustums, a
wing is not a body of revolution, and "where does the landing gear go"
has no answer in a stack model.

The goal is a builder where a player makes a simple SSTO **no harder
than KSP** — roughly six gestures — while an advanced player can author
an arbitrary fuselage cross-section, a multi-segment wing, a Space
Shuttle forebody, and a precisely partitioned interior. Simple by
default, deep on demand, **one data model** underneath both (the "easy"
tiers are constrained editors over the full model, never a separate
system).

## 2. Where we start (current shipyard)

`crates/shipyard/` — relevant pieces this design builds on:

- `attach.rs` — `AttachNode { diameter, offset }`, `Attachment {
  parent, parent_node, my_node }`, `Ship { root }`. Node-mating is the
  existing connection mechanism; the construction model **keeps and
  generalises it**.
- `part.rs` — `Part` marker + concrete component types (`CommandPod`,
  `Engine`, `FuelTank`, `Decoupler`, `Adapter`). These become "modules
  with end-nodes" (§3).
- `sizing.rs::propagate_node_sizes` — child node diameter inherits from
  parent at the mated node. This *is* the end-cap seam-match (§4.2);
  the construction model reuses it and **scopes it to stack/end-cap
  nodes only** (surface mounts opt out — see §4.3).
- `blueprint.rs` — `ShipBlueprint { root, parts, connections }`,
  `PartParams`, RON round-trip. The serialization format extends to
  carry lofts, modules, morphs, and the interior partition.
- `resource.rs` / `stats.rs` / `staging.rs` — `ResourcePool`,
  `ShipStats` (dry mass, Δv via Tsiolkovsky), decoupler-topology
  staging. These are the **shared sink** (§5.3): both structural tanks
  and internal fuel fills feed the same resource/mass/stats layer.
- `material.rs` + `shaders/ship_part.wgsl` — procedural panel/rivet
  detail. **Caveat:** the shader maps detail by *cylindrical* coords
  (`radius_top`/`radius_bottom`); a non-circular loft skin needs UVs
  generated from the loft. Not a blocker, but it won't transfer for
  free.

## 3. The three layers

The construction model is three layers over one shared hull geometry:

1. **Assembly layer** — structure and external geometry. Lofts,
   end-caps, footprint modules (wings, gear, engines, cockpit shells),
   connectors. Owns hull geometry, external features, structural mass,
   connectivity, staging. **Outside-in** interaction.
2. **Internal / loadout layer** — what fills the enclosed volume:
   bulkheads, decks (deferred — §6.3), compartments, and role-fills
   (crew / cargo / fuel / avionics). Owns interior partitioning and
   capacities. **Slice-view** interaction. *Mostly separate* from the
   assembly layer.
3. **Resource / mass / stats** — the existing shared sink (§2). Both
   upper layers feed it: a structural tank and an internal fuel fill
   both just produce `ResourcePool`s.

The coupling is deliberately **one-way**: the internal layer *reads*
hull interior volume from the assembly layer (minus volume that
structural modules **reserve** — cockpit shell, wing box, gear bays)
and **never edits geometry**. You can re-loadout a finished airframe
without touching the airframe; reshaping the hull just re-flows the
compartments.

## 4. Assembly layer

### 4.1 One primitive: the Module

Everything in the assembly layer is a **Module**. Modules differ only
by which **capabilities** they turn on:

- **end-node** — exposes attach node(s) for end-to-end / stack
  connection (the existing `AttachNode` mechanism). Nosecones,
  decouplers, station segments, today's pod/tank/engine.
- **footprint + morph** — mounts onto another module's *surface* at a
  (station, angle) point, deforming the host skin to accept it (§4.3).
  Wings, gear, engines, footprint cockpits, doors.
- **end-cap** — *replaces/terminates* a host's end with its own loft,
  seam-matched to the host's end diameter (§4.2). Noses, end-cap
  cockpits, Starship nosecone.
- **host** — *can be mounted onto*. A capability, not a special root:
  a fuselage hosts wings; a **wing hosts an engine**. This is what
  makes "engine on wing" and "module on station truss" uniform.
- **auto-connector** — generated bridging geometry between a footprint
  module and its host: a **fillet** (wing root), a **pylon** (engine
  under wing), a **strut**. One concept, different shapes, driven by
  the module↔host gap.
- **internal reservation** — claims interior volume the internal layer
  must treat as occupied (cockpit crew space, wing box, gear well).

A given module can combine these (an end-cap cockpit = end-cap +
internal-reservation + windows; a wing = footprint + host + connector +
internal-reservation).

### 4.2 Host lofts

The fuselage — and a wing — is a **stationed loft**: a centerline
(start straight-axis with per-station vertical offset; full 3-D spline
deferred — §6.1) with ordered cross-section **stations**. The skin is
lofted between consecutive stations.

Cross-section, simple → deep:

- **Default: superellipse** (width, height, corner roundness, vertical
  offset). Covers circle → ellipse → rounded-rectangle → flat-bottomed
  belly with four sliders — ~90% of real fuselages.
- **Promote to a control-point loop** (Bezier/Catmull) for arbitrary
  shapes. A promoted station stores points instead of four scalars;
  same loft, richer station.

A wing is the same primitive with airfoil-style stations (root/tip
chord, thickness, twist) along a spanwise spline. Multi-segment wings
(cranked deltas, gull wings) are just more stations. One wing module
serves main wing, h-stab, and v-stab — different size/placement/symmetry.

Lofts must **locally re-tessellate** around a footprint so a morph
(fillet, recess) has resolution to resolve — placing a module inserts/
refines stations near its footprint.

### 4.3 Placement modes

Three ways a module attaches; the first is what exists today.

- **End-node stack** — mate `my_node` to a parent's node. Diameter
  propagates (`propagate_node_sizes`). Rockets, station chains.
- **End-cap** — replace a host's *end*. The end-cap is a short loft
  whose seam station **inherits the host's end diameter** (the same
  diameter-propagation as a stack node) and tapers to its own profile
  (ogive / conic / blunt / droop / chined). No morph — an end-cap *is*
  the skin there, there is no underlying host surface to displace; it
  is governed by **profile + seam-match**.
- **Footprint** — sit on host *skin* at (station, angle). Parameterised
  by a **signed morph** (§4.4) and an auto-connector (§4.1). Surface
  mounts **opt out of diameter propagation** — a wing must not inherit
  the fuselage's local diameter as its size.

### 4.4 Morph: signed local displacement, no CSG

A footprint module deforms the host skin over its u-v footprint by a
**signed scalar displacement field** — *not* boolean/CSG mesh surgery:

- **+ bulge** — 747 upper-deck hump, fighter bubble canopy, sensor
  blister.
- **0 flush** — 777 flight deck (the nose taper + a windshield region
  do the visual work; no displacement).
- **− recess** — landing-gear bay, weapons bay, sunken/faired-in
  canopy.

The gear-bay recess and the canopy hump are the **same mechanism with
opposite sign**. Field displacement is chosen over CSG for robustness
(displacing a parameterised surface never falls apart on edge cases),
speed, determinism, and composition: overlapping morphs blend by
**smooth min/max**, so order mostly does not matter. Limitation: it
cannot punch arbitrary topology (a clean hole through the body) — but
that is rarely needed; an intake is a **tunnel module**, not a hole.

### 4.5 Symmetry and connectors

- **Symmetry / mirror** is first-class and default-on for off-centerline
  mounts (gear, wings, engines come in pairs). Place once → mirrored
  pair. Centerline mounts (v-stab, dorsal door) don't mirror.
- **Connectors** (fillet / pylon / strut) are generated from the
  module↔host relationship: a fillet when a footprint meets the skin at
  an angle (wing root), a pylon when a module stands off from its host
  (engine under wing).

## 5. Internal / loadout layer

### 5.1 Compartments from dividers

The interior is the hull loft partitioned into **compartments** by:

- **Bulkheads** — transverse cuts at a station. On a simple tube this
  is the *entire* model (a 1-D front-to-back ruler).
- **Decks** — horizontal floors at a waterline, optionally over a
  station range. A deck splits a segment vertically (747 main deck +
  belly cargo; upper deck). **Deferred** (§6.3) — the simple case needs
  only bulkheads; decks slot into the same system later.

A compartment is a cell bounded by two bulkheads, the skin, and
optionally a deck. **Modules don't connect to each other internally —
they fill compartments, and adjacency in the partition is the
connection.** (Functional connections — crew corridors, fuel crossfeed
between non-adjacent tanks — are a thin opt-in graph over compartments,
generalising the existing `FuelCrossfeed`.)

### 5.2 Volume by integration

Compartment capacity is a **numeric integral**: slice the compartment
axially, compute each cross-section polygon's area clipped by the deck
planes, sum × thickness. This works identically for a superellipse tube
and a double-bubble with arbitrary control-loop sections — which is what
makes the model scale past a 747 with no special-casing.

### 5.3 Role-fills and the shared sink

A compartment is assigned a **role**: crew, cargo, fuel, avionics,
ballast, empty. Fuel produces a `ResourcePool` sized from the integrated
volume × packing efficiency — exactly like the existing
`recompute_tank_state` path, but volume comes from the loft, not a
cylinder. Crew/cargo become mass + capability. Everything feeds the
existing `ResourcePool` / `ShipStats` sink (§3 layer 3).

Fuel therefore exists **two ways**, both feeding the same pools:

- **Structural tank** (Starship, today's `FuelTank`) — the hull segment
  *is* the tank. Assembly layer.
- **Internal fuel fill** (airliner wet wing / belly tank) — a
  compartment with the fuel role. Internal layer.

Catalog storage is now whitelist-based: a part's `storage` list says exactly
which resources it can physically house and how capacity is computed. Fixed
capacity (`units`) covers batteries and service modules; volume-scaled capacity
(`units_per_m3`) covers tanks and future loft compartments. The same
`PartResources` map is still the runtime sink, but blueprint serialization
distinguishes omitted resources (catalog default loadout) from explicit maps
(the user's selected active pools).

### 5.4 Optional internals, default auto-loadout

**Closely managing internals must be optional.** The internal layer
always carries a complete, valid **auto-loadout**, so opening the slice
view is never *required*:

- Leftover volume (minus structural reservations) defaults to **fuel**
  for a rocket/SSTO — the KSP-equivalent of "the hull is a tank,"
  maximising Δv with zero clicks.
- **Payload is the one expected action** and is a single gesture:
  "add payload bay" carves a compartment out of the default fuel volume
  and gives it a default door (§5.5).
- Everything beyond that — multiple bays, crew/cargo/avionics splits,
  precise volumes, decks — is opt-in in the slice view. Templates
  (airliner = pax cabin + belly cargo + wet wings) pre-set richer
  loadouts for people who start there.

A simple SSTO is then ~6 assembly gestures (tube + end-cap cockpit +
wings + engines + gear) → hull auto-fuels → "add payload bay". Not
harder than KSP.

### 5.5 Cargo doors

A door is a **skin region** (a u-v patch, §4 region concept) plus a
**kinematic descriptor**. Because the door panel is a piece of the hull
loft, it conforms to whatever shape the skin has there — every style
falls out of three parameters, no per-style geometry authoring:

- **Footprint** — where on the skin (top / side / belly / end), which
  compartment it serves, length + angular span.
- **Kinematic** — hinge (axis + swing range), slide, or fold-down ramp.
- **Leaf count** — single vs double (clamshell).

Named styles are presets over those:

| Style | Footprint | Kinematic | Leaves |
|---|---|---|---|
| Shuttle payload bay | dorsal, bay length | hinge along top edges, swing outboard | double |
| 747 nose visor | nose end-cap | hinge near top, swing up | single |
| 747F side cargo | upper side | top-edge hinge, swing up/out | single |
| C-17/C-130 rear ramp | tail belly | bottom-edge hinge, fold to ground | single (or split) |
| Bomb / belly bay | belly | longitudinal-edge hinge, swing down | double |

Two unifications: cargo doors are the **same mechanism as gear-bay
doors** at a different scale, and a door is an **assembly-layer skin
feature bound to an internal-layer compartment** it grants access to
(same cross-layer link as the cockpit's volume reservation). Author from
either side — drop a door on the skin (auto-binds to the bay behind it)
or create a bay and request a door. The kinematic is a real deploy
state gameplay drives later (Shuttle deploying a satellite through open
doors), not just cosmetic.

### 5.6 Cockpit

A **full** cockpit (flight deck, crew capacity, windows), organised as
**size class × {end-cap, footprint}**:

- **End-cap variant** (777, Cessna, fighter) — brings its own nose,
  seam-matches the tube diameter. The *default*. Mechanically "a
  nosecone that inherits the diameter, plus crew + a windshield region."
- **Footprint variant** (747, Shuttle) — mounts on/in a separately
  authored forebody, ends untouched. Uses a signed morph (+bulge =
  747 hump).

Either way the cockpit is a structure-layer module that **reserves** its
crew volume (the §3 handshake into the internal layer). The nose is a
first-class authorable end-cap loft, **decoupled** from the cockpit, so
you can also author a bare nose and place a footprint cockpit behind it.

## 6. Worked examples

- **Simple SSTO** — tube + end-cap cockpit + wings + engines + gear
  (~6 gestures), hull auto-fuels, "add payload bay". KSP-parity.
- **777** — tube; **end-cap cockpit** (brings nose, flush); main + tail
  wings (mirrored); engines on wings (auto-pylon); gear (mirrored);
  cabin = crew, belly = cargo, wings = fuel.
- **747** — authored nose + **footprint cockpit** with +bulge (the
  hump); decks split main deck / belly cargo (deck tier); nose-visor
  cargo door.
- **Space Shuttle** — authored chined forebody; footprint/integrated
  cockpit; dorsal double-leaf payload-bay doors; delta wing.
- **Starship** — stack of **host-loft segments** (end-node connected) +
  **footprint fins** + **end-cap nosecone**. The fuselage primitives,
  vertical.
- **Space station** — modules connected purely by **end-nodes**, no loft
  host. (`host` and `end-node` being independent capabilities is what
  makes this free.)

## 7. Mechanism summary

- **Morph** = signed scalar displacement field, smooth-min/max
  composition, no CSG.
- **Connector** = generated bridge (fillet/pylon/strut) from the
  module↔host gap.
- **Interior** = loft partitioned by bulkheads (+ decks later) →
  compartments; capacity by clipped cross-section integration.
- **Shared sink** = `ResourcePool` / `ShipStats`; structural tanks and
  internal fills both feed it.

## 8. Deferred / computed-now

**Defer:**

- Full 3-D spline centerline (§6.1) — start straight-axis + per-station
  vertical offset.
- Decks / multi-deck interiors (§6.3) — bulkheads-only carries fighters,
  GA, landers, most cargo.
- Aero *simulation* — M6; Thalos has no atmosphere yet.
- Runtime door/gear animation + payload deploy/load gameplay.

**Free now (geometry-derived editor feedback, no flight model needed):**

- Mass, center of mass (the stationed model makes CoM a straight
  integral, like the existing Δv stat).
- Wing area, mean aerodynamic chord, estimated center of lift — so a
  player gets "will this fly" intuition before there is a sim to test
  it in.
- Compartment volumes / capacities.

## 9. Open decisions

1. **Centerline** — straight-axis (proposed start) vs 3-D spline.
   Curvature interacts with morphs (a fillet on a curved tail is
   harder).
2. **Conflict resolution** — when two module footprints overlap, or a
   bay won't fit the available volume: reject / blend / push. Smooth-min
   handles the *skin*; *volume* contention needs an explicit rule.
3. **Bulkheads** — explicit player-placed vs implied by modules. Leaning
   module-implied with optional manual bulkheads.
4. **Auto-loadout role inference** — default everything to fuel +
   manual payload (proposed), vs template/cockpit/wing-driven inference
   (airliner → pax + cargo + wet wing).
