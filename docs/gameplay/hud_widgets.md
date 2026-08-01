# HUD widgets — permanent launcher and floating workspace

## Target direction: instruments are always within reach

The flight HUD has one compact, **permanent widget launcher panel**. It is the
catalogue for ND, TRAJ, DOCK, transfer plotting, and later instruments: selecting
an entry opens or focuses that widget as a separate floating window on the HUD.
The launcher itself stays fixed and cannot be closed. It hides only when the
flight HUD as a whole hides (photo mode, map/editor/non-flight contexts).

Floating widgets are immediately manipulable during normal flight. A header
drags the window, a close control removes it from the current workspace, and
resizing is constrained by the widget's declared minimum/maximum size and aspect
policy. No edit mode is required for these essential operations. Closing does
not forget placement; reopening restores the last position. Selecting an
already-open launcher entry focuses it rather than creating accidental copies;
an explicit duplicate action may create a second instance of the same kind.

The three separate concepts are:

- **widget kind** — ND, TRAJ, DOCK, transfer, and so on;
- **widget instance** — a stable ID with its own position, size, binding, and
  view state (for example its own ND range);
- **HUD workspace** — the persisted collection and stacking order of instances.

Several kinds, and deliberately several instances of one kind, may coexist.
Context and craft capability choose the first-run defaults and what the launcher
offers; they never rearrange an explicit pilot workspace. An optional AUTO
instance may switch its own content without spawning, closing, or moving other
windows.

All launcher and window actions route through one `HudLayoutRequest` command
path (`Add`, `Focus`, `Move`, `Resize`, `Duplicate`, `Close`, `Reset`) with one
layout owner. Widget state is per instance, not a resource that encodes "the one
ND" or "the active widget". Widgets remain projections: ND still reads the one
navigation authority and emits `RouteRequest`; TRAJ reads the canonical
prediction and maneuver state. The workspace owns presentation, never flight
truth.

Placement is free screen-space layout rather than a mandatory docking grid.
Saved placement is normalised to the usable viewport, clamped into the screen on
load and after resolution/UI-scale changes. Gentle edge/widget snapping belongs
to the responsive-layout follow-up. The workspace persists through the unified
`settings.ron` seam, with a draft during dragging and one settings commit when
the interaction ends.

Decision: [ADR-20260801T070234Z](../adr/20260801T070234Z-permanent-widget-launcher-and-floating-workspace.md).

### Workspace boundary

Self-contained boxed instruments belong in the workspace: ND, TRAJ, DOCK,
transfer plotting, and later staging/orbit/system panels where that improves the
flight deck. Camera-aligned PFD ladders/tapes, flight-director and world
symbology, global time/view controls, and transient safety annunciations remain
fixed overlays.

## Current implementation: launcher plus floating windows

The permanent **WIDGETS** launcher sits at the top-right of the ship-view HUD.
Its craft-filtered buttons open or focus TRAJ, ND, DOCK, and XFER. Each kind is
an independent floating window with an always-available drag header and close
button, so ND and TRAJ can remain on screen together. Closing retains its last
position; selecting the launcher entry restores it there.

The old contextual MFD selection remains only as first-run policy: the initial
workspace opens ND for a winged craft and TRAJ for a rocket/capsule. After that,
context never closes, opens, or moves a pilot-arranged window.

Code: `crates/gameplay/hud/src/mfd/` (`mod.rs` + `widgets/`). Added by `MfdPlugin`
from `HudPlugin`.

### Current shape

- **`HudWorkspaceSettings`** is the persisted `settings.ron` section: stable
  window IDs, open state, normalised placement, and stacking order. Loaded
  positions are repaired/clamped before use.
- **`HudWorkspaceRuntime`** is the drag draft. Pointer motion changes it only;
  release commits the final position to settings, so a drag does not write the
  file every frame.
- **`HudLayoutRequest`** is the one mutation path: initialise, open/focus, move,
  and close. `apply_layout_requests` is the sole layout reducer.
- **`ActiveWidgets`** is the derived visibility set consumed by the widget
  update/input systems. It replaces the scalar `ActiveWidget`, allowing several
  kinds to update in one frame.
- **Widgets** (`mfd/widgets/<kind>.rs`) still expose `build`, pure contextual
  `relevance` for the first-run default, and optional live update/input systems.
- **`FlightContext`** remains one derived answer for craft capability and the
  first-run choice; it no longer owns ongoing workspace selection.

## Invariants

- **The launcher is permanent, windows are optional.** The launcher is fixed
  and cannot close; every instrument can. All carry `HudPanel`, so photo mode
  and non-flight contexts suppress the complete workspace through the existing
  visibility contract.
- **Opening is additive.** Selecting ND cannot close TRAJ. Selecting an already
  open entry focuses it; close is the only ordinary removal path.
- **Unavailable is dormant, not deleted.** ND is absent from a rocket's
  launcher and its saved window is hidden, but the arrangement returns intact
  when a winged craft is active again.
- **Dragging owns the pointer.** Headers are native interactive UI, so the one
  `UiPointerGate` prevents the same motion from steering the craft or camera.
- **One pass derives visibility.** `sync_workspace_visuals` projects runtime
  layout, craft capability, and ship/map view into launcher/window/content
  visibility and the `ActiveWidgets` set.

## Widgets (current)

| Kind | Relevance | Notes |
|------|-----------|-------|
| `Trajectory` | `!in_atmosphere && prediction_shown && (recently_burning \|\| has_nodes)` → 60 | Top-down orbital schematic (`system_map.wgsl`). The `!in_atmosphere` gate is the fix for the old panel popping up in cruise. |
| `NavDisplay` | `in_atmosphere` → 100; else low over a runway → 90 | Airliner heading-up ND (`nav_display.wgsl`): compass rose, craft, **true-scale** runways with threshold bars, and the armed approach route. Owned by [navigation](navigation.md); the widget is a projection of `route::RouteState`. |
| `Docking` | `None` (stub) | Placeholder until craft/port targets exist. |
| `Interplanetary` | `None` (stub) | Placeholder for transfer-window / heliocentric plotting. |

## Adding a widget

1. Add a variant to `WidgetKind` (and its catalogue label/title, default
   placement, availability, and relevance dispatch).
2. Create `mfd/widgets/<kind>.rs` with `build` + `relevance` (+ `update` if it
   has live content). Root carries `MfdWidgetRoot { kind }` and starts
   `Visibility::Hidden`.
3. If the widget renders through a UI-material shader, register the material in
   `MfdPlugin::build` (`UiMaterialPlugin::<…>`), build it in `setup_workspace`, and
   add its `update` to the plugin's `Update` chain.
4. Extend `FlightContext` if the widget needs a context signal not already
   present (single writer: `update_flight_context`).

## Navigation-display projection

**The ND draws navigation; it does not compute it.** The route, the deviations,
and the runway selection all come from `crate::route` / `thalos_navigation` —
see [navigation.md](navigation.md), which is the spec for everything below the
symbology. What lives here is only the projection and the drawing.

One `RouteFrame` anchored at the craft does every projection: runways, route
points, and waypoints go through it into local east/north metres, then get rotated
heading-up and divided by the plot range. That frame's basis is built exactly like
the shared `hud::geo::local_enu_basis` (`up` = radial-out, `north` = world-Y
projected onto the tangent plane with an X-axis fallback at the poles,
`east = north × up`), so ND and PFD headings agree by construction.

- **Runways draw at true scale** from `StructureKind::Runway`'s half-extents, with
  a bar across the end being landed on. Only the *width* has a floor (a 90 m strip
  is sub-pixel at 20 km); length stays true. A strip longer than the plot still
  draws — culling tests the nearest point of the strip, not its centre.
- **The route is a real polyline** (arcs tessellated), with the final approach
  segment in its own brighter colour and waypoint symbols at the join point,
  threshold, and aim point.
- **Selection has two paths, one writer.** Clicking a strip on the plot arms it
  (clicking the armed one again lands the other way); the `< > FLIP CLR` row does
  the same for an off-plot strip. Both send a `RouteRequest` — `crate::route`
  stays the sole writer of the selection.
- Range is a 500 m – 300 km ladder. AUTO frames **what is still ahead** (the
  remaining route plus the armed threshold) with step-down hysteresis; `−` / `+`
  / the scroll wheel pin a rung and `AUTO` releases it (`NavZoom`, sole writer
  `handle_zoom`).
- The panel is laid out in blocks so it has a reading order: a header carrying
  the armed runway, phase, and distance-to-go; the plot; the zoom row; a
  guidance block pairing a **centring dot** (keep it in the middle: it carries
  localizer left/right and glideslope up/down in one glyph) with the secondary
  HDG / TRK / XTK / G-S readouts; then the runway selector.

Assembly is the pure `nav_display_data(&NavScene)` over a scene built by the pure
`build_nav_scene(&NavSceneInputs)`, which is what lets **`just nd-preview`**
render the real pipeline headlessly (see navigation.md § Verification).

`nav_display.wgsl` mirrors `NavDisplayData` field-for-field and draws everything
as signed-distance shapes in the normalised `[-1, 1]` plot space. Route points are
packed two per `vec4`; the `MAX_*` counts must match on both sides.

## Orbit-target control surface

The top-centre orbital widget (`hud/orbital_panel.rs`) is separate from the MFD
slot and remains the owner-facing surface for ORBIT configuration. Its compact
form continues to show live AP/PE. Clicking the orbital half expands an editor
in place for circular/elliptical altitude, inclination, direction, plane policy,
plan/execute, and cancel.

`AUTO SET` is a button, not a value. It populates the visible inclination and
direction from the live situation: the lowest directly reachable prograde
plane on a ground/ascent start, or the current orbit's concrete values in
space. The fields always show numbers/direction (`8°`, `PROGRADE`) afterward.
The adjacent policy selector toggles `NEAREST` / `PRESERVE`.

After deliberate pilot takeover, `EXEC ORBIT` becomes `RESUME ORBIT`. It keeps
the target and continues from the live ascent phase; it does not restart the
pad plan or test remaining fuel against the full original launch cost.

It follows the ND selection architecture rather than copying its state:

- widget interactions emit `OrbitTargetRequest`;
- the orbit-program system is the sole writer of target and plan state;
- the widget projects target/phase state, while the `Trajectory` MFD and map
  project the ordinary generated maneuver nodes;
- generated in-space burns are ordinary maneuver nodes owned and executed by
  the existing maneuver path.

The expanded editor is anchored below the compact panel and does not participate
in the balancing row's width, so opening it cannot shift the central altitude
readout. Full behavior and ground/ascent semantics:
[orbit_autopilot.md](../simulation/orbit_autopilot.md) §5.

## Verification

- `just game cruise` (airliner in atmosphere) → the launcher is permanent and
  the first-run workspace opens **ND**, not the orbital plot.
- A plane with no active route/burn starts on **ND**; a rocket or capsule starts
  on **TRAJ**, including while sitting on the pad.
- `just nd-preview` → eight ND panels from real approach plans (agent-runnable;
  covers scale, route drawing, and threshold marking, **not** ECS wiring).
- `just game runway-approach` → ND shows the heading rose, craft centred, and both
  runways at true size; arm one and confirm the drawn route and the PFD's
  localizer/glideslope needles agree and track together as the aircraft
  maneuvers.
- A vacuum burn / pending node (`just game orbit`) → the first-run workspace
  opens **Trajectory**.
- `THALOS_SCREENSHOT_HUD=1 just screenshot interstage` captures the expanded
  ground editor deterministically. The ordinary clean-scene shot keeps the HUD
  hidden.
- Workspace: open ND and TRAJ together; both launcher entries highlight and both
  windows remain live. Drag each by its header, close/reopen it, and restart the
  game; placement and open state must survive. A rocket hides ND without erasing
  its saved placement. DOCK/XFER show their honest `NO DATA` stubs.
- Deterministic multi-window evidence:
  `THALOS_SCREENSHOT_HUD=1 THALOS_SCREENSHOT_WIDGETS=traj,nd just screenshot
  runway-atmosphere`.
