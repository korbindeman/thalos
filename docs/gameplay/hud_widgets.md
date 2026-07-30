# HUD widgets — the MFD slot

The ship-view HUD has one **Multi-Function Display (MFD)** slot (top-right):
a single panel that hosts exactly one *widget* at a time and, by default,
auto-selects the widget most relevant to the current flight context. A small
tab row lets the pilot pin a specific widget or hide the slot.

This replaced the old single hardcoded "TRAJECTORY" panel (`system_map_panel`),
whose show-gate (`throttle > 0.02` ⇒ "recently burning") popped an orbital
schematic up during atmospheric flight — useless in an airplane. The MFD makes
"which instrument is relevant right now" a first-class, data-driven decision and
gives a home to future cockpit displays (navigation display, docking, transfer
plotting, …) without another bespoke panel each time.

Code: `crates/runtime/game/src/hud/mfd/` (`mod.rs` + `widgets/`). Added by `MfdPlugin`
from `HudPlugin`.

## Shape

- **Slot container** (`mfd/mod.rs`) owns the panel frame, the selector tab row,
  and a *widget area* that the widget roots parent into. It is built in one
  system (`setup_mfd`) so widget roots never race a deferred parent insert.
- **Widgets** (`mfd/widgets/<kind>.rs`) each expose:
  - `build(area, theme, …)` — spawn the widget's root (a `MfdWidgetRoot { kind }`
    node, `Visibility::Hidden`) and children under the area.
  - `relevance(&FlightContext) -> Option<i32>` — a pure priority. `None` =
    "not applicable now"; higher number wins auto-selection.
  - `update(...)` (optional) — a system, gated on being the active widget, that
    refreshes the widget's contents.
- **`FlightContext`** (`Resource`, sole writer `update_flight_context`) — the
  per-frame situation the `relevance` functions read. Derived only from
  always-available signals (no dependency on the regime bubble, which can be
  absent): `in_atmosphere` (`AeroReadout.density_kgm3`), `prediction_shown`,
  `recently_burning`, `has_nodes`, `altitude_m`, `nearest_runway_m`.
- **`MfdSelection`** (`Resource`, Reflect): `Auto`,
  `Pinned(WidgetKind)`, or `Hidden`. Sole writer `handle_tab_clicks`.
- **`ActiveWidget`** (`Resource`) — the resolved widget. Sole writer
  `select_active_widget`.

## Invariants

- **`HudPanel` lives on the slot container only**, never on widget roots.
  `hide_in_photo_mode` flips every `HudPanel`'s visibility; container-only
  tagging means photo mode hides the whole slot through inheritance while the
  selector keeps sole ownership of which widget shows (no one-frame flashes).
- **One pass, one visible.** `select_active_widget` resolves the selection,
  then in a single pass sets the chosen root `Inherited` and every other root
  (and the slot container, when nothing is active) — all diff-writes so it
  coexists with the photo-mode / shipyard-editor visibility writers without
  ordering constraints.
- **The selector bezel stays reachable.** In ship view the slot container (the
  selector tab row) is always visible, even when no widget is shown (`Hidden`,
  or `Auto` with nothing relevant) — otherwise turning the slot off would make
  it impossible to turn back on. Only the widget content below the tabs
  collapses. The whole slot hides only in map view (the map already draws the
  full 3D trajectory) and in photo mode / the editor (via the container's
  `HudPanel`).
- **Auto-pick priority.** `Auto` chooses the highest-`relevance` widget, ties
  broken by `WidgetKind::ALL` order (earlier wins). `Pinned(k)` always shows
  `k` (its own "no data" state if `k` wouldn't otherwise be picked). `Hidden`
  shows no widget (but keeps the selector).

## Widgets (current)

| Kind | Relevance | Notes |
|------|-----------|-------|
| `Trajectory` | `!in_atmosphere && prediction_shown && (recently_burning \|\| has_nodes)` → 60 | Top-down orbital schematic (`system_map.wgsl`). The `!in_atmosphere` gate is the fix for the old panel popping up in cruise. |
| `NavDisplay` | `in_atmosphere` → 100; else low over a runway → 90 | Airliner heading-up ND (`nav_display.wgsl`): compass rose, craft, **true-scale** runways with threshold bars, and the armed approach route. Owned by [navigation](navigation.md); the widget is a projection of `route::RouteState`. |
| `Docking` | `None` (stub) | Placeholder until craft/port targets exist. |
| `Interplanetary` | `None` (stub) | Placeholder for transfer-window / heliocentric plotting. |

## Adding a widget

1. Add a variant to `WidgetKind` (and its `ALL`, `tab_label`, `relevance`
   dispatch).
2. Create `mfd/widgets/<kind>.rs` with `build` + `relevance` (+ `update` if it
   has live content). Root carries `MfdWidgetRoot { kind }` and starts
   `Visibility::Hidden`.
3. If the widget renders through a UI-material shader, register the material in
   `MfdPlugin::build` (`UiMaterialPlugin::<…>`), build it in `setup_mfd`, and
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
- Range snaps to a 2 km – 300 km ladder containing the armed route (or the nearest
  runway when nothing is armed), with step-down hysteresis.

Assembly is the pure `nav_display_data(&NavScene)` over a scene built by the pure
`build_nav_scene(&NavSceneInputs)`, which is what lets **`just nd-preview`**
render the real pipeline headlessly (see navigation.md § Verification).

`nav_display.wgsl` mirrors `NavDisplayData` field-for-field and draws everything
as signed-distance shapes in the normalised `[-1, 1]` plot space. Route points are
packed two per `vec4`; the `MAX_*` counts must match on both sides.

## Verification

- `just game cruise` (airliner in atmosphere) → the slot shows **ND**, not the
  orbital plot (the headline bug fix).
- `just nd-preview` → eight ND panels from real approach plans (agent-runnable;
  covers scale, route drawing, and threshold marking, **not** ECS wiring).
- `just game runway-approach` → ND shows the heading rose, craft centred, and both
  runways at true size; arm one and confirm the drawn route and the PFD's
  localizer/glideslope needles agree and track together as the aircraft
  maneuvers.
- A vacuum burn / pending node (`just game orbit`) → the slot shows
  **Trajectory**.
- Selector: pin each tab, confirm it stays regardless of context (DOCK/IPL show
  "NO DATA"); OFF blanks the slot; AUTO resumes context picking. `MfdSelection`
  is Reflect-registered.
