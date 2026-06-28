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

Code: `crates/game/src/hud/mfd/` (`mod.rs` + `widgets/`). Added by `MfdPlugin`
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
| `NavDisplay` | `in_atmosphere` → 100; else low over a runway → 90 | Airliner heading-up ND (`nav_display.wgsl`): compass rose, craft, runways + extended-centerline approach. |
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

The ND reuses the shared **`hud::geo::local_enu_basis(craft_pos, body_pos)`**
(extracted from the PFD's `attitude_angles`, so ND and PFD headings agree):
`up` = radial-out, `north` = world-Y projected onto the tangent plane (X-axis
fallback at the poles), `east = north × up`.

Per runway from `StructureRegistry::sites_on(dominant)` (filtered to
`StructureKind::Runway`):

- Surface point (inertial), mirroring the runway's own placement, via
  `mfd::runway_surface_inertial`:
  `body_pos + body_orientation * (anchor_dir * (radius + elevation))`, where
  `elevation` comes from `placement` (`FlattenTo { elevation_m, .. }`).
- Heading `psi = atan2(nose·east, nose·north)` (0 = north, 90° = east).
- Heading-up screen coords for a ground offset `(e, n)`:
  `x = e·cos psi − n·sin psi`, `y_fwd = e·sin psi + n·cos psi`,
  `y_screen = −y_fwd`, then scale by the adaptive view range.
- Runway orientation: the projected `heading_tangent` rotated the same way; the
  shader draws the runway rect along it and dashes the extended centerline back
  from the threshold.

The range snaps to a 2 km – 150 km ladder containing the nearest runway, with
hysteresis. `nav_display.wgsl` mirrors `NavDisplayData` and draws everything as
signed-distance shapes in the normalised `[-1, 1]` plot space.

## Verification

- `just game cruise` (airliner in atmosphere) → the slot shows **ND**, not the
  orbital plot (the headline bug fix).
- `just game runway-approach` → ND shows the heading rose, craft centred, the
  runway at correct relative bearing/heading, and the dashed approach line;
  bearing/heading track as the aircraft maneuvers.
- A vacuum burn / pending node (`just game orbit`) → the slot shows
  **Trajectory**.
- Selector: pin each tab, confirm it stays regardless of context (DOCK/IPL show
  "NO DATA"); OFF blanks the slot; AUTO resumes context picking. `MfdSelection`
  is Reflect-registered.
