# Input

Thalos routes semantic player input through `bevy_enhanced_input`
contexts owned by `thalos_input`. Raw Bevy input should remain only for
spatial data and third-party UI internals: cursor position, ray
projection, Bevy picking hover/click/drag events, egui widget input, and
shipyard pinch gestures.

## Crate Boundary

`thalos_input` is a Bevy-facing workspace crate shared by:

- `thalos_game`
- `thalos_planet_editor`
- `thalos_shipyard`'s `ship_editor` binary

The crate owns:

- the checked-in RON schema and loader for `assets/input.ron`
- keyboard/mouse default bindings
- action/context definitions for each binary
- per-binary input intent resources collected after
  `EnhancedInputSystems::Apply`

Gameplay systems consume intent resources, not `ButtonInput`:

- `GameInputIntent`
- `PlanetEditorInputIntent`
- `ShipyardInputIntent`

## Binding File

`assets/input.ron` is the editable source of default bindings. It uses
`version: 1` and separate sections for `game`, `planet_editor`, and
`shipyard`. Missing individual actions fall back to code defaults.
Unknown action names, unknown axis names, and unknown source names fail
load with an error that includes the binding path.

Keyboard and mouse are the only checked-in defaults today. Keep the
schema extensible for gamepad sources, but do not add gamepad defaults
until gamepad behavior is designed.

## Context Model

Each binary spawns one controller entity with layered contexts.

Game:

- `GameSystemContext` for Escape and screenshot
- `GameFlightContext` for attitude, SAS, and throttle
- `GameWarpContext` for sim-time meta-controls (pause, warp speed,
  warp-to-maneuver) — always active except during egui text input, so
  pause is reachable from EVA, freecam, etc.
- `GameViewContext` for view and camera mode toggles
- `GameCameraContext` for orbit drag and zoom
- `GameManeuverContext` for node placement and deletion
- `GameManeuverPrecisionContext` for Shift/Ctrl precision while a
  maneuver drag is active

Planet editor:

- `PlanetEditorContext` for orbit camera, fullbright, placement click,
  and overlay suppression

Shipyard:

- `ShipyardContext` for orbit camera, wheel pan/zoom, primary pointer,
  and precision modifier

Use `ActionSettings::consume_input` for semantic keyboard actions that
should not bleed into lower-priority actions. Mouse camera actions stay
pass-through and are gated by pointer/egui spatial state in the
consumer systems.

## Escape Policy

`GameSystemContext` emits the Escape intent, and `pause_menu` owns the
priority policy:

1. close the pause menu
2. cancel an active maneuver interaction
3. clear the current target
4. open the pause menu

Do not reintroduce `ButtonInput::clear_just_pressed` to arbitrate this.

## Scheduling

Enhanced input updates in `PreUpdate`. Intent resources are reset before
`EnhancedInputSystems::Apply` and collected immediately after it.
Simulation and presentation systems then read those intent resources
during their normal `Update` sets.

