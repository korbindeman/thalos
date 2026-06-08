# Input

Thalos routes semantic player input through `bevy_enhanced_input`
contexts owned by `thalos_input`. Raw Bevy input should remain only for
spatial data and third-party UI internals: cursor position, ray
projection, Bevy picking hover/click/drag events, egui widget input, and
shipyard pinch gestures.

## Crate Boundary

`thalos_input` is a Bevy-facing workspace crate shared by:

- `thalos_game`
- `thalos_body_editor`
- `thalos_shipyard`'s `ship_editor` binary

The crate owns:

- the checked-in RON schema and loader for `assets/input.ron`
- keyboard/mouse default bindings
- action/context definitions for each binary
- per-binary input intent resources collected after
  `EnhancedInputSystems::Apply`

Gameplay systems consume intent resources, not `ButtonInput`:

- `GameInputIntent`
- `BodyEditorInputIntent`
- `ShipyardInputIntent`

## Binding File

`assets/input.ron` is the editable source of default bindings. It uses
`version: 1` and separate sections for `game`, `body_editor`, and
`shipyard`. Missing individual actions fall back to code defaults.
Unknown action names, unknown axis names, and unknown source names fail
load with an error that includes the binding path.

Keyboard and mouse remain the enabled checked-in defaults. The binding
schema also accepts Bevy `GamepadButton(...)` and `GamepadAxis(...)`
sources so HOTAS/gamepad buttons can be mapped onto the existing
actions without bypassing context gating.

Continuous HOTAS flight axes use the separate `game.hotas` block,
disabled by default. It is intentionally profile-shaped rather than a
single "active gamepad" switch because flight sticks and throttles often
enumerate as separate Bevy `Gamepad` entities. The supported semantic
axes are:

- `pitch`, `yaw`, `roll` — signed attitude commands merged into
  `GameInputIntent.attitude`
- `throttle` — absolute `[0, 1]` throttle command; when connected it is
  the source of truth and keyboard ramp/full/cut becomes a fallback only

Each HOTAS axis binding names a Bevy `GamepadAxis`, optionally overrides
the device selector, and carries calibration fields:

```ron
hotas: (
    enabled: true,
    device: NameContains("T.16000M"), // or Any / Usb(...)
    axes: {
        "pitch": (axis: LeftStickY, invert: true, deadzone: 0.05),
        "roll": (axis: LeftStickX, deadzone: 0.05),
        "yaw": (axis: RightZ, deadzone: 0.05),
        "throttle": (axis: LeftZ, min: -1.0, max: 1.0),
    },
),
```

For multi-device setups, put `device: ...` on an individual axis to bind
that axis to a different physical device than the block default. Bevy's
standard HOTAS-ish axes are `LeftZ` for throttle and `RightZ` for yaw;
non-standard sliders and extra axes arrive as `Other(n)`.

## Context Model

Each binary spawns one controller entity with layered contexts.

Game:

- `GameSystemContext` for Escape, screenshot, and freecam
- `GameViewContext` for HUD toggle, map toggle, and camera cycle
- `GameFlightContext` for attitude, SAS, throttle, and HOTAS flight axes
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

