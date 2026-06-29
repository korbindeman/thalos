# Input

Thalos routes semantic player input through `bevy_enhanced_input`
contexts owned by `thalos_input`. Raw Bevy input should remain only for
spatial data and UI internals: cursor position, ray projection, Bevy
picking hover/click/drag events, native-UI text-field input
(`thalos_game::ui_widgets`), and shipyard pinch gestures. (The game UI is
native Bevy UI; `bevy_egui` is no longer a `thalos_game` dependency — only
the standalone `just shipyard` binary uses egui.)

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
enumerate as separate devices. The supported semantic axes are:

- `pitch`, `yaw`, `roll` — signed attitude commands merged into
  `GameInputIntent.attitude`
- `throttle` — absolute `[0, 1]` throttle command; when connected it is
  the source of truth and keyboard ramp/full/cut becomes a fallback only

### Why raw axis codes, not `GamepadAxis`

HOTAS axes are **not** read through Bevy's gamepad layer. Bevy's
`bevy_gilrs` converter (`convert_axis`) maps only the standard named
gamepad axes and discards every axis gilrs labels `Axis::Unknown` —
which, on a bare flight stick with no SDL gamepad profile (e.g. a
Thrustmaster T.16000M), is precisely the twist (rudder) and the throttle
slider. Those axes never reach Bevy's `Gamepad` state, so they cannot be
bound through `GamepadAxis` at all.

Thalos works around this in `thalos_input::joystick`: it runs its **own**
`gilrs::Gilrs` instance alongside Bevy's, reads every axis by its raw
platform `Code` (`Code::into_u32`), and snapshots `code -> value` per
device into the `RawJoystickState` resource each frame
(`poll_joysticks`). `collect_hotas_intent` reads that snapshot. So a HOTAS
axis binding names a **raw `u32` code**, not a `GamepadAxis`:

```ron
hotas: (
    enabled: true,
    device: NameContains("T16000M"), // or Any / Usb(...)
    axes: {
        // codes below are a T.16000M on Windows
        "pitch": (code: 65537, invert: true, deadzone: 0.05),
        "roll":  (code: 65536, deadzone: 0.05),
        "yaw":   (code: 65538, deadzone: 0.05), // the twist Bevy drops
        // "throttle": (code: 65539, min: -1.0, max: 1.0),
    },
),
```

Raw codes are **platform-specific** (a Windows code does not equal a
Linux one) and per-device. Discover them for your hardware with the probe
tool, which prints each axis's raw code and value range as you move it:

```bash
cargo run -p thalos_input --example gamepad_axes
```

For multi-device setups, put `device: ...` on an individual axis to bind
that axis to a different physical device than the block default.

## Context Model

Each binary spawns one controller entity with layered contexts.

Game:

- `GameSystemContext` for Escape, screenshot, and freecam
- `GameViewContext` for HUD toggle, map toggle, and camera cycle
- `GameFlightContext` for attitude, SAS, throttle, and HOTAS flight axes
- `GameWarpContext` for sim-time meta-controls (pause, warp speed,
  warp-to-maneuver) — always active except while a UI text field is focused,
  so pause is reachable from EVA, freecam, etc.
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
  and precision modifier. The **game adds this context too** (alongside
  its own contexts) for the in-game shipyard editor: the gate in
  `crates/game/src/input.rs` keeps it inactive unless the editor is open,
  and deactivates every gameplay context (flight, warp, view, eva,
  maneuver) while it is. While the editor's ship-name field has focus, the
  keyboard action source is disabled entirely and the field consumes raw
  key events (including Escape).

Use `ActionSettings::consume_input` for semantic keyboard actions that
should not bleed into lower-priority actions. Mouse camera actions stay
pass-through and are gated by the Bevy-UI pointer state
(`hud::UiPointerGate`) in the consumer systems.

## Escape Policy

`GameSystemContext` emits the Escape intent, and `pause_menu` owns the
priority policy:

1. close the settings overlay
2. close the shipyard editor
3. close the pause menu
4. cancel an active maneuver interaction
5. clear the current target
6. open the pause menu

Do not reintroduce `ButtonInput::clear_just_pressed` to arbitrate this.

## Scheduling

Enhanced input updates in `PreUpdate`. Intent resources are reset before
`EnhancedInputSystems::Apply` and collected immediately after it.
Simulation and presentation systems then read those intent resources
during their normal `Update` sets.

