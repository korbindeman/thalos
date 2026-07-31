# Input

Thalos routes semantic player input through `bevy_enhanced_input`
contexts owned by `thalos_input`. Raw Bevy input should remain only for
spatial data and UI internals: cursor position, ray projection, Bevy
picking hover/click/drag events, native-UI text-field input
(`thalos_runtime::ui_widgets`), and shipyard pinch gestures. (The game UI is
native Bevy UI; `bevy_egui` is no longer a dependency anywhere in the
workspace.)

## Crate Boundary

`thalos_input` is a Bevy-facing workspace crate shared by:

- `thalos_game` (including its in-game shipyard editor's `ShipyardContext`)
- `thalos_body_editor`

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

- `GameSystemContext` for Escape, desktop screenshot, F8 viewpoint-manager
  toggle, F9 viewpoint quick-save, and freecam
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

Priority matters here, and one pair of keys is shared: Shift/Ctrl are the
throttle ramp in `GameFlightContext` (priority 20) *and* the precision
modifiers in `GameManeuverPrecisionContext` (90, consuming). The precision
context therefore has to be genuinely off outside a drag, or the throttle ramp
goes dead with no error anywhere — **a mode resource that gates an input
context must be released by a system that cannot decline to run**
(`end_drag_on_release`, INC-20260730T222419Z).

Applying any saved or agent-scripted entry from the F8 viewpoint manager
activates freecam at that body-fixed camera pose. Closing the manager therefore
leaves the developer in freecam at the selected view instead of handing the
camera back to the orbit-focus driver.

F9 saves the current view without opening the manager: the prompt's name field
takes the keyboard (through the shared text-entry gate below, so no keystroke
reaches flight or system bindings while it is up), and Enter or Escape closes
it. Because the field owns Escape, F9 never stacks the pause menu.

### Freecam

F4 hands the ship camera to `freecam` (debug builds). Movement is WASD +
R/F (up/down), look is LMB-drag, and the wheel sets the cruise speed —
all raw `ButtonInput<KeyCode>` reads, because the flight context is
suspended while freecam owns the camera (see *UI input ownership* below
for how those reads still respect a focused text field).

Two flight modes, both **on by default**, both toggleable from the panel
or the keyboard:

- **Level to planet up** (`L`) — the pose is constrained to the local
  vertical at the camera's current position: yaw turns about that
  vertical, pitch stops short of the poles, roll is zero. The constraint
  is re-derived every frame against the up direction *where the camera
  now is*, so the horizon stays level while flying across a body instead
  of tipping over as the vertical rotates underneath.

  Its strength is an **authority** in `0..=1` set by how large the body
  *looks* from the camera — the angular diameter `2·asin(R/r)` it
  subtends — smoothstepped from rigid at 120° down to fully 6-DOF at 45°.
  Apparent size, not altitude: the rule is independent of where the camera
  points and of the lens (panning or zooming must never change the flight
  model), and being a pure function of `r/R` it fits a 190 km moonlet and
  a 3186 km planet with one constant. Full authority reaches ~493 km over
  Thalos and ~134 km over Mira, against the 80 km Kármán line and 43 km
  airless ceiling the old rule released at.

  Nothing switches at a boundary. Authority is smoothstepped, so it has
  zero slope at both ends of the band; the pose *eases* toward level at a
  rate of `2 Hz · a/(1−a)`, which diverges at full authority (rigid, as
  before) and vanishes at zero; and mouse-look's yaw axis and R/F's climb
  axis interpolate between the camera's own up and the local vertical by
  the same authority, so the control feel crosses the band as continuously
  as the pose does. Q/E roll stands down above half authority and returns
  below it. The checkbox remains remembered throughout.
- **Stop at the ground** (`C`) — the camera's radius is clamped to the
  terrain height beneath it plus a small clearance. A *floor*, not a
  swept collision: it stops the camera sinking through the surface it is
  parked on, but one fast frame can still cross a ridge. Off = the old
  fly-through-a-planet behaviour.

Both constraints only touch a pose freecam produced, or the flight-camera
pose it inherited on F4 entry. A pose *handed* to it — an applied
viewpoint, a headless capture framing — is reproduced exactly until the
user moves the camera, so authored roll and authored framing survive
replay and capture baselines don't shift under them.

`freecam::panel` draws the matching control surface on the left flank
while freecam is active: the cruise speed with a real-world reference for
scale, a log-scale drag slider over the whole 1 m/s – 10 000 km/s range,
the two mode switches, and the shared camera lens. The lens control is a
logarithmic 12–400 mm full-frame-equivalent slider with common focal-length
marks plus horizontal/vertical angle-of-view readouts. Holding `Z` temporarily
multiplies the effective focal length by four; releasing it returns to the
slider's base lens.

F4 converts the currently presented framing into that physical lens on entry,
then restores the receiving flight rig's optics on exit, so neither transition
should jump or leak a photographic edit into normal flight. The exit frame also
keeps freecam's final pointer delta out of the receiving orbit controller while
allowing that controller to rebuild its own pose immediately. Panel and
keyboard are two surfaces on **one** state — movement settings push into
`FreeCam`, while lens controls edit the one `CameraOptics` component on
`ShipCamera`; readback is value-guarded so the surfaces do not chase each other.

## UI input ownership

Two UIs draw over the same 3-D view — native Bevy UI (HUD, `thalos_ui` panels
and text fields) and the egui F8 viewpoint manager — and either can take the
pointer or the keyboard. `hud::input_gate` folds both into **one resource per
device**, written by `update_ui_input_gates`:

- `UiPointerGate.hovered` — a UI surface owns the pointer.
- `UiKeyboardGate.text_entry()` — a text field owns the keyboard.

Everything that reads input consults these rather than testing one UI system:
`gate_enhanced_input_sources` disables the whole keyboard action source while
`text_entry()` is true, and the handful of systems that must read
`ButtonInput<KeyCode>` raw — freecam translation/roll/zoom and the god-view
WASD pan, both of which run while their enhanced-input context is suspended —
check it themselves. A new text surface is wired in at `update_ui_input_gates`,
not chased through every keyboard reader.

Planet editor:

- `PlanetEditorContext` for orbit camera, fullbright, placement click,
  and overlay suppression

Shipyard:

- `ShipyardContext` for orbit camera, wheel pan/zoom, primary pointer,
  and precision modifier. The **game adds this context too** (alongside
  its own contexts) for the in-game shipyard editor: the gate in
  `crates/runtime/game/src/input.rs` keeps it inactive unless the editor is open,
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

