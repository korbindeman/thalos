# Quality profiles

Developer and machine-fit graphics bundles. This is not a shipping Low/Medium/High
ladder. The first named profile exists so Thalos can be run and developed on a
Mac without the Showcase path melting the machine.

Capture and screenshots always use Showcase defaults. A laptop session must not
leak into `just screenshot` evidence.

## Presets

Selecting **Showcase** or **Laptop** stamps every knob those bundles own.
Editing a stamped knob moves the selector to **Custom**. Reset returns to
Showcase.

| Knob | Showcase | Laptop |
|---|---|---|
| Frame cap | Off | 30 Hz |
| Foliage | On | Off |
| MSAA | Off (post-process AA) | Off |
| Clouds | On | Off |
| Grass | On | Off |
| Terrain detail | 1.00× | 0.50× |
| Shadow cascades | 4 | 2 |

Laptop may look rough. That is the point.

Laptop does not change the window. Mode and size stay on the Window page
(default is borderless fullscreen). UI stays at the OS HiDPI scale. Do not
write `scale_factor_override` for this profile: that made the HUD 1×.
`THALOS_SCALE` still pins one session.

## First run

A machine with no `preferences.ron` and no `settings.ron` is a first run.

- **macOS** writes Laptop graphics and keeps the default borderless window,
  then persists that choice.
- **Other platforms** write Showcase.

An existing file is never rewritten by this path. Changing a Showcase install
to Laptop is a settings or `THALOS_QUALITY` action.

On **macOS**, a bare `just game` / `just korsou` pins Laptop for that session
even when `preferences.ron` already exists. Existing files stay as they are;
the pin is not written back. Use `quality=showcase` or `THALOS_QUALITY=showcase`
when you want the canonical look.

## Session pin

```bash
just game orbit
just game orbit quality=showcase
THALOS_QUALITY=laptop just korsou
```

`THALOS_QUALITY=showcase|laptop` stamps the in-memory knobs for one process and
is not written back. On macOS the just recipes default that pin to laptop.
Capture ignores it: the headless host starts from Showcase and each request
patches only the typed capture overrides.

## Where the knobs live

Shared knobs (preset, render scale, frame cap, MSAA, foliage) live in
`thalos_preferences` and appear on the common Graphics page. Game-only knobs
(clouds, grass, terrain detail, shadow cascades) live in `GraphicsSettings` and
appear on the game Graphics page. Picking a named preset stamps both files.

## Later work

Measured optimization of the Showcase path stays a separate track. Shipping
Low/Medium/High can reuse this stamp-and-custom model once Laptop has taught
us which knobs actually move Mac frame time.
