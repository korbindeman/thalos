# Quality profiles

Developer and machine-fit graphics bundles. This is not a shipping Low/Medium/High
ladder. The first named profile exists so Thalos can be run and developed on a
Mac without the Showcase path melting the machine.

Capture and screenshots always use Showcase defaults. A laptop session must not
leak into `just screenshot` evidence. To reproduce a Laptop-only 3D-scale
defect headlessly, pin `THALOS_SCREENSHOT_RENDER_SCALE=0.5` on that shot
(INC-20260817T014132Z).

## Presets

Selecting **Showcase** or **Laptop** stamps every knob those bundles own.
Editing a stamped knob moves the selector to **Custom**. Reset returns to
Showcase.

| Knob | Showcase | Laptop |
|---|---|---|
| Render scale | 1.00× | 0.50× |
| Frame cap | Off | 30 Hz |
| Foliage | On | On |
| MSAA | Off (post-process AA) | Off |
| Clouds | On | On |
| Grass | On | Off |
| Terrain detail | 1.00× | 0.50× |
| Shadow quality | High (4 cascades) | Low (2 cascades) |

Laptop keeps clouds and woody foliage because both are part of the game's
signature world fidelity. Its primary savings are the 0.50× 3D render scale
and 30 Hz cap; grass, terrain detail, and shadow range remain secondary
reductions.

### Shadow quality

Shadow quality is an independent player-facing tier, not a raw renderer count:

| Tier | Active cascades | Effect |
|---|---:|---|
| Off | 0 | Disables the custom sun-shadow rig |
| Low | 2 | Keeps near and mid shadows; parks the two broadest views |
| Medium | 3 | Adds broad terrain/foliage grounding; parks only the farthest view |
| High | 4 | Full Showcase shadow range |

Every active cascade remains 4096². The broad-cascade benchmark showed that
caster geometry, not map fill, dominated cost, so lowering resolution would
weaken near detail without addressing the measured bottleneck
(ADR-20260814T201228Z). An inactive cascade's depth and throwaway colour targets
shrink to 1×1, returning a nominal 80 MiB per parked cascade as well as removing
its cull, queue, depth pass, and depth copy. The measured forest-stand ladder
saved about 11.9 ms/frame from High to Low on the machine that produced the
current report; rerun `just perf-shadow-bisect forest-stand` for the current
hardware and source.

`THALOS_SHADOW_QUALITY=off|low|medium|high` pins a tier for one process.
`THALOS_SHADOW_CASCADES=0..4` remains the lower-level diagnostic override; a
one-cascade run is deliberately reported as Custom rather than exposed in the
menu. `just compare <scene> shadow-quality` captures the four named tiers.

Laptop does not change the window. Mode and size stay on the Window page
(default is borderless fullscreen). Render scale shrinks the 3D main target
and Bevy upscales it to the swapchain; UI stays on a full-resolution camera
at OS HiDPI. Picking and `world_to_viewport` keep window-logical coordinates.
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

Shared knobs (preset, render scale, frame cap, MSAA, foliage, clouds) live in
`thalos_preferences` and appear on the common Graphics page when the
application supplies that adapter. Game-only knobs (clouds until folded in,
grass, terrain detail, shadow quality) live in `GraphicsSettings` and
appear on the game Graphics page. Picking a named preset stamps both files.

## Later work

Measured optimization of the Showcase path stays a separate track. Shipping
Low/Medium/High can reuse this stamp-and-custom model once Laptop has taught
us which knobs actually move Mac frame time.
