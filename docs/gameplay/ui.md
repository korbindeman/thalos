# Thalos UI kit (`thalos_ui`)

The one home for the game interface's look and building blocks. Every screen
— menus, editors, overlays, dialogs — composes this crate's tokens and
widgets; **no screen defines its own colours, fonts, paddings, or interaction
styling**. The direction is *modern, light, compact*: frosted dark-glass
surfaces, hairline light strokes, a near-white text ramp, one warm amber
accent — polished game UI, not a debug overlay.

Added 2026-07-05 as part of the architecture-cleanup sprint: it replaced
four parallel widget/styling implementations (the game's `ui_widgets.rs`,
the shipyard editor's private `ui/widgets.rs`, per-screen
`update_button_visuals` clones in the main/pause/scenario/space-center
menus, and hand-rolled panel frames everywhere).

## Iterating on the look

```bash
just ui-preview          # headless kitchen sink → artifacts/visual/latest/ui_preview.png
just ui-preview-window   # interactive variant (hover/press/typing; S = screenshot)
```

The kitchen sink (`crates/interface/ui/examples/kitchen_sink.rs`) lays out every token
and widget over a colourful 3D scene so the frost has something to blur.
Agents iterate by editing the kit and **reading the PNG** — same
self-inspection loop as `just preview` / `just screenshot`. Add any new
widget to the sink in the same change that introduces it.

## Structure

| Module | Contents |
|---|---|
| `tokens` | The design tokens: colour palette (`TEXT_*`, `ACCENT`, `STROKE*`, `FILL_*`, `GLASS_TINT*`, `SCREEN_BG`), spacing scale (`SPACE_XS…XL`), radii (`RADIUS_PANEL/CTRL`), control heights (`CTRL_H*`), type scale (`FS_*`), and the [`UiTheme`] resource (font handles + shared glass material handles + text-bundle helpers `display/title/heading/body/body_strong/small/faint/mono/mono_dim`). |
| `glass` | [`GlassMaterial`] (a `UiMaterial`: rounded-rect SDF, 16-tap jittered spiral blur over the scene copy, tint, top sheen, grain, hairline stroke) and [`UiBackdropPlugin`] (a `Core3d` render system, after `PostProcess` / before `ui_pass`, that blits the [`UiBackdropSource`]-marked camera's view target into a half-res image). No marked camera / no backdrop ⇒ panels fall back to a translucent tint, so tools and world-less screens still work. |
| `widgets` | `spawn_button` (Ghost / Primary / Danger / Bare variants) + `spawn_menu_row`, `UiSlider` + `spawn_slider_row` (a visionOS-style *filled bar*: label inside-left, mono value inside-right, lighter fill sweeping from the left with a bright caret line at its lip, faint tick dots; drag anywhere on the bar), `UiCheckbox`, `UiCycle` (`‹ value ›` picker), `UiTextField` + `TextFieldFocus`, `ScrollableColumn`, toasts, and the static pieces (`panel_node`/`floating_panel_node`, `spawn_heading`, `spawn_divider`, `spawn_key_hint`, `spawn_value_row`). |

`ThalosUiPlugin` adds the material + backdrop pass, loads the theme
(`init_ui_theme` — order consumer `Startup` UI builders `.after` it), and
runs all widget interaction/visual systems globally.

## Fonts

- **Inter** (Light / Regular / SemiBold, OFL — `assets/fonts/Inter-OFL.txt`)
  carries all interface text: the refined neutral grotesque of the visionOS
  references, with full glyph coverage (Δ, ▶, ‹›, ×). Chosen 2026-07-05 over
  Titillium Web (deleted; the user disliked it) via a kitchen-sink bake-off.
- **Barlow** (Regular / SemiBold, OFL) ships alongside as the characterful
  aerospace-signage alternative — swap the three handles in `init_ui_theme`
  to try it (note: it lacks ▶ and Δ).
- **Fira Code** stays for numeric/mono readouts (tabular digits) and the
  flight HUD. Keep Δ-strings and aligned numerals on mono by convention.

## Invariants

- **Tokens are the single styling authority.** A colour/spacing/radius used
  by more than one screen lives in `tokens`, not at the call site. The flight
  HUD's `HudTheme` (`crates/runtime/game/src/hud/theme.rs`) is a *projection* of the
  same tokens (it re-points its palette at them and keeps only the Fira Code
  face + HUD-specific datum colours) — change tokens, both worlds move.
- **One interaction-styling system.** Anything clickable carries
  `UiButton { variant, latched, selected }` and lets `style_buttons` drive
  its visuals (labels marked `ButtonLabel`/`ButtonDesc`). Never write another
  per-screen `update_button_visuals`.
- **Only panels are glass.** Buttons/rows/fills inside a panel are plain
  translucent `BackgroundColor` nodes layered on top — one blur per surface.
  Attach glass via `theme.glass()` (regular) / `theme.glass_heavy()`
  (dominant modal dialogs); both share one material asset per style.
- **Text fields share one focus.** `TextFieldFocus` is the keyboard owner;
  the game input gate (`crates/runtime/game/src/input.rs`) reads it to suppress
  keyboard bindings while typing. New editable fields must be
  `UiTextField`s, not bespoke key readers.
- **The loading screen depends on token consts only** (no `UiTheme`
  resource) so it renders on frame 1.

## In-game wiring

- `main.rs` adds `ThalosUiPlugin`; the ship camera (`camera.rs`) and the
  shipyard editor camera (`shipyard_editor/scene.rs`) carry
  `UiBackdropSource` (inactive cameras are ignored, so exactly one feeds the
  frost at a time).
- Screens on the kit: main menu, pause menu, settings, scenario (destruction)
  picker, space-center hub panel, base-editor palette, loading screen, and
  the whole shipyard editor (top bar / palette / inspector / staging /
  hangar overlay / pending pill / status toasts).
- Flight-HUD panels (hud/, navball, body tree, maneuver editor) keep
  `HudTheme` text (mono readouts) but share the kit's buttons via
  `hud::theme::hud_button` and the token palette — and their panel chrome
  **is the same frosted glass**: `hud::theme::panel_frame` returns the shared
  `GlassMaterial` (HudTheme carries the `UiTheme::glass_regular` handle;
  `init_theme` is ordered after `init_ui_theme`), so every panel in the game
  is one surface.
- **HUD screenshots**: the headless capture normally hides the HUD; set
  `THALOS_SCREENSHOT_HUD=1` to keep it visible when iterating on HUD chrome,
  then use `just screenshot` so the shared dynamic-link dev path stays active
  (PowerShell: `$env:THALOS_SCREENSHOT_HUD='1'; just screenshot`).

## Shipyard editor UX (2026-07-05 pass)

- The bottom status bar is gone. `EditorState::status` changes surface as
  **transient toasts** (single-slot: a new message replaces the pill);
  the pending-part state shows as a floating **hint pill** under the top bar
  with its CANCEL button.
- **HANGAR** (top bar) opens the craft load/save overlay: every `ships/*.ron`
  as a row (click to load, `×` to delete), refreshed on open. SAVE stores the
  current build under the name field's name. The saved-ship list is no longer
  buried under the parts palette.
- The build-orientation toggle is labelled **HORIZONTAL** (it was "HANGAR",
  which now means the overlay).
- The ship-name field is a kit `UiTextField` with two-way model sync
  (`top_bar::sync_ship_name`).

## Known follow-ups

- Runtime-verify the whole pass in a live session (`just game`) — landed
  compile-clean + kitchen-sink/hub-screenshot-verified only.
- Frost quality knobs (blur radius, tint) may want tuning per screen once
  seen over real scenes; both live in `GlassMaterial::new`.
- The navball and map-view overlays draw their own custom chrome outside
  `panel_frame`; folding them onto the kit is a candidate later polish pass.
