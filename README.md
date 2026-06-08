# Thalos

![Auron](screenshots/auron.jpg)

Thalos (working title) is a spaceflight simulation game.

I'm aiming for a more physically grounded take on the genre, with a realistic scaling, simulation that aims for physical plausibility while still being fun to play, and a solar system whose nature reveals itself through exploration.

## Quick Start

Prebuilt Windows and macOS builds are available from the [GitHub Releases page](https://github.com/korbindeman/thalos/releases).

Run the game:

```bash
cargo run -p thalos_game --release
```

Run the ship editor:

```bash
cargo run -p thalos_shipyard --bin ship_editor
```

Run the planet editor:

```bash
cargo run -p thalos_planet_editor
```


The repository also includes a `justfile` with shortcuts for common commands.

## Playing

You start in a low orbit around Thalos, the homeworld, flying a prebuilt spacecraft.

Controls:

- `W` / `S` pitch
- `A` / `D` yaw
- `Q` / `E` roll
- `Shift` / `Ctrl` raise or lower throttle
- `Z` full throttle
- `X` cut throttle
- `T` toggle SAS
- `Space` stage (ignite the next stage's engines, jettison spent stages)
- `.` increase time warp
- `,` decrease time warp (steps down into pause below 1x)
- `\` reset time warp to 1x
- `Esc` pause menu
- `M` toggle map view
- `V` cycle ship camera mode
- `F2` save a screenshot to `~/Desktop/thalos`
- `F1` hide / show the HUD
- Left drag rotates the camera
- Scroll zooms the camera
- Double-click a body or ship marker to focus it
- `N` place a maneuver node
- `Delete` / `Backspace` delete the selected maneuver node
- `P` toggle photo mode

`Cmd` + left-click (Mac) or `Ctrl` + left-click (Windows) a body in the map-view navigator to move the ship into a low orbit around that body.

Debug surface drop: click a body's `drop` button in the map-view navigator, aim the terrain cursor, then left-click the surface to mount the ship there.

## Project Status

This project is in a very early stage. You're welcome to look through the code, but the internals are changing quickly and there is no public-facing documentation yet.

## License

Thalos is fully source-available, with a deliberate split — **you can't sell the game; you can sell content for it.**

- **Code** — [PolyForm Noncommercial 1.0.0](LICENSE): use, modify, fork, and redistribute for any noncommercial purpose. Selling the game is reserved to the copyright holder.
- **Assets** (art, audio, and authored content under `assets/` and `ships/`) — [CC BY 4.0](LICENSE-ASSETS): share and adapt for any purpose, **including commercially** (e.g. paid planet/part packs), with attribution.
- **Vendored crates** under `crates/` keep their upstream licenses (`avian_fdm` is LGPL-3.0-or-later; `udlod` and `volumetric_clouds` are MIT/Apache-2.0).
- The **"Thalos" name and logo** are not licensed.

See [LICENSING.md](LICENSING.md) for the full rationale and contribution terms.

## Acknowledgements

Kerbal Space Program was a major influence on me and the main inspiration for this project.
