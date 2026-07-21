# Thalos — orbital mechanics sandbox
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]
set dotenv-load := true
set dotenv-filename := ".env.just"

# Every dev entry point that boots Bevy defaults to dynamic linking. Bevy links
# once into `bevy_dylib`; subsequent game/screenshot/preview iterations relink
# only Thalos crates. Release/build/trace paths stay static. Override the whole
# game command in `.env.just` to opt out locally.
game_command := env_var_or_default("THALOS_GAME_COMMAND", "cargo run -p thalos_game --features bevy/dynamic_linking")

# Run the game. Bare `just game` boots to the start screen (scenario
# picker / shipyard / settings); naming a mode skips it and launches
# directly: `just game orbit` (ship in low equatorial Thalos orbit),
# `just game polar` (same altitude, polar / i≈90°), `just game eva`
# (on foot on the Thalos surface), `just game landing` (powered-descent
# approach over Thalos land), `just game final` (very low over a flat dry
# patch for touchdown practice), `just game runway` (aircraft parked on the
# Thalos surface runway), `just game runway-approach` (short final lined up
# with that runway), `just game cruise` (Meridian at ~15,000 ft flying
# level), `just game shipyard` (straight into the in-game ship editor —
# also reachable via the pause menu's SHIPYARD button), `just game hub`
# (straight into the space-center hub over the spaceport, no craft — the
# PLAY path without the start screen). `THALOS_AUTO_RUN=1`
# also skips the start screen (agents keep a one-shot launch flow).
# Set a persistent default with THALOS_SPAWN in `.env.just`.
game mode=env_var_or_default("THALOS_SPAWN", "menu"):
    {{game_command}} -- {{mode}}

# The ship editor is the in-game Bevy-UI editor: `just game shipyard`
# (or the pause menu's SHIPYARD button). There is no standalone editor binary.

# Headless procedural-object gallery: renders each object (trees, conifer,
# shrub; rocks etc. later) to a PNG under tools/preview/out/, then exits. No
# window — both a human and an agent can run it and inspect the images. Lit with
# the real TreeMaterial + sky model so it matches the in-game look. Add objects
# in crates/body_render/examples/object_preview.rs.
preview:
    cargo run -p thalos_body_render --features bevy/dynamic_linking --example object_preview

# UI kitchen sink: renders every thalos_ui token/widget over a test scene to
# tools/ui_preview/kitchen_sink.png headlessly, then exits — agents iterate on
# the UI kit by reading the PNG. See crates/ui/examples/kitchen_sink.rs.
ui-preview:
    cargo run -p thalos_ui --features bevy/dynamic_linking,bevy/wayland,bevy/jpeg --example kitchen_sink

# Interactive window variant of `just ui-preview` (hover/press/typing feel;
# S saves the same screenshot). User-run (opens a window).
ui-preview-window:
    cargo run -p thalos_ui --features bevy/dynamic_linking,bevy/wayland,bevy/jpeg --example kitchen_sink -- --window

# Interactive window variant of `just preview`: opens a window with an orbit
# camera — drag to orbit, scroll to zoom, ←/→ cycle objects, S saves a
# screenshot to tools/preview/out/<object>_view.png.
preview-window:
    cargo run -p thalos_body_render --features bevy/dynamic_linking --example object_preview -- --window

# Headless screenshot: boots the game off-screen (no window), builds the world
# for the preset, poses the camera, and writes a PNG to tools/screenshots/, then
# exits — agent-runnable like `just preview`, but of a whole composed scene.
# `just screenshot` does the spaceport aerial; `just screenshot runway-atmosphere`
# captures a low near-horizontal view inside the atmosphere; `just screenshot hub` captures
# the space-center hub exactly as PLAY presents it (spaceport built, no craft
# placed — the regression probe for view-anchored surface detail);
# `just screenshot dry-belt` (aliases: dry / desert / biome) surveys the driest
# sunlit desert site it can find — the verification probe for terrain-per-biome
# work (landcover palette + the tree/scatter biome gate); `just screenshot ocean`
# captures a low-sun, eye-level deep-ocean material probe and `ocean-slopes`
# captures its resolved-slope / mip-roughness diagnostic. `mira-orbit` and
# `mira-surface` verifies the landmark crater/package/Hapke view; `mira-eva`
# reproduces the canonical eye-level EVA spawn and its horizon/LOD coverage.
# `earth-reference` is the 3:2 ISS-like atmosphere calibration view; it defaults
# to Bevy raymarching, while THALOS_SCREENSHOT_ATMOSPHERE=custom captures the
# matched legacy BodySky A/B.
# Override the framing
# without recompiling via env vars, e.g. (PowerShell):
#   $env:THALOS_SCREENSHOT_ELEVATION='90'; $env:THALOS_SCREENSHOT_DISTANCE='6000'; just screenshot
# Other knobs: THALOS_SCREENSHOT_AZIMUTH, _SIZE (1920x1080), _OUT, _WARMUP,
# _ATMOSPHERE (configured/custom/bevy).
# Cloud probes additionally accept _CAMERA_ALTITUDE, _LOOK_ELEVATION,
# _SUN_ELEVATION, _CLOUD_QUALITY (low/baseline/high/reference),
# _CLOUD_TEMPORAL (on/off), _CLOUD_COVERAGE, and _REPORT (JSONL). Ocean probes
# additionally accept THALOS_SCREENSHOT_OCEAN_TIME for deterministic phase.
screenshot preset="spaceport-aerial":
    {{ if os() == "windows" { "$env:THALOS_SCREENSHOT='" + preset + "'; " } else { "THALOS_SCREENSHOT='" + preset + "' " } }}{{game_command}}

# Deterministic visual A/B or N-way comparison. Builds the normal dynamic-link
# game once, then the lightweight orchestrator launches that exact binary in a
# clean process per typed variant. Outputs full captures + contact sheet +
# diffs/wipes + manifest under the disposable agent scratch tree at
# tools/agent_scratch/screenshots/comparisons/<preset>/<axis>/.
# Axes: atmosphere (custom/bevy), ssao (off/on/raw). See docs/visual_testing.md.
compare preset="earth-reference" axis="atmosphere":
    cargo build -p thalos_game --features bevy/dynamic_linking --bin thalos_game --example visual_compare
    {{ if os() == "windows" { "target/debug/examples/visual_compare.exe" } else { "./target/debug/examples/visual_compare" } }} "{{preset}}" "{{axis}}"

# Offline authored terrain package. The MVP producer is the deterministic
# airless compiler; ADR-20260720T211046Z-offline-terrain-packages's diffusion producer will emit the same package
# boundary. Output: assets/terrain_packages/<body>.bin.
bake body="Mira":
    cargo run --release -p thalos_terrain_baker -- {{body}}

# Validate package schema, content key, node/blob bounds, checksums, and payload.
validate-bake body="Mira":
    cargo run --release -p thalos_terrain_baker -- validate {{body}}

# Whole-planet biome map export (headless, agent-readable): renders the true
# in-game macro palette + a flat biome-class map with per-biome area stats to
# target/world_map.png + target/world_biomes.png, then exits. Defaults to
# web-mercator; knobs (set as env vars): WORLD_PROJ=equirect, WORLD_MODE=hypso
# (legacy ramp), WORLD_W, WORLD_SEED, WORLD_RADIUS_KM, and the WORLD_ZOOM /
# WORLD_TRANSECT probe modes. See crates/terrain/examples/world_map.rs.
map:
    cargo run --release -p thalos_terrain --example world_map

# CLOUD-0's repeatable five-view baseline. Each preset writes a PNG and a
# same-named JSONL report under tools/screenshots/. Use the single-preset
# `screenshot` recipe with overrides for 1440p, temporal-off, or quality sweeps.
cloud-baseline:
    $presets = @('cloud-runway', 'cloud-cruise', 'cloud-interior', 'cloud-limb', 'cloud-sunset'); foreach ($preset in $presets) { $env:THALOS_SCREENSHOT = $preset; cargo run -p thalos_game; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }

# Build everything
build:
    cargo build --workspace

# Bump, commit, tag, and push a release. Usage: just release patch|minor|major|0.2.0
release kind="patch":
    #!/usr/bin/env bash
    set -euo pipefail
    scripts/bump-version.sh "{{kind}}"
    version="$(scripts/bump-version.sh --current)"
    branch="$(git branch --show-current)"
    if [[ -z "${branch}" ]]; then
        echo "Cannot release from detached HEAD" >&2
        exit 1
    fi
    git add Cargo.toml Cargo.lock
    git commit -m "release v${version}"
    git tag "v${version}"
    git push origin "HEAD:${branch}"
    git push origin "v${version}"
    printf 'Published release tag v%s.\n' "${version}"

# Run tests
test:
    cargo test -p thalos_physics_canonical

# Lint
clippy:
    cargo clippy --workspace

# Profile the game with Tracy. Requires a running Tracy client (tracy-profiler
# or tracy-capture) listening on localhost before launch.
trace:
    cargo run --release -p thalos_game --features profile-tracy
