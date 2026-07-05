# Thalos — orbital mechanics sandbox
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]
set dotenv-load := true
set dotenv-filename := ".env.just"

# Dev run defaults to Bevy dynamic linking (cross-platform iteration speedup;
# dev-only, never reaches `just build`/`just trace`/release). Override the whole
# command in `.env.just` to opt out locally.
game_command := env_var_or_default("THALOS_GAME_COMMAND", "cargo run -p thalos_game --features bevy/dynamic_linking")

# Run the game. Bare `just game` boots to the start screen (scenario
# picker / shipyard / settings); naming a mode skips it and launches
# directly: `just game orbit` (ship in low Thalos orbit), `just game eva`
# (on foot on the Thalos surface), `just game landing` (powered-descent
# approach over Thalos land), `just game final` (very low over a flat dry
# patch for touchdown practice), `just game runway` (aircraft parked on the
# Thalos surface runway), `just game runway-approach` (short final lined up
# with that runway), `just game cruise` (Meridian at ~15,000 ft flying
# level), `just game shipyard` (straight into the in-game ship editor —
# also reachable via the pause menu's SHIPYARD button). `THALOS_AUTO_RUN=1`
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
    cargo run -p thalos_body_render --example object_preview

# Interactive window variant of `just preview`: opens a window with an orbit
# camera — drag to orbit, scroll to zoom, ←/→ cycle objects, S saves a
# screenshot to tools/preview/out/<object>_view.png.
preview-window:
    cargo run -p thalos_body_render --example object_preview -- --window

# Headless screenshot: boots the game off-screen (no window), builds the world
# for the preset, poses the camera, and writes a PNG to tools/screenshots/, then
# exits — agent-runnable like `just preview`, but of a whole composed scene.
# `just screenshot` does the spaceport aerial; override the framing without
# recompiling via env vars, e.g. (PowerShell):
#   $env:THALOS_SCREENSHOT_ELEVATION='90'; $env:THALOS_SCREENSHOT_DISTANCE='6000'; just screenshot
# Other knobs: THALOS_SCREENSHOT_AZIMUTH, _SIZE (1920x1080), _OUT, _WARMUP.
screenshot preset="spaceport-aerial":
    $env:THALOS_SCREENSHOT='{{preset}}'; cargo run -p thalos_game

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

