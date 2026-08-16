# Thalos — orbital mechanics sandbox
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]
set dotenv-load := true
set dotenv-filename := ".env.just"

# Every dev entry point that boots Bevy defaults to dynamic linking. Bevy links
# once into `bevy_dylib`; subsequent game/screenshot/preview iterations relink
# only Thalos crates. Release/build/trace paths stay static. Override the whole
# game command in `.env.just` to opt out locally.
game_command := env_var_or_default("THALOS_GAME_COMMAND", "cargo run -p thalos_game --features dev-renderer")
capture_command := env_var_or_default("THALOS_CAPTURE_COMMAND", "cargo run -p thalos_capture_host --features dev-renderer")
# Existing Mac checkouts already have preferences.ron, so first-run Laptop
# never fires. A bare `just game` on macOS pins Laptop for the session.
default_quality := env_var_or_default("THALOS_QUALITY", if os() == "macos" { "laptop" } else { "" })

# Run the game. Bare `just game` boots to the start screen (scenario
# picker / shipyard / settings); naming a mode skips it and launches
# directly: `just game orbit` (ship in low equatorial Thalos orbit),
# `just game polar` (same altitude, polar / i≈90°), `just game eva`
# (on foot on the Thalos surface), `just game landing` (powered-descent
# approach over Thalos land), `just game final` (very low over a flat dry
# patch for touchdown practice), `just game runway` (aircraft parked on the
# Thalos surface runway), `just game runway-approach` (short final lined up
# with that runway), `just game cruise` (Meridian at ~15,000 ft flying
# level), `just game launch` (Saturn standing vertically on the spaceport
# launchpad), `just game shipyard` (straight into the in-game ship editor —
# also reachable via the pause menu's SHIPYARD button), `just game hub`
# (straight into the space-center hub over the spaceport, no craft — the
# PLAY path without the start screen). `THALOS_AUTO_RUN=1`
# also skips the start screen (agents keep a one-shot launch flow).
# Set a persistent default with THALOS_SPAWN in `.env.just`. The stop is the
# friendly handoff from an idle capture host; the process-level renderer lease
# remains the race-proof authority if another renderer starts concurrently.
# On macOS, `quality` defaults to laptop (cheap knobs; window mode unchanged).
# Override with `quality=showcase` or THALOS_QUALITY in `.env.just`.
game mode=env_var_or_default("THALOS_SPAWN", "menu") quality=default_quality:
    cargo run -p thalos_capture --bin thalos_capture -- stop
    {{ if quality != "" { "THALOS_QUALITY='" + quality + "' " } else { "" } }}{{game_command}} -- {{mode}}

# Run the lightweight Kòrsou explorer. Hand off from an idle Thalos capture
# host first; the machine-wide renderer lease remains the race-proof authority.
# On macOS this also defaults to the Laptop developer profile.
korsou quality=default_quality:
    cargo run -p thalos_capture --bin thalos_capture -- stop
    {{ if quality != "" { "THALOS_QUALITY='" + quality + "' " } else { "" } }}cargo run -p korsou --features dev-renderer

# The ship editor is the in-game Bevy-UI editor: `just game shipyard`
# (or the pause menu's SHIPYARD button). There is no standalone editor binary.

# Headless procedural-object gallery: renders each object (trees, conifer,
# shrub; rocks etc. later) to a PNG under artifacts/visual/latest/object_preview/, then exits. No
# window — both a human and an agent can run it and inspect the images. Lit with
# the real TreeMaterial + sky model so it matches the in-game look. Add objects
# in crates/rendering/render/examples/object_preview.rs.
preview:
    cargo run -p thalos_body_render --features bevy/dynamic_linking --example object_preview

# UI kitchen sink: renders every thalos_ui token/widget over a test scene to
# artifacts/visual/latest/ui_preview.png headlessly, then exits — agents iterate on
# the UI kit by reading the PNG. See crates/interface/ui/examples/kitchen_sink.rs.
ui-preview:
    cargo run -p thalos_ui --features bevy/dynamic_linking,bevy/wayland,bevy/jpeg --example kitchen_sink

# Navigation-display preview: renders the ND in eight approach situations
# (straight-in, offset intercept, overflown, short final, reciprocal end,
# crosswind strip, 60 km out, idle) to artifacts/visual/latest/nav_preview.png
# headlessly, then exits. Every panel is a real approach plan drawn by the real
# shader, so it is evidence about geometry/scale/symbology — not about ECS
# wiring. See crates/runtime/game/examples/nav_preview.rs and
# docs/gameplay/navigation.md.
nd-preview:
    cargo run -p thalos_game_runtime --features bevy/dynamic_linking --example nav_preview

# Loading-screen preview: renders the real loading screen — bar, status line,
# and the GPU/VRAM/residency readout — to
# artifacts/visual/latest/loading_preview.png headlessly, then exits. This is the
# ONLY way to look at that screen: it despawns before any capture preset can
# shoot it. See crates/runtime/game/examples/loading_preview.rs.
loading-preview:
    cargo run -p thalos_game_runtime --features bevy/dynamic_linking --example loading_preview

# Interactive window variant of `just ui-preview` (hover/press/typing feel;
# S saves the same screenshot). User-run (opens a window).
ui-preview-window:
    cargo run -p thalos_ui --features bevy/dynamic_linking,bevy/wayland,bevy/jpeg --example kitchen_sink -- --window

# Interactive window variant of `just preview`: opens a window with an orbit
# camera — drag to orbit, scroll to zoom, ←/→ cycle objects, S saves a
# screenshot to artifacts/visual/latest/object_preview/<object>_view.png.
preview-window:
    cargo run -p thalos_body_render --features bevy/dynamic_linking --example object_preview -- --window

# Headless screenshot: the first call boots one persistent off-screen renderer;
# later calls reuse it while Bevy reloads both file-backed and embedded WGSL in
# place (~3 s to a fresh PNG). Rust/manifest edits restart the host through an
# automatic rebuild on the next call — there is no in-process Rust reload
# (hot-patching retired, ADR-20260724T153619Z). Presets with the same body,
# spawn scenario, hub mode, and viewport reuse one booted world; incompatible
# scenes and viewport changes perform a managed restart.
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
# `earth-reference` is the 3:2 ISS-like custom-atmosphere calibration view.
# F8 opens the viewpoint manager backed by assets/viewpoints.json. A named
# catalog entry can be captured directly (`just screenshot <viewpoint-id>`);
# `just screenshot latest` selects the newest entry and writes
# artifacts/visual/latest/latest_perspective.png.
# Override the framing
# without recompiling via env vars, e.g. (PowerShell):
#   $env:THALOS_SCREENSHOT_ELEVATION='90'; $env:THALOS_SCREENSHOT_DISTANCE='6000'; just screenshot
# Other knobs: THALOS_SCREENSHOT_AZIMUTH, _SIZE (1920x1080), _OUT, _WARMUP,
# _TIME (canonical simulation seconds; overrides a saved viewpoint's time),
# _GRAPHICS (clouds=off,grass=on,foliage=off; cold-capture compatibility adapter),
# Cloud probes additionally accept _CAMERA_ALTITUDE, _LOOK_ELEVATION,
# _SUN_ELEVATION, _CLOUD_QUALITY (low/baseline/high/reference),
# _CLOUD_TEMPORAL (on/off), _CLOUD_COVERAGE, and _REPORT (JSONL). Ocean probes
# additionally accept THALOS_SCREENSHOT_OCEAN_TIME for deterministic phase.
screenshot preset="spaceport-aerial" *options:
    cargo run -p thalos_capture --bin thalos_capture -- shot "{{preset}}" {{options}}

# Capture several scenes through one controller invocation. Compatible scenes
# reuse the same world/GPU; the controller restarts only at boot-world boundaries.
capture *presets:
    cargo run -p thalos_capture --bin thalos_capture -- shot {{presets}}

# Authoritative one-shot capture: clean process, full preset warmup, then exit.
screenshot-cold preset="spaceport-aerial":
    {{ if os() == "windows" { "$env:THALOS_SCREENSHOT='" + preset + "'; " } else { "THALOS_SCREENSHOT='" + preset + "' " } }}{{capture_command}}

capture-status:
    cargo run -p thalos_capture --bin thalos_capture -- status

capture-stop:
    cargo run -p thalos_capture --bin thalos_capture -- stop

# Full dev-lane reset when the build tree disagrees with itself (link errors
# naming `anon.*.llvm.*` internal symbols). Stops the host, drops
# target/debug/incremental, and cleans the dynamic-linking crate set as ONE
# unit. Never hand-roll `cargo clean -p bevy_dylib` (or any subset): cleaning
# the dylib while dependent incremental caches survive is what creates that
# corruption. The capture client also self-heals this once automatically —
# reach for this only when it reports the retry failed too.
build-reset:
    cargo run -p thalos_capture --bin thalos_capture -- reset

# Publish a visual report from its non-browser-renderable `.html.in` input.
# The publisher embeds every image, validates that no live image or placeholder
# remains, and prints the one canonical `.html` file agents may open.
publish-report report:
    python3 scripts/present_embed.py "{{report}}"

# Deterministic visual A/B or N-way comparison. The lightweight orchestrator
# sends every variant to the same persistent renderer used by `just screenshot`,
# resetting temporal histories between captures. Outputs full captures + contact sheet +
# diffs/wipes + manifest under the disposable agent scratch tree at
# artifacts/visual/runs/comparisons/<preset>/<axis>/.
# Axes include ssao (off/on/raw), terrain-lighting, terrain-culling,
# terrain-regolith-filter, and cloud-reconstruction. See docs/development/visual_testing.md.
compare preset="spaceport-aerial" axis="ssao":
    cargo run -p thalos_capture --bin thalos_capture -- compare "{{preset}}" "{{axis}}"

# Clean-process evidence lane. Use after exploratory iteration and for structural
# pipeline axes (terrain-culling automatically falls back here).
compare-cold preset="spaceport-aerial" axis="ssao":
    cargo build -p thalos_capture_host --features dev-renderer
    cargo run -p thalos_capture --bin thalos_capture -- compare "{{preset}}" "{{axis}}" --cold

# Materialize the runtime-ready learned terrain payloads in a clone where Git
# LFS smudging was unavailable or deliberately skipped, then verify the store.
terrain-assets:
    git lfs install --local
    git lfs pull --include="assets/terrain_packages/thalos_diffusion/thalos_site_detail_*.f32"
    git lfs fsck --objects

# Offline authored terrain package. The MVP producer is the deterministic
# airless compiler; ADR-20260720T211046Z-offline-terrain-packages's diffusion producer will emit the same package
# boundary. Output: assets/terrain_packages/<body>.bin.
bake body="Mira":
    cargo run --release -p thalos_terrain_baker -- {{body}}

# Validate package schema, content key, node/blob bounds, checksums, and payload.
validate-bake body="Mira":
    cargo run --release -p thalos_terrain_baker -- validate {{body}}

# Rebuild the versioned vegetation atlases consumed by the runtime renderer.
texgen:
    cargo run --release -p thalos_texgen_tool

# Whole-planet map export (headless, agent-readable): renders the true in-game
# macro palette, a climate-independent signed-elevation relief map, and a flat
# biome-class map with per-biome area stats to target/world_{map,relief,biomes}.png,
# then exits. Defaults to web-mercator; knobs (set as env vars):
# WORLD_PROJ=equirect, WORLD_MODE=hypso (legacy ramp), WORLD_W, WORLD_SEED,
# WORLD_RADIUS_KM, and the WORLD_ZOOM / WORLD_TRANSECT probe modes. See
# tools/world_map/src/main.rs.
map:
    cargo run --release -p thalos_world_map

# Diagnostics triage: read the last <hours> of the diagnostics lane (runtime +
# tools, rotated files included) and report only what crossed a threshold.
# A healthy window prints its header and "no action needed". `--json` for
# machine use; thresholds and their rationale live in tools/diag/src/finding.rs.
diag hours="24" *args:
    cargo run --release -q -p thalos_diag -- --since {{hours}} {{args}}

# Offline perf report: renders one session of the runtime.jsonl perf lane
# (frame gauges, spikes, optional THALOS_PERF_RECORD full-rate blocks) to
# artifacts/diagnostics/reports/<session>/report.html + summary.json.
# `session` = a `<pid>-<unix_ms>` id, `latest` (default), or `--list`.
perf-report session="latest":
    cargo run --release -p thalos_perfreport -- {{session}}

# Agent-runnable render-cost differential. Unlike `just compare`, this keeps
# the real offscreen game render graph running continuously without screenshot
# readback or the capture host's 60 Hz pacing, waits for terrain/scene stability,
# and records a four-cell foliage × custom-shadows matrix.
perf-bisect preset="forest-stand":
    cargo run -p thalos_capture --bin thalos_capture -- stop
    RUST_LOG=warn,thalos::diagnostic=info THALOS_SCREENSHOT={{preset}} THALOS_SCREENSHOT_WARMUP=1000000 THALOS_SCREENSHOT_SIZE=1600x900 THALOS_SCREENSHOT_GRAPHICS=clouds=off,grass=off,foliage=on THALOS_CAPTURE_CLOCK=driven:60 THALOS_HEADLESS_PERF=matrix THALOS_PERF_FOLIAGE=on THALOS_SHADOW_CASCADES=4 {{capture_command}}
    cargo run --release -p thalos_perfreport -- --headless-matrix

# Marginal cost of each custom shadow camera, measured 4 -> 0 in one warmed
# scene with foliage held resident.
perf-shadow-bisect preset="forest-stand":
    cargo run -p thalos_capture --bin thalos_capture -- stop
    RUST_LOG=warn,thalos::diagnostic=info THALOS_SCREENSHOT={{preset}} THALOS_SCREENSHOT_WARMUP=1000000 THALOS_SCREENSHOT_SIZE=1600x900 THALOS_SCREENSHOT_GRAPHICS=clouds=off,grass=off,foliage=on THALOS_CAPTURE_CLOCK=driven:60 THALOS_HEADLESS_PERF=shadow-cascades THALOS_PERF_FOLIAGE=on THALOS_SHADOW_CASCADES=4 {{capture_command}}
    cargo run --release -p thalos_perfreport -- --headless-shadow-cascades

# CLOUD-0's repeatable five-view baseline. Each preset writes a PNG under
# artifacts/visual/latest/ and a same-named JSONL report under artifacts/diagnostics/.
# Use the single-preset `screenshot` recipe with overrides for 1440p,
# temporal-off, or quality sweeps.
cloud-baseline:
    $presets = @('cloud-runway', 'cloud-cruise', 'cloud-interior', 'cloud-limb', 'cloud-sunset'); foreach ($preset in $presets) { cargo run -p thalos_capture --bin thalos_capture -- shot $preset; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }

# Fast edit-loop type check. Pass another package when working below the game
# composition boundary: `just check thalos_body_render`.
check package="thalos_game":
    cargo check -p "{{package}}"

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
