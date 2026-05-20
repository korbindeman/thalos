# Thalos — orbital mechanics sandbox
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# Run the game
game:
    cargo run -p thalos_game

# Edit a planet's terrain. Usage: just edit auron
edit body:
    cargo run -p thalos_planet_editor -- {{body}}

# Run the ship editor (shipyard crate)
shipyard:
    cargo run -p thalos_shipyard --bin ship_editor

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

# Wipe the editor's on-disk terrain cache. The game and `just bake`
# don't use this directory — only the planet editor does, for
# iteration speed. Source-tree edits already invalidate cached entries
# via the build-time hash key in `crates/terrain/src/cache.rs`, so
# you rarely need to wipe manually.
clear-terrain-cache:
    rm -rf target/terrain_cache

# Headless terrain bake.
#
# Default (full): writes the local bake to `target/bakes/<body>.bin`
# (ignored by Git, what your local game loads) plus full-resolution
# equirect PNGs to `stage-bakes/<body>/full/`. Slow.
#
# `--preview`: 512² preview run. PNGs only at
# `stage-bakes/<body>/preview/`, no local game bake. Fast iteration loop —
# read the PNG to inspect compiler output without launching the game.
#
# Body name is case-insensitive; pass `all` to bake every body with a
# terrain block.
#
# Examples:
#   just bake Thalos              # full-res local bake + PNGs
#   just bake Thalos --preview    # fast preview PNGs, no local game bake
#   just bake all
#   just bake all --preview
bake body *args:
    cargo run --release -p thalos_bake_dump -- {{body}} {{args}}
