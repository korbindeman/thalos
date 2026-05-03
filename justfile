# Thalos — orbital mechanics sandbox

# Run the game
game:
    cargo run -p thalos_game

# Edit a planet's terrain (default: Mira). Usage: just edit auron
edit body="":
    cargo run -p thalos_planet_editor {{ if body != "" { "-- " + body } else { "" } }}

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
    cargo test -p thalos_physics -p thalos_terrain_gen

# Lint
clippy:
    cargo clippy --workspace

# Profile the game with Tracy. Requires a running Tracy client (tracy-profiler
# or tracy-capture) listening on localhost before launch.
trace:
    cargo run --release -p thalos_game --features profile-tracy

# Wipe the on-disk terrain cache. Run this after changing stage code;
# stage param changes invalidate the cache key automatically.
clear-terrain-cache:
    rm -rf target/terrain_cache

# Headless terrain bake + PNG dump. Writes to `stage-bakes/<body>/`.
# Body name is case-insensitive; pass `all` to bake every body with a
# terrain block.
#
# Examples:
#   just bake Thalos
#   just bake thalos
#   just bake all
bake body:
    cargo run --release -p thalos_bake_dump -- {{body}}
