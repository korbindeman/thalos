# Thalos — orbital mechanics sandbox

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
    cargo test -p thalos_physics

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
# terrain block. Default resolution is 512² for fast iteration; pass
# `--full` for the body's authored/derived resolution, or
# `--cubemap-resolution N` to force a specific size.
#
# Examples:
#   just bake Thalos
#   just bake thalos
#   just bake all
#   just bake Thalos --full
#   just bake Mira --cubemap-resolution 2048
bake body *args:
    cargo run --release -p thalos_bake_dump -- {{body}} {{args}}
