# Thalos — orbital mechanics sandbox

# Run the game (dev build with dynamic linking for fast iteration)
game:
    cargo run -p thalos_game --features dev

# Edit a planet's terrain (default: Mira). Usage: just edit auron
edit body="":
    cargo run -p thalos_planet_editor --features dev {{ if body != "" { "-- " + body } else { "" } }}

# Run the ship editor (shipyard crate)
shipyard:
    cargo run -p thalos_shipyard --bin ship_editor

# Build everything
build:
    cargo build --workspace

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

# Headless terrain bake + PNG dump. Writes to `target/stage-bakes/<body>/`.
# Body name is case-insensitive; pass `all` to bake every body with a
# generator block. Pass `stage=N` to run only the first N stages.
#
# Examples:
#   just bake Thalos
#   just bake thalos
#   just bake all
#   just bake Mira stage=3
bake body stage="":
    cargo run --release -p thalos_bake_dump -- {{body}} \
        {{ if stage != "" { "--up-to-stage " + stage } else { "" } }}
