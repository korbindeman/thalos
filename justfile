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
