# Thalos — orbital mechanics sandbox

# Run the game
game:
    cargo run -p thalos_game

# Run the planet editor
editor:
    cargo run -p thalos_planet_editor

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
