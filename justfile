# Thalos — orbital mechanics sandbox

# Run the game
run:
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
