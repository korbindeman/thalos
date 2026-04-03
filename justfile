# Thalos — N-body orbital mechanics sandbox

# Run the game (loads pre-generated ephemeris if available)
run:
    cargo run -p thalos_game

# Generate ephemeris. Defaults to 10,000 years. Resumable and extensible:
#   just generate 1000    # start with 1,000 years
#   just generate 5000    # extend to 5,000 years
#   just generate         # extend to 10,000 years
generate years="10000":
    cargo run -p thalos_physics --release --bin generate_ephemeris -- {{years}}

# Validate energy conservation over the given span (default 10,000 years)
validate years="10000":
    cargo run -p thalos_physics --release --bin generate_ephemeris -- {{years}} --validate

# Build everything
build:
    cargo build --workspace

# Run physics tests
test:
    cargo test -p thalos_physics

# Lint
clippy:
    cargo clippy --workspace
