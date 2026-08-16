#!/usr/bin/env bash
set -euo pipefail

# The facade supports four compositions. Keep this intentionally small: capability
# features are product bundles, not a power-set of individual plugins.
cargo check -p thalos_runtime --no-default-features
cargo check -p thalos_runtime --no-default-features --features interactive
cargo check -p thalos_runtime --no-default-features --features game
cargo check -p thalos_runtime --no-default-features --features game,capture
cargo check -p thalos_game
cargo check -p korsou
cargo check -p thalos_capture_host

readonly forbidden_packages='thalos_(game_runtime|physics_canonical|physics_local|control|navigation|game_state|hud|map|shipyard|shipyard_editor|structures|udlod|capture_protocol|capture_runtime)'

check_light_tree() {
    local package="$1"
    shift
    local dependency_tree
    local offenders

    dependency_tree="$(cargo tree -p "$package" -e normal "$@")"
    offenders="$(printf '%s\n' "$dependency_tree" | grep -E "(^| )${forbidden_packages} v" || true)"
    if [[ -n "$offenders" ]]; then
        printf 'error: %s pulled in disabled simulation/gameplay/capture packages:\n%s\n' \
            "$package" "$offenders" >&2
        return 1
    fi
    printf 'OK: %s excludes simulation, gameplay, and capture packages\n' "$package"
}

check_light_tree thalos_runtime --no-default-features --features interactive
check_light_tree korsou
