#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  scripts/bump-version.sh patch|minor|major|VERSION
  scripts/bump-version.sh --current

VERSION may be X.Y.Z or vX.Y.Z.
USAGE
}

current_version() {
    awk '
        /^\[workspace\.package\]/ { in_section = 1; next }
        /^\[/ { in_section = 0 }
        in_section && /^[[:space:]]*version[[:space:]]*=/ {
            value = $0
            sub(/^[^"]*"/, "", value)
            sub(/".*$/, "", value)
            print value
            exit
        }
    ' Cargo.toml
}

current="$(current_version)"
if [[ -z "${current}" ]]; then
    echo "Could not read [workspace.package] version from Cargo.toml" >&2
    exit 1
fi

if [[ "${1:-}" == "--current" ]]; then
    echo "${current}"
    exit 0
fi

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 64
fi

if [[ ! "${current}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    echo "Current version must be X.Y.Z, got ${current}" >&2
    exit 1
fi

major="${BASH_REMATCH[1]}"
minor="${BASH_REMATCH[2]}"
patch="${BASH_REMATCH[3]}"

case "$1" in
    patch)
        next="${major}.${minor}.$((patch + 1))"
        ;;
    minor)
        next="${major}.$((minor + 1)).0"
        ;;
    major)
        next="$((major + 1)).0.0"
        ;;
    v[0-9]*.[0-9]*.[0-9]* | [0-9]*.[0-9]*.[0-9]*)
        next="${1#v}"
        if [[ ! "${next}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            usage >&2
            exit 64
        fi
        ;;
    *)
        usage >&2
        exit 64
        ;;
esac

if [[ "${next}" == "${current}" ]]; then
    echo "Workspace version is already ${current}"
    exit 0
fi

NEXT_VERSION="${next}" perl -0pi -e '
    my $next = $ENV{"NEXT_VERSION"};
    s/(\[workspace\.package\][\s\S]*?^[[:space:]]*version[[:space:]]*=[[:space:]]*")([^"]+)(")/$1$next$3/m
        or die "failed to update workspace package version\n";
' Cargo.toml

cargo metadata --format-version 1 --no-deps >/dev/null

echo "Bumped workspace version: ${current} -> ${next}"
echo "Release tag: v${next}"
