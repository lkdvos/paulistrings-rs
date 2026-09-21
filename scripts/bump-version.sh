#!/usr/bin/env bash
# Bump the single version shared by Cargo.toml (workspace.package) and
# pyproject.toml ([project]) — CLAUDE.md's "Releasing" policy requires both to
# move together, and release.yml / crates-publish.yml both refuse to run
# otherwise.
#
#   scripts/bump-version.sh 0.2.0
set -euo pipefail

cd "$(dirname "$0")/.."

version=${1:-}
if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "usage: scripts/bump-version.sh X.Y.Z" >&2
  exit 1
fi

sed -i -E "0,/^version = /s/^version = \".*\"/version = \"$version\"/" Cargo.toml
sed -i -E "0,/^version = /s/^version = \".*\"/version = \"$version\"/" pyproject.toml

echo "bumped Cargo.toml and pyproject.toml to $version"
git diff --stat -- Cargo.toml pyproject.toml
