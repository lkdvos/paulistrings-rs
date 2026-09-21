#!/usr/bin/env bash
# Fail if Cargo.toml and pyproject.toml disagree on version — CLAUDE.md's
# Releasing policy requires the two to move together. Shared by ci.yml,
# release.yml and crates-publish.yml so the check has one definition.
set -euo pipefail

cd "$(dirname "$0")/.."

cargo_version="$(grep -m1 '^version = ' Cargo.toml | sed -E 's/version = "(.*)"/\1/')"
pyproject_version="$(grep -m1 '^version = ' pyproject.toml | sed -E 's/version = "(.*)"/\1/')"

if [[ "$cargo_version" != "$pyproject_version" ]]; then
  echo "::error::Cargo.toml ($cargo_version) and pyproject.toml ($pyproject_version) disagree — release both together, e.g. scripts/bump-version.sh"
  exit 1
fi

echo "$cargo_version"
