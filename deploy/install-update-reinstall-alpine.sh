#!/bin/bash

set -euo pipefail

readonly cInstallerDistribution=alpine
readonly cDefaultBackend=cpu
vScriptDirectory="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
. "${vScriptDirectory}/install-common.sh"

if ! fMain "$@"; then
  printf '%s\n' 'Alpine installation failed.' >&2
  exit 1
fi
