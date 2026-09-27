#!/usr/bin/env bash
# Stop the CADAC workbench started by ./start.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${ROOT}/workbench/kill.sh" "$@"
