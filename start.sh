#!/usr/bin/env bash
# Start the CADAC workbench (API :8001, Vite :5174).
# Delegates to workbench/start.sh (kills any previous instance first).
# Stop only: ./kill.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${ROOT}/workbench/start.sh" "$@"
