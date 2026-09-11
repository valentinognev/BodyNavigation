#!/usr/bin/env bash
# Stop processes started by start.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${ROOT}/.run"
API_PORT="${API_PORT:-8001}"
WEB_PORT="${WEB_PORT:-5174}"

kill_pidfile() {
  local name="$1"
  local pidfile="${RUN}/${name}.pid"
  if [[ ! -f "${pidfile}" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${pidfile}")"
  if [[ -z "${pid}" ]]; then
    rm -f "${pidfile}"
    return 0
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    # Session leader from setsid in start.sh: kill the whole group.
    kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
    local i
    for i in 1 2 3 4 5; do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -9 -- "-${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
    fi
    echo "Stopped ${name} (pid ${pid})"
  else
    echo "${name} pid ${pid} is not running"
  fi
  rm -f "${pidfile}"
}

kill_port() {
  local port="$1"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true)"
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    return 0
  fi
  if [[ -n "${pids}" ]]; then
    echo "Stopping leftover listeners on port ${port}: ${pids}"
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    sleep 0.2
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
  fi
}

kill_pidfile api
kill_pidfile web
kill_port "${API_PORT}"
kill_port "${WEB_PORT}"
echo "CADAC workbench stopped."
