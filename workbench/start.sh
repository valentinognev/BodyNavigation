#!/usr/bin/env bash
# Start the CADAC workbench FastAPI backend and Vite frontend.
#
# Once (from this directory):
#   (cd api && python -m venv .venv && .venv/bin/pip install -e .)
#   (cd web && npm install)
# Then: ./start.sh   # API http://127.0.0.1:8001  Vite UI http://127.0.0.1:5174
# Re-run ./start.sh to stop the previous instance and start again.
# Stop only: ./kill.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${ROOT}/.run"
API_PORT="${API_PORT:-8001}"
WEB_PORT="${WEB_PORT:-5174}"
API_HOST="${API_HOST:-127.0.0.1}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"

mkdir -p "${RUN}"

echo "Stopping any previous instance..."
bash "${ROOT}/kill.sh"

start_api() {
  local pidfile="${RUN}/api.pid"
  local logfile="${RUN}/api.log"
  if [[ ! -f "${ROOT}/api/cadac_web/app.py" ]]; then
    echo "API not started: ${ROOT}/api/cadac_web/app.py missing."
    return 1
  fi
  local py="python3"
  if [[ -x "${ROOT}/api/.venv/bin/python" ]]; then
    py="${ROOT}/api/.venv/bin/python"
  fi
  (
    cd "${ROOT}/api"
    export PYTHONPATH="${ROOT}/api${PYTHONPATH:+:${PYTHONPATH}}"
    exec setsid "${py}" -m uvicorn cadac_web.app:app \
      --host "${API_HOST}" --port "${API_PORT}" --reload
  ) >"${logfile}" 2>&1 &
  echo $! >"${pidfile}"
  echo "API started pid $(cat "${pidfile}") → http://${API_HOST}:${API_PORT}/  (log ${logfile})"
}

start_web() {
  local pidfile="${RUN}/web.pid"
  local logfile="${RUN}/web.log"
  if [[ ! -f "${ROOT}/web/package.json" ]]; then
    echo "Web not started: ${ROOT}/web/package.json missing."
    return 1
  fi
  if [[ ! -d "${ROOT}/web/node_modules" ]]; then
    echo "Installing web dependencies..."
    (cd "${ROOT}/web" && npm install)
  fi
  (
    cd "${ROOT}/web"
    export CHOKIDAR_USEPOLLING=1
    exec setsid npm run dev -- --host "${WEB_HOST}" --port "${WEB_PORT}"
  ) >"${logfile}" 2>&1 &
  echo $! >"${pidfile}"
  echo "Web started pid $(cat "${pidfile}") → http://${WEB_HOST}:${WEB_PORT}/  (log ${logfile})"
}

api_ok=0
web_ok=0
start_api && api_ok=1 || true
start_web && web_ok=1 || true

if [[ "${api_ok}" -eq 0 && "${web_ok}" -eq 0 ]]; then
  echo "Nothing started. Check api/cadac_web/app.py and web/package.json, then re-run ./start.sh"
  exit 1
fi

if [[ "${api_ok}" -eq 1 && "${web_ok}" -eq 0 ]]; then
  echo "Partial start: API only. Run ./kill.sh before retrying after the web app exists."
  exit 0
fi
if [[ "${api_ok}" -eq 0 && "${web_ok}" -eq 1 ]]; then
  echo "Partial start: web only. Run ./kill.sh before retrying after the API exists."
  exit 0
fi

echo "CADAC workbench running. Stop with ${ROOT}/kill.sh"
