#!/usr/bin/env bash
# Start the CADAC workbench FastAPI backend and Vite frontend.
#
# API http://127.0.0.1:8001  Vite UI http://127.0.0.1:5174
# Creates api/.venv and pip install -e . if cadac is missing (npm install if node_modules missing).
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

ensure_api_venv() {
  local venv="${ROOT}/api/.venv"
  if [[ ! -x "${venv}/bin/python" ]]; then
    echo "Creating API venv..."
    python3 -m venv "${venv}" || return 1
  fi
  if ! "${venv}/bin/python" -c "import cadac" 2>/dev/null; then
    echo "Installing API dependencies..."
    (cd "${ROOT}/api" && "${venv}/bin/python" -m pip install -e .) || return 1
  fi
  "${venv}/bin/python" -c "import cadac, uvicorn" || return 1
}

wait_for_catalog() {
  local url="http://${API_HOST}:${API_PORT}/catalog"
  local py="${ROOT}/api/.venv/bin/python"
  local i
  for i in $(seq 1 50); do
    if "${py}" -c "import urllib.request; urllib.request.urlopen('${url}', timeout=0.4)" 2>/dev/null; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

start_api() {
  local pidfile="${RUN}/api.pid"
  local logfile="${RUN}/api.log"
  if [[ ! -f "${ROOT}/api/cadac_web/app.py" ]]; then
    echo "API not started: ${ROOT}/api/cadac_web/app.py missing."
    return 1
  fi
  ensure_api_venv || return 1
  local py="${ROOT}/api/.venv/bin/python"
  (
    cd "${ROOT}/api"
    export PYTHONPATH="${ROOT}/api${PYTHONPATH:+:${PYTHONPATH}}"
    exec setsid "${py}" -m uvicorn cadac_web.app:app \
      --host "${API_HOST}" --port "${API_PORT}" --reload
  ) >"${logfile}" 2>&1 &
  echo $! >"${pidfile}"
  if ! wait_for_catalog; then
    echo "API failed to serve http://${API_HOST}:${API_PORT}/catalog (log ${logfile})"
    tail -n 40 "${logfile}" || true
    return 1
  fi
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
