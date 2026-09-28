from __future__ import annotations

import json
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from cadac import run_scenario
from cadac.io import jsonc

from cadac_web.paths import CASES_ROOT, CasePathError, resolve_case

RUN_TIMEOUT_S = 120.0

# Honest v1 cancel/timeout: POST /run/{id}/cancel and the worker-side 120 s
# timer set a flag. They do not kill the worker thread (unsafe). Both mean
# abandon the client wait; the worker may still finish. If cancel was
# requested before completion (including before timeout), GET reports
# status=cancelled and ok=false. If the timer fires first, GET reports
# status=error, ok=false, error="timeout". The UI must not plot that result.
RunRegistry: dict[str, dict] = {}
_lock = threading.Lock()


def _copy_program_jsonc(program: str, dest: Path) -> None:
    src = CASES_ROOT() / program
    dest.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        return
    for path in src.glob("*.jsonc"):
        shutil.copy2(path, dest / path.name)


def _patch_end_time(scenario_path: Path, end_time: float) -> None:
    data = jsonc.loads(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenario is not an object")
    data["end_time"] = end_time
    scenario_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run_case(
    program: str,
    stem: str,
    end_time: float | None = None,
    scenario: dict | None = None,
) -> dict:
    try:
        src_path = resolve_case(program, stem)
    except CasePathError as exc:
        return {"ok": False, "error": str(exc)}
    if not src_path.is_file():
        return {"ok": False, "error": f"unknown stem {stem!r}"}

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir)
        _copy_program_jsonc(program, dest)
        temp_path = dest / src_path.name
        try:
            if scenario is not None:
                if not isinstance(scenario, dict):
                    return {"ok": False, "error": "scenario is not an object"}
                temp_path.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
            elif not temp_path.is_file():
                shutil.copy2(src_path, temp_path)
            if end_time is not None:
                _patch_end_time(temp_path, end_time)
            result = run_scenario(temp_path)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        rows = result.plot_rows
        columns = list(rows[0].keys()) if rows else []
        return {
            "ok": True,
            "columns": columns,
            "rows": rows,
            "vehicles": list(getattr(result, "tracks", None) or []),
            "modules": dict(getattr(result, "column_modules", None) or {}),
        }


def _cannot_start(program: str, stem: str) -> dict | None:
    try:
        src_path = resolve_case(program, stem)
    except CasePathError as exc:
        return {"ok": False, "error": str(exc)}
    if not src_path.is_file():
        return {"ok": False, "error": f"unknown stem {stem!r}"}
    return None


def _cancel_timer(entry: dict) -> None:
    timer = entry.get("timer")
    if timer is not None:
        timer.cancel()


def _mark_timeout(run_id: str) -> None:
    with _lock:
        entry = RunRegistry.get(run_id)
        if entry is None:
            return
        if entry["event"].is_set() or entry["status"] == "cancelled":
            return
        if entry["status"] == "running":
            entry["status"] = "error"
            entry["result"] = {"ok": False, "error": "timeout"}


def _worker(
    run_id: str,
    program: str,
    stem: str,
    end_time: float | None,
    scenario: dict | None,
) -> None:
    try:
        result = run_case(program, stem, end_time, scenario)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    with _lock:
        entry = RunRegistry[run_id]
        _cancel_timer(entry)
        if entry["status"] == "cancelled":
            entry["result"] = {"ok": False}
            return
        if entry["status"] != "running":
            return
        if entry["event"].is_set():
            entry["status"] = "cancelled"
            entry["result"] = {"ok": False}
            return
        if result.get("ok"):
            entry["status"] = "done"
            entry["result"] = result
        else:
            entry["status"] = "error"
            entry["result"] = result


def start_run(
    program: str,
    stem: str,
    end_time: float | None = None,
    timeout: float | None = None,
    scenario: dict | None = None,
) -> dict:
    failed = _cannot_start(program, stem)
    if failed is not None:
        return failed
    run_id = str(uuid.uuid4())
    limit = RUN_TIMEOUT_S if timeout is None else timeout
    timer = threading.Timer(limit, _mark_timeout, args=(run_id,))
    timer.daemon = True
    with _lock:
        RunRegistry[run_id] = {
            "event": threading.Event(),
            "status": "running",
            "result": None,
            "timer": timer,
        }
    thread = threading.Thread(
        target=_worker,
        args=(run_id, program, stem, end_time, scenario),
        daemon=True,
    )
    thread.start()
    timer.start()
    return {"ok": True, "runId": run_id}


def run_status(run_id: str) -> dict | None:
    with _lock:
        entry = RunRegistry.get(run_id)
        if entry is None:
            return None
        status = entry["status"]
        result = entry["result"] or {}
        if entry["event"].is_set() and status == "running":
            status = "cancelled"
        payload: dict = {
            "status": status,
            "ok": bool(status == "done" and result.get("ok")),
        }
        if status == "done":
            payload["columns"] = result.get("columns", [])
            payload["rows"] = result.get("rows", [])
            payload["vehicles"] = result.get("vehicles", [])
            payload["modules"] = result.get("modules", {})
        if status == "error":
            payload["error"] = result.get("error", "error")
        return payload


def cancel_run(run_id: str) -> bool:
    with _lock:
        entry = RunRegistry.get(run_id)
        if entry is None:
            return False
        entry["event"].set()
        _cancel_timer(entry)
        if entry["status"] == "running":
            entry["status"] = "cancelled"
            entry["result"] = {"ok": False}
        return True
