from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from cadac.aero_map import AeroPayload, MapResult, map_payload
from cadac.io import jsonc

from cadac_web.paths import CasePathError, resolve_case

SESSION_TTL_S = 3600.0
CALLBACK_BASE = "http://127.0.0.1:8001/handshake/sessions"


@dataclass
class Session:
    id: str
    vehicle: str
    family: str | None
    vtype: str
    program: str
    stem: str
    created_at: float
    payload: dict | None = None


_sessions: dict[str, Session] = {}
_lock = threading.Lock()


def create_session(
    vehicle: str,
    family: str | None,
    vtype: str,
    program: str,
    stem: str,
) -> dict:
    sid = str(uuid.uuid4())
    session = Session(
        id=sid,
        vehicle=vehicle,
        family=family,
        vtype=vtype,
        program=program,
        stem=stem,
        created_at=time.time(),
    )
    with _lock:
        _sessions[sid] = session
    return {
        "id": sid,
        "callback": f"{CALLBACK_BASE}/{sid}/complete",
    }


def _find(session_id: str) -> Session | None:
    with _lock:
        return _sessions.get(session_id)


def _age_status(session: Session) -> str:
    if time.time() - session.created_at >= SESSION_TTL_S:
        return "expired"
    if session.payload is not None:
        return "complete"
    return "open"


def _payload_from_body(body: dict) -> AeroPayload:
    return AeroPayload(
        source=body["source"],
        solver=body["solver"],
        axes=body["axes"],
        tables=body["tables"],
        ref=body["ref"] if body.get("ref") else {},
    )


def complete_session(session_id: str, body: dict) -> dict | None:
    session = _find(session_id)
    if session is None:
        return None
    payload = _payload_from_body(body)
    stored = asdict(payload)
    with _lock:
        current = _sessions.get(session_id)
        if current is None:
            return None
        current.payload = stored
    return {"ok": True}


def _vehicle_from_case(session: Session) -> dict | None:
    try:
        path = resolve_case(session.program, session.stem)
    except CasePathError:
        return None
    if not path.is_file():
        return None
    try:
        scenario = jsonc.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(scenario, dict):
        return None
    for vehicle in scenario.get("vehicles") or []:
        if isinstance(vehicle, dict) and vehicle.get("name") == session.vehicle:
            return vehicle
    return None


def _sibling_deck_path(session: Session) -> Path | None:
    vehicle = _vehicle_from_case(session)
    if vehicle is None:
        return None
    deck_name = vehicle.get("aero_deck")
    if not isinstance(deck_name, str) or not deck_name:
        return None
    try:
        case_path = resolve_case(session.program, session.stem)
    except CasePathError:
        return None
    return case_path.parent / Path(deck_name).name


def _template_for(session: Session) -> dict | None:
    deck_path = _sibling_deck_path(session)
    if deck_path is None or not deck_path.is_file():
        return None
    try:
        template = jsonc.loads(deck_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return template if isinstance(template, dict) else None


def _payload_obj(stored: dict) -> AeroPayload:
    return AeroPayload(
        source=stored["source"],
        solver=stored["solver"],
        axes=stored["axes"],
        tables=stored["tables"],
        ref=stored.get("ref") or {},
    )


def _map_session(session: Session) -> MapResult:
    payload = _payload_obj(session.payload or {})
    template = _template_for(session)
    return map_payload(session.family, session.vtype, payload, template)


def get_session(session_id: str) -> dict | None:
    session = _find(session_id)
    if session is None:
        return None
    status = _age_status(session)
    if status == "open":
        return {"status": "open", "can_confirm": False}
    if status == "expired":
        return {"status": "expired", "can_confirm": False}
    try:
        result = _map_session(session)
    except Exception as exc:
        return {"ok": False, "error": str(exc) or "map failed"}
    return {
        "status": "complete",
        "payload": session.payload,
        "preview": asdict(result),
        "can_confirm": result.can_confirm,
    }


def confirm_session(session_id: str) -> tuple[int, dict]:
    session = _find(session_id)
    if session is None:
        return 404, {}
    if _age_status(session) != "complete":
        return 400, {"ok": False, "error": "cannot confirm"}
    try:
        result = _map_session(session)
    except Exception:
        return 400, {"ok": False, "error": "cannot confirm"}
    if not result.can_confirm:
        return 400, {"ok": False, "error": "cannot confirm"}
    dest = _sibling_deck_path(session)
    if dest is None:
        return 400, {"ok": False, "error": "cannot confirm"}
    dest.write_text(json.dumps(result.deck, indent=2) + "\n", encoding="utf-8")
    return 200, {"ok": True}
