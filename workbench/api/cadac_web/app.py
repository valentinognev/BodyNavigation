from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from cadac.io import jsonc
from cadac.io.scenario import load_scenario
from cadac.io.translate import translate_scenario_asc

from cadac_web import handshake
from cadac_web.paths import CASES_ROOT, CasePathError, resolve_case
from cadac_web.runs import cancel_run, run_status, start_run

PROGRAM_ORDER = (
    "HYPER3",
    "FALCON5",
    "FALCON6",
    "HYPER5",
    "HYPER6",
    "AIM5",
    "CRUISE5",
    "MAGSIX",
    "ROCKET6",
    "SAM6",
    "SRAAM6",
    "AGM6",
)

CORS_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScenarioBody(BaseModel):
    scenario: dict


class RunBody(BaseModel):
    program: str
    stem: str
    end_time: float | None = None
    scenario: dict | None = None


class HandshakeSessionBody(BaseModel):
    vehicle: str
    family: str | None = None
    type: str
    program: str
    stem: str


class HandshakeCompleteBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    source: str
    solver: str
    axes: dict
    tables: dict
    ref: dict = {}


def _pretty_jsonc(data: dict) -> str:
    return json.dumps(data, indent=2) + "\n"


def _case_from_jsonc(path: Path) -> dict[str, str] | None:
    try:
        data = jsonc.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    title = data.get("title")
    if not isinstance(title, str) or not title:
        return None
    if not (data.get("vehicles") or data.get("modules")):
        return None
    return {"stem": path.stem, "title": title}


def _bad_path() -> JSONResponse:
    return JSONResponse(status_code=400, content={"ok": False, "error": "invalid path"})


def _load_error(exc: BaseException) -> JSONResponse:
    return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})


@app.get("/catalog")
def get_catalog() -> dict:
    programs: list[dict] = []
    for label in PROGRAM_ORDER:
        folder = CASES_ROOT() / label.lower()
        if not folder.is_dir():
            continue
        cases = []
        for path in sorted(folder.glob("*.jsonc")):
            entry = _case_from_jsonc(path)
            if entry is not None:
                cases.append(entry)
        if not cases:
            continue
        programs.append({"id": label.lower(), "label": label, "cases": cases})
    return {"programs": programs}


@app.get("/cases/{program}/{stem}")
def get_case(program: str, stem: str):
    try:
        path = resolve_case(program, stem)
    except CasePathError:
        raise HTTPException(status_code=400, detail="invalid path") from None
    if not path.is_file():
        raise HTTPException(status_code=404)
    try:
        scenario = jsonc.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=404) from None
    if not isinstance(scenario, dict):
        raise HTTPException(status_code=404)
    return {"ok": True, "path": str(path), "scenario": scenario}


@app.put("/cases/{program}/{stem}")
def put_case(program: str, stem: str, body: ScenarioBody):
    try:
        path = resolve_case(program, stem)
    except CasePathError:
        return _bad_path()
    previous = path.read_text(encoding="utf-8") if path.is_file() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".jsonc.tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(_pretty_jsonc(body.scenario))
        try:
            load_scenario(tmp_path)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            return _load_error(exc)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    try:
        load_scenario(path)
    except Exception as exc:
        if previous is not None:
            path.write_text(previous, encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
        return _load_error(exc)
    return {"ok": True, "path": str(path)}


@app.post("/cases/validate")
def validate_case(body: ScenarioBody):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "scenario.jsonc"
        path.write_text(_pretty_jsonc(body.scenario), encoding="utf-8")
        try:
            load_scenario(path)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    return {"ok": True}


def _import_fail(exc: BaseException) -> dict:
    return {"ok": False, "error": str(exc)}


@app.post("/cases/import")
async def import_case(file: UploadFile = File(...)):
    name = file.filename or "imported"
    suffix = Path(name).suffix.lower()
    raw = await file.read()
    try:
        if suffix == ".jsonc":
            scenario = jsonc.loads(raw.decode("utf-8"))
        elif suffix == ".asc":
            with tempfile.TemporaryDirectory() as tmpdir:
                src = Path(tmpdir) / Path(name).name
                src.write_bytes(raw)
                dst_dir = Path(tmpdir) / "out"
                translate_scenario_asc(src, dst_dir, family=None)
                dest = dst_dir / f"{src.stem}.jsonc"
                scenario = jsonc.loads(dest.read_text(encoding="utf-8"))
        else:
            return {"ok": False, "error": f"unsupported type {suffix}"}
    except Exception as exc:
        return _import_fail(exc)
    if not isinstance(scenario, dict):
        return {"ok": False, "error": "not an object"}
    return {"ok": True, "scenario": scenario}


@app.post("/run")
def post_run(body: RunBody):
    return start_run(body.program, body.stem, body.end_time, scenario=body.scenario)


@app.get("/run/{run_id}")
def get_run(run_id: str):
    payload = run_status(run_id)
    if payload is None:
        raise HTTPException(status_code=404)
    return payload


@app.post("/run/{run_id}/cancel")
def post_cancel(run_id: str):
    if not cancel_run(run_id):
        raise HTTPException(status_code=404)
    return {"ok": True}


@app.post("/handshake/sessions")
def post_handshake_session(body: HandshakeSessionBody):
    return handshake.create_session(
        vehicle=body.vehicle,
        family=body.family,
        vtype=body.type,
        program=body.program,
        stem=body.stem,
    )


@app.post("/handshake/sessions/{session_id}/complete")
def post_handshake_complete(session_id: str, body: HandshakeCompleteBody):
    result = handshake.complete_session(session_id, body.model_dump())
    if result is None:
        raise HTTPException(status_code=404)
    return result


@app.get("/handshake/sessions/{session_id}")
def get_handshake_session(session_id: str):
    result = handshake.get_session(session_id)
    if result is None:
        raise HTTPException(status_code=404)
    if result.get("ok") is False:
        return JSONResponse(status_code=400, content=result)
    return result


@app.post("/handshake/sessions/{session_id}/confirm")
def post_handshake_confirm(session_id: str):
    status, payload = handshake.confirm_session(session_id)
    if status == 404:
        raise HTTPException(status_code=404)
    if status != 200:
        return JSONResponse(status_code=status, content=payload)
    return payload
