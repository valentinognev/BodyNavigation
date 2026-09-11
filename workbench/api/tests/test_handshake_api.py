import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi.testclient import TestClient

from cadac.aero_map.payload import payload_from_mdt_rows
from cadac.io import jsonc
from cadac_web.app import app

REPO = Path(__file__).resolve().parents[3]
SRAAM_SRC = REPO / "Python" / "cases" / "sraam6"
LIBRARY_DECK = SRAAM_SRC / "sraam6_aero_deck.jsonc"

GRID_ROWS = [
    {"mach": 0.8, "alpha": 0.0, "cn": 0.1, "cm": -0.01, "ca": 0.3},
    {"mach": 0.8, "alpha": 4.0, "cn": 0.5, "cm": -0.02, "ca": 0.31},
    {"mach": 1.2, "alpha": 0.0, "cn": 0.12, "cm": -0.011, "ca": 0.4},
    {"mach": 1.2, "alpha": 4.0, "cn": 0.6, "cm": -0.03, "ca": 0.41},
]

SESSION_BODY = {
    "vehicle": "Missile",
    "family": "sraam6",
    "type": "MISSILE6",
    "program": "sraam6",
    "stem": "input_1v1",
}


def _mdt_payload() -> dict:
    return asdict(payload_from_mdt_rows(GRID_ROWS))


def _copy_sraam(tmp_path, monkeypatch) -> Path:
    dest = tmp_path / "sraam6"
    shutil.copytree(SRAAM_SRC, dest)
    monkeypatch.setenv("CADAC_CASES", str(tmp_path))
    return dest


def test_session_complete_preview_sraam(tmp_path, monkeypatch):
    _copy_sraam(tmp_path, monkeypatch)
    client = TestClient(app)
    created = client.post("/handshake/sessions", json=SESSION_BODY)
    assert created.status_code == 200
    body = created.json()
    sid = body["id"]
    uuid.UUID(sid)
    assert body["callback"] == f"http://127.0.0.1:8001/handshake/sessions/{sid}/complete"

    complete = client.post(f"/handshake/sessions/{sid}/complete", json=_mdt_payload())
    assert complete.status_code == 200
    assert complete.json() == {"ok": True}

    got = client.get(f"/handshake/sessions/{sid}")
    assert got.status_code == 200
    preview = got.json()
    assert preview["status"] == "complete"
    assert preview["can_confirm"] is True
    assert preview["payload"]["source"] == "misdc"
    assert preview["payload"]["solver"] == "mdt"
    assert preview["payload"]["tables"]["cn"][0][1] == 0.5
    rows = {r["name"]: r["status"] for r in preview["preview"]["rows"]}
    assert rows["cn0_vs_mach_alpha"] == "mapped"
    assert rows["clm0_vs_mach_alpha"] == "mapped"
    assert rows["ca0_vs_mach"] == "mapped"
    assert rows["clmq_vs_mach"] == "merged"


def test_confirm_writes_aero_deck_under_tmp(tmp_path, monkeypatch):
    dest = _copy_sraam(tmp_path, monkeypatch)
    library_before = LIBRARY_DECK.read_bytes()
    client = TestClient(app)
    sid = client.post("/handshake/sessions", json=SESSION_BODY).json()["id"]
    client.post(f"/handshake/sessions/{sid}/complete", json=_mdt_payload())
    confirm = client.post(f"/handshake/sessions/{sid}/confirm")
    assert confirm.status_code == 200
    written = dest / "sraam6_aero_deck.jsonc"
    assert written.is_file()
    assert not (dest / "aero_deck.jsonc").exists()
    deck = jsonc.loads(written.read_text(encoding="utf-8"))
    cn = next(t for t in deck["tables"] if t["name"] == "cn0_vs_mach_alpha")
    assert cn["values"] == [[0.1, 0.5], [0.12, 0.6]]
    assert LIBRARY_DECK.read_bytes() == library_before


def test_open_session_cannot_confirm():
    client = TestClient(app)
    created = client.post("/handshake/sessions", json=SESSION_BODY)
    sid = created.json()["id"]
    got = client.get(f"/handshake/sessions/{sid}")
    assert got.status_code == 200
    body = got.json()
    assert body["status"] == "open"
    assert body["can_confirm"] is False
    confirm = client.post(f"/handshake/sessions/{sid}/confirm")
    assert confirm.status_code == 400


def test_unknown_session_404():
    client = TestClient(app)
    sid = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/handshake/sessions/{sid}").status_code == 404
    assert client.post(f"/handshake/sessions/{sid}/complete", json=_mdt_payload()).status_code == 404
    assert client.post(f"/handshake/sessions/{sid}/confirm").status_code == 404


def test_malformed_payload_get_and_confirm_are_400(tmp_path, monkeypatch):
    _copy_sraam(tmp_path, monkeypatch)
    client = TestClient(app)
    sid = client.post("/handshake/sessions", json=SESSION_BODY).json()["id"]
    bad = {
        "source": "misdc",
        "solver": "mdt",
        "axes": {},
        "tables": {"ca": [[0.3]]},
        "ref": {},
    }
    complete = client.post(f"/handshake/sessions/{sid}/complete", json=bad)
    assert complete.status_code == 200
    got = client.get(f"/handshake/sessions/{sid}")
    assert got.status_code == 400
    assert got.json()["ok"] is False
    confirm = client.post(f"/handshake/sessions/{sid}/confirm")
    assert confirm.status_code == 400
    assert confirm.json()["ok"] is False


def test_session_expires_after_3600s(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr("cadac_web.handshake.time.time", lambda: clock["now"])
    client = TestClient(app)
    sid = client.post("/handshake/sessions", json=SESSION_BODY).json()["id"]
    clock["now"] = 4601.0
    got = client.get(f"/handshake/sessions/{sid}")
    assert got.status_code == 200
    assert got.json()["status"] == "expired"
    assert got.json()["can_confirm"] is False
