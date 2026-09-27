import json
from pathlib import Path

from fastapi.testclient import TestClient

from cadac.io.jsonc import leading_comment, loads
from cadac_web.app import app
from cadac_web.paths import CASES_ROOT

_VALID = {
    "title": "x",
    "options": {},
    "modules": [],
    "timing": {"int_step": 0.01},
    "end_time": 1,
    "vehicles": [],
}


def test_get_hyper3_climb():
    r = TestClient(app).get("/cases/hyper3/input_climb")
    assert r.status_code == 200
    assert r.json()["scenario"]["vehicles"][0]["type"] == "CRUISE3"


def test_validate_unknown_option():
    r = TestClient(app).post("/cases/validate", json={"scenario": {
        "title": "x", "options": {"nope": True}, "modules": [],
        "timing": {"int_step": 0.01}, "end_time": 1, "vehicles": []
    }})
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert "nope" in r.json()["error"]


def test_put_rejects_dotdot():
    r = TestClient(app).put("/cases/hyper3/../secret", json={"scenario": {}})
    assert r.status_code in (400, 404)


def test_get_missing_returns_404():
    r = TestClient(app).get("/cases/hyper3/no_such_stem_zzzz")
    assert r.status_code == 404


def test_get_returns_leading_comment_as_description(tmp_path, monkeypatch):
    monkeypatch.setattr("cadac_web.paths.CASES_ROOT", lambda: tmp_path)
    folder = tmp_path / "hyper3"
    folder.mkdir()
    note = "/* HYPER3, 3-DOF round earth. One CRUISE3. Two-phase climb. */\n"
    (folder / "input_climb.jsonc").write_text(note + json.dumps(_VALID), encoding="utf-8")
    r = TestClient(app).get("/cases/hyper3/input_climb")
    assert r.status_code == 200
    assert r.json()["description"] == "HYPER3, 3-DOF round earth. One CRUISE3. Two-phase climb."


def test_get_description_empty_without_leading_comment(tmp_path, monkeypatch):
    monkeypatch.setattr("cadac_web.paths.CASES_ROOT", lambda: tmp_path)
    folder = tmp_path / "hyper3"
    folder.mkdir()
    (folder / "input_climb.jsonc").write_text(json.dumps(_VALID), encoding="utf-8")
    r = TestClient(app).get("/cases/hyper3/input_climb")
    assert r.status_code == 200
    assert r.json()["description"] == ""


def test_get_hyper3_climb_shape():
    r = TestClient(app).get("/cases/hyper3/input_climb")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["path"].endswith("input_climb.jsonc")
    assert "hyper3" in body["path"].replace("\\", "/")


def test_validate_ok_does_not_persist():
    before = set(CASES_ROOT().rglob("*.jsonc"))
    r = TestClient(app).post("/cases/validate", json={"scenario": _VALID})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    after = set(CASES_ROOT().rglob("*.jsonc"))
    assert after == before


def test_put_writes_pretty_jsonc(tmp_path, monkeypatch):
    monkeypatch.setattr("cadac_web.paths.CASES_ROOT", lambda: tmp_path)
    monkeypatch.setattr("cadac_web.app.CASES_ROOT", lambda: tmp_path)
    (tmp_path / "hyper3").mkdir()
    r = TestClient(app).put("/cases/hyper3/my_case", json={"scenario": _VALID})
    assert r.status_code == 200
    dest = tmp_path / "hyper3" / "my_case.jsonc"
    text = dest.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text == json.dumps(_VALID, indent=2) + "\n"


def test_put_preserves_leading_comment(tmp_path, monkeypatch):
    monkeypatch.setattr("cadac_web.paths.CASES_ROOT", lambda: tmp_path)
    monkeypatch.setattr("cadac_web.app.CASES_ROOT", lambda: tmp_path)
    folder = tmp_path / "hyper3"
    folder.mkdir()
    dest = folder / "my_case.jsonc"
    note = "/* HYPER3, 3-DOF round earth. One CRUISE3.\n   Two-phase climb. */\n"
    dest.write_text(note + json.dumps(_VALID, indent=2) + "\n", encoding="utf-8")
    edited = {**_VALID, "title": "edited"}
    r = TestClient(app).put("/cases/hyper3/my_case", json={"scenario": edited})
    assert r.status_code == 200
    text = dest.read_text(encoding="utf-8")
    assert text.startswith(note)
    assert leading_comment(text) == "HYPER3, 3-DOF round earth. One CRUISE3.\nTwo-phase climb."
    assert loads(text)["title"] == "edited"


def test_put_invalid_restores_previous(tmp_path, monkeypatch):
    monkeypatch.setattr("cadac_web.paths.CASES_ROOT", lambda: tmp_path)
    monkeypatch.setattr("cadac_web.app.CASES_ROOT", lambda: tmp_path)
    folder = tmp_path / "hyper3"
    folder.mkdir()
    dest = folder / "my_case.jsonc"
    original = json.dumps(_VALID, indent=2) + "\n"
    dest.write_text(original, encoding="utf-8")
    bad = {**_VALID, "options": {"nope": True}}
    r = TestClient(app).put("/cases/hyper3/my_case", json={"scenario": bad})
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False
    assert "nope" in body["error"]
    assert dest.read_text(encoding="utf-8") == original


def test_import_asc():
    repo = Path(__file__).resolve().parents[3]
    path = repo / "CADAC_Simulations/HYPER3_250114/HYPER3/input_climb.asc"
    before = set(CASES_ROOT().rglob("*.jsonc"))
    r = TestClient(app).post("/cases/import", files={"file": (path.name, path.read_bytes())})
    assert r.status_code == 200
    assert r.json()["ok"]
    assert r.json()["scenario"]["vehicles"][0]["type"] == "CRUISE3"
    after = set(CASES_ROOT().rglob("*.jsonc"))
    assert after == before


def test_import_jsonc():
    payload = b'{ "title": "imported", /* c */ "vehicles": [{"type": "CRUISE3", "name": "x"}] }\n'
    r = TestClient(app).post("/cases/import", files={"file": ("climb.jsonc", payload)})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["scenario"]["title"] == "imported"
    assert body["scenario"]["vehicles"][0]["type"] == "CRUISE3"
    assert body["description"] == ""


def test_import_jsonc_returns_leading_comment():
    payload = (
        b"/* Two vehicles. Horizontal engagement. */\n"
        b'{ "title": "imported", "vehicles": [{"type": "AIM5", "name": "Missile"}] }\n'
    )
    r = TestClient(app).post("/cases/import", files={"file": ("hori.jsonc", payload)})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["description"] == "Two vehicles. Horizontal engagement."


def test_import_parse_fail_is_http_200():
    r = TestClient(app).post("/cases/import", files={"file": ("bad.jsonc", b"not jsonc")})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"]
