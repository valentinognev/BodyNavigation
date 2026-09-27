from pathlib import Path

from fastapi.testclient import TestClient
from cadac_web.app import app


def test_catalog_lists_hyper3_climb():
    client = TestClient(app)
    r = client.get("/catalog")
    assert r.status_code == 200
    programs = {p["id"]: p for p in r.json()["programs"]}
    assert "hyper3" in programs
    stems = {c["stem"] for c in programs["hyper3"]["cases"]}
    assert "input_climb" in stems
    assert "ghame3_aero_deck" not in stems


def test_cases_root_ignores_cadac_install_location(monkeypatch):
    monkeypatch.delenv("CADAC_CASES", raising=False)
    monkeypatch.setattr(
        "cadac_web.paths.repo_root", lambda: Path("/not-the-repo"), raising=False
    )
    from cadac_web.paths import CASES_ROOT

    climb = CASES_ROOT() / "hyper3" / "input_climb.jsonc"
    assert climb.is_file(), CASES_ROOT()


def test_catalog_lists_hyper3_climb():
    client = TestClient(app)
    r = client.get("/catalog")
    assert r.status_code == 200
    programs = {p["id"]: p for p in r.json()["programs"]}
    assert "hyper3" in programs
    stems = {c["stem"] for c in programs["hyper3"]["cases"]}
    assert "input_climb" in stems
    assert "ghame3_aero_deck" not in stems
