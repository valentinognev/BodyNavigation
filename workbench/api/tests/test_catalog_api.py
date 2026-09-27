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


def test_catalog_stamps_dimension_and_subgroup():
    client = TestClient(app)
    r = client.get("/catalog")
    programs = {p["id"]: p for p in r.json()["programs"]}
    expected = {
        "hyper3": ("3", "round"),
        "falcon5": ("5", "flat"),
        "falcon6": ("6", "flat"),
        "hyper5": ("5", "round"),
        "hyper6": ("6", "round"),
        "aim5": ("5", "flat"),
        "cruise5": ("5", "round"),
        "magsix": ("xz", "spinner"),
        "rocket6": ("6", "round"),
        "sam6": ("6", "flat"),
        "sraam6": ("6", "flat"),
        "agm6": ("6", "flat"),
    }
    assert set(expected) <= set(programs)
    for program_id, (dimension, subgroup) in expected.items():
        assert programs[program_id]["dimension"] == dimension
        assert programs[program_id]["subgroup"] == subgroup


def test_cases_root_ignores_cadac_install_location(monkeypatch):
    monkeypatch.delenv("CADAC_CASES", raising=False)
    monkeypatch.setattr(
        "cadac_web.paths.repo_root", lambda: Path("/not-the-repo"), raising=False
    )
    from cadac_web.paths import CASES_ROOT

    climb = CASES_ROOT() / "hyper3" / "input_climb.jsonc"
    assert climb.is_file(), CASES_ROOT()
