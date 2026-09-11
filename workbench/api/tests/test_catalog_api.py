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
