from fastapi.testclient import TestClient

from cadac_web.app import app
from cadac_web.browse import ask_open_path
from cadac_web.paths import CASES_ROOT


def test_browse_returns_the_dialog_path(monkeypatch):
    monkeypatch.setattr("cadac_web.browse.ask_open_path", lambda start: "/tmp/aero.jsonc")
    response = TestClient(app).post("/browse", json={"program": "hyper3", "stem": "input_climb"})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "path": "/tmp/aero.jsonc"}


def test_browse_reports_cancel(monkeypatch):
    monkeypatch.setattr("cadac_web.browse.ask_open_path", lambda start: None)
    response = TestClient(app).post("/browse", json={"program": "hyper3", "stem": "input_climb"})
    assert response.json() == {"ok": False, "cancelled": True}


def test_browse_starts_in_the_case_folder(monkeypatch):
    seen: dict = {}

    def fake(start):
        seen["start"] = start
        return None

    monkeypatch.setattr("cadac_web.browse.ask_open_path", fake)
    TestClient(app).post("/browse", json={"program": "hyper3", "stem": "input_climb"})
    assert seen["start"] == CASES_ROOT() / "hyper3"


def test_browse_without_a_case_starts_at_the_cases_root(monkeypatch):
    seen: dict = {}

    def fake(start):
        seen["start"] = start
        return None

    monkeypatch.setattr("cadac_web.browse.ask_open_path", fake)
    TestClient(app).post("/browse", json={})
    assert seen["start"] == CASES_ROOT()


def test_zenity_returns_the_chosen_path(monkeypatch):
    monkeypatch.setattr(
        "cadac_web.browse.shutil.which",
        lambda name: "/usr/bin/zenity" if name == "zenity" else None,
    )

    def run(cmd, **kwargs):
        assert cmd[:2] == ["zenity", "--file-selection"]
        assert f"--filename={CASES_ROOT() / 'hyper3'}/" in cmd
        return type("Proc", (), {"returncode": 0, "stdout": "/tmp/aero.jsonc\n"})()

    monkeypatch.setattr("cadac_web.browse.subprocess.run", run)
    assert ask_open_path(CASES_ROOT() / "hyper3") == "/tmp/aero.jsonc"


def test_zenity_cancel_does_not_open_another_dialog(monkeypatch):
    calls: list[str] = []

    def which(name):
        calls.append(name)
        return "/usr/bin/zenity" if name == "zenity" else None

    monkeypatch.setattr("cadac_web.browse.shutil.which", which)
    monkeypatch.setattr(
        "cadac_web.browse.subprocess.run",
        lambda cmd, **kwargs: type("Proc", (), {"returncode": 1, "stdout": ""})(),
    )
    assert ask_open_path(CASES_ROOT()) is None
    assert calls == ["zenity"]
