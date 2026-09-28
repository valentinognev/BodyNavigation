from pathlib import Path

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


def _capture_start(monkeypatch) -> dict:
    seen: dict = {}

    def fake(start):
        seen["start"] = start
        return None

    monkeypatch.setattr("cadac_web.browse.ask_open_path", fake)
    return seen


def _case_folder(tmp_path, monkeypatch) -> Path:
    folder = tmp_path / "hyper3"
    folder.mkdir()
    (folder / "input_climb.jsonc").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("CADAC_CASES", str(tmp_path))
    return folder


def test_browse_starts_in_the_case_folder(monkeypatch):
    seen = _capture_start(monkeypatch)
    TestClient(app).post("/browse", json={"program": "hyper3", "stem": "input_climb"})
    assert seen["start"] == CASES_ROOT() / "hyper3"


def test_blank_current_still_starts_in_the_case_folder(monkeypatch):
    seen = _capture_start(monkeypatch)
    TestClient(app).post(
        "/browse",
        json={"program": "hyper3", "stem": "input_climb", "current": "   "},
    )
    assert seen["start"] == CASES_ROOT() / "hyper3"


def test_relative_existing_file_is_the_dialog_start(tmp_path, monkeypatch):
    folder = _case_folder(tmp_path, monkeypatch)
    deck = folder / "ghame3_aero_deck.jsonc"
    deck.write_text("{}\n", encoding="utf-8")
    seen = _capture_start(monkeypatch)
    TestClient(app).post(
        "/browse",
        json={"program": "hyper3", "stem": "input_climb", "current": "ghame3_aero_deck.jsonc"},
    )
    assert seen["start"] == deck


def test_relative_existing_directory_is_the_dialog_start(tmp_path, monkeypatch):
    folder = _case_folder(tmp_path, monkeypatch)
    decks = folder / "decks"
    decks.mkdir()
    seen = _capture_start(monkeypatch)
    TestClient(app).post(
        "/browse",
        json={"program": "hyper3", "stem": "input_climb", "current": "decks"},
    )
    assert seen["start"] == decks


def test_relative_missing_file_starts_in_its_folder(tmp_path, monkeypatch):
    folder = _case_folder(tmp_path, monkeypatch)
    decks = folder / "decks"
    decks.mkdir()
    seen = _capture_start(monkeypatch)
    TestClient(app).post(
        "/browse",
        json={"program": "hyper3", "stem": "input_climb", "current": "decks/missing.jsonc"},
    )
    assert seen["start"] == decks


def test_absolute_existing_file_is_the_dialog_start(tmp_path, monkeypatch):
    _case_folder(tmp_path, monkeypatch)
    deck = tmp_path / "elsewhere" / "aero.jsonc"
    deck.parent.mkdir()
    deck.write_text("{}\n", encoding="utf-8")
    seen = _capture_start(monkeypatch)
    TestClient(app).post(
        "/browse",
        json={"program": "hyper3", "stem": "input_climb", "current": str(deck)},
    )
    assert seen["start"] == deck


def test_absolute_missing_file_starts_in_its_parent(tmp_path, monkeypatch):
    _case_folder(tmp_path, monkeypatch)
    parent = tmp_path / "elsewhere"
    parent.mkdir()
    seen = _capture_start(monkeypatch)
    TestClient(app).post(
        "/browse",
        json={
            "program": "hyper3",
            "stem": "input_climb",
            "current": str(parent / "missing.jsonc"),
        },
    )
    assert seen["start"] == parent


def test_missing_parent_falls_back_to_the_case_folder(tmp_path, monkeypatch):
    folder = _case_folder(tmp_path, monkeypatch)
    seen = _capture_start(monkeypatch)
    TestClient(app).post(
        "/browse",
        json={"program": "hyper3", "stem": "input_climb", "current": "no/such/deck.jsonc"},
    )
    assert seen["start"] == folder


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


def test_zenity_highlights_a_file_without_a_trailing_slash(monkeypatch, tmp_path):
    chosen = tmp_path / "ghame3_aero_deck.jsonc"
    chosen.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "cadac_web.browse.shutil.which",
        lambda name: "/usr/bin/zenity" if name == "zenity" else None,
    )
    seen: dict = {}

    def run(cmd, **kwargs):
        seen["cmd"] = cmd
        return type("Proc", (), {"returncode": 0, "stdout": f"{chosen}\n"})()

    monkeypatch.setattr("cadac_web.browse.subprocess.run", run)
    assert ask_open_path(chosen) == str(chosen)
    assert seen["cmd"][2] == f"--filename={chosen}"


def test_kdialog_receives_a_file_start(monkeypatch, tmp_path):
    chosen = tmp_path / "deck.jsonc"
    chosen.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "cadac_web.browse.shutil.which",
        lambda name: "/usr/bin/kdialog" if name == "kdialog" else None,
    )
    seen: dict = {}

    def run(cmd, **kwargs):
        seen["cmd"] = cmd
        return type("Proc", (), {"returncode": 0, "stdout": f"{chosen}\n"})()

    monkeypatch.setattr("cadac_web.browse.subprocess.run", run)
    assert ask_open_path(chosen) == str(chosen)
    assert seen["cmd"] == ["kdialog", "--getopenfilename", str(chosen)]


def test_tkinter_highlights_a_file_by_name(monkeypatch, tmp_path):
    chosen = tmp_path / "deck.jsonc"
    chosen.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("cadac_web.browse.shutil.which", lambda name: None)
    seen: dict = {}

    class Root:
        def withdraw(self):
            return None

        def destroy(self):
            return None

    def askopenfilename(**kwargs):
        seen.update(kwargs)
        return str(chosen)

    monkeypatch.setattr("tkinter.Tk", lambda: Root())
    monkeypatch.setattr("tkinter.filedialog.askopenfilename", askopenfilename)
    assert ask_open_path(chosen) == str(chosen)
    assert seen == {"initialdir": str(chosen.parent), "initialfile": chosen.name}


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
