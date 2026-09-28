from pathlib import Path
import time
import types
import uuid

from fastapi.testclient import TestClient

from cadac_web.app import app
from cadac_web.paths import CASES_ROOT


def _wait_run(client: TestClient, run_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/run/{run_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} still running after {timeout}s")


def test_run_hyper3_short():
    client = TestClient(app)
    started = client.post("/run", json={"program": "hyper3", "stem": "input_climb", "end_time": 0.05})
    assert started.status_code == 200
    started_body = started.json()
    assert started_body["ok"] is True
    run_id = started_body["runId"]
    uuid.UUID(run_id)
    body = _wait_run(client, run_id)
    assert body["status"] == "done"
    assert body["ok"] is True
    assert "time" in body["columns"]
    assert "alt" in body["columns"]
    assert len(body["rows"]) >= 1
    assert body["rows"][0]["time"] == 0 or body["rows"][0]["time"] >= 0
    assert body["modules"]["alt"] == "newton"
    assert body["vehicles"][0]["modules"]["alt"] == "newton"


def test_run_status_includes_column_modules(monkeypatch):
    def spy(path, on_progress=None):
        return types.SimpleNamespace(
            plot_rows=[{"time": 0.0, "alt": 1.0}],
            tracks=[
                {
                    "name": "Missile",
                    "columns": ["time", "alt", "mach"],
                    "rows": [{"time": 0.0, "alt": 1.0, "mach": 0.8}],
                    "modules": {"alt": "newton", "mach": "environment"},
                }
            ],
            column_modules={"alt": "newton"},
        )

    monkeypatch.setattr("cadac_web.runs.run_scenario", spy)
    client = TestClient(app)
    started = client.post("/run", json={"program": "hyper3", "stem": "input_climb", "end_time": 0.05})
    body = _wait_run(client, started.json()["runId"])
    assert body["modules"]["alt"] == "newton"
    assert body["vehicles"][0]["modules"]["mach"] == "environment"


def test_run_returns_every_vehicle_track(monkeypatch):
    def spy(path, on_progress=None):
        return types.SimpleNamespace(
            plot_rows=[{"time": 0.0, "SBEL1": 0.0}],
            tracks=[
                {
                    "name": "Missile 1",
                    "columns": ["SBEL1", "SBEL2", "SBEL3"],
                    "rows": [{"SBEL1": 0.0, "SBEL2": 1.0, "SBEL3": -2.0}],
                },
                {
                    "name": "Target 1",
                    "columns": ["SAEL1", "SAEL2", "SAEL3"],
                    "rows": [{"SAEL1": 3.0, "SAEL2": 4.0, "SAEL3": -5.0}],
                },
            ],
        )

    monkeypatch.setattr("cadac_web.runs.run_scenario", spy)
    client = TestClient(app)
    started = client.post("/run", json={"program": "sraam6", "stem": "input_2v2", "end_time": 0.05})
    body = _wait_run(client, started.json()["runId"])
    assert body["status"] == "done"
    assert [item["name"] for item in body["vehicles"]] == ["Missile 1", "Target 1"]
    assert body["vehicles"][1]["columns"] == ["SAEL1", "SAEL2", "SAEL3"]


def test_run_unknown_stem():
    r = TestClient(app).post("/run", json={"program": "hyper3", "stem": "no_such"})
    assert r.json()["ok"] is False


def test_run_posted_scenario_is_written_to_temp(monkeypatch):
    from cadac.io import jsonc

    captured: dict = {}

    def spy(path, on_progress=None):
        p = Path(path)
        captured["data"] = jsonc.loads(p.read_text(encoding="utf-8"))
        captured["siblings"] = sorted(x.name for x in p.parent.glob("*.jsonc"))
        return types.SimpleNamespace(plot_rows=[{"time": 0.0, "alt": 0.0}])

    monkeypatch.setattr("cadac_web.runs.run_scenario", spy)
    lib = CASES_ROOT() / "hyper3" / "input_climb.jsonc"
    before = lib.read_text(encoding="utf-8")
    scenario = jsonc.loads(before)
    scenario["title"] = "patched-in-memory"
    scenario["end_time"] = 0.05
    client = TestClient(app)
    started = client.post(
        "/run",
        json={
            "program": "hyper3",
            "stem": "input_climb",
            "end_time": 0.05,
            "scenario": scenario,
        },
    )
    assert started.status_code == 200
    started_body = started.json()
    assert started_body["ok"] is True
    body = _wait_run(client, started_body["runId"])
    assert body["status"] == "done"
    assert captured["data"]["title"] == "patched-in-memory"
    assert captured["data"]["end_time"] == 0.05
    assert "ghame3_aero_deck.jsonc" in captured["siblings"]
    assert "ghame3_prop_deck.jsonc" in captured["siblings"]
    assert lib.read_text(encoding="utf-8") == before


def test_run_does_not_mutate_library_cases():
    folder = CASES_ROOT() / "hyper3"
    plot = folder / "plot.csv"
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in folder.iterdir()}
    existed = plot.exists()
    client = TestClient(app)
    started = client.post("/run", json={"program": "hyper3", "stem": "input_climb", "end_time": 0.05})
    assert started.status_code == 200
    started_body = started.json()
    assert started_body["ok"] is True
    body = _wait_run(client, started_body["runId"])
    assert body["ok"] is True
    after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in folder.iterdir()}
    assert after == before
    if not existed:
        assert not plot.exists()


def test_run_value_error_http_200(monkeypatch):
    def boom(path, on_progress=None):
        raise ValueError("unknown vehicle type 'NO_SUCH_TYPE'")

    monkeypatch.setattr("cadac_web.runs.run_scenario", boom)
    client = TestClient(app)
    started = client.post("/run", json={"program": "hyper3", "stem": "input_climb"})
    assert started.status_code == 200
    started_body = started.json()
    assert started_body["ok"] is True
    body = _wait_run(client, started_body["runId"])
    assert body["status"] == "error"
    assert body["ok"] is False
    assert "NO_SUCH_TYPE" in body["error"]


def test_cancel_marks_run(monkeypatch):
    def slow(path, on_progress=None):
        time.sleep(2.0)
        return types.SimpleNamespace(plot_rows=[{"time": 0.0, "alt": 0.0}])

    monkeypatch.setattr("cadac_web.runs.run_scenario", slow)
    client = TestClient(app)
    t0 = time.monotonic()
    started = client.post("/run", json={"program": "hyper3", "stem": "input_climb", "end_time": 5})
    assert time.monotonic() - t0 < 0.5
    assert started.status_code == 200
    run_id = started.json()["runId"]
    assert client.get(f"/run/{run_id}").json()["status"] == "running"
    cancelled = client.post(f"/run/{run_id}/cancel")
    assert cancelled.status_code == 200
    st = client.get(f"/run/{run_id}").json()
    assert st["status"] == "cancelled"
    assert st["ok"] is False


def test_run_worker_timeout(monkeypatch):
    def slow(path, on_progress=None):
        time.sleep(2.0)
        return types.SimpleNamespace(plot_rows=[{"time": 0.0, "alt": 0.0}])

    monkeypatch.setattr("cadac_web.runs.run_scenario", slow)
    monkeypatch.setattr("cadac_web.runs.RUN_TIMEOUT_S", 0.05)
    client = TestClient(app)
    t0 = time.monotonic()
    started = client.post("/run", json={"program": "hyper3", "stem": "input_climb"})
    assert time.monotonic() - t0 < 0.5
    assert started.status_code == 200
    started_body = started.json()
    assert started_body["ok"] is True
    body = _wait_run(client, started_body["runId"])
    assert body["status"] == "error"
    assert body["ok"] is False
    assert body["error"] == "timeout"


def test_run_progress_while_running_and_done(monkeypatch):
    def progress_then_sleep(path, on_progress=None):
        if on_progress is not None:
            on_progress(0.4)
        time.sleep(0.3)
        return types.SimpleNamespace(plot_rows=[{"time": 0.0, "alt": 0.0}])

    monkeypatch.setattr("cadac_web.runs.run_scenario", progress_then_sleep)
    client = TestClient(app)
    started = client.post("/run", json={"program": "hyper3", "stem": "input_climb", "end_time": 0.05})
    assert started.status_code == 200
    run_id = started.json()["runId"]
    deadline = time.monotonic() + 2.0
    mid = None
    while time.monotonic() < deadline:
        mid = client.get(f"/run/{run_id}").json()
        if mid["status"] == "running" and mid.get("progress") == 0.4:
            break
        time.sleep(0.02)
    assert mid is not None
    assert mid["status"] == "running"
    assert mid["progress"] == 0.4
    body = _wait_run(client, run_id)
    assert body["status"] == "done"
    assert body["progress"] == 1.0


def test_run_done_progress_is_one():
    client = TestClient(app)
    started = client.post("/run", json={"program": "hyper3", "stem": "input_climb", "end_time": 0.05})
    assert started.status_code == 200
    body = _wait_run(client, started.json()["runId"])
    assert body["status"] == "done"
    assert body["progress"] == 1.0


def test_late_cancel_does_not_overwrite_timeout(monkeypatch):
    finished = []

    def slow(path, on_progress=None):
        time.sleep(0.4)
        finished.append(True)
        return types.SimpleNamespace(plot_rows=[{"time": 0.0, "alt": 0.0}])

    monkeypatch.setattr("cadac_web.runs.run_scenario", slow)
    monkeypatch.setattr("cadac_web.runs.RUN_TIMEOUT_S", 0.05)
    client = TestClient(app)
    started = client.post("/run", json={"program": "hyper3", "stem": "input_climb"})
    assert started.status_code == 200
    run_id = started.json()["runId"]
    body = _wait_run(client, run_id)
    assert body["status"] == "error"
    assert body["error"] == "timeout"
    cancelled = client.post(f"/run/{run_id}/cancel")
    assert cancelled.status_code == 200
    deadline = time.monotonic() + 2.0
    while not finished and time.monotonic() < deadline:
        time.sleep(0.05)
    assert finished
    time.sleep(0.05)
    st = client.get(f"/run/{run_id}").json()
    assert st["status"] == "error"
    assert st["ok"] is False
    assert st["error"] == "timeout"
