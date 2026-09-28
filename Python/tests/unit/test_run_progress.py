import json
import shutil
from pathlib import Path

from cadac.cli import make_run_progress, run_scenario
from cadac.io.jsonc import loads


def test_synthetic_percent_buckets():
    seen = []
    consider = make_run_progress(1.0, seen.append)
    for i in range(1001):
        consider(i * 0.001, final=(i == 1000))
    assert seen
    assert all(0.0 <= value <= 1.0 for value in seen)
    assert all(seen[i] <= seen[i + 1] for i in range(len(seen) - 1))
    assert seen[-1] == 1.0
    assert len(seen) <= 101


def test_end_time_zero_reports_one():
    seen = []
    consider = make_run_progress(0.0, seen.append)
    consider(0.0, final=True)
    assert seen == [1.0]


def test_run_scenario_reports_progress(tmp_path: Path):
    src = Path(__file__).resolve().parents[2] / "cases" / "hyper3"
    for path in src.glob("*.jsonc"):
        shutil.copy2(path, tmp_path / path.name)
    scenario = tmp_path / "input_climb.jsonc"
    data = loads(scenario.read_text(encoding="utf-8"))
    data["end_time"] = 0.2
    scenario.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    seen = []
    run_scenario(scenario, on_progress=seen.append)
    assert seen
    assert all(0.0 <= value <= 1.0 for value in seen)
    assert all(seen[i] <= seen[i + 1] for i in range(len(seen) - 1))
    assert seen[-1] == 1.0
