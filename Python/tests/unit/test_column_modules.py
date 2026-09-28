import json
import shutil
from pathlib import Path

from cadac.cli import run_scenario
from cadac.io.jsonc import loads
from cadac.io.plot import column_modules
from cadac.kernel.state import Field, StateStore


def test_column_modules_uses_vector_stem_and_skips_time():
    store = StateStore()
    store.define(Field("SBEL", (0.0, 0.0, 0.0), "vec", "state", "newton", ("plot",)))
    store.define(Field("alt", 1.0, "real", "out", "newton", ("plot",)))
    store.define(Field("sbeg", (0.0, 0.0, 0.0), "vec", "state", "newton", ("plot",)))
    store.define(Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces", ("plot",)))
    assert column_modules(store, ["time", "alt", "SBEL1", "SBEG2", "FSPV3", "missing"]) == {
        "alt": "newton",
        "SBEL1": "newton",
        "SBEG2": "newton",
        "FSPV3": "forces",
    }


def test_run_scenario_attaches_column_modules(tmp_path: Path):
    src = Path(__file__).resolve().parents[2] / "cases" / "hyper3"
    for path in src.glob("*.jsonc"):
        shutil.copy2(path, tmp_path / path.name)
    scenario = tmp_path / "input_climb.jsonc"
    data = loads(scenario.read_text(encoding="utf-8"))
    data["end_time"] = 0.05
    scenario.write_text(json.dumps(data) + "\n", encoding="utf-8")
    result = run_scenario(scenario)
    assert result.column_modules["alt"] == "newton"
    assert result.column_modules["SBEG1"] == "newton"
    assert result.column_modules["FSPV1"] == "forces"
    assert "time" not in result.column_modules
    assert result.tracks[0]["modules"]["alt"] == "newton"
    assert result.tracks[0]["modules"]["mach"] == "environment"
