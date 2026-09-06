import json
from pathlib import Path

from cadac import run_scenario
from cadac.io.jsonc import loads
from cadac.io.scenario import load_scenario
from cadac.io.translate import translate_scenario_asc

ROOT = Path(__file__).resolve().parents[3]
MAGSIX_ASC = ROOT / "CADAC_Simulations/MAGSIX_231111/MAGSIX"
CASES = Path(__file__).resolve().parents[2] / "cases" / "magsix"


def test_committed_trajectory_case():
    cfg = load_scenario(CASES / "input_trajectoryMR1.jsonc")
    assert cfg.end_time == 50
    assert cfg.timing["int_step"] == 0.001
    assert cfg.timing["plot_step"] == 0.01
    assert [m.name for m in cfg.modules] == ["environment", "trajectory"]
    v = cfg.vehicles[0]
    assert v.type == "ROTOR"
    assert v.family == "magsix"
    assert v.params["hbe"] == 1000
    assert v.params["dvbe"] == 16.6
    assert v.params["omega_rpm"] == 850
    assert "nonlinear" not in v.params
    assert "moi_trans" not in v.params
    assert v.aero_deck is None


def test_trajectory_smoke_hbe_dvbe(tmp_path: Path):
    translate_scenario_asc(
        MAGSIX_ASC / "input_trajectoryMR1.asc", tmp_path, family="magsix"
    )
    path = tmp_path / "input_trajectoryMR1.jsonc"
    data = loads(path.read_text(encoding="utf-8"))
    data["end_time"] = 0.1
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    result = run_scenario(path)
    row = result.plot_rows[-1]
    assert 990.0 < row["hbe"] < 1010.0
    assert "dvbe" in row
    assert "omega_rpm" in row
    assert "sim_time" in row
