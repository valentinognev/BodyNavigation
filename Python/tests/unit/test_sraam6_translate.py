from pathlib import Path

from cadac.io.jsonc import loads
from cadac.io.translate import translate_scenario_asc

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "CADAC_Simulations/SRAAM6_250130/SRAAM6/input_1v1.asc"


def test_translate_1v1_family_on_each_vehicle(tmp_path: Path):
    translate_scenario_asc(SRC, tmp_path, family="sraam6")
    data = loads((tmp_path / "input_1v1.jsonc").read_text(encoding="utf-8"))
    assert data.get("family") in (None, "sraam6")
    assert [v["family"] for v in data["vehicles"]] == ["sraam6", "sraam6"]
    assert data["end_time"] == 12
    assert data["vehicles"][0]["type"] == "MISSILE6"
    assert data["vehicles"][1]["type"] == "TARGET3"
    params = data["vehicles"][0]["params"]
    assert "GAUSS" not in params
    assert params.get("biast", 0) == 0
    assert params.get("biasp", 0) == 0
    assert params.get("biaseh", 0) == 0
    assert params["tgt_num"] == 1
    assert params["mseek"] == 2
    assert params["ms1dyn"] == 1
    assert params["mprop"] == 1
    assert params["mact"] == 2
    assert params["maut"] == 2
    assert params["sbel3"] == -5000
    assert params["dvbe"] == 250
    tgt = data["vehicles"][1]["params"]
    assert tgt["tgt_option"] == 1
    assert tgt["sael1"] == 10000
    assert tgt["dvae"] == 250
    assert "aero_deck" not in data["vehicles"][1]
    assert data["vehicles"][0]["events"][0]["set"] == {"maut": 3, "mguid": 3}


def test_translate_without_family_omits_vehicle_key(tmp_path: Path):
    translate_scenario_asc(SRC, tmp_path)
    data = loads((tmp_path / "input_1v1.jsonc").read_text(encoding="utf-8"))
    assert all("family" not in v for v in data["vehicles"])
