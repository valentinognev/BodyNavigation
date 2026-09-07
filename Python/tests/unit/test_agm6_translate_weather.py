import json
from pathlib import Path

from cadac.io.scenario import VehicleSpec, load_scenario
from cadac.io.translate import translate_scenario_asc


def _weather_asc(path: Path) -> None:
    path.write_text(
        "TITLE weather stoch\n"
        "OPTIONS y_plot\n"
        "MODULES\n"
        "environment def,exec\n"
        "END\n"
        "TIMING\n"
        "int_step 0.1\n"
        "END\n"
        "VEHICLES 1\n"
        "MISSILE6 m1\n"
        "WEATHER_DECK weather_deck.asc\n"
        "GAUSS biasal 0 5\n"
        "MARKOV randal 2 5\n"
        "RAYL dvae 5\n"
        "END\n"
        "ENDTIME 1\n"
        "STOP\n",
        encoding="utf-8",
        newline="\n",
    )


def test_translate_weather_deck_and_stochastic_means(tmp_path):
    src = tmp_path / "input.asc"
    _weather_asc(src)
    translate_scenario_asc(src, tmp_path)
    data = json.loads((tmp_path / "input.jsonc").read_text(encoding="utf-8"))
    vehicle = data["vehicles"][0]
    assert vehicle["weather_deck"] == "weather_deck.jsonc"
    params = vehicle["params"]
    assert params["biasal"] == 0
    assert params["randal"] == 0
    assert params["dvae"] == 5
    assert "GAUSS" not in params
    assert "MARKOV" not in params
    assert "RAYL" not in params

    cfg = load_scenario(tmp_path / "input.jsonc")
    spec = cfg.vehicles[0]
    assert spec.weather_deck == tmp_path / "weather_deck.jsonc"
    assert spec.params["biasal"] == 0
    assert spec.params["randal"] == 0
    assert spec.params["dvae"] == 5


def test_translate_weather_keeps_family_kwarg(tmp_path):
    src = tmp_path / "input.asc"
    _weather_asc(src)
    translate_scenario_asc(src, tmp_path, family="agm6")
    data = json.loads((tmp_path / "input.jsonc").read_text(encoding="utf-8"))
    assert data["family"] == "agm6"
    cfg = load_scenario(tmp_path / "input.jsonc")
    assert cfg.family == "agm6"
    assert cfg.vehicles[0].family == "agm6"
    assert cfg.vehicles[0].weather_deck == tmp_path / "weather_deck.jsonc"


def test_load_scenario_weather_deck_none_when_absent(tmp_path):
    path = tmp_path / "s.jsonc"
    path.write_text(
        json.dumps(
            {
                "title": "t",
                "options": {},
                "modules": [],
                "timing": {"int_step": 0.1},
                "end_time": 0.0,
                "vehicles": [{"type": "PLANE6", "name": "p", "params": {}, "events": []}],
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    spec = load_scenario(path).vehicles[0]
    assert spec.weather_deck is None
    assert spec.family is None


def test_vehicle_spec_weather_deck_defaults_none():
    spec = VehicleSpec(
        type="PLANE6",
        name="p",
        aero_deck=None,
        prop_deck=None,
        params={},
        events=[],
    )
    assert spec.weather_deck is None
    assert spec.family is None
