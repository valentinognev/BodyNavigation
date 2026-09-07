import json
from pathlib import Path

from cadac.io.scenario import load_scenario
from cadac.io.translate import translate_scenario_asc

_MINIMAL_ASC = (
    "TITLE t\nOPTIONS y_plot n_csv\nMODULES\n\tkinematics\tdef,exec\nEND\n"
    "TIMING\n\tint_step 0.001\nEND\nVEHICLES 1\n"
    "\tHYPER6 SLV\n"
    "\t\tmair  012\n"
    "\tEND\nENDTIME 1\nSTOP\n"
)


def test_translate_family_and_stoch(tmp_path: Path):
    src = tmp_path / "in.asc"
    src.write_text(
        "TITLE t\nOPTIONS y_plot n_csv\nMODULES\n\tkinematics\tdef,exec\nEND\n"
        "TIMING\n\tint_step 0.001\nEND\nVEHICLES 1\n"
        "\tHYPER6 SLV\n"
        "\t\tmair  012\n"
        "\t\tWEATHER_DECK  weather_deck_Wallops.asc\n"
        "\t\tAERO_DECK aero_deck_SLV.asc\n"
        "\t\tRAYL dvae  5\n"
        "\t\tGAUSS ucbias_error  0  3\n"
        "\t\tMARKOV pr1_noise  0.25  0.002\n"
        "\tEND\nENDTIME 190\nSTOP\n",
        encoding="utf-8",
        newline="\n",
    )
    translate_scenario_asc(src, tmp_path, family="rocket6")
    raw = json.loads((tmp_path / "in.jsonc").read_text(encoding="utf-8"))
    assert raw["family"] == "rocket6"
    assert raw["vehicles"][0]["family"] == "rocket6"
    cfg = load_scenario(tmp_path / "in.jsonc")
    v = cfg.vehicles[0]
    assert v.type == "HYPER6"
    assert v.family == "rocket6"  # scenario default resolved onto VehicleSpec
    assert v.params["mair"] == 12
    assert v.params["dvae"] == 5
    assert v.params["ucbias_error"] == 0
    assert v.params["pr1_noise"] == 0
    assert v.weather_deck == tmp_path / "weather_deck_Wallops.jsonc"
    assert v.aero_deck == tmp_path / "aero_deck_SLV.jsonc"
    assert v.prop_deck is None


def test_translate_without_family_writes_no_family_key(tmp_path: Path):
    src = tmp_path / "in.asc"
    src.write_text(_MINIMAL_ASC, encoding="utf-8", newline="\n")
    translate_scenario_asc(src, tmp_path)
    raw = json.loads((tmp_path / "in.jsonc").read_text(encoding="utf-8"))
    assert "family" not in raw
    assert "family" not in raw["vehicles"][0]
    assert load_scenario(tmp_path / "in.jsonc").vehicles[0].family is None
