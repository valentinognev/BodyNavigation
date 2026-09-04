from pathlib import Path

from cadac.io.scenario import load_scenario
from cadac.io.translate import translate_scenario_asc

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "CADAC_Simulations/HYPER3_250114/HYPER3/input_climb.asc"


def test_translate_input_climb_header_and_params(tmp_path: Path):
    translate_scenario_asc(SRC, tmp_path)
    out = tmp_path / "input_climb.jsonc"
    raw = out.read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")

    cfg = load_scenario(out)
    assert cfg.end_time == 90
    assert cfg.options["scrn"] is True
    assert cfg.options["events"] is True
    assert cfg.options["plot"] is True
    assert cfg.options["doc"] is True
    assert cfg.options["csv"] is True
    assert cfg.timing["int_step"] == 0.01
    assert cfg.modules[0].name == "environment"
    assert "init" in cfg.modules[0].phases

    v = cfg.vehicles[0]
    assert v.type == "CRUISE3"
    assert v.name == "HYPER3"
    assert v.params["lonx"] == -80.55
    assert v.params["alphax"] == 7
    assert v.params["mprop"] == 1
    assert "qhold" not in v.params
    assert "tq" not in v.params
    assert v.aero_deck == tmp_path / "ghame3_aero_deck.jsonc"
    assert v.prop_deck == tmp_path / "ghame3_prop_deck.jsonc"


def test_n_option_maps_to_false(tmp_path: Path):
    src = tmp_path / "n_opt.asc"
    src.write_text(
        "TITLE n-options\n"
        "OPTIONS n_scrn n_plot y_csv\n"
        "MODULES\n"
        "\tenvironment\tdef,init,exec\n"
        "END\n"
        "TIMING\n"
        "\tint_step 0.01\n"
        "END\n"
        "VEHICLES 1\n"
        "\tCRUISE3 TEST\n"
        "\t\tlonx  0\n"
        "\tEND\n"
        "END\n"
        "ENDTIME 1\n"
        "STOP\n",
        encoding="utf-8",
        newline="\n",
    )
    translate_scenario_asc(src, tmp_path)
    cfg = load_scenario(tmp_path / "n_opt.jsonc")
    assert cfg.options["scrn"] is False
    assert cfg.options["plot"] is False
    assert cfg.options["csv"] is True
