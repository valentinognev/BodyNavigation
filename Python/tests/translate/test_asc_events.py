from pathlib import Path

from cadac.io.scenario import load_scenario
from cadac.io.translate import translate_scenario_asc
from cadac.kernel.events import EventSpec

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "CADAC_Simulations/HYPER3_250114/HYPER3/input_climb.asc"


def test_input_climb_yields_two_events(tmp_path: Path):
    translate_scenario_asc(SRC, tmp_path)
    events = load_scenario(tmp_path / "input_climb.jsonc").vehicles[0].events
    assert events == [
        EventSpec(
            when={"time": {">": 10}},
            set={"mprop": 2, "qhold": 50000, "tq": 1, "alphax": 2.5},
        ),
        EventSpec(
            when={"time": {">": 50}},
            set={"alphax": 2.4},
        ),
    ]


def test_if_less_than_and_equals_ops(tmp_path: Path):
    src = tmp_path / "ops.asc"
    src.write_text(
        "TITLE ops\n"
        "OPTIONS y_scrn\n"
        "MODULES\n"
        "\tenvironment\tdef,init,exec\n"
        "END\n"
        "TIMING\n"
        "\tint_step 0.01\n"
        "END\n"
        "VEHICLES 1\n"
        "\tCRUISE3 TEST\n"
        "\t\tIF alt < 1000\n"
        "\t\t\tmprop  0\n"
        "\t\tENDIF\n"
        "\t\tIF wp_flag = -1\n"
        "\t\t\tthrottle  0.5\n"
        "\t\tENDIF\n"
        "\tEND\n"
        "END\n"
        "ENDTIME 1\n"
        "STOP\n",
        encoding="utf-8",
        newline="\n",
    )
    translate_scenario_asc(src, tmp_path)
    events = load_scenario(tmp_path / "ops.jsonc").vehicles[0].events
    assert events == [
        EventSpec(when={"alt": {"<": 1000}}, set={"mprop": 0}),
        EventSpec(when={"wp_flag": {"=": -1}}, set={"throttle": 0.5}),
    ]
