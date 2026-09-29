"""y_comscrn → comscrn.asc (C++ execution.cpp comscrn_data / combus dump)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cadac.cli import run_scenario
from cadac.kernel.events import EventEngine, EventSpec
from cadac.kernel.state import Field, StateStore


class _Rocket:
    type = "ROCKET5"
    name = "r1"

    def __init__(self):
        self.store = StateStore()
        self.store.define(
            Field("time", 0.0, "real", "exec", "kinematics", ("com", "plot"))
        )
        self.store.define(
            Field("alt", 1000.0, "real", "out", "newton", ("com", "plot"))
        )
        self.com_names = ["time", "alt"]
        self.modules = []
        self.events = EventEngine(
            [EventSpec(when={"time": {">": -1.0}}, set={})]
        )
        self.event_time = 0.0
        self.health = 1
        self.event_epoch = False

    def define(self):
        return None


def test_y_comscrn_writes_dump(tmp_path: Path):
    """options['comscrn'] creates CADAC-shaped combus dump; plot.csv equations unchanged."""
    path = tmp_path / "case.jsonc"
    path.write_text("{}", encoding="utf-8", newline="\n")
    vehicle = _Rocket()
    cfg = SimpleNamespace(
        title="comscrn case",
        options={
            "scrn": False,
            "events": False,
            "plot": False,
            "doc": False,
            "csv": False,
            "tabout": False,
            "merge": False,
            "comscrn": True,
            "traj": False,
            "stat": False,
        },
        modules=[],
        timing={"int_step": 0.1, "com_step": 0.1},
        end_time=0.0,
        vehicles=[
            SimpleNamespace(
                type="ROCKET5",
                name="r1",
                family="sam6",
                aero_deck=None,
                prop_deck=None,
                weather_deck=None,
                sam_deck=None,
                srmb_deck=None,
                params={},
                events=[],
            )
        ],
        iseed=0,
        nmonte=0,
    )
    with (
        patch("cadac.cli.load_scenario", return_value=cfg),
        patch("cadac.cli._build_vehicle", return_value=vehicle),
    ):
        run_scenario(path)

    dump = tmp_path / "comscrn.asc"
    assert dump.is_file()
    text = dump.read_text(encoding="utf-8")
    assert "time =" in text
    assert "combus" in text
    assert "ROCKET5" in text
    assert "alt" in text
    assert "r_" in text or "*** r_" in text
    assert "1000" in text or "1000.0" in text
    assert not (tmp_path / "plot.csv").exists()
