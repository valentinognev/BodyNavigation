"""y_scrn → timed console tables (C++ execution.cpp + Vehicle::scrn_banner/scrn_data)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cadac.cli import run_scenario
from cadac.kernel.events import EventEngine, EventSpec
from cadac.kernel.state import Field, StateStore


class _Plane:
    type = "PLANE"
    name = "p1"

    def __init__(self):
        self.store = StateStore()
        self.store.define(
            Field("time", 0.0, "real", "exec", "kinematics", ("scrn", "plot"))
        )
        self.store.define(
            Field("alt", 1000.0, "real", "out", "newton", ("scrn", "plot"))
        )
        self.modules = []
        self.events = EventEngine(
            [EventSpec(when={"time": {">": -1.0}}, set={})]
        )
        self.event_time = 0.0
        self.health = 1
        self.event_epoch = False

    def define(self):
        return None


def _cfg(*, scrn: bool, end_time: float = 0.1):
    return SimpleNamespace(
        title="scrn case",
        options={
            "scrn": scrn,
            "events": False,
            "plot": False,
            "doc": False,
            "csv": False,
            "tabout": False,
            "merge": False,
            "comscrn": False,
            "traj": False,
            "stat": False,
        },
        modules=[],
        timing={"int_step": 0.1, "scrn_step": 0.1},
        end_time=end_time,
        vehicles=[
            SimpleNamespace(
                type="PLANE",
                name="p1",
                family=None,
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


def test_y_scrn_emits_timed_table(tmp_path: Path, capsys):
    """options['scrn'] prints CADAC-like banner+data on schedule; default off is quiet."""
    path = tmp_path / "case.jsonc"
    path.write_text("{}", encoding="utf-8", newline="\n")
    vehicle = _Plane()

    with (
        patch("cadac.cli.load_scenario", return_value=_cfg(scrn=False)),
        patch("cadac.cli._build_vehicle", return_value=vehicle),
    ):
        run_scenario(path)
    quiet = capsys.readouterr()
    assert quiet.out == ""
    assert not (tmp_path / "plot.csv").exists()

    vehicle = _Plane()
    with (
        patch("cadac.cli.load_scenario", return_value=_cfg(scrn=True)),
        patch("cadac.cli._build_vehicle", return_value=vehicle),
    ):
        run_scenario(path)
    out = capsys.readouterr().out
    assert "Vehicle:" in out
    assert "time" in out
    assert "alt" in out or "1000" in out
    # vehicle name row + timed dumps (init + t≈0 + t=scrn_step)
    assert out.count("p1") >= 3
    assert not (tmp_path / "plot.csv").exists()
