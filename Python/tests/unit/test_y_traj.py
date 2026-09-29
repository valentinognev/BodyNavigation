"""y_traj → traj.asc (C++ execution.cpp traj_banner + traj_data / combus)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cadac.cli import run_scenario
from cadac.kernel.events import EventEngine, EventSpec
from cadac.kernel.state import Field, StateStore


class _Cruise:
    type = "CRUISE3"
    name = "c1"

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


def test_y_traj_writes_traj_asc(tmp_path: Path):
    """options['traj'] creates CADAC-shaped traj.asc; plot.csv equations unchanged."""
    path = tmp_path / "case.jsonc"
    path.write_text("{}", encoding="utf-8", newline="\n")
    vehicle = _Cruise()
    cfg = SimpleNamespace(
        title="traj case",
        options={
            "scrn": False,
            "events": False,
            "plot": False,
            "doc": False,
            "csv": False,
            "tabout": False,
            "merge": False,
            "comscrn": False,
            "traj": True,
            "stat": False,
        },
        modules=[],
        timing={"int_step": 0.1, "traj_step": 0.1},
        end_time=0.0,
        vehicles=[
            SimpleNamespace(
                type="CRUISE3",
                name="c1",
                family="cruise5",
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

    traj = tmp_path / "traj.asc"
    assert traj.is_file()
    text = traj.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith("1traj case")
    assert lines[1].strip().startswith("0  0")
    assert any(line.strip().startswith("time") or "time" in line for line in lines[2:5])
    assert any("alt_c1" in line for line in lines[2:5])
    # at least one numeric data row and a CADAC Studio merge endblock (-1.0)
    data = "\n".join(lines[3:])
    assert "1000" in data or "1000.0" in data
    assert any(line.strip().startswith("-1") for line in lines)
    assert not (tmp_path / "plot.csv").exists()
