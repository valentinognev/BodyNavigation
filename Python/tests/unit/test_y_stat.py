"""y_stat → stati.asc / stat.asc (C++ execution.cpp + Missile::stat_data)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cadac.cli import run_scenario
from cadac.kernel.events import EventEngine, EventSpec
from cadac.kernel.state import Field, StateStore


class _Missile:
    type = "MISSILE6"
    name = "m1"

    def __init__(self):
        self.store = StateStore()
        self.store.define(
            Field("time", 0.0, "real", "exec", "environment", ("plot",))
        )
        self.store.define(Field("hbe", 5000.0, "real", "diag", "newton", ("plot",)))
        self.modules = []
        self.events = EventEngine(
            [EventSpec(when={"time": {">": -1.0}}, set={})]
        )
        self.event_time = 0.0
        self.health = 1
        self.event_epoch = False

    def define(self):
        return None


def test_y_stat_writes_stati_asc(tmp_path: Path):
    """options['stat'] creates CADAC-shaped stati.asc; plot.csv equations unchanged."""
    path = tmp_path / "case.jsonc"
    path.write_text("{}", encoding="utf-8", newline="\n")
    vehicle = _Missile()
    cfg = SimpleNamespace(
        title="stat case",
        options={
            "scrn": False,
            "events": False,
            "plot": False,
            "doc": False,
            "csv": False,
            "tabout": False,
            "merge": False,
            "comscrn": False,
            "traj": False,
            "stat": True,
        },
        modules=[],
        timing={"int_step": 0.1},
        end_time=0.0,
        vehicles=[
            SimpleNamespace(
                type="MISSILE6",
                name="m1",
                family="sraam6",
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

    stati = tmp_path / "stat1.asc"
    assert stati.is_file()
    text = stati.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith("1stat case")
    assert "m1" in lines[0]
    assert lines[1].strip().startswith("0  0")
    # banner labels five-across, then at least one data block with MC# / object#
    assert any(line.strip().startswith("time") or "time" in line for line in lines[2:5])
    data_lines = [ln for ln in lines[2:] if ln.strip() and not ln.strip().startswith("time") and "hbe" not in ln]
    assert data_lines, "expected at least one numeric data row"
    # trailing MC index (nmc+1) and vehicle_slot+1
    assert "1" in data_lines[-1]
    assert not (tmp_path / "plot.csv").exists()
