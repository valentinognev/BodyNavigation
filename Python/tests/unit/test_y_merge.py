"""y_merge → ploti.asc / plot.asc and stati → stat.asc (C++ merge_plot_files / merge_stat_files)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cadac.cli import run_scenario
from cadac.kernel.events import EventEngine, EventSpec
from cadac.kernel.state import Field, StateStore


def _missile(name: str, hbe: float):
    class _Missile:
        type = "MISSILE6"

        def __init__(self):
            self.name = name
            self.store = StateStore()
            self.store.define(
                Field("time", 0.0, "real", "exec", "environment", ("plot",))
            )
            self.store.define(
                Field("hbe", hbe, "real", "diag", "newton", ("plot",))
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

    return _Missile()


def _spec(name: str):
    return SimpleNamespace(
        type="MISSILE6",
        name=name,
        family="sraam6",
        aero_deck=None,
        prop_deck=None,
        weather_deck=None,
        sam_deck=None,
        srmb_deck=None,
        params={},
        events=[],
    )


def test_y_merge_writes_ploti_stati(tmp_path: Path):
    """options merge+plot+stat produce ploti/plot.asc and stati/stat.asc; plot.csv unchanged."""
    path = tmp_path / "case.jsonc"
    path.write_text("{}", encoding="utf-8", newline="\n")
    vehicles = [_missile("m1", 5000.0), _missile("m2", 6000.0)]
    cfg = SimpleNamespace(
        title="merge case",
        options={
            "scrn": False,
            "events": False,
            "plot": True,
            "doc": False,
            "csv": False,
            "tabout": False,
            "merge": True,
            "comscrn": False,
            "traj": False,
            "stat": True,
        },
        modules=[],
        timing={"int_step": 0.1, "plot_step": 0.1},
        end_time=0.0,
        vehicles=[_spec("m1"), _spec("m2")],
        iseed=0,
        nmonte=0,
    )
    with (
        patch("cadac.cli.load_scenario", return_value=cfg),
        patch("cadac.cli._build_vehicle", side_effect=vehicles),
    ):
        run_scenario(path)

    plot1 = tmp_path / "plot1.asc"
    plot2 = tmp_path / "plot2.asc"
    plot_merged = tmp_path / "plot.asc"
    assert plot1.is_file()
    assert plot2.is_file()
    assert plot_merged.is_file()
    p1 = plot1.read_text(encoding="utf-8")
    assert p1.startswith("1merge case")
    assert "m1" in p1.splitlines()[0]
    assert "5000" in p1 or "5000.0" in p1
    # CADAC Studio endblock: time replaced with -1.0
    assert any(line.strip().startswith("-1") for line in p1.splitlines())

    merged = plot_merged.read_text(encoding="utf-8")
    assert merged.startswith("1merge case")
    # vehicle name stripped from merge title line
    assert "'m1" not in merged.splitlines()[0]
    assert "5000" in merged or "5000.0" in merged
    assert "6000" in merged or "6000.0" in merged

    stat1 = tmp_path / "stat1.asc"
    assert stat1.is_file()
    stat_merged = tmp_path / "stat.asc"
    assert stat_merged.is_file()
    assert "5000" in stat_merged.read_text(encoding="utf-8") or "5000.0" in (
        stat_merged.read_text(encoding="utf-8")
    )

    # csv off → no plot.csv; slot-0 selection path untouched when csv later enabled
    assert not (tmp_path / "plot.csv").exists()


def test_y_merge_plot_csv_slot0_unchanged(tmp_path: Path):
    """plot+csv still writes slot-0 plot.csv; merge is an additional artifact."""
    path = tmp_path / "case.jsonc"
    path.write_text("{}", encoding="utf-8", newline="\n")
    vehicles = [_missile("m1", 1111.0), _missile("m2", 2222.0)]
    cfg = SimpleNamespace(
        title="csv merge",
        options={
            "scrn": False,
            "events": False,
            "plot": True,
            "doc": False,
            "csv": True,
            "tabout": False,
            "merge": True,
            "comscrn": False,
            "traj": False,
            "stat": False,
        },
        modules=[],
        timing={"int_step": 0.1, "plot_step": 0.1},
        end_time=0.0,
        vehicles=[_spec("m1"), _spec("m2")],
        iseed=0,
        nmonte=0,
    )
    with (
        patch("cadac.cli.load_scenario", return_value=cfg),
        patch("cadac.cli._build_vehicle", side_effect=vehicles),
    ):
        run_scenario(path)

    assert (tmp_path / "plot.asc").is_file()
    csv_text = (tmp_path / "plot.csv").read_text(encoding="utf-8")
    assert "1111" in csv_text or "1111.0" in csv_text
    assert "2222" not in csv_text and "2222.0" not in csv_text
