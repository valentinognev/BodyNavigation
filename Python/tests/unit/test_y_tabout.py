"""y_tabout → tabout.asc (C++ execution.cpp + Aim/Plane::tabout_*)."""

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


def test_y_tabout_writes_tabout_asc(tmp_path: Path):
    """options['tabout'] creates CADAC-shaped tabout.asc; plot.csv equations unchanged."""
    path = tmp_path / "case.jsonc"
    path.write_text("{}", encoding="utf-8", newline="\n")
    vehicle = _Plane()
    cfg = SimpleNamespace(
        title="tabout case",
        options={
            "scrn": False,
            "events": False,
            "plot": False,
            "doc": False,
            "csv": False,
            "tabout": True,
            "merge": False,
            "comscrn": False,
            "traj": False,
            "stat": False,
        },
        modules=[],
        timing={"int_step": 0.1, "scrn_step": 0.1},
        end_time=0.0,
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
    with (
        patch("cadac.cli.load_scenario", return_value=cfg),
        patch("cadac.cli._build_vehicle", return_value=vehicle),
    ):
        run_scenario(path)

    tabout = tmp_path / "tabout.asc"
    assert tabout.is_file()
    text = tabout.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert any("tabout case" in line for line in lines)
    assert any("Vehicle:" in line for line in lines)
    assert any(line.strip().startswith("time") or "time" in line for line in lines)
    assert any(line.strip() == "p1" or line.strip().startswith("p1") for line in lines)
    assert not (tmp_path / "plot.csv").exists()


class _Aim:
    """Primary vehicle with scrn outputs (AIM5-shaped)."""

    type = "AIM5"
    name = "m1"

    def __init__(self):
        self.store = StateStore()
        self.store.define(
            Field("time", 0.0, "real", "exec", "kinematics", ("scrn", "plot"))
        )
        self.store.define(
            Field("alt", 5000.0, "real", "out", "newton", ("scrn", "plot"))
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


class _AircraftStub:
    """Trailing companion with no scrn outputs (AIRCRAFT3 empty stubs)."""

    type = "AIRCRAFT3"
    name = "a1"

    def __init__(self):
        self.store = StateStore()
        self.store.define(Field("time", 0.0, "real", "exec", "kinematics", ()))
        self.modules = []
        self.events = EventEngine(
            [EventSpec(when={"time": {">": -1.0}}, set={})]
        )
        self.event_time = 0.0
        self.health = 1
        self.event_epoch = False

    def define(self):
        return None


def test_y_tabout_scrn_time_advances_with_trailing_no_scrn(tmp_path: Path):
    """Last vehicle without scrn must still advance scrn_time (AIM5 AIRCRAFT3 last)."""
    path = tmp_path / "case.jsonc"
    path.write_text("{}", encoding="utf-8", newline="\n")
    aim = _Aim()
    aircraft = _AircraftStub()
    specs = [
        SimpleNamespace(
            type="AIM5",
            name="m1",
            family="aim5",
            aero_deck=None,
            prop_deck=None,
            weather_deck=None,
            sam_deck=None,
            srmb_deck=None,
            params={},
            events=[],
        ),
        SimpleNamespace(
            type="AIRCRAFT3",
            name="a1",
            family="aim5",
            aero_deck=None,
            prop_deck=None,
            weather_deck=None,
            sam_deck=None,
            srmb_deck=None,
            params={},
            events=[],
        ),
    ]
    built = [aim, aircraft]
    cfg = SimpleNamespace(
        title="aim5 multi tabout",
        options={
            "scrn": False,
            "events": False,
            "plot": False,
            "doc": False,
            "csv": False,
            "tabout": True,
            "merge": False,
            "comscrn": False,
            "traj": False,
            "stat": False,
        },
        modules=[],
        timing={"int_step": 0.1, "scrn_step": 0.1},
        end_time=0.1,
        vehicles=specs,
        iseed=0,
        nmonte=0,
    )

    def _build(_path, spec):
        return built[specs.index(spec)]

    with (
        patch("cadac.cli.load_scenario", return_value=cfg),
        patch("cadac.cli._build_vehicle", side_effect=_build),
    ):
        run_scenario(path)

    text = (tmp_path / "tabout.asc").read_text(encoding="utf-8")
    # Init dump + timed dumps at t=0 and t=scrn_step (0.1). Bug: only init + t≈0.
    m1_blocks = sum(1 for line in text.splitlines() if line.strip() == "m1")
    assert m1_blocks >= 3, f"expected >=3 m1 data blocks (init+2 timed), got {m1_blocks}"
    assert "a1" not in text  # no-scrn companion never writes tabout_data
