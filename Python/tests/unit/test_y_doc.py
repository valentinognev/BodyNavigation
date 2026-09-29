"""y_doc → doc.asc (C++ Vehicle::document + document_input)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cadac.cli import run_scenario
from cadac.kernel.events import EventEngine, EventSpec
from cadac.kernel.state import Field, StateStore


class _Plane:
    type = "PLANE6"
    name = "p1"

    def __init__(self):
        self.store = StateStore()
        self.store.define(
            Field("time", 0.0, "real", "exec", "environment", ("plot",))
        )
        self.store.define(
            Field("alt", 1000.0, "real", "out", "newton", ("scrn", "plot"))
        )
        self.store.define(Field("mprop", 0, "int", "data", "propulsion"))
        self.modules = []
        self.events = EventEngine(
            [EventSpec(when={"time": {">": -1.0}}, set={})]
        )
        self.event_time = 0.0
        self.health = 1
        self.event_epoch = False

    def define(self):
        return None


def test_y_doc_writes_doc_asc(tmp_path: Path):
    """options['doc'] creates CADAC-shaped doc.asc; plot.csv equations unchanged."""
    path = tmp_path / "case.jsonc"
    path.write_text("{}", encoding="utf-8", newline="\n")
    vehicle = _Plane()
    cfg = SimpleNamespace(
        title="doc case",
        options={
            "scrn": False,
            "events": False,
            "plot": False,
            "doc": True,
            "csv": False,
            "tabout": False,
            "merge": False,
            "comscrn": False,
            "traj": False,
            "stat": False,
        },
        modules=[],
        timing={"int_step": 0.1},
        end_time=0.0,
        vehicles=[
            SimpleNamespace(
                type="PLANE6",
                name="p1",
                family="falcon6",
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

    doc = tmp_path / "doc.asc"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert any("*" * 20 in line for line in lines[:5])
    assert any("PLANE6" in line for line in lines[:5])
    assert any("doc case" in line for line in lines[:10])
    assert any("NAME" in line and "MODULE" in line for line in lines)
    assert any("time" in line for line in lines)
    assert any("alt" in line for line in lines)
    assert any("mprop" in line and "int" in line for line in lines)
    assert any("newton" in line for line in lines)
    assert not (tmp_path / "plot.csv").exists()
