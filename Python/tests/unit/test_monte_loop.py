"""MONTE parse + outer Monte Carlo loop (C++ execution.cpp semantics)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cadac.cli import run_scenario
from cadac.io.scenario import load_scenario
from cadac.io.translate import parse_scenario_asc


def test_parse_monte_sets_nmonte_and_iseed(tmp_path: Path):
    asc = tmp_path / "monte.asc"
    asc.write_text(
        "TITLE monte case\n"
        "MONTE 30 1234\n"
        "OPTIONS y_plot\n"
        "MODULES\n"
        "environment def,exec\n"
        "END\n"
        "TIMING\n"
        "int_step 0.01\n"
        "END\n"
        "VEHICLES 1\n"
        "MISSILE6 m1\n"
        "END\n"
        "ENDTIME 1\n",
        encoding="utf-8",
        newline="\n",
    )
    payload = parse_scenario_asc(asc)
    assert payload["nmonte"] == 30
    assert payload["iseed"] == 1234


def test_load_scenario_reads_nmonte(tmp_path: Path):
    p = tmp_path / "monte.jsonc"
    p.write_text(
        '{ "title": "t", "options": {}, "modules": [], "timing": {}, '
        '"end_time": 1, "nmonte": 30, "iseed": 1234, '
        '"vehicles": [ { "type": "CRUISE3", "name": "v", '
        '"params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    cfg = load_scenario(p)
    assert cfg.nmonte == 30
    assert cfg.iseed == 1234


def test_run_scenario_monte_2_invokes_body_twice(tmp_path: Path):
    """C++ do/while(nmc < nmonte): nmonte=2 → two executes; srand only when nmc==0."""
    cfg = SimpleNamespace(
        title="t",
        options={"plot": False, "csv": False},
        modules=[],
        timing={"int_step": 0.01},
        end_time=0.0,
        vehicles=[],
        iseed=99,
        nmonte=2,
    )
    seeds = []
    loops = []
    with (
        patch("cadac.cli.seed", side_effect=lambda iseed=0: seeds.append(iseed)),
        patch("cadac.cli.load_scenario", return_value=cfg),
        patch(
            "cadac.cli.run_loop",
            side_effect=lambda *a, **k: loops.append(1),
        ),
    ):
        run_scenario(tmp_path / "x.jsonc")
    assert len(loops) == 2
    # C++: if(!nmc) srand(iseed); no iseed bump between MC runs
    assert seeds == [99]


def test_run_scenario_nmonte_0_invokes_body_once(tmp_path: Path):
    """C++ nmonte==0 still executes the body once (deterministic means)."""
    cfg = SimpleNamespace(
        title="t",
        options={"plot": False, "csv": False},
        modules=[],
        timing={"int_step": 0.01},
        end_time=0.0,
        vehicles=[],
        iseed=0,
        nmonte=0,
    )
    loops = []
    with (
        patch("cadac.cli.seed"),
        patch("cadac.cli.load_scenario", return_value=cfg),
        patch(
            "cadac.cli.run_loop",
            side_effect=lambda *a, **k: loops.append(1),
        ),
    ):
        run_scenario(tmp_path / "x.jsonc")
    assert len(loops) == 1
