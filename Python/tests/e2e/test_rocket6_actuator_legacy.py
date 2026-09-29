"""ROCKET6 legacy fin actuator e2e vs optional old-C++ plot golden.

Golden at goldens/rocket6/actuator_legacy.plot.csv is harvested from the old
Simulations_C++/ROCKET6 tree (legacy-binary) with mact/dlimx set for rocket fins.
Absent golden → skip (same pattern as other deferred e2e goldens).
"""

from pathlib import Path

import pytest

from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "rocket6" / "input_actuator_legacy.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "rocket6" / "actuator_legacy.plot.csv"


def test_rocket6_actuator_legacy_load_scenario():
    cfg = load_scenario(CASE)
    assert cfg.family == "rocket6"
    assert cfg.vehicles[0].type == "HYPER6"
    assert cfg.vehicles[0].params.get("mact") == 22
    names = [m.name for m in cfg.modules]
    assert "actuator" in names
    assert names.index("rcs") < names.index("actuator") < names.index("tvc")


def test_rocket6_actuator_legacy_golden_or_skip():
    if not GOLDEN.is_file():
        pytest.skip(
            "legacy-binary golden actuator_legacy.plot.csv is absent "
            "(harvest old Simulations_C++/ROCKET6 plot)"
        )
