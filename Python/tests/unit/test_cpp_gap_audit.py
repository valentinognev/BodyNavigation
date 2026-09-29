"""Per-vehicle lock of C++ modes the inventory marks missing or stubbed.

GAP_MATRIX values are ``ported here (path:line)``, ``raises here (path:line)``,
or ``not ported (path:line)`` when the mode is absent and execute does not raise
``unknown <flag>``. Later tasks read this dict for scope.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from cadac.cli import _resolve_vehicle
from cadac.tables.lookup import Datadeck

# Global types use family None. program.lower() is only right for cruise5/sam6.
_GLOBAL_FAMILY = {
    "HYPER6": None,
    "ROCKET6": "rocket6",
    "HYPER5": None,
    "FALCON5": None,
    "FALCON6": None,
}


def _vehicle(program, vtype, **kw):
    family = _GLOBAL_FAMILY.get(program, program.lower())
    cls = _resolve_vehicle(family, vtype)
    return cls("probe", **kw)


def _new(program, vtype):
    deck = Datadeck.from_tables({})
    veh = _vehicle(program, vtype, aero_deck=deck, prop_deck=deck, events=[])
    veh.define()
    return veh


def _execute_module(veh, name, method="execute", ctx=None):
    if ctx is None:
        ctx = SimpleNamespace(int_step=0.01)
    for module in veh.modules:
        if module.name == name:
            getattr(module, method)(veh, ctx)
            return
    raise AssertionError(f"module {name!r} not on vehicle")


def _here(kind, path, line):
    return f"{kind} ({path}:{line})"


_H6_GUIDE = "Python/src/cadac/vehicles/round6/hyper6/guidance.py"
_H6_AERO = "Python/src/cadac/vehicles/round6/hyper6/aero.py"
_H6_CTRL = "Python/src/cadac/vehicles/round6/hyper6/control.py"
_R6_GUIDE = "Python/src/cadac/vehicles/round6/rocket6/guidance.py"
_ROUND6 = "Python/src/cadac/eom/round6.py"
_CLI = "Python/src/cadac/cli.py"
_F6_GUIDE = "Python/src/cadac/vehicles/flat6/falcon6/guidance.py"
_F6_CTRL = "Python/src/cadac/vehicles/flat6/falcon6/control.py"
_C5_GUIDE = "Python/src/cadac/vehicles/round3/cruise5/guidance.py"
_C5_CTRL = "Python/src/cadac/vehicles/round3/cruise5/control.py"
_H5_GUIDE = "Python/src/cadac/vehicles/round3/hyper5/guidance.py"
_H5_CTRL = "Python/src/cadac/vehicles/round3/hyper5/control.py"
_F5_GUIDE = "Python/src/cadac/vehicles/flat3/falcon5/guidance.py"
_F5_CTRL = "Python/src/cadac/vehicles/flat3/falcon5/control.py"
_SAM_INS = "Python/src/cadac/vehicles/flat6/sam6/ins.py"
_SAM_INT = "Python/src/cadac/vehicles/flat6/sam6/intercept.py"

GAP_MATRIX: dict[tuple[str, str, int | str], str] = {
    ("CRUISE5", "mcontrol", 1): _here("ported here", _C5_CTRL, 84),
    ("CRUISE5", "mcontrol", 10): _here("ported here", _C5_CTRL, 87),
    ("CRUISE5", "mcontrol", 11): _here("ported here", _C5_CTRL, 91),
    ("CRUISE5", "mguidance", 3): _here("ported here", _C5_GUIDE, 63),
    ("CRUISE5", "mguidance", 6): _here("ported here", _C5_GUIDE, 75),
    ("CRUISE5", "mguidance", 60): _here("ported here", _C5_GUIDE, 71),
    ("FALCON5", "mcontrol", 1): _here("ported here", _F5_CTRL, 80),
    ("FALCON5", "mcontrol", 10): _here("ported here", _F5_CTRL, 85),
    ("FALCON5", "mcontrol", 11): _here("ported here", _F5_CTRL, 90),
    ("FALCON5", "mguidance", 3): _here("ported here", _F5_GUIDE, 55),
    ("FALCON5", "mguidance", 6): _here("raises here", _F5_GUIDE, 68),
    ("FALCON5", "mguidance", 60): _here("raises here", _F5_GUIDE, 68),
    ("FALCON6", "mauty", 3): _here("ported here", _F6_CTRL, 122),
    ("FALCON6", "mauty", 4): _here("ported here", _F6_CTRL, 126),
    ("FALCON6", "mguid", 30): _here("ported here", _F6_GUIDE, 77),
    ("FALCON6", "mguid", 33): _here("ported here", _F6_GUIDE, 79),
    ("HYPER5", "mcontrol", 1): _here("ported here", _H5_CTRL, 84),
    ("HYPER5", "mcontrol", 10): _here("ported here", _H5_CTRL, 87),
    ("HYPER5", "mcontrol", 11): _here("ported here", _H5_CTRL, 91),
    ("HYPER5", "mguidance", 3): _here("ported here", _H5_GUIDE, 85),
    ("HYPER5", "mguidance", 6): _here("ported here", _H5_GUIDE, 97),
    ("HYPER5", "mguidance", 60): _here("ported here", _H5_GUIDE, 91),
    ("HYPER6", "maero", 1): _here("ported here", _H6_AERO, 177),
    ("HYPER6", "maero", 2): _here("ported here", _H6_AERO, 111),
    ("HYPER6", "matmo", 1): _here("ported here", _ROUND6, 344),
    ("HYPER6", "mauty", 3): _here("ported here", _H6_CTRL, 127),
    ("HYPER6", "mauty", 4): _here("ported here", _H6_CTRL, 140),
    ("HYPER6", "mguide", 3): _here("ported here", _H6_GUIDE, 206),
    ("HYPER6", "mguide", 4): _here("ported here", _H6_GUIDE, 228),
    ("HYPER6", "mguide", 5): _here("ported here", _H6_GUIDE, 230),
    ("HYPER6", "mguide", 6): _here("ported here", _H6_GUIDE, 233),
    ("HYPER6", "mguide", 7): _here("ported here", _H6_GUIDE, 237),
    ("HYPER6", "mguide", 8): _here("ported here", _H6_GUIDE, 241),
    ("HYPER6", "mguide", 30): _here("ported here", _H6_GUIDE, 196),
    ("HYPER6", "mguide", 33): _here("ported here", _H6_GUIDE, 216),
    ("HYPER6", "minit", 0): _here("ported here", _ROUND6, 776),
    ("HYPER6", "minit", 1): _here("ported here", _ROUND6, 725),
    ("HYPER6", "vehicle", "RADAR0"): _here("ported here", _CLI, 49),
    ("HYPER6", "vehicle", "SAT3"): _here("ported here", _CLI, 48),
    ("ROCKET6", "matmo", 1): _here("ported here", _ROUND6, 344),
    ("ROCKET6", "mguide", 5): _here("ported here", _R6_GUIDE, 135),
    ("SAM6", "mins", 2): _here("ported here", _SAM_INS, 236),
    ("SAM6", "mins", 3): _here("ported here", _SAM_INS, 245),
    ("SAM6", "mterm", -1): _here("ported here", _SAM_INT, 155),
}


def test_audit_hyper6_sat_radar_resolve_ground0_raises():
    assert _resolve_vehicle(None, "SAT3").type == "SAT3"
    assert _resolve_vehicle(None, "RADAR0").type == "RADAR0"
    with pytest.raises(ValueError, match="Ground0"):
        _resolve_vehicle(None, "GROUND0")


def test_audit_falcon5_mguidance_6_and_60_raise():
    for mguidance in (6, 60):
        veh = _new("FALCON5", "PLANE")
        veh.store.set("mguidance", mguidance)
        with pytest.raises(ValueError, match="unknown mguidance"):
            _execute_module(veh, "guidance")


def test_audit_hyper6_maero():
    veh = _new("HYPER6", "HYPER6")
    veh.store.set("maero", 2)
    _execute_module(veh, "aerodynamics")
    assert veh.store.get("cy") == 0.0

    veh = _new("HYPER6", "HYPER6")
    veh.store.set("maero", 1)
    with pytest.raises(KeyError, match="cd0_vs_alpha_mach"):
        _execute_module(veh, "aerodynamics")


def test_audit_hyper6_minit_0_executes():
    veh = _new("HYPER6", "HYPER6")
    veh.store.set("minit", 0)
    _execute_module(veh, "newton", method="initialize")


def test_audit_rocket6_mguide_5_executes_today():
    veh = _new("ROCKET6", "HYPER6")
    veh.store.set("mguide", 5)
    _execute_module(veh, "guidance")
    assert veh.store.get("init_flag") == 0


def test_audit_sam6_mins_executes():
    for mins in (2, 3):
        veh = _new("SAM6", "MISSILE6")
        veh.store.set("mins", mins)
        _execute_module(veh, "ins")


def test_audit_sam6_mterm_minus1_executes():
    veh = _new("SAM6", "MISSILE6")
    veh.store.set("mterm", -1)
    veh.store.set("mseek", 4)
    veh.store.set("alt", 1000.0)
    veh.store.set("hbe", 1000.0)
    veh.store.set("ip_sltrange", 1.0e9)
    veh.store.set("SBEL", (0.0, 0.0, 0.0))
    veh.store.set("SBMEL", (1.0, 0.0, 0.0))
    veh.store.set("STEL", (100.0, 0.0, 0.0))
    veh.store.set("VBEL", (0.0, 0.0, 0.0))
    veh.store.set("VTEL", (10.0, 0.0, 0.0))
    ctx = SimpleNamespace(
        int_step=0.01,
        vehicle_slot=0,
        combus=[SimpleNamespace(status=1)],
    )
    _execute_module(veh, "intercept", ctx=ctx)
    assert veh.store.get("miss") == 100.0
    assert veh.health == 0


def test_gap_matrix_cites_real_lines():
    root = Path(__file__).resolve().parents[3]
    assert len(GAP_MATRIX) == 44
    raises = {key for key, text in GAP_MATRIX.items() if text.startswith("raises here")}
    assert raises == {
        ("FALCON5", "mguidance", 6),
        ("FALCON5", "mguidance", 60),
    }
    ported = {key for key, text in GAP_MATRIX.items() if text.startswith("ported here")}
    assert ported == set(GAP_MATRIX) - raises
    assert not any(text.startswith("not ported") for text in GAP_MATRIX.values())
    for key, text in GAP_MATRIX.items():
        assert text.startswith(("ported here (", "raises here (", "not ported (")), key
        path_s, line_s = text[text.index("(") + 1 : -1].rsplit(":", 1)
        lines = (root / path_s).read_text(encoding="utf-8").splitlines()
        line_no = int(line_s)
        assert 1 <= line_no <= len(lines), key
        assert "Python/src/cadac/" in path_s
