"""SRAAM5 propulsion A2 — boost motor thrust/mass/CG vs time (Fortran MODULE.FOR)."""

from cadac.kernel.state import Field
from cadac.tables.lookup import Datadeck, Table
from cadac.vehicles.flat5.sraam5.stubs import StubModule

# Fortran A2 DATA (English: THRUST lbf, WGT lb, CGLOC inch).
_PROTIM = (
    0.0,
    0.05,
    0.269,
    0.538,
    0.807,
    1.076,
    1.345,
    1.614,
    1.883,
    2.152,
    2.421,
    2.690,
    6.000,
)
_THRUST = (
    0.0,
    7141.0,
    7747.0,
    8083.0,
    8109.0,
    7970.0,
    7796.0,
    7630.0,
    7463.0,
    7272.0,
    7031.0,
    0.0,
    0.0,
)
_WGT = (
    202.73,
    200.70,
    196.10,
    187.80,
    178.85,
    171.10,
    162.77,
    154.71,
    146.96,
    139.52,
    131.55,
    124.11,
    124.11,
)
_CGLOC = (
    60.46,
    60.39,
    59.32,
    58.18,
    57.04,
    55.87,
    54.69,
    53.42,
    52.12,
    50.77,
    49.47,
    48.05,
    48.05,
)

_BURN_END = 2.69


def _motor_deck() -> Datadeck:
    return Datadeck.from_tables(
        [
            Table("thrust_vs_time", 1, list(_PROTIM), None, None, list(_THRUST)),
            Table("wgt_vs_time", 1, list(_PROTIM), None, None, list(_WGT)),
            Table("cg_vs_time", 1, list(_PROTIM), None, None, list(_CGLOC)),
        ]
    )


class Sraam5Propulsion(StubModule):
    name = "propulsion"
    _fields = (
        Field("mprop", 0, "int", "out", "propulsion"),
        Field("fthalt", 0.0, "real", "out", "propulsion"),
        Field("amass", 0.0, "real", "data", "propulsion"),
        Field("xcgin", 0.0, "real", "out", "propulsion"),
    )

    def __init__(self):
        self._deck = _motor_deck()

    def execute(self, vehicle, ctx):
        store = vehicle.store
        time = store.get("time")
        optmet = store.get("optmet")
        press = store.get("press")

        if time < _BURN_END:
            mprop = 1
            fthsl = self._deck.look_up("thrust_vs_time", time)
            weight = self._deck.look_up("wgt_vs_time", time)
            xcgin = self._deck.look_up("cg_vs_time", time)
            aexit = 0.1351 * (1.0 - 0.9071 * optmet)
            amass = weight * (1.0 + 13.59 * optmet) / 32.174
            altcor = ((2116.0 + 99208.0 * optmet) - press) * aexit
            fthalt = fthsl * (1.0 + 3.45 * optmet) + altcor
            store.set("amass", amass)
            store.set("xcgin", xcgin)
        else:
            mprop = 0
            fthalt = 0.0

        store.set("mprop", mprop)
        store.set("fthalt", fthalt)
