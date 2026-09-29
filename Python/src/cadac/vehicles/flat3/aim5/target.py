"""AIM5 target module — Fortran MODULE.FOR G1I / G1."""

from __future__ import annotations

import math

import numpy as np

from cadac.constants import RAD
from cadac.kernel.state import Field

_ZEROS3 = (0.0, 0.0, 0.0)


def _matcar(magnitude: float, azimuth: float, elevation: float) -> np.ndarray:
    """Fortran UTL MATCAR: cartesian from polar (mag, heading, flight-path)."""
    cel = math.cos(elevation)
    return np.array(
        [
            magnitude * cel * math.cos(azimuth),
            magnitude * cel * math.sin(azimuth),
            -magnitude * math.sin(elevation),
        ],
        dtype=float,
    )


class Aim5Target:
    """Zipfel AIM5 flat-Earth target (Fortran ``G1I`` / ``G1``)."""

    name = "target"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("mtarg", 0, "int", "data", "target"),
            Field("dhtb", 0.0, "real", "data", "target"),
            Field("hte", 0.0, "real", "data", "target"),
            Field("aztlx", 0.0, "real", "data", "target"),
            Field("ST1EL0", _ZEROS3, "vec", "data", "target"),
            Field("VT1EL", _ZEROS3, "vec", "data", "target"),
            Field("ST1EL", _ZEROS3, "vec", "out", "target"),
            Field("STBL", _ZEROS3, "vec", "diag", "target"),
            Field("dbt1", 0.0, "real", "diag", "target"),
        ):
            if field.name not in store:
                store.define(field)
        if "SBEL" not in store:
            store.define(Field("SBEL", _ZEROS3, "vec", "state", "newton"))

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        mtarg = int(store.get("mtarg"))
        st1el0 = np.asarray(store.get("ST1EL0"), dtype=float).reshape(3).copy()

        if mtarg == 1:
            sbel = np.asarray(store.get("SBEL"), dtype=float).reshape(3)
            hte = float(store.get("hte"))
            dhtb = float(store.get("dhtb"))
            aztlx = float(store.get("aztlx"))
            # G1I: DUMH=SBEL(3)+HTE; THTTL0=ATAN2(DUMH,DHTB); DBT1; MATCAR(STBL,…)
            dumh = float(sbel[2]) + hte
            thttl0 = math.atan2(dumh, dhtb)
            dbt1 = math.sqrt(dhtb * dhtb + dumh * dumh)
            stbl = _matcar(dbt1, aztlx * RAD, thttl0)
            st1el0 = stbl + sbel
            store.set("STBL", stbl)
            store.set("dbt1", dbt1)
            store.set("ST1EL0", st1el0)
        elif mtarg != 0:
            raise ValueError(f"unsupported mtarg={mtarg}")

        store.set("ST1EL", st1el0.copy())

    def execute(self, vehicle, ctx):
        store = vehicle.store
        st1el0 = np.asarray(store.get("ST1EL0"), dtype=float).reshape(3)
        vt1el = np.asarray(store.get("VT1EL"), dtype=float).reshape(3)
        t = float(ctx.sim_time)
        store.set("ST1EL", st1el0 + t * vt1el)

    def terminate(self, vehicle, ctx):
        pass
