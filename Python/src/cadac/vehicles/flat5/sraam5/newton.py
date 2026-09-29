"""SRAAM5 Newton module — Fortran ``MODULE.FOR`` subroutine ``D1`` / ``D1I``."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from cadac.constants import AGRAV, DEG
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr
from cadac.vehicles.flat5.sraam5.stubs import StubModule

_ZEROS3 = (0.0, 0.0, 0.0)
_ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _matcar(dvbe: float, psivl: float, thtvl: float) -> np.ndarray:
    """Fortran ``MATCAR`` — cartesian from polar speed / heading / flight-path."""
    cel = math.cos(thtvl)
    return np.array(
        [
            dvbe * cel * math.cos(psivl),
            dvbe * cel * math.sin(psivl),
            -dvbe * math.sin(thtvl),
        ],
        dtype=float,
    )


class Sraam5Newton(StubModule):
    """Zipfel SRAAM5 flat-Earth polar Newton (Fortran ``D1``)."""

    name = "newton"
    _fields = (
        Field("mterm", 0, "int", "data", "newton"),
        Field("dvbe", 0.0, "real", "state", "newton"),
        Field("dvbed", 0.0, "real", "state", "newton"),
        Field("psivl", 0.0, "real", "state", "newton"),
        Field("psivld", 0.0, "real", "state", "newton"),
        Field("thtvl", 0.0, "real", "state", "newton"),
        Field("thtvld", 0.0, "real", "state", "newton"),
        Field("SBELS", _ZEROS3, "vec", "state", "newton"),
        Field("SBELSD", _ZEROS3, "vec", "state", "newton"),
        Field("SBEL", _ZEROS3, "vec", "out", "newton"),
        Field("VBEL", _ZEROS3, "vec", "out", "newton"),
        Field("WVEV", _ZEROS3, "vec", "out", "newton"),
        Field("TVL", _ZEROS33, "mat", "out", "newton"),
        Field("hbe", 0.0, "real", "out", "newton"),
        Field("psivlx", 0.0, "real", "diag", "newton"),
        Field("thtvlx", 0.0, "real", "diag", "newton"),
        Field("dbe", 0.0, "real", "diag", "newton"),
        Field("gndtck", 0.0, "real", "diag", "newton"),
        Field("hg", 0.0, "real", "diag", "newton"),
        Field("icoor", 0, "int", "exec", "newton"),
    )

    def initialize(self, vehicle: Any, ctx: Any) -> None:
        store = vehicle.store
        psivl = float(store.get("psivl"))
        thtvl = float(store.get("thtvl"))
        store.set("TVL", mat2tr(psivl, thtvl))

    def execute(self, vehicle: Any, ctx: Any) -> None:
        store = vehicle.store
        fspv = np.asarray(store.get("FSPV"), dtype=float).reshape(3)
        agrav = AGRAV
        crad = DEG
        icoor = int(store.get("icoor"))
        int_step = float(ctx.int_step)

        dvbe = float(store.get("dvbe"))
        psivl = float(store.get("psivl"))
        thtvl = float(store.get("thtvl"))
        sbels = np.asarray(store.get("SBELS"), dtype=float).reshape(3).copy()
        sbel = np.asarray(store.get("SBEL"), dtype=float).reshape(3).copy()
        dvbed = float(store.get("dvbed"))
        psivld = float(store.get("psivld"))
        thtvld = float(store.get("thtvld"))
        sbelsd = np.asarray(store.get("SBELSD"), dtype=float).reshape(3).copy()
        gndtck = float(store.get("gndtck"))

        # Fortran: CALL MATEQL(SBELM,SBEL,3,1)
        sbelm = sbel.copy()

        # Equations of motion (derivatives at current polar state).
        dvbed_new = fspv[0] - math.sin(thtvl) * agrav
        psivld_new = fspv[1] / (dvbe * math.cos(thtvl))
        thtvld_new = -(fspv[2] + math.cos(thtvl) * agrav) / dvbe

        # Kinematics slice: MATCAR → VBEL; SBELSD = VBEL; SBEL ← SBELS.
        vbel = _matcar(dvbe, psivl, thtvl)
        thtvlx = thtvl * crad
        psivlx = psivl * crad
        sbelsd_new = vbel.copy()
        sbel = sbels.copy()
        dbe = float(np.linalg.norm(sbel))
        hbe = -sbel[2]

        tvl = mat2tr(psivl, thtvl)
        wvev = np.array(
            [
                -math.sin(thtvl) * psivld_new,
                thtvld_new,
                math.cos(thtvl) * psivld_new,
            ],
            dtype=float,
        )

        if icoor >= 0:
            dum3 = sbel - sbelm
            dum3[2] = 0.0
            gndtck = float(np.linalg.norm(dum3)) + gndtck

        # Fold CADAC executive trapezoidal integrate of D1 states.
        dvbe = integrate(dvbed_new, dvbed, dvbe, int_step)
        psivl = integrate(psivld_new, psivld, psivl, int_step)
        thtvl = integrate(thtvld_new, thtvld, thtvl, int_step)
        sbels = integrate(sbelsd_new, sbelsd, sbels, int_step)
        # Publish advanced position on SBEL (one full executive step).
        sbel = sbels.copy()
        hbe = -sbel[2]
        dbe = float(np.linalg.norm(sbel))

        store.set("dvbed", dvbed_new)
        store.set("dvbe", dvbe)
        store.set("psivld", psivld_new)
        store.set("psivl", psivl)
        store.set("thtvld", thtvld_new)
        store.set("thtvl", thtvl)
        store.set("SBELSD", sbelsd_new)
        store.set("SBELS", sbels)
        store.set("SBEL", sbel)
        store.set("VBEL", vbel)
        store.set("WVEV", wvev)
        store.set("TVL", tvl)
        store.set("hbe", hbe)
        store.set("psivlx", psivlx)
        store.set("thtvlx", thtvlx)
        store.set("dbe", dbe)
        store.set("gndtck", gndtck)
