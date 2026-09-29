"""SRAAM5 Force Module A3 / A3TRA (Fortran MODULE.FOR)."""

from math import cos, sin

import numpy as np

from cadac.kernel.state import Field
from cadac.vehicles.flat5.sraam5.stubs import StubModule

_ZEROS3 = (0.0, 0.0, 0.0)
_ZEROS33 = tuple((0.0, 0.0, 0.0) for _ in range(3))


def a3tra(mturn: int, alpha: float, beta: float, phibv: float) -> np.ndarray:
    """Transformation matrix of body wrt flight-path axes (Fortran A3TRA)."""
    tbv = np.zeros((3, 3), dtype=float)
    if mturn == 0:
        # Skid-to-turn / yaw-to-turn: ALPHA, BETA
        calp = cos(alpha)
        salp = sin(alpha)
        cbet = cos(beta)
        sbet = sin(beta)
        tbv[0, 0] = calp * cbet
        tbv[0, 1] = -calp * sbet
        tbv[0, 2] = -salp
        tbv[1, 0] = sbet
        tbv[1, 1] = cbet
        tbv[1, 2] = 0.0
        tbv[2, 0] = salp * cbet
        tbv[2, 1] = -salp * sbet
        tbv[2, 2] = calp
    else:
        # Bank-to-turn: ALPHA, PHIBV
        calp = cos(alpha)
        salp = sin(alpha)
        cphi = cos(phibv)
        sphi = sin(phibv)
        tbv[0, 0] = calp
        tbv[0, 1] = salp * sphi
        tbv[0, 2] = -salp * cphi
        tbv[1, 0] = 0.0
        tbv[1, 1] = cphi
        tbv[1, 2] = sphi
        tbv[2, 0] = salp
        tbv[2, 1] = -calp * sphi
        tbv[2, 2] = calp * cphi
    return tbv


class Sraam5Forces(StubModule):
    name = "forces"
    _fields = (
        Field("fraca", 0.0, "real", "data", "forces"),
        Field("fracn", 0.0, "real", "data", "forces"),
        Field("fracy", 0.0, "real", "data", "forces"),
        Field("FSPB", _ZEROS3, "vec", "out", "forces"),
        Field("TBV", _ZEROS33, "mat", "out", "forces"),
        Field("FSPV", _ZEROS3, "vec", "out", "forces"),
        Field("ABEL", _ZEROS3, "vec", "out", "forces"),
        Field("FAB", _ZEROS3, "vec", "diag", "forces"),
        Field("al", 0.0, "real", "diag", "forces"),
        Field("an", 0.0, "real", "diag", "forces"),
    )

    def execute(self, vehicle, ctx):
        store = vehicle.store
        fraca = store.get("fraca")
        fracn = store.get("fracn")
        fracy = store.get("fracy")
        ca = (1.0 + fraca) * store.get("ca")
        cn = (1.0 + fracn) * store.get("cn")
        cy = (1.0 + fracy) * store.get("cy")
        pdynmc = store.get("pdynmc")
        area = store.get("area")
        fthalt = store.get("fthalt")
        amass = store.get("amass")
        agrav = store.get("agrav")
        tlb = np.asarray(store.get("TLB"), dtype=float)

        fab = np.array(
            [
                fthalt - ca * pdynmc * area,
                cy * pdynmc * area,
                -cn * pdynmc * area,
            ],
            dtype=float,
        )
        fspb = fab / amass
        al = fspb[1] / agrav
        an = -fspb[2] / agrav
        abel = tlb @ fspb

        mturn = int(store.get("mturn"))
        alpha = store.get("alpha")
        beta = store.get("beta")
        phibv = store.get("phibv")
        tbv = a3tra(mturn, alpha, beta, phibv)
        fspv = tbv.T @ fspb

        store.set("FAB", fab)
        store.set("FSPB", fspb)
        store.set("al", al)
        store.set("an", an)
        store.set("ABEL", abel)
        store.set("TBV", tbv)
        store.set("FSPV", fspv)
