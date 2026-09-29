"""ROCKET3 Newton module — Fortran ``MODULE.FOR`` subroutines ``D1`` / ``D1I``."""

from __future__ import annotations

from typing import Any

import numpy as np

from cadac.constants import DEG, RAD, REARTH, WEII3
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.earth import cadsph, cadtei, cadtge
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.round3.rocket3.stubs import StubModule

_ZEROS3 = (0.0, 0.0, 0.0)
_ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _matcar(dvbe: float, psivg: float, thtvg: float) -> np.ndarray:
    """Fortran ``MATCAR`` — cartesian from polar speed / heading / flight-path."""
    cel = np.cos(thtvg)
    return np.array(
        [
            dvbe * cel * np.cos(psivg),
            dvbe * cel * np.sin(psivg),
            -dvbe * np.sin(thtvg),
        ],
        dtype=float,
    )


class Rocket3Newton(StubModule):
    """Zipfel ROCKET3 round-Earth Cartesian inertial Newton (Fortran ``D1``)."""

    name = "newton"
    _fields = (
        Field("blon", 0.0, "real", "data", "newton"),
        Field("blat", 0.0, "real", "data", "newton"),
        Field("balt", 0.0, "real", "data", "newton"),
        Field("balt0", 0.0, "real", "save", "newton"),
        Field("dvbe", 0.0, "real", "data", "newton"),
        Field("psivgx", 0.0, "real", "data", "newton"),
        Field("thtvgx", 0.0, "real", "data", "newton"),
        Field("dvbi", 0.0, "real", "diag", "newton"),
        Field("psivigx", 0.0, "real", "diag", "newton"),
        Field("thtvigx", 0.0, "real", "diag", "newton"),
        Field("SBII", _ZEROS3, "vec", "state", "newton"),
        Field("VBII", _ZEROS3, "vec", "state", "newton"),
        Field("abii", _ZEROS3, "vec", "state", "newton"),
        Field("VBEG", _ZEROS3, "vec", "out", "newton"),
        Field("VBIG", _ZEROS3, "vec", "diag", "newton"),
        Field("TGV", _ZEROS33, "mat", "init", "newton"),
        Field("TIG", _ZEROS33, "mat", "init", "newton"),
        Field("WEII", _ZEROS33, "mat", "init", "newton"),
        Field("TGE", _ZEROS33, "mat", "out", "newton"),
    )

    def initialize(self, vehicle: Any, ctx: Any) -> None:
        """Fortran ``D1I`` — geographic IC → inertial SBII/VBII, TGV, TIG."""
        store = vehicle.store
        blon = float(store.get("blon"))
        blat = float(store.get("blat"))
        balt = float(store.get("balt"))
        dvbe = float(store.get("dvbe"))
        psivgx = float(store.get("psivgx"))
        thtvgx = float(store.get("thtvgx"))
        sim_time = float(getattr(ctx, "sim_time", 0.0))

        # SBIE from spherical geographic; at t=0 TEI=I so SBII=SBIE (D1I MATUNI TIE).
        sbie = np.array(
            [
                (balt + REARTH) * np.cos(blat) * np.cos(blon),
                (balt + REARTH) * np.cos(blat) * np.sin(blon),
                (balt + REARTH) * np.sin(blat),
            ],
            dtype=float,
        )
        tei = cadtei(sim_time)
        sbii = tei.T @ sbie

        psivg = psivgx * RAD
        thtvg = thtvgx * RAD
        vbeg = _matcar(dvbe, psivg, thtvg)

        weii = np.zeros((3, 3), dtype=float)
        weii[0, 1] = -WEII3
        weii[1, 0] = WEII3

        tge = cadtge(blon, blat)
        teg = tge.T
        tig = tei.T @ teg
        vbii = tig @ vbeg + weii @ sbii
        tgv = mat2tr(psivg, thtvg).T

        store.set("SBII", sbii)
        store.set("VBII", vbii)
        store.set("abii", np.zeros(3, dtype=float))
        store.set("VBEG", vbeg)
        store.set("TGV", tgv)
        store.set("TIG", tig)
        store.set("WEII", weii)
        store.set("TGE", tge)
        store.set("balt0", balt)

    def execute(self, vehicle: Any, ctx: Any) -> None:
        """Fortran ``D1`` RHS + folded trapezoidal integrate of SBII/VBII."""
        store = vehicle.store
        fspv = np.asarray(store.get("FSPV"), dtype=float).reshape(3)
        grav = float(store.get("grav"))
        int_step = float(ctx.int_step)
        sim_time = float(getattr(ctx, "sim_time", 0.0))

        sbii = np.asarray(store.get("SBII"), dtype=float).reshape(3).copy()
        vbii = np.asarray(store.get("VBII"), dtype=float).reshape(3).copy()
        abii = np.asarray(store.get("abii"), dtype=float).reshape(3).copy()
        tgv = np.asarray(store.get("TGV"), dtype=float).copy()
        tig = np.asarray(store.get("TIG"), dtype=float).copy()
        weii = np.asarray(store.get("WEII"), dtype=float).copy()

        # RHS: FSPG=TGV*FSPV; ACCG=FSPG+AGRAVG; AI=TIG*ACCG → VBIID; SBIID=VBII.
        grav_vec = np.array([0.0, 0.0, grav], dtype=float)
        abii_new = tig @ ((tgv @ fspv) + grav_vec)
        vbii_new = integrate(abii_new, abii, vbii, int_step)
        sbii = integrate(vbii_new, vbii, sbii, int_step)
        vbii = vbii_new
        abii = abii_new

        # Geographic update (Fortran CADTEI3 / CADSPH3 / CADTGE3 / MATPOL).
        tei = cadtei(sim_time)
        sbie = tei @ sbii
        blon, blat, balt = cadsph(sbie)
        tge = cadtge(blon, blat)
        tgi = tge @ tei
        vbeg = tgi @ (vbii - weii @ sbii)
        polar = polar_from_cart(vbeg)
        dvbe = float(polar[0])
        psivg = float(polar[1])
        thtvg = float(polar[2])
        psivgx = psivg * DEG
        thtvgx = thtvg * DEG

        tig = tgi.T
        tgv = mat2tr(psivg, thtvg).T

        vbig = tgi @ vbii
        inert = polar_from_cart(vbig)
        dvbi = float(inert[0])
        psivigx = float(inert[1]) * DEG
        thtvigx = float(inert[2]) * DEG

        store.set("SBII", sbii)
        store.set("VBII", vbii)
        store.set("abii", abii)
        store.set("VBEG", vbeg)
        store.set("VBIG", vbig)
        store.set("TGV", tgv)
        store.set("TIG", tig)
        store.set("TGE", tge)
        store.set("blon", blon)
        store.set("blat", blat)
        store.set("balt", balt)
        store.set("dvbe", dvbe)
        store.set("psivgx", psivgx)
        store.set("thtvgx", thtvgx)
        store.set("dvbi", dvbi)
        store.set("psivigx", psivigx)
        store.set("thtvigx", thtvigx)
