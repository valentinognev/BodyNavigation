"""SRAAM5 target module — Fortran G1I (init) / G1 (exec stub until later)."""

from __future__ import annotations

import math

import numpy as np

from cadac.constants import DEG
from cadac.kernel.state import Field
from cadac.vehicles.flat5.sraam5.stubs import StubModule

_ZEROS3 = np.zeros(3)


def _matcar(dbe: float, azbel: float, elbel: float) -> np.ndarray:
    """Fortran UTL3 MATCAR: cartesian from polar (speed, heading, flight-path)."""
    celb = math.cos(elbel)
    selb = math.sin(elbel)
    cazb = math.cos(azbel)
    sazb = math.sin(azbel)
    return np.array([dbe * celb * cazb, dbe * celb * sazb, -dbe * selb])


class Sraam5Target(StubModule):
    name = "target"
    _fields = (
        Field("mtarg", 0, "int", "data", "target"),
        Field("an1c", 0.0, "real", "data", "target"),
        Field("dvt1e", 0.0, "real", "data", "target"),
        Field("ht1e", 0.0, "real", "data", "target"),
        Field("tauhx", 0.0, "real", "data", "target"),
        Field("dvt2e", 0.0, "real", "data", "target"),
        Field("an2c", 0.0, "real", "data", "target"),
        Field("sighx", 0.0, "real", "data", "target"),
        Field("wloadt2", 0.0, "real", "data", "target"),
        Field("clat2", 0.0, "real", "data", "target"),
        Field("ht2e", 0.0, "real", "data", "target"),
        Field("rhl", 0.0, "real", "data", "target"),
        Field("mstop", 0, "int", "data", "target"),
        Field("phit1lcx", 0.0, "real", "data", "target"),
        Field("phit2lcx", 0.0, "real", "data", "target"),
        Field("alpt2x", 0.0, "real", "data", "target"),
        Field("alamhx", 0.0, "real", "diag", "target"),
        Field("psit1lx", 0.0, "real", "init/diag", "target"),
        Field("thtt1lx", 0.0, "real", "init/diag", "target"),
        Field("psit2lx", 0.0, "real", "init/diag", "target"),
        Field("thtt2lx", 0.0, "real", "init/diag", "target"),
        Field("anuhx", 0.0, "real", "diag", "target"),
        Field("ST1EL", _ZEROS3.copy(), "vec", "state", "target"),
        Field("ST2EL", _ZEROS3.copy(), "vec", "state", "target"),
        Field("VT1EL", _ZEROS3.copy(), "vec", "state", "target"),
        Field("VT2EL", _ZEROS3.copy(), "vec", "state", "target"),
        Field("SBEL", _ZEROS3.copy(), "vec", "state", "target"),
        Field("SBELS", _ZEROS3.copy(), "vec", "state", "target"),
        Field("VBEL", _ZEROS3.copy(), "vec", "state", "target"),
        Field("alp", 0.0, "real", "out", "target"),
        Field("bet", 0.0, "real", "out", "target"),
        Field("dvbe", 0.0, "real", "out", "target"),
        Field("psivl", 0.0, "real", "out", "target"),
        Field("thtvl", 0.0, "real", "out", "target"),
        Field("hbe", 0.0, "real", "out", "target"),
    )

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        mtarg = int(store.get("mtarg"))
        an1c = float(store.get("an1c"))
        dvt1e = float(store.get("dvt1e"))
        ht1e = float(store.get("ht1e"))
        tauhx = float(store.get("tauhx"))
        dvt2e = float(store.get("dvt2e"))
        an2c = float(store.get("an2c"))
        sighx = float(store.get("sighx"))
        wloadt2 = float(store.get("wloadt2"))
        clat2 = float(store.get("clat2"))
        ht2e = float(store.get("ht2e"))
        rhl = float(store.get("rhl"))
        alpt2x = float(store.get("alpt2x"))
        psit1lx = float(store.get("psit1lx"))
        thtt1lx = float(store.get("thtt1lx"))
        psit2lx = float(store.get("psit2lx"))
        thtt2lx = float(store.get("thtt2lx"))
        alamhx = float(store.get("alamhx"))
        phit1lcx = float(store.get("phit1lcx"))
        phit2lcx = float(store.get("phit2lcx"))
        anuhx = float(store.get("anuhx"))

        st1el = np.asarray(store.get("ST1EL"), dtype=float).copy()
        st2el = np.asarray(store.get("ST2EL"), dtype=float).copy()

        mtargc = int(mtarg / 10)
        mtargm = mtarg - mtargc * 10

        if mtarg != 0:
            # Shooter position (LAR-1: MTARGC=2, MTARGM=1)
            if mtargc == 1:
                st2el = np.array([0.0, 0.0, -ht2e])
            elif mtargc == 2:
                if mtargm == 1:
                    st2el = np.array(
                        [
                            rhl * math.cos(tauhx / DEG),
                            rhl * math.sin(tauhx / DEG),
                            -ht2e,
                        ]
                    )
                else:
                    raise ValueError(f"unsupported mtarg {mtarg}")
            else:
                raise ValueError(f"unsupported mtarg {mtarg}")

            # Shooter alpha (skip for circle engmts MTARGM>=2 under MTARGC=2)
            if not (mtargc == 2 and mtargm >= 2):
                if alpt2x == 0.0:
                    rhot2 = 1.225 * (1.0 + st2el[2] / 41900.0) ** 4
                    alpt2x = 2.0 * an2c / (rhot2 * dvt2e**2) * wloadt2 / clat2

            # Target-centered LAR-1 / UK circles (MTARGC=2)
            if mtargc == 2:
                thtt2lx = 0.0
                if mtargm == 1:
                    phit2lx = 0.0
                    psit2lx = -180.0 + tauhx - sighx
                    alamhx = sighx - alpt2x
                else:
                    raise ValueError(f"unsupported mtarg {mtarg}")
                phit2lcx = phit2lx

            # Target position
            if mtargc == 1:
                st1el = np.array(
                    [
                        rhl * math.cos(alamhx / DEG),
                        rhl * math.sin(alamhx / DEG),
                        -ht1e,
                    ]
                )
            elif mtargc == 2:
                st1el = np.array([0.0, 0.0, -ht1e])

            # Target angles (MTARGC=2 LAR-1)
            if mtargc == 2:
                psit1lx = 0.0
                thtt1lx = 0.0
                if mtargm == 1:
                    phit1lx = DEG * math.atan(an1c)
                    if an1c <= 1.0:
                        phit1lx = 0.0
                else:
                    raise ValueError(f"unsupported mtarg {mtarg}")
                phit1lcx = phit1lx

            anuhx = psit1lx - psit2lx

        vt2el = _matcar(dvt2e, psit2lx / DEG, thtt2lx / DEG)
        vt1el = _matcar(dvt1e, psit1lx / DEG, thtt1lx / DEG)

        store.set("ST2EL", st2el)
        store.set("ST1EL", st1el)
        store.set("VT2EL", vt2el)
        store.set("VT1EL", vt1el)
        store.set("alpt2x", alpt2x)
        store.set("alamhx", alamhx)
        store.set("psit2lx", psit2lx)
        store.set("thtt2lx", thtt2lx)
        store.set("psit1lx", psit1lx)
        store.set("thtt1lx", thtt1lx)
        store.set("phit1lcx", phit1lcx)
        store.set("phit2lcx", phit2lcx)
        store.set("anuhx", anuhx)
        store.set("dvt1e", dvt1e)
        store.set("dvt2e", dvt2e)

        if mtarg != 0:
            # Missile incidence + state from shooter (G1I)
            alp = math.atan(math.cos(-phit2lcx / DEG) * math.tan(alpt2x / DEG))
            bet = math.asin(math.sin(-phit2lcx / DEG) * math.sin(alpt2x / DEG))
            store.set("alp", alp)
            store.set("bet", bet)
            store.set("dvbe", dvt2e)
            store.set("psivl", psit2lx / DEG)
            store.set("thtvl", thtt2lx / DEG)
            store.set("SBELS", st2el.copy())
            store.set("SBEL", st2el.copy())
            store.set("VBEL", vt2el.copy())
            store.set("hbe", ht2e)
