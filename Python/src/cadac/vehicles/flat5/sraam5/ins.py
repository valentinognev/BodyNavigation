"""SRAAM5 INS — Fortran MODULE.FOR subroutines S4I / S4 / S4GYRO / S4ACCL / S4ALT."""

import math

import numpy as np

from cadac.constants import AGRAV, REARTH
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import skew
from cadac.stoch import gauss

# Fortran S4I DATA PP0 (9x9). Units: m, m/s, milli-rad. Row-major as written.
PP0 = np.array(
    [
        [
            20.701,
            0.12317,
            0.10541,
            6.3213e-02,
            2.2055e-03,
            1.7234e-03,
            1.0633e-03,
            3.4941e-02,
            -3.5179e-02,
        ],
        [
            0.12317,
            20.696,
            -0.27174,
            4.8366e-03,
            5.9463e-02,
            -1.3367e-03,
            -3.4903e-02,
            2.6112e-03,
            -4.2663e-02,
        ],
        [
            0.10541,
            -0.27174,
            114.12,
            5.6373e-04,
            -8.3147e-03,
            5.4059e-02,
            1.5496e-02,
            7.6463e-02,
            -3.5302e-03,
        ],
        [
            6.3213e-02,
            4.8366e-03,
            5.6373e-04,
            1.9106e-03,
            8.0945e-05,
            1.9810e-06,
            2.5755e-04,
            2.8346e-03,
            -5.6482e-04,
        ],
        [
            2.2055e-03,
            5.9463e-02,
            -8.3147e-03,
            8.0945e-05,
            1.7201e-03,
            -1.5760e-05,
            -2.8341e-03,
            2.6478e-04,
            -1.0781e-03,
        ],
        [
            1.7234e-03,
            -1.3367e-03,
            5.4059e-02,
            1.9810e-06,
            -1.5760e-05,
            3.0070e-03,
            4.1963e-04,
            -1.3297e-04,
            4.1190e-05,
        ],
        [
            1.0638e-03,
            -3.4903e-02,
            1.5496e-02,
            2.5755e-04,
            -2.8341e-03,
            4.1963e-04,
            5.4490e-02,
            -1.8695e-03,
            8.9868e-04,
        ],
        [
            3.4941e-02,
            2.6112e-03,
            7.6463e-02,
            2.8346e-03,
            2.6478e-04,
            -1.3297e-04,
            -1.8695e-03,
            5.2819e-02,
            1.0990e-02,
        ],
        [
            -3.5179e-02,
            -4.2663e-02,
            -3.5302e-03,
            -5.6482e-04,
            -1.0781e-03,
            4.1190e-05,
            8.9868e-04,
            1.0990e-02,
            0.1291,
        ],
    ],
    dtype=float,
)

# Fortran S4I: DO I=1,100 DISCARD=FNGAUS(0.,1.) then 9 unit draws.
_S4I_DISCARD = 100


def _cholesky(mat):
    """CADAC MATCHO — lower-triangular Cholesky factor."""
    a = np.asarray(mat, dtype=float)
    dim = a.shape[0]
    out = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            if j < i:
                total = 0.0
                if j > 0:
                    for k in range(j):
                        total += out[i, k] * out[j, k]
                if out[j, j] == 0.0:
                    out[i, j] = 0.0
                else:
                    out[i, j] = (a[i, j] - total) / out[j, j]
            elif j == i:
                total = 0.0
                if i > 0:
                    for k in range(i):
                        total += out[i, k] * out[i, k]
                out[i, j] = math.sqrt(a[i, i] - total)
            else:
                out[i, j] = 0.0
    return out


def _diamat(vec):
    return np.diag(np.asarray(vec, dtype=float))


class Sraam5Ins:
    name = "ins"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for field in (
            Field("mins", 0, "int", "data", "ins"),
            Field("frax", 0.0, "real", "data", "ins"),
            Field("hbem", 0.0, "real", "out", "ins"),
            Field("VBELC", zeros3, "vec", "out", "ins"),
            Field("SBELC", zeros3, "vec", "out", "ins"),
            Field("WBECB", zeros3, "vec", "out", "ins"),
            Field("EWALKG", zeros3, "vec", "data", "ins"),
            Field("EUNBG", zeros3, "vec", "data", "ins"),
            Field("EMISG", zeros3, "vec", "data", "ins"),
            Field("ESCALG", zeros3, "vec", "data", "ins"),
            Field("EBIASG", zeros3, "vec", "data", "ins"),
            Field("biasal", 0.0, "real", "data", "ins"),
            Field("randal", 0.0, "real", "data", "ins"),
            Field("ehbe", 0.0, "real", "out", "ins"),
            Field("TBLC", zeros33, "mat", "out", "ins"),
            Field("EWALKA", zeros3, "vec", "data", "ins"),
            Field("EMISA", zeros3, "vec", "data", "ins"),
            Field("ESCALA", zeros3, "vec", "data", "ins"),
            Field("EBIASA", zeros3, "vec", "data", "ins"),
            Field("EUG", zeros3, "vec", "diag", "ins"),
            Field("EMISCG", zeros3, "vec", "diag", "ins"),
            Field("EWG", zeros3, "vec", "diag", "ins"),
            Field("EWBEB", zeros3, "vec", "diag", "ins"),
            Field("EFSPB", zeros3, "vec", "diag", "ins"),
            Field("tanlat", 0.0, "real", "init", "ins"),
            Field("dvbec", 0.0, "real", "out", "ins"),
            Field("FSPCB", zeros3, "vec", "out", "ins"),
            Field("RECED", zeros3, "vec", "state", "ins"),
            Field("RECE", zeros3, "vec", "state", "ins"),
            Field("EVBED", zeros3, "vec", "state", "ins"),
            Field("EVBE", zeros3, "vec", "state", "ins"),
            Field("ESTTCD", zeros3, "vec", "state", "ins"),
            Field("ESTTC", zeros3, "vec", "state", "ins"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        """Fortran S4I."""
        store = vehicle.store
        mins = store.get("mins")
        self.ins_alt(vehicle)
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)
        if mins == 0:
            store.set("SBELC", sbel.copy())
            store.set("VBELC", vbel.copy())
            return
        if mins != 1:
            raise ValueError(f"unknown mins {mins}")
        frax = store.get("frax")
        app0 = _cholesky(PP0)
        for _ in range(_S4I_DISCARD):
            gauss(0.0, 1.0)
        draws = np.array([gauss(0.0, 1.0) for _ in range(9)], dtype=float)
        xx0 = app0 @ draws * (1.0 + frax)
        esttc = np.array([xx0[0], xx0[1], xx0[2]], dtype=float)
        evbe = np.array([xx0[3], xx0[4], xx0[5]], dtype=float)
        rece = np.array([xx0[6], xx0[7], xx0[8]], dtype=float) * 0.001
        store.set("ESTTC", esttc)
        store.set("EVBE", evbe)
        store.set("RECE", rece)
        store.set("VBELC", evbe + vbel)
        store.set("SBELC", esttc + sbel)

    def execute(self, vehicle, ctx):
        """Fortran S4."""
        store = vehicle.store
        mins = store.get("mins")
        if mins not in (0, 1):
            raise ValueError(f"unknown mins {mins}")
        int_step = ctx.int_step
        self.ins_alt(vehicle)

        reced = np.asarray(store.get("RECED"), dtype=float).copy()
        rece = np.asarray(store.get("RECE"), dtype=float).copy()
        evbed = np.asarray(store.get("EVBED"), dtype=float).copy()
        evbe = np.asarray(store.get("EVBE"), dtype=float).copy()
        esttcd = np.asarray(store.get("ESTTCD"), dtype=float).copy()
        esttc = np.asarray(store.get("ESTTC"), dtype=float).copy()
        ewbeb = np.zeros(3)
        efspb = np.zeros(3)

        if mins == 0:
            tblc = np.asarray(store.get("TBL"), dtype=float).copy()
            fspcb = np.asarray(store.get("FSPB"), dtype=float).copy()
            wbecb = np.asarray(store.get("WBEB"), dtype=float).copy()
            sbelc = np.asarray(store.get("SBEL"), dtype=float).copy()
            vbelc = np.asarray(store.get("VBEL"), dtype=float).copy()
            dvbec = float(store.get("dvbe"))
        else:
            fspcb, efspb = self.ins_accl(vehicle)
            ewbeb, wbecb = self.ins_gyro(vehicle, int_step)
            tlb = np.asarray(store.get("TLB"), dtype=float)
            ewbel = tlb @ ewbeb
            tanlat = store.get("tanlat")
            reced_new = np.array(
                [
                    ewbel[0] + evbe[1] / REARTH,
                    ewbel[1] - evbe[0] / REARTH,
                    ewbel[2] - evbe[1] * tanlat / REARTH,
                ],
                dtype=float,
            )
            rece = integrate(reced_new, reced, rece, int_step)
            reced = reced_new
            rere = skew(rece)
            tllc = rere + np.eye(3)
            tbl = np.asarray(store.get("TBL"), dtype=float)
            tblc = tbl @ tllc
            tlcb = tblc.T
            fsplc = tlcb @ fspcb
            ef = tlcb @ efspb - rere @ fsplc
            evbed_new = np.array(
                [
                    ef[0],
                    ef[1],
                    ef[2] + 2.0 * AGRAV * esttc[2] / REARTH,
                ],
                dtype=float,
            )
            evbe = integrate(evbed_new, evbed, evbe, int_step)
            evbed = evbed_new
            esttcd_new = evbe.copy()
            esttc = integrate(esttcd_new, esttcd, esttc, int_step)
            esttcd = esttcd_new
            sbel = np.asarray(store.get("SBEL"), dtype=float)
            vbel = np.asarray(store.get("VBEL"), dtype=float)
            sbelc = esttc + sbel
            vbelc = evbe + vbel
            dvbec = float(np.linalg.norm(vbelc))

        store.set("RECED", reced)
        store.set("RECE", rece)
        store.set("EVBED", evbed)
        store.set("EVBE", evbe)
        store.set("ESTTCD", esttcd)
        store.set("ESTTC", esttc)
        store.set("VBELC", vbelc)
        store.set("WBECB", wbecb)
        store.set("TBLC", tblc)
        store.set("dvbec", dvbec)
        store.set("FSPCB", fspcb)
        store.set("SBELC", sbelc)
        store.set("EWBEB", ewbeb)
        store.set("EFSPB", efspb)

    def ins_gyro(self, vehicle, int_step):
        """Fortran S4GYRO."""
        store = vehicle.store
        ewalkg = np.asarray(store.get("EWALKG"), dtype=float)
        eunbg = np.asarray(store.get("EUNBG"), dtype=float)
        emisg = np.asarray(store.get("EMISG"), dtype=float)
        escalg = np.asarray(store.get("ESCALG"), dtype=float)
        ebiasg = np.asarray(store.get("EBIASG"), dtype=float)
        fspb = np.asarray(store.get("FSPB"), dtype=float)
        wbeb = np.asarray(store.get("WBEB"), dtype=float)
        egb = _diamat(escalg) + skew(emisg)
        emiscg = egb @ wbeb
        emsbg = ebiasg + emiscg
        eug = np.array(
            [eunbg[0] * fspb[0], eunbg[1] * fspb[1], eunbg[2] * fspb[2]],
            dtype=float,
        )
        ewg = ewalkg * (1.0 / math.sqrt(int_step))
        ewbeb = emsbg + eug + ewg
        wbecb = wbeb + ewbeb
        store.set("EUG", eug)
        store.set("EMISCG", emiscg)
        store.set("EWG", ewg)
        return ewbeb, wbecb

    def ins_accl(self, vehicle):
        """Fortran S4ACCL — returns (FSPCB, EFSPB)."""
        store = vehicle.store
        emisa = np.asarray(store.get("EMISA"), dtype=float)
        escala = np.asarray(store.get("ESCALA"), dtype=float)
        ebiasa = np.asarray(store.get("EBIASA"), dtype=float)
        ewalka = np.asarray(store.get("EWALKA"), dtype=float)
        fspb = np.asarray(store.get("FSPB"), dtype=float)
        eab = _diamat(escala) + skew(emisa)
        efspb = ebiasa + eab @ fspb
        fspcb = ewalka + efspb + fspb
        return fspcb, efspb

    def ins_alt(self, vehicle):
        """Fortran S4ALT."""
        store = vehicle.store
        biasal = store.get("biasal")
        randal = store.get("randal")
        hbe = store.get("hbe")
        ehbe = biasal + randal
        hbem = hbe + ehbe
        store.set("hbem", hbem)
        store.set("ehbe", ehbe)

    def terminate(self, vehicle, ctx):
        pass
