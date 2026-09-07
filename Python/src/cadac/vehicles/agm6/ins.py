import math

import numpy as np

from cadac.constants import AGRAV, DEG, PI, REARTH
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field

# C++ Missile::init_ins PP0 (9x9, row-major as written).
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


def _cadac_sign(variable):
    if variable < 0.0:
        return -1
    return 1


def _skew(vec):
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _diamat(vec):
    return np.diag(np.asarray(vec, dtype=float))


def _cholesky(mat):
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


class Agm6Ins:
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
            Field("EWG", zeros3, "vec", "diag", "ins"),
            Field("EWBEB", zeros3, "vec", "diag", "ins"),
            Field("EFSPB", zeros3, "vec", "diag", "ins"),
            Field("tanlat", 0.0, "real", "init", "ins"),
            Field("thtblc", 0.0, "real", "out", "ins"),
            Field("thtblcx", 0.0, "real", "out", "ins"),
            Field("dvbec", 0.0, "real", "out", "ins"),
            Field("thtvlc", 0.0, "real", "out", "ins"),
            Field("thtvlcx", 0.0, "real", "out", "ins"),
            Field("psivlcx", 0.0, "real", "out", "ins"),
            Field("FSPCB", zeros3, "vec", "out", "ins"),
            Field("phiblcx", 0.0, "real", "out", "ins"),
            Field("RECED", zeros3, "vec", "state", "ins"),
            Field("RECE", zeros3, "vec", "state", "ins"),
            Field("EVBED", zeros3, "vec", "state", "ins"),
            Field("EVBE", zeros3, "vec", "state", "ins"),
            Field("ESTTCD", zeros3, "vec", "state", "ins"),
            Field("ESTTC", zeros3, "vec", "state", "ins"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        mins = store.get("mins")
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)
        if mins == 0:
            store.set("SBELC", sbel.copy())
            store.set("VBELC", vbel.copy())
            return
        if mins != 1:
            raise ValueError(f"unknown mins {mins}")
        frax = store.get("frax")
        app_init = _cholesky(PP0)
        zero_draw = np.zeros(9)
        xx_init = app_init @ zero_draw
        xx_init = xx_init * (1.0 + frax)
        esttc = np.array([xx_init[0], xx_init[1], xx_init[2]], dtype=float)
        evbe = np.array([xx_init[3], xx_init[4], xx_init[5]], dtype=float)
        rece = np.array([xx_init[6], xx_init[7], xx_init[8]], dtype=float)
        rece = rece * 0.001
        store.set("VBELC", evbe + vbel)
        store.set("RECE", rece)
        store.set("EVBE", evbe)
        store.set("ESTTC", esttc)
        store.set("SBELC", esttc + sbel)

    def execute(self, vehicle, ctx):
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
            phiblcx = float(store.get("phiblx"))
        else:
            efspb = self.ins_accl(vehicle)
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
            rere = _skew(rece)
            tllc = rere + np.eye(3)
            tbl = np.asarray(store.get("TBL"), dtype=float)
            tblc = tbl @ tllc
            tlcb = tblc.T
            ewalka = np.asarray(store.get("EWALKA"), dtype=float)
            fspb = np.asarray(store.get("FSPB"), dtype=float)
            fspcb = ewalka + efspb + fspb
            ef = tlcb @ efspb - rere @ (tlcb @ fspcb)
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

        vbelc1 = float(vbelc[0])
        vbelc2 = float(vbelc[1])
        vbelc3 = float(vbelc[2])
        if vbelc1 == 0.0 and vbelc2 == 0.0:
            psivlc = 0.0
            thtvlc = 0.0
        else:
            psivlc = math.atan2(vbelc2, vbelc1)
            thtvlc = math.atan2(
                -vbelc3, math.sqrt(vbelc1 * vbelc1 + vbelc2 * vbelc2)
            )
        psivlcx = psivlc * DEG
        thtvlcx = thtvlc * DEG

        tblc13 = float(tblc[0, 2])
        if math.fabs(tblc13) < 1.0:
            thtblc = math.asin(-tblc13)
        else:
            thtblc = PI / 2.0 * _cadac_sign(-tblc13)
        thtblcx = thtblc * DEG

        tblc23 = float(tblc[1, 2])
        tblc33 = float(tblc[2, 2])
        phiblcx = math.atan2(tblc23, tblc33) * DEG

        store.set("RECED", reced)
        store.set("RECE", rece)
        store.set("EVBED", evbed)
        store.set("EVBE", evbe)
        store.set("ESTTCD", esttcd)
        store.set("ESTTC", esttc)
        store.set("VBELC", vbelc)
        store.set("WBECB", wbecb)
        store.set("TBLC", tblc)
        store.set("thtblc", thtblc)
        store.set("thtblcx", thtblcx)
        store.set("dvbec", dvbec)
        store.set("thtvlc", thtvlc)
        store.set("thtvlcx", thtvlcx)
        store.set("psivlcx", psivlcx)
        store.set("FSPCB", fspcb)
        store.set("phiblcx", phiblcx)
        store.set("SBELC", sbelc)
        store.set("EWBEB", ewbeb)
        store.set("EFSPB", efspb)

    def ins_gyro(self, vehicle, int_step):
        store = vehicle.store
        ewalkg = np.asarray(store.get("EWALKG"), dtype=float)
        eunbg = np.asarray(store.get("EUNBG"), dtype=float)
        emisg = np.asarray(store.get("EMISG"), dtype=float)
        escalg = np.asarray(store.get("ESCALG"), dtype=float)
        ebiasg = np.asarray(store.get("EBIASG"), dtype=float)
        fspb = np.asarray(store.get("FSPB"), dtype=float)
        wbeb = np.asarray(store.get("WBEB"), dtype=float)
        egb = _diamat(escalg) + _skew(emisg)
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
        store.set("EWG", ewg)
        return ewbeb, wbecb

    def ins_accl(self, vehicle):
        store = vehicle.store
        emisa = np.asarray(store.get("EMISA"), dtype=float)
        escala = np.asarray(store.get("ESCALA"), dtype=float)
        ebiasa = np.asarray(store.get("EBIASA"), dtype=float)
        fspb = np.asarray(store.get("FSPB"), dtype=float)
        eab = _diamat(escala) + _skew(emisa)
        return ebiasa + eab @ fspb

    def ins_alt(self, vehicle):
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
