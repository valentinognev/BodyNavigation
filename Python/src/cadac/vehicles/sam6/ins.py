import math

import numpy as np

from cadac.constants import AGRAV, DEG, EPS, PI, REARTH
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import skew
from cadac.stoch import gauss, uniform

# Initial covariance after GPS transfer alignment (C++ PP0). Units: m, m/s, mrad.
_PP0 = np.array(
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


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _gauss(mean, sig):
    return gauss(mean, sig)


def _uniform(low, high):
    return uniform(low, high)


def _gauss3_rtl(sig):
    # g++ evaluates Variable::init(v1, v2, v3) arguments right-to-left.
    third = gauss(0.0, sig)
    second = gauss(0.0, sig)
    first = gauss(0.0, sig)
    return np.array([first, second, third], dtype=float)


def _diamat(vec):
    return np.diag(np.asarray(vec, dtype=float))


def _cholesky(mat):
    dim = mat.shape[0]
    sqrtmat = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            if j < i:
                total = 0.0
                if j > 0:
                    for k in range(j):
                        total += sqrtmat[i, k] * sqrtmat[j, k]
                if sqrtmat[j, j] == 0:
                    sqrtmat[i, j] = 0.0
                else:
                    sqrtmat[i, j] = (mat[i, j] - total) / sqrtmat[j, j]
            elif j == i:
                total = 0.0
                if i > 0:
                    for k in range(i):
                        total += sqrtmat[i, k] * sqrtmat[i, k]
                sqrtmat[i, j] = math.sqrt(mat[i, i] - total)
            else:
                sqrtmat[i, j] = 0.0
    return sqrtmat


class Sam6Ins:
    name = "ins"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        plot = ("plot",)
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
            Field("thtvlcx", 0.0, "real", "out", "ins", plot),
            Field("psivlcx", 0.0, "real", "out", "ins", plot),
            Field("FSPCB", zeros3, "vec", "out", "ins"),
            Field("phiblcx", 0.0, "real", "out", "ins", plot),
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
        if mins not in (0, 1):
            raise ValueError(f"unknown mins {mins}")
        frax = store.get("frax")
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)
        if mins == 1:
            store.set("EUNBG", np.zeros(3))
            store.set("EMISG", _gauss3_rtl(1.1e-4))
            store.set("ESCALG", _gauss3_rtl(2.5e-5))
            store.set("EBIASG", _gauss3_rtl(3.2e-6))
            store.set("EWALKA", _gauss3_rtl(8.35e-4))
            store.set("EMISA", _gauss3_rtl(1.1e-4))
            store.set("ESCALA", _gauss3_rtl(5e-4))
            store.set("EBIASA", _gauss3_rtl(3.56e-3))
        gauss_init = np.array([_gauss(0.0, 1.0) for _ in range(9)], dtype=float)
        xx_init = _cholesky(_PP0) @ gauss_init
        xx_init = xx_init * (1.0 + frax)
        esttc = np.array([xx_init[0], xx_init[1], xx_init[2]], dtype=float)
        evbe = np.array([xx_init[3], xx_init[4], xx_init[5]], dtype=float)
        rece = np.array([xx_init[6], xx_init[7], xx_init[8]], dtype=float) * 0.001
        vbelc = evbe + vbel
        sbelc = esttc + sbel
        store.set("VBELC", vbelc)
        store.set("RECE", rece)
        store.set("EVBE", evbe)
        store.set("ESTTC", esttc)
        store.set("SBELC", sbelc)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mins = store.get("mins")
        if mins not in (0, 1):
            raise ValueError(f"unknown mins {mins}")
        int_step = ctx.int_step
        tbl = np.asarray(store.get("TBL"), dtype=float)
        wbeb = np.asarray(store.get("WBEB"), dtype=float)
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        fspb = np.asarray(store.get("FSPB"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)
        dvbe = store.get("dvbe")
        reced = np.asarray(store.get("RECED"), dtype=float)
        rece = np.asarray(store.get("RECE"), dtype=float)
        evbed = np.asarray(store.get("EVBED"), dtype=float)
        evbe = np.asarray(store.get("EVBE"), dtype=float)
        esttcd = np.asarray(store.get("ESTTCD"), dtype=float)
        esttc = np.asarray(store.get("ESTTC"), dtype=float)

        self.ins_alt(vehicle)

        ewbeb = np.zeros(3, dtype=float)
        efspb = np.zeros(3, dtype=float)
        if mins == 0:
            tblc = np.array(tbl, copy=True)
            fspcb = np.array(fspb, copy=True)
            wbecb = np.array(wbeb, copy=True)
            sbelc = np.array(sbel, copy=True)
            vbelc = np.array(vbel, copy=True)
            dvbec = dvbe
        else:
            tanlat = store.get("tanlat")
            ewalka = np.asarray(store.get("EWALKA"), dtype=float)
            tlb = np.asarray(store.get("TLB"), dtype=float)
            efspb = self.ins_accl(vehicle)
            ewbeb, wbecb = self.ins_gyro(vehicle, int_step)
            ewbel = tlb @ ewbeb
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
            tblc = tbl @ tllc
            tlcb = tblc.T
            # g++ evaluates Matrix::build_vec3 arguments right-to-left.
            u3 = _uniform(-ewalka[2], ewalka[2])
            u2 = _uniform(-ewalka[1], ewalka[1])
            u1 = _uniform(-ewalka[0], ewalka[0])
            walka = np.array([u1, u2, u3], dtype=float)
            fspcb = walka + efspb + fspb
            ef = tlcb @ efspb - rere @ tlcb @ fspcb
            evbed_new = np.array(
                [ef[0], ef[1], ef[2] + 2.0 * AGRAV * esttc[2] / REARTH],
                dtype=float,
            )
            evbe = integrate(evbed_new, evbed, evbe, int_step)
            evbed = evbed_new
            esttcd_new = evbe
            esttc = integrate(esttcd_new, esttcd, esttc, int_step)
            esttcd = esttcd_new
            sbelc = esttc + sbel
            vbelc = evbe + vbel
            dvbec = float(np.linalg.norm(vbelc))

        vbelc1 = float(vbelc[0])
        vbelc2 = float(vbelc[1])
        vbelc3 = float(vbelc[2])
        if vbelc1 == 0 and vbelc2 == 0:
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
        if math.fabs(tblc13) < 1:
            thtblc = math.asin(-tblc13)
            cthtblc = math.cos(thtblc)
        else:
            thtblc = PI / 2 * _sign(-tblc13)
            cthtblc = EPS
        thtblcx = thtblc * DEG

        tblc23 = float(tblc[1, 2])
        tblc33 = float(tblc[2, 2])
        cphic = tblc33 / cthtblc
        if math.fabs(cphic) >= 1:
            cphic = (1 - EPS) * _sign(cphic)
        phiblc = math.acos(cphic) * _sign(tblc23)
        phiblcx = phiblc * DEG

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
        store.set("EWG", ewg)
        return ewbeb, wbecb

    def ins_accl(self, vehicle):
        store = vehicle.store
        emisa = np.asarray(store.get("EMISA"), dtype=float)
        escala = np.asarray(store.get("ESCALA"), dtype=float)
        ebiasa = np.asarray(store.get("EBIASA"), dtype=float)
        fspb = np.asarray(store.get("FSPB"), dtype=float)
        eab = _diamat(escala) + skew(emisa)
        return ebiasa + eab @ fspb

    def ins_alt(self, vehicle):
        store = vehicle.store
        biasal = store.get("biasal")
        randal = store.get("randal")
        alt = store.get("alt")
        ehbe = biasal + randal
        hbem = alt + ehbe
        store.set("hbem", hbem)
        store.set("ehbe", ehbe)

    def terminate(self, vehicle, ctx):
        pass
