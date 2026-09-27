import math

import numpy as np

from cadac.constants import DEG, EPS, PI, WEII3
from cadac.kernel.state import Field
from cadac.math.frames import cadac_matmul, cadac_sign
from cadac.math.wgs84 import cad_geo84_in, cad_tdi84


class Hyper6Ins:
    name = "ins"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        plot = ("plot",)
        for field in (
            Field("mins", 0, "int", "data", "ins"),
            Field("frax", 0.0, "real", "data", "ins"),
            Field("VBIIC", zeros3, "vec", "out", "ins"),
            Field("SBIIC", zeros3, "vec", "out", "ins"),
            Field("WBICI", zeros3, "vec", "out", "ins"),
            Field("WBICB", zeros3, "vec", "out", "ins"),
            Field("EWALKG", zeros3, "vec", "data", "ins"),
            Field("EUNBG", zeros3, "vec", "data", "ins"),
            Field("EMISG", zeros3, "vec", "data", "ins"),
            Field("ESCALG", zeros3, "vec", "data", "ins"),
            Field("EBIASG", zeros3, "vec", "data", "ins"),
            Field("EUG", zeros3, "vec", "diag", "ins"),
            Field("EWG", zeros3, "vec", "diag", "ins"),
            Field("TBIC", zeros33, "mat", "out", "ins"),
            Field("EWALKA", zeros3, "vec", "data", "ins"),
            Field("EMISA", zeros3, "vec", "data", "ins"),
            Field("ESCALA", zeros3, "vec", "data", "ins"),
            Field("EBIASA", zeros3, "vec", "data", "ins"),
            Field("ppcx", 0.0, "real", "out", "ins"),
            Field("qqcx", 0.0, "real", "out", "ins"),
            Field("rrcx", 0.0, "real", "out", "ins"),
            Field("EWBIB", zeros3, "vec", "diag", "ins"),
            Field("EFSPB", zeros3, "vec", "diag", "ins"),
            Field("loncx", 0.0, "real", "out", "ins"),
            Field("latcx", 0.0, "real", "out", "ins"),
            Field("altc", 0.0, "real", "out", "ins"),
            Field("VBECD", zeros3, "vec", "out", "ins"),
            Field("dvbec", 0.0, "real", "out", "ins"),
            Field("TDCI", zeros33, "mat", "out", "ins"),
            Field("thtvdcx", 0.0, "real", "out", "ins"),
            Field("psivdcx", 0.0, "real", "out", "ins"),
            Field("FSPCB", zeros3, "vec", "out", "ins"),
            Field("alphacx", 0.0, "real", "diag", "ins"),
            Field("betacx", 0.0, "real", "diag", "ins"),
            Field("phibdcx", 0.0, "real", "out", "ins"),
            Field("thtbdcx", 0.0, "real", "out", "ins"),
            Field("psibdcx", 0.0, "real", "out", "ins"),
            Field("alppcx", 0.0, "real", "out", "ins"),
            Field("phipcx", 0.0, "real", "diag", "ins"),
            Field("RICID", zeros3, "vec", "state", "ins"),
            Field("RICI", zeros3, "vec", "state", "ins", plot),
            Field("EVBID", zeros3, "vec", "state", "ins"),
            Field("EVBI", zeros3, "vec", "state", "ins"),
            Field("ESBID", zeros3, "vec", "state", "ins"),
            Field("ESBI", zeros3, "vec", "state", "ins", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        mins = vehicle.store.get("mins")
        if mins != 0:
            raise ValueError(f"unknown mins {mins}")

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mins = store.get("mins")
        if mins != 0:
            raise ValueError(f"unknown mins {mins}")

        tbi = store.get("TBI")
        fspb = store.get("FSPB")
        wbib = store.get("WBIB")
        wbii = store.get("WBII")
        sbii = store.get("SBII")
        vbii = store.get("VBII")
        time = store.get("time")

        tbic = np.asarray(tbi, dtype=float).copy()
        fspcb = np.asarray(fspb, dtype=float).copy()
        wbici = np.asarray(wbii, dtype=float).copy()
        wbicb = np.asarray(wbib, dtype=float).copy()
        sbiic = np.asarray(sbii, dtype=float).copy()
        vbiic = np.asarray(vbii, dtype=float).copy()

        veic = np.array(
            [-WEII3 * sbiic[1], WEII3 * sbiic[0], 0.0],
            dtype=float,
        )
        vbeic = vbiic - veic
        vbecb = cadac_matmul(tbic, vbeic)
        dvbec = float(np.linalg.norm(vbecb))

        ppcx = wbicb[0] * DEG
        qqcx = wbicb[1] * DEG
        rrcx = wbicb[2] * DEG

        alphac = math.atan2(vbecb[2], vbecb[0])
        betac = math.asin(vbecb[1] / dvbec)
        alphacx = alphac * DEG
        betacx = betac * DEG

        dum = vbecb[0] / dvbec
        if math.fabs(dum) > 1.0:
            dum = 1.0 * cadac_sign(dum)
        alppc = math.acos(dum)
        if vbecb[1] == 0.0 and vbecb[2] == 0.0:
            phipc = 0.0
        elif math.fabs(vbecb[1]) < EPS:
            phipc = 0.0
            if vbecb[2] > 0.0:
                phipc = 0.0
            if vbecb[2] < 0.0:
                phipc = PI
        else:
            phipc = math.atan2(vbecb[1], vbecb[2])
        alppcx = alppc * DEG
        phipcx = phipc * DEG

        lonc, latc, altc = cad_geo84_in(sbiic, time)
        tdci = cad_tdi84(lonc, latc, altc, time)
        loncx = lonc * DEG
        latcx = latc * DEG
        vbecd = cadac_matmul(tdci, vbeic)

        if vbecd[0] == 0.0 and vbecd[1] == 0.0:
            psivdc = 0.0
            thtvdc = 0.0
        else:
            psivdc = math.atan2(vbecd[1], vbecd[0])
            thtvdc = math.atan2(
                -vbecd[2], math.sqrt(vbecd[0] * vbecd[0] + vbecd[1] * vbecd[1])
            )
        psivdcx = psivdc * DEG
        thtvdcx = thtvdc * DEG

        tbd = cadac_matmul(tbic, tdci.T.copy())
        tbd13 = tbd[0, 2]
        tbd11 = tbd[0, 0]
        tbd33 = tbd[2, 2]
        tbd12 = tbd[0, 1]
        tbd23 = tbd[1, 2]
        if math.fabs(tbd13) < 1.0 - 1e-14:
            thtbdc = math.asin(-tbd13)
            cthtbd = math.cos(thtbdc)
        else:
            thtbdc = PI / 2.0 * cadac_sign(-tbd13)
            cthtbd = EPS
        cpsi = tbd11 / cthtbd
        if math.fabs(cpsi) > 1.0:
            cpsi = 1.0 * cadac_sign(cpsi)
        cphi = tbd33 / cthtbd
        if math.fabs(cphi) > 1.0:
            cphi = 1.0 * cadac_sign(cphi)
        psibdc = math.acos(cpsi) * cadac_sign(tbd12)
        phibdc = math.acos(cphi) * cadac_sign(tbd23)
        psibdcx = DEG * psibdc
        thtbdcx = DEG * thtbdc
        phibdcx = DEG * phibdc

        ewbib = np.zeros(3)
        efspb = np.zeros(3)

        store.set("VBIIC", vbiic)
        store.set("SBIIC", sbiic)
        store.set("WBICI", wbici)
        store.set("WBICB", wbicb)
        store.set("TBIC", tbic)
        store.set("ppcx", ppcx)
        store.set("qqcx", qqcx)
        store.set("rrcx", rrcx)
        store.set("loncx", loncx)
        store.set("latcx", latcx)
        store.set("altc", altc)
        store.set("VBECD", vbecd)
        store.set("dvbec", dvbec)
        store.set("TDCI", tdci)
        store.set("thtvdcx", thtvdcx)
        store.set("psivdcx", psivdcx)
        store.set("FSPCB", fspcb)
        store.set("phibdcx", phibdcx)
        store.set("thtbdcx", thtbdcx)
        store.set("psibdcx", psibdcx)
        store.set("alppcx", alppcx)
        store.set("EWBIB", ewbib)
        store.set("EFSPB", efspb)
        store.set("alphacx", alphacx)
        store.set("betacx", betacx)
        store.set("phipcx", phipcx)

    def terminate(self, vehicle, ctx):
        pass
