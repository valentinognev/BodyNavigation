import math

import numpy as np

from cadac.constants import DEG, EPS, PI, WEII3
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.wgs84 import GM, cad_geo84_in, cad_tdi84


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


class Rocket6Ins:
    name = "ins"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        plot = ("plot",)
        scrn_plot = ("scrn", "plot")
        for field in (
            Field("mins", 0, "int", "data", "ins"),
            Field("frax_algnmnt", 0.0, "real", "data", "ins"),
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
            Field("dbic", 0.0, "real", "out", "ins"),
            Field("alphacx", 0.0, "real", "out", "ins", plot),
            Field("betacx", 0.0, "real", "out", "ins", plot),
            Field("phibdcx", 0.0, "real", "out", "ins", plot),
            Field("thtbdcx", 0.0, "real", "out", "ins", plot),
            Field("psibdcx", 0.0, "real", "out", "ins", plot),
            Field("alppcx", 0.0, "real", "out", "ins"),
            Field("phipcx", 0.0, "real", "diag", "ins"),
            Field("RICID", zeros3, "vec", "state", "ins"),
            Field("RICI", zeros3, "vec", "state", "ins", plot),
            Field("EVBID", zeros3, "vec", "state", "ins"),
            Field("EVBI", zeros3, "vec", "state", "ins", plot),
            Field("ESBID", zeros3, "vec", "state", "ins"),
            Field("ESBI", zeros3, "vec", "state", "ins", plot),
            Field("ins_pos_err", 0.0, "real", "diag", "ins", scrn_plot),
            Field("ins_vel_err", 0.0, "real", "diag", "ins", scrn_plot),
            Field("ins_tilt_err", 0.0, "real", "diag", "ins", scrn_plot),
            Field("frax_transfer", 0.0, "real", "data", "ins"),
            Field("eunbg", 0.0, "real", "data", "ins"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        mins = vehicle.store.get("mins")
        if mins == 0:
            return
        if mins == 1:
            zeros = np.zeros(3)
            vehicle.store.set("ESBI", zeros)
            vehicle.store.set("EVBI", zeros)
            vehicle.store.set("RICI", zeros)
            return
        raise ValueError(f"unknown mins {mins}")

    def ins_gyro(self, vehicle, int_step):
        store = vehicle.store
        ewalkg = np.asarray(store.get("EWALKG"), dtype=float)
        emibg = np.asarray(store.get("EMISG"), dtype=float)
        escalg = np.asarray(store.get("ESCALG"), dtype=float)
        ebiasg = np.asarray(store.get("EBIASG"), dtype=float)
        eunbg_s = store.get("eunbg")
        wbib = np.asarray(store.get("WBIB"), dtype=float)
        fspb = np.asarray(store.get("FSPB"), dtype=float)
        egb = np.diag(escalg) + _skew(emibg)
        emiscg = egb @ wbib
        emsbg = ebiasg + emiscg
        eunbg = np.array([eunbg_s, eunbg_s, eunbg_s], dtype=float)
        eug = np.array(
            [eunbg[0] * fspb[0], eunbg[1] * fspb[1], eunbg[2] * fspb[2]],
            dtype=float,
        )
        ewg = ewalkg * (1.0 / math.sqrt(int_step))
        ewbib = emsbg + eug + ewg
        wbicb = wbib + ewbib
        store.set("EUG", eug)
        store.set("EWG", ewg)
        return ewbib, wbicb

    def ins_accl(self, vehicle):
        store = vehicle.store
        emisa = np.asarray(store.get("EMISA"), dtype=float)
        escala = np.asarray(store.get("ESCALA"), dtype=float)
        ebiasa = np.asarray(store.get("EBIASA"), dtype=float)
        fspb = np.asarray(store.get("FSPB"), dtype=float)
        eab = np.diag(escala) + _skew(emisa)
        return ebiasa + eab @ fspb

    def ins_grav(self, vehicle, esbi, sbiic):
        dbi = vehicle.store.get("dbi")
        dbic = float(np.linalg.norm(sbiic))
        ed = dbic - dbi
        dum = GM / dbic**3
        return np.asarray(esbi, dtype=float) * (-dum) - np.asarray(
            sbiic, dtype=float
        ) * (3.0 * ed * dum / dbic)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mins = store.get("mins")
        if mins not in (0, 1):
            raise ValueError(f"unknown mins {mins}")

        tbi = np.asarray(store.get("TBI"), dtype=float)
        fspb = np.asarray(store.get("FSPB"), dtype=float)
        wbib = np.asarray(store.get("WBIB"), dtype=float)
        wbii = np.asarray(store.get("WBII"), dtype=float)
        sbii = np.asarray(store.get("SBII"), dtype=float)
        vbii = np.asarray(store.get("VBII"), dtype=float)
        time = store.get("time")
        names = store.names()
        mroll = store.get("mroll") if "mroll" in names else 0
        int_step = ctx.int_step

        if mins == 0:
            tbic = tbi.copy()
            fspcb = fspb.copy()
            wbici = wbii.copy()
            wbicb = wbib.copy()
            sbiic = sbii.copy()
            vbiic = vbii.copy()
            dbic = float(np.linalg.norm(sbiic))
            ewbib = np.zeros(3)
            efspb = np.zeros(3)
            ins_pos_err = 0.0
            ins_vel_err = 0.0
            ins_tilt_err = 0.0
        else:
            esbi = np.asarray(store.get("ESBI"), dtype=float).copy()
            evbi = np.asarray(store.get("EVBI"), dtype=float).copy()
            rici = np.asarray(store.get("RICI"), dtype=float).copy()
            ricid = np.asarray(store.get("RICID"), dtype=float).copy()
            evbid = np.asarray(store.get("EVBID"), dtype=float).copy()
            esbid = np.asarray(store.get("ESBID"), dtype=float).copy()
            ewalka = np.asarray(store.get("EWALKA"), dtype=float)
            vbiic = np.asarray(store.get("VBIIC"), dtype=float).copy()

            sbiic = esbi + sbii
            dbic = float(np.linalg.norm(sbiic))

            ewbib, wbicb = self.ins_gyro(vehicle, int_step)
            ricid_new = tbi.T @ ewbib
            rici = integrate(ricid_new, ricid, rici, int_step)
            ricid = ricid_new

            if "mstar" in names and store.get("mstar") == 3 and "URIC" in names:
                rici = rici - np.asarray(store.get("URIC"), dtype=float)
                store.set("mstar", 2)

            tiic = np.eye(3) - _skew(rici)
            tbic = tbi @ tiic

            efspb = self.ins_accl(vehicle)
            fspcb = ewalka + efspb + fspb
            egravi = self.ins_grav(vehicle, esbi, sbiic)
            ticb = tbic.T
            evbid_new = ticb @ efspb - _skew(rici) @ ticb @ fspcb + egravi
            evbi = integrate(evbid_new, evbid, evbi, int_step)
            evbid = evbid_new

            esbid_new = evbi.copy()
            esbi = integrate(esbid_new, esbid, esbi, int_step)
            esbid = esbid_new

            if (
                "mgps" in names
                and store.get("mgps") == 3
                and "SXH" in names
                and "VXH" in names
            ):
                sxh = np.asarray(store.get("SXH"), dtype=float)
                vxh = np.asarray(store.get("VXH"), dtype=float)
                sbiic = sbiic - sxh
                vbiic = vbiic - vxh
                esbi = esbi - sxh
                evbi = evbi - vxh
                store.set("mgps", 2)

            sbiic = esbi + sbii
            vbiic = evbi + vbii
            wbici = tbic.T @ wbicb

            ins_pos_err = float(np.linalg.norm(esbi))
            ins_vel_err = float(np.linalg.norm(evbi))
            ins_tilt_err = float(np.linalg.norm(rici))

            store.set("RICID", ricid)
            store.set("RICI", rici)
            store.set("EVBID", evbid)
            store.set("EVBI", evbi)
            store.set("ESBID", esbid)
            store.set("ESBI", esbi)

        veic = np.array(
            [-WEII3 * sbiic[1], WEII3 * sbiic[0], 0.0],
            dtype=float,
        )
        vbeic = vbiic - veic
        vbecb = tbic @ vbeic
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
            dum = 1.0 * _cadac_sign(dum)
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
        vbecd = tdci @ vbeic

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

        tbd = tbic @ tdci.T
        tbd13 = tbd[0, 2]
        tbd11 = tbd[0, 0]
        tbd33 = tbd[2, 2]
        tbd12 = tbd[0, 1]
        tbd23 = tbd[1, 2]
        if math.fabs(tbd13) < 1.0:
            thtbdc = math.asin(-tbd13)
            cthtbd = math.cos(thtbdc)
        else:
            thtbdc = PI / 2.0 * _cadac_sign(-tbd13)
            cthtbd = EPS
        cpsi = tbd11 / cthtbd
        if math.fabs(cpsi) > 1.0:
            cpsi = 1.0 * _cadac_sign(cpsi)
        cphi = tbd33 / cthtbd
        if math.fabs(cphi) > 1.0:
            cphi = 1.0 * _cadac_sign(cphi)
        psibdc = math.acos(cpsi) * _cadac_sign(tbd12)
        if mroll == 0 or mroll == 1:
            phibdc = math.acos(cphi) * _cadac_sign(tbd23)
        elif mroll == 2:
            phibdc = math.acos(-cphi) * _cadac_sign(-tbd23)
        else:
            phibdc = 0.0
        psibdcx = DEG * psibdc
        thtbdcx = DEG * thtbdc
        phibdcx = DEG * phibdc

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
        store.set("dbic", dbic)
        store.set("alphacx", alphacx)
        store.set("betacx", betacx)
        store.set("phibdcx", phibdcx)
        store.set("thtbdcx", thtbdcx)
        store.set("psibdcx", psibdcx)
        store.set("alppcx", alppcx)
        store.set("EWBIB", ewbib)
        store.set("EFSPB", efspb)
        store.set("phipcx", phipcx)
        store.set("ins_pos_err", ins_pos_err)
        store.set("ins_vel_err", ins_vel_err)
        store.set("ins_tilt_err", ins_tilt_err)

    def terminate(self, vehicle, ctx):
        pass
