from math import acos, atan2, cos, sin, sqrt, tan

import numpy as np

from cadac.constants import DEG
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr, polar_from_cart, skew
from cadac.vehicles.flat6.sraam6.ai_radar import Sraam6AiRadar

SMALL = 1e-7

_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_ZEROS3 = (0.0, 0.0, 0.0)
_ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


class Sraam6Seeker:
    name = "seeker"

    def __init__(self):
        self._ai_radar = Sraam6AiRadar()

    def define(self, vehicle):
        store = vehicle.store
        plot = ("scrn", "plot")
        scrn = ("scrn",)
        com = ("com",)
        for field in (
            Field("tgt_num", 0, "int", "data", "combus"),
            Field("STEL", _ZEROS3, "vec", "out", "combus"),
            Field("VTEL", _ZEROS3, "vec", "out", "combus"),
            Field("TTL", _IDENTITY, "mat", "", "combus"),
            Field("tgt_com_slot", 0, "int", "out", "combus"),
            Field("sht_num", 0, "int", "data", "combus"),
            Field("SSEL", _ZEROS3, "vec", "", "combus"),
            Field("dts", 0.0, "real", "out", "combus", plot),
            Field("STSL", _ZEROS3, "vec", "out", "combus"),
            Field("mseek", 0, "int", "data/diag", "seeker", com),
            Field("ms1dyn", 0, "int", "data", "seeker"),
            Field("isets1", 0, "int", "init", "seeker"),
            Field("epchac", 0.0, "real", "save", "seeker"),
            Field("ibreak", 0, "int", "init", "seeker"),
            Field("dblind", 0.0, "real", "data", "seeker"),
            Field("racq", 0.0, "real", "data", "seeker"),
            Field("dtimac", 0.0, "real", "data", "seeker"),
            Field("dbt", 0.0, "real", "diag", "seeker", plot),
            Field("gk", 0.0, "real", "data", "seeker"),
            Field("zetak", 0.0, "real", "data", "seeker"),
            Field("wnk", 0.0, "real", "data", "seeker"),
            Field("biast", 0.0, "real", "data", "seeker"),
            Field("randt", 0.0, "real", "data", "seeker"),
            Field("biasp", 0.0, "real", "data", "seeker"),
            Field("randp", 0.0, "real", "data", "seeker"),
            Field("wlq1d", 0.0, "real", "state", "seeker"),
            Field("wlq1", 0.0, "real", "state", "seeker"),
            Field("wlqd", 0.0, "real", "state", "seeker"),
            Field("wlq", 0.0, "real", "state", "seeker"),
            Field("wlr1d", 0.0, "real", "state", "seeker"),
            Field("wlr1", 0.0, "real", "state", "seeker"),
            Field("wlrd", 0.0, "real", "state", "seeker"),
            Field("wlr", 0.0, "real", "state", "seeker"),
            Field("wlq2d", 0.0, "real", "state", "seeker"),
            Field("wlq2", 0.0, "real", "state", "seeker"),
            Field("wlr2d", 0.0, "real", "state", "seeker"),
            Field("wlr2", 0.0, "real", "state", "seeker"),
            Field("fovyaw", 0.0, "real", "data", "seeker"),
            Field("fovpitch", 0.0, "real", "data", "seeker"),
            Field("dba", 0.0, "real", "diag", "seeker"),
            Field("daim", 0.0, "real", "data", "seeker"),
            Field("BIASAI", (1.0, 0.5, 0.2), "vec", "data", "seeker"),
            Field("BIASSC", _ZEROS3, "vec", "data", "seeker"),
            Field("RANDSC", _ZEROS3, "vec", "data", "seeker"),
            Field("epy", 0.0, "real", "diag", "seeker"),
            Field("epz", 0.0, "real", "diag", "seeker"),
            Field("thtpb", 0.0, "real", "out", "seeker"),
            Field("psipb", 0.0, "real", "out", "seeker"),
            Field("ththb", 0.0, "real", "diag", "seeker"),
            Field("phihb", 0.0, "real", "diag", "seeker"),
            Field("ththbx", 0.0, "real", "diag", "seeker", scrn),
            Field("phihbx", 0.0, "real", "diag", "seeker", scrn),
            Field("TPB", _ZEROS33, "mat", "init", "seeker"),
            Field("THB", _ZEROS33, "mat", "init", "seeker"),
            Field("timeac", 0.0, "real", "diag", "seeker"),
            Field("psiot1", 0.0, "real", "diag", "seeker"),
            Field("thtot1", 0.0, "real", "diag", "seeker"),
            Field("dvbtc", 0.0, "real", "diag", "seeker"),
            Field("EAHH", _ZEROS3, "vec", "diag", "seeker"),
            Field("EPHH", _ZEROS3, "vec", "diag", "seeker"),
            Field("EAPH", _ZEROS3, "vec", "diag", "seeker"),
            Field("thtpbx", 0.0, "real", "diag", "seeker", plot),
            Field("psipbx", 0.0, "real", "diag", "seeker", plot),
            Field("sigdpy", 0.0, "real", "out", "seeker"),
            Field("sigdpz", 0.0, "real", "out", "seeker"),
            Field("biaseh", 0.0, "real", "data", "seeker"),
            Field("randeh", 0.0, "real", "data", "seeker"),
            Field("psihlx", 0.0, "real", "diag", "seeker"),
            Field("ththlx", 0.0, "real", "diag", "seeker"),
            Field("phihlx", 0.0, "real", "diag", "seeker"),
            Field("SBTL", _ZEROS3, "vec", "diag", "seeker"),
        ):
            store.define(field)
        self._ai_radar.define(vehicle)

    def initialize(self, vehicle, ctx):
        self._ai_radar.initialize(vehicle, ctx)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        tgt_num = store.get("tgt_num")
        sht_num = store.get("sht_num")
        mseek = store.get("mseek")
        ms1dyn = store.get("ms1dyn")
        isets1 = store.get("isets1")
        racq = store.get("racq")
        dtimac = store.get("dtimac")
        fovyaw = store.get("fovyaw")
        fovpitch = store.get("fovpitch")
        epchac = store.get("epchac")
        time = ctx.sim_time
        sbel = store.get("SBEL")
        mguid = store.get("mguid")
        thb = np.array(store.get("THB"), dtype=float, copy=True)

        if mseek not in (0, 2, 3, 4, 5):
            raise ValueError(f"unknown mseek {mseek}")

        stel = np.zeros(3)
        vtel = np.zeros(3)
        ssel = np.zeros(3)
        tgt_com_slot = 0
        packets = ctx.combus or ()
        targets = [
            (slot, packet)
            for slot, packet in enumerate(packets)
            if packet.type == "TARGET3"
        ]
        if sht_num >= 1:
            idx = sht_num - 1
            if idx < len(targets):
                _slot, packet = targets[idx]
                ssel = np.asarray(packet.vars["SAEL"], dtype=float)
        if tgt_num >= 1:
            idx = tgt_num - 1
            if idx < len(targets):
                tgt_com_slot, packet = targets[idx]
                stel = np.asarray(packet.vars["SAEL"], dtype=float)
                vtel = np.asarray(packet.vars["VAEL"], dtype=float)
        store.set("STEL", stel)
        store.set("VTEL", vtel)

        stsl = stel - ssel
        dts = float(np.linalg.norm(stsl))
        sbtl = sbel - stel
        ttl = np.eye(3)
        sbtt = ttl @ sbtl
        polar = polar_from_cart(sbtt)
        dbt = float(polar[0])
        psiot1 = float(polar[1])
        thtot1 = float(polar[2])

        psipb = 0.0
        thtpb = 0.0
        sigdy = 0.0
        sigdz = 0.0
        sigdpy = 0.0
        sigdpz = 0.0
        timeac = 0.0

        if mseek == 2:
            isets1 = 1
            if dbt < racq:
                mseek = 3
        if mseek == 3:
            if ms1dyn not in (0, 1):
                raise ValueError(f"unknown ms1dyn {ms1dyn}")
            if isets1 == 1:
                thtpb, psipb, sigdy, sigdz = self.seeker_kin(vehicle, sbtl, vtel, dbt)
                ththb, phihb = self.seeker_uthpb(psipb, thtpb)
                thb = self.seeker_thb(ththb, phihb)
                isets1 = 0
                epchac = time
            if ms1dyn == 1:
                mseek, mguid, thtpb, psipb, sigdy, sigdz, ehz, ehy, thb = self.seeker_dyn(
                    vehicle, mseek, mguid, thb, sbtl, dbt, ctx.int_step
                )
                if abs(ehz) <= fovyaw and abs(ehy) <= fovpitch:
                    timeac = time - epchac
                    if timeac > dtimac:
                        mseek = 4
            else:
                thtpb, psipb, sigdy, sigdz = self.seeker_kin(vehicle, sbtl, vtel, dbt)
                timeac = time - epchac
                if timeac > dtimac:
                    mseek = 4
        if mseek == 4:
            if ms1dyn not in (0, 1):
                raise ValueError(f"unknown ms1dyn {ms1dyn}")
            mguid = 6
            if ms1dyn == 1:
                mseek, mguid, thtpb, psipb, sigdy, sigdz, ehz, ehy, thb = self.seeker_dyn(
                    vehicle, mseek, mguid, thb, sbtl, dbt, ctx.int_step
                )
            else:
                thtpb, psipb, sigdy, sigdz = self.seeker_kin(vehicle, sbtl, vtel, dbt)
            sigdpy = sigdy
            sigdpz = sigdz

        thtpbx = thtpb * DEG
        psipbx = psipb * DEG
        store.set("STEL", stel)
        store.set("VTEL", vtel)
        store.set("tgt_com_slot", tgt_com_slot)
        store.set("SSEL", ssel)
        store.set("dts", dts)
        store.set("STSL", stsl)
        store.set("thtpb", thtpb)
        store.set("psipb", psipb)
        store.set("sigdpy", sigdpy)
        store.set("sigdpz", sigdpz)
        store.set("mguid", mguid)
        store.set("THB", thb)
        store.set("epchac", epchac)
        store.set("mseek", mseek)
        store.set("isets1", isets1)
        store.set("dbt", dbt)
        store.set("timeac", timeac)
        store.set("psiot1", psiot1)
        store.set("thtot1", thtot1)
        store.set("thtpbx", thtpbx)
        store.set("psipbx", psipbx)
        store.set("SBTL", sbtl)
        # Fortran S2 AI radar (Band D) — may set mnav 2→3 and bias STEL/VTEL
        self._ai_radar.execute(vehicle, ctx)

    def seeker_kin(self, vehicle, sbtl, vtel, dbt):
        store = vehicle.store
        tbl = store.get("TBL")
        vbel = store.get("VBEL")
        stbl = np.asarray(sbtl, dtype=float) * (-1.0)
        stbb = tbl @ stbl
        dum1 = 1.0 / dbt
        utbl = stbl * dum1
        vtbl = np.asarray(vtel, dtype=float) - vbel
        dvbtc = abs(float(utbl @ vtbl))
        woeb = tbl @ (skew(utbl) @ vtbl) * dum1
        polar = polar_from_cart(stbb)
        psipb = float(polar[1])
        thtpb = float(polar[2])
        tpb = mat2tr(psipb, thtpb)
        woep = tpb @ woeb
        sigdy = float(woep[1])
        sigdz = float(woep[2])
        store.set("dvbtc", dvbtc)
        return thtpb, psipb, sigdy, sigdz

    def seeker_dyn(self, vehicle, mseek, mguid, thb, sbtl, dbt, int_step):
        store = vehicle.store
        dblind = store.get("dblind")
        gk = store.get("gk")
        zetak = store.get("zetak")
        wnk = store.get("wnk")
        biast = store.get("biast")
        randt = store.get("randt")
        biasp = store.get("biasp")
        randp = store.get("randp")
        biaseh = store.get("biaseh")
        randeh = store.get("randeh")
        tpb = np.array(store.get("TPB"), dtype=float, copy=True)
        ttl = np.array(store.get("TTL"), dtype=float, copy=True)
        tbl = np.array(store.get("TBL"), dtype=float, copy=True)
        wbeb = np.asarray(store.get("WBEB"), dtype=float)
        trcond = store.get("trcond")
        trtht = store.get("trtht")
        trthtd = store.get("trthtd")
        trphid = store.get("trphid")
        trate = store.get("trate")
        wlq1d = store.get("wlq1d")
        wlq1 = store.get("wlq1")
        wlqd = store.get("wlqd")
        wlq = store.get("wlq")
        wlr1d = store.get("wlr1d")
        wlr1 = store.get("wlr1")
        wlrd = store.get("wlrd")
        wlr = store.get("wlr")
        wlq2d = store.get("wlq2d")
        wlq2 = store.get("wlq2")
        wlr2d = store.get("wlr2d")
        wlr2 = store.get("wlr2")

        thl = thb @ tbl
        sbth = thl @ np.asarray(sbtl, dtype=float)
        sath = self.seeker_aimp(vehicle, thl, ttl, dbt)
        sabh = sath - sbth
        ey = atan2(-sabh[2], sabh[0])
        ez = atan2(sabh[1], sabh[0])
        ehy = ey + biaseh + randeh
        ehz = ez + biaseh + randeh
        eahh = np.array([0.0, ehz, -ehy])
        tbh = thb.T
        tph = tpb @ tbh
        thp = tph.T
        u1pp = np.array([1.0, 0.0, 0.0])
        u1hh = np.array([1.0, 0.0, 0.0])
        ephh = thp @ u1pp - u1hh
        eaph = eahh - ephh
        eapp = tph @ eaph
        epy = -float(eapp[2])
        epz = float(eapp[1])

        wsq = wnk * wnk
        gg = gk * wsq
        wlr1d_new = wlr2
        wlr1 = integrate(wlr1d_new, wlr1d, wlr1, int_step)
        wlr1d = wlr1d_new
        wlr2d_new = gg * epz - 2.0 * zetak * wnk * wlr1d - wsq * wlr1
        wlr2 = integrate(wlr2d_new, wlr2d, wlr2, int_step)
        wlr2d = wlr2d_new
        wlq1d_new = wlq2
        wlq1 = integrate(wlq1d_new, wlq1d, wlq1, int_step)
        wlq1d = wlq1d_new
        wlq2d_new = gg * epy - 2.0 * zetak * wnk * wlq1d - wsq * wlq1
        wlq2 = integrate(wlq2d_new, wlq2d, wlq2, int_step)
        wlq2d = wlq2d_new
        sigdz = wlr1
        sigdy = wlq1

        wbep = tpb @ wbeb
        wlrd_new = wlr1 - float(wbep[2])
        wlr = integrate(wlrd_new, wlrd, wlr, int_step)
        wlrd = wlrd_new
        psipb = wlr
        wlqd_new = wlq1 - float(wbep[1])
        wlq = integrate(wlqd_new, wlqd, wlq, int_step)
        wlqd = wlqd_new
        thtpb = wlq
        thtpbd = wlqd
        tpb = mat2tr(psipb, thtpb)
        ththbc, phihbc = self.seeker_uthpb(psipb, thtpb)
        ththb = ththbc + biast + randt
        phihb = phihbc + biasp + randp
        thb = self.seeker_thb(ththb, phihb)
        ththbx = ththb * DEG
        phihbx = phihb * DEG

        if mseek == 4:
            ibreak = 0
            phihbd = -thtpbd * sin(psipb)
            eh = sqrt(ehy * ehy + ehz * ehz)
            if abs(ththb) > trtht:
                trcond = 6
                ibreak = 1
            elif abs(thtpbd) > trthtd:
                trcond = 7
                ibreak = 1
            elif abs(phihbd) > trphid:
                trcond = 8
                ibreak = 1
            elif eh > trate:
                trcond = 9
                ibreak = 1
            if ibreak == 1:
                mseek = 2
                mguid = 3
            if dbt < dblind:
                mseek = 5

        store.set("wlq1d", wlq1d)
        store.set("wlq1", wlq1)
        store.set("wlqd", wlqd)
        store.set("wlq", wlq)
        store.set("wlr1d", wlr1d)
        store.set("wlr1", wlr1)
        store.set("wlrd", wlrd)
        store.set("wlr", wlr)
        store.set("wlq2d", wlq2d)
        store.set("wlq2", wlq2)
        store.set("wlr2d", wlr2d)
        store.set("wlr2", wlr2)
        store.set("trcond", trcond)
        store.set("epy", epy)
        store.set("epz", epz)
        store.set("ththb", ththb)
        store.set("phihb", phihb)
        store.set("ththbx", ththbx)
        store.set("phihbx", phihbx)
        store.set("TPB", tpb)
        store.set("EAHH", eahh)
        store.set("EPHH", ephh)
        store.set("EAPH", eaph)
        return mseek, mguid, thtpb, psipb, sigdy, sigdz, ehz, ehy, thb

    def seeker_aimp(self, vehicle, thl, ttl, dbt):
        store = vehicle.store
        daim = store.get("daim")
        biasai = np.asarray(store.get("BIASAI"), dtype=float)
        biassc = np.asarray(store.get("BIASSC"), dtype=float)
        randsc = np.asarray(store.get("RANDSC"), dtype=float)
        tht = thl @ np.asarray(ttl, dtype=float).T
        if dbt < daim:
            return tht @ biasai
        return tht @ (biassc + randsc)

    def seeker_uthpb(self, psipb, thtpb):
        ththb = acos(cos(thtpb) * cos(psipb))
        sinpsi = sin(psipb)
        tantht = tan(thtpb)
        if abs(sinpsi) and abs(tantht) < SMALL:
            phihb = 0.0
        else:
            phihb = atan2(sinpsi, tantht)
        return ththb, phihb

    def seeker_thb(self, tht, phi):
        thb = np.zeros((3, 3))
        thb[0, 0] = cos(tht)
        thb[2, 0] = sin(tht)
        thb[1, 1] = cos(phi)
        thb[1, 2] = sin(phi)
        thb[0, 1] = thb[2, 0] * thb[1, 2]
        thb[0, 2] = (-thb[2, 0]) * thb[1, 1]
        thb[2, 1] = (-thb[0, 0]) * thb[1, 2]
        thb[2, 2] = thb[0, 0] * thb[1, 1]
        thb[1, 0] = 0.0
        return thb

    def terminate(self, vehicle, ctx):
        pass
