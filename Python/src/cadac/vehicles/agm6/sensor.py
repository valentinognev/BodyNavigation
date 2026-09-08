from math import acos, atan2, cos, fabs, sin, sqrt, tan

import numpy as np

from cadac.constants import DEG
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr, polar_from_cart

SMALL = 1.e-7


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


def _vec3(vars_, primary, fallback):
    if primary in vars_:
        return np.array(vars_[primary], dtype=float, copy=True)
    if fallback in vars_:
        return np.array(vars_[fallback], dtype=float, copy=True)
    return np.zeros(3)


def _sensor_ir_uthpb(psipb, thtpb):
    ththb = acos(cos(thtpb) * cos(psipb))
    sinpsi = sin(psipb)
    tantht = tan(thtpb)
    if fabs(sinpsi) and fabs(tantht) < SMALL:
        phihb = 0.0
    else:
        phihb = atan2(sinpsi, tantht)
    return ththb, phihb


def _sensor_ir_thb(tht, phi):
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


class Agm6Sensor:
    name = "sensor"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        plot = ("plot",)
        scrn = ("scrn",)
        com = ("com",)
        for field in (
            Field("tgt_num", 0, "int", "data", "sensor"),
            Field("STEL", zeros3, "vec", "", "sensor"),
            Field("VTEL", zeros3, "vec", "", "sensor"),
            Field("tgt_com_slot", 0, "int", "out", "sensor"),
            Field("mseek", 0, "int", "data/diag", "sensor", com),
            Field("skr_dyn", 0, "int", "data", "sensor"),
            Field("isets1", 0, "int", "init", "sensor"),
            Field("epchac", 0.0, "real", "init", "sensor"),
            Field("ibreak", 0, "int", "init", "sensor"),
            Field("dblind", 0.0, "real", "data", "sensor"),
            Field("racq", 0.0, "real", "data", "sensor"),
            Field("dtimac", 0.0, "real", "data", "sensor"),
            Field("gk", 0.0, "real", "data", "sensor"),
            Field("zetak", 0.0, "real", "data", "sensor"),
            Field("wnk", 0.0, "real", "data", "sensor"),
            Field("biast", 0.0, "real", "data", "sensor"),
            Field("randt", 0.0, "real", "data", "sensor"),
            Field("biasp", 0.0, "real", "data", "sensor"),
            Field("randp", 0.0, "real", "data", "sensor"),
            Field("wlq1d", 0.0, "real", "state", "sensor"),
            Field("wlq1", 0.0, "real", "state", "sensor"),
            Field("wlqd", 0.0, "real", "state", "sensor"),
            Field("wlq", 0.0, "real", "state", "sensor"),
            Field("wlr1d", 0.0, "real", "state", "sensor"),
            Field("wlr1", 0.0, "real", "state", "sensor"),
            Field("wlrd", 0.0, "real", "state", "sensor"),
            Field("wlr", 0.0, "real", "state", "sensor"),
            Field("wlq2d", 0.0, "real", "state", "sensor"),
            Field("wlq2", 0.0, "real", "state", "sensor"),
            Field("wlr2d", 0.0, "real", "state", "sensor"),
            Field("wlr2", 0.0, "real", "state", "sensor"),
            Field("fovyaw", 0.0, "real", "data", "sensor"),
            Field("fovpitch", 0.0, "real", "data", "sensor"),
            Field("dba", 0.0, "real", "diag", "sensor"),
            Field("daim", 0.0, "real", "data", "sensor"),
            Field("BIASAI", zeros3, "vec", "data", "sensor"),
            Field("BIASSC", zeros3, "vec", "data", "sensor"),
            Field("RANDSC", zeros3, "vec", "data", "sensor"),
            Field("epy", 0.0, "real", "diag", "sensor", plot),
            Field("epz", 0.0, "real", "diag", "sensor", plot),
            Field("thtpb", 0.0, "real", "out", "sensor", plot),
            Field("psipb", 0.0, "real", "out", "sensor", plot),
            Field("ththb", 0.0, "real", "diag", "sensor", plot),
            Field("phihb", 0.0, "real", "diag", "sensor", plot),
            Field("TPB", zeros33, "mat", "init", "sensor"),
            Field("THB", zeros33, "mat", "init", "sensor"),
            Field("dvbtc", 0.0, "real", "diag", "sensor"),
            Field("EAHH", zeros3, "vec", "diag", "sensor"),
            Field("EPHH", zeros3, "vec", "diag", "sensor"),
            Field("EAPH", zeros3, "vec", "diag", "sensor"),
            Field("thtpbx", 0.0, "real", "diag", "sensor"),
            Field("psipbx", 0.0, "real", "diag", "sensor"),
            Field("sigdpy", 0.0, "real", "out", "sensor", plot),
            Field("sigdpz", 0.0, "real", "out", "sensor", plot),
            Field("biaseh", 0.0, "real", "data", "sensor"),
            Field("randeh", 0.0, "real", "data", "sensor"),
            Field("SBTL", zeros3, "vec", "diag", "sensor", scrn),
            Field("epaz_saved", 0.0, "real", "save", "sensor"),
            Field("epel_saved", 0.0, "real", "save", "sensor"),
            Field("range_saved", 0.0, "real", "save", "sensor"),
            Field("rate_saved", 0.0, "real", "save", "sensor"),
            Field("timeac", 0.0, "real", "save", "sensor"),
            Field("dbtk", 0.0, "real", "diag", "sensor"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def terminate(self, vehicle, ctx):
        pass

    def sensor_ir_kin(self, vehicle, sbtl, vtel, dbtk):
        store = vehicle.store
        tbl = np.asarray(store.get("TBL"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)
        stbl = -sbtl
        stbb = tbl @ stbl
        utbl = stbl / dbtk
        vtbl = vtel - vbel
        dvbtc = abs(float(utbl @ vtbl))
        woeb = tbl @ _skew(utbl) @ vtbl / dbtk
        polar = polar_from_cart(stbb)
        psipb = float(polar[1])
        thtpb = float(polar[2])
        tpb = mat2tr(psipb, thtpb)
        woep = tpb @ woeb
        store.set("dvbtc", dvbtc)
        return thtpb, psipb, float(woep[1]), float(woep[2])

    def sensor_ir_aimp(self, vehicle, thl, ttl, dbtk):
        store = vehicle.store
        daim = store.get("daim")
        biasai = np.asarray(store.get("BIASAI"), dtype=float)
        biassc = np.asarray(store.get("BIASSC"), dtype=float)
        randsc = np.asarray(store.get("RANDSC"), dtype=float)
        tht = thl @ ttl.T
        if dbtk < daim:
            return tht @ biasai
        return tht @ (biassc + randsc)

    def sensor_ir_dyn(self, vehicle, sbtl, dbtk, int_step, mseek, mguid, thb, trcond):
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
        tbl = np.asarray(store.get("TBL"), dtype=float)
        wbecb = np.asarray(store.get("WBECB"), dtype=float)
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
        if "TTL" in store:
            ttl = np.asarray(store.get("TTL"), dtype=float)
        else:
            ttl = np.eye(3)

        thl = thb @ tbl
        sbth = thl @ sbtl
        sath = self.sensor_ir_aimp(vehicle, thl, ttl, dbtk)
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
        epy = float(-eapp[2])
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

        wbep = tpb @ wbecb
        wlrd_new = wlr1 - wbep[2]
        wlr = integrate(wlrd_new, wlrd, wlr, int_step)
        wlrd = wlrd_new
        psipb = wlr
        psipbd = wlrd
        wlqd_new = wlq1 - wbep[1]
        wlq = integrate(wlqd_new, wlqd, wlq, int_step)
        wlqd = wlqd_new
        thtpb = wlq
        thtpbd = wlqd
        tpb = mat2tr(psipb, thtpb)
        ththbc, phihbc = _sensor_ir_uthpb(psipb, thtpb)
        ththb = ththbc + biast + randt
        phihb = phihbc + biasp + randp
        thb = _sensor_ir_thb(ththb, phihb)

        if mseek == 4:
            ibreak = 0
            phihbd = -thtpbd * sin(psipb)
            eh = sqrt(ehy * ehy + ehz * ehz)
            if fabs(ththb) > trtht:
                trcond = 6
                ibreak = 1
            elif fabs(thtpbd) > trthtd:
                trcond = 7
                ibreak = 1
            elif fabs(phihbd) > trphid:
                trcond = 8
                ibreak = 1
            elif eh > trate:
                trcond = 9
                ibreak = 1
            if ibreak == 1:
                mseek = 2
                mguid = 40
            if dbtk < dblind:
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
        store.set("epy", epy)
        store.set("epz", epz)
        store.set("ththb", ththb)
        store.set("phihb", phihb)
        store.set("TPB", tpb)
        store.set("EAHH", eahh)
        store.set("EPHH", ephh)
        store.set("EAPH", eaph)
        return mseek, mguid, thtpb, psipb, sigdy, sigdz, ehz, ehy, thb, trcond

    def execute(self, vehicle, ctx):
        store = vehicle.store
        tgt_num = store.get("tgt_num")
        mseek = store.get("mseek")
        skr_dyn = store.get("skr_dyn")
        isets1 = store.get("isets1")
        racq = store.get("racq")
        dtimac = store.get("dtimac")
        epchac = store.get("epchac")
        timeac = store.get("timeac")
        thb = np.array(store.get("THB"), dtype=float, copy=True)
        time = store.get("time")
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        trcond = store.get("trcond") if "trcond" in store else 0
        mguid = store.get("mguid") if "mguid" in store else 0

        stel = np.zeros(3)
        vtel = np.zeros(3)
        tgt_com_slot = 0
        count = 0
        for i, packet in enumerate(ctx.combus or ()):
            if packet is None or packet.type != "TARGET3":
                continue
            count += 1
            if count == tgt_num:
                tgt_com_slot = i
                stel = _vec3(packet.vars, "SAEL", "SBEL")
                vtel = _vec3(packet.vars, "VAEL", "VBEL")
                break

        sbtl = sbel - stel
        dbtk = float(polar_from_cart(sbtl)[0])
        thtpb = 0.0
        psipb = 0.0
        sigdy = 0.0
        sigdz = 0.0
        sigdpy = 0.0
        sigdpz = 0.0
        ehz = 0.0
        ehy = 0.0
        int_step = ctx.int_step

        if mseek not in (0, 2, 3, 4, 5):
            raise ValueError(f"unknown mseek {mseek}")
        if skr_dyn not in (0, 1) and mseek not in (0, 5):
            raise ValueError(f"unknown skr_dyn {skr_dyn}")

        if mseek == 2:
            isets1 = 1
            if dbtk < racq:
                mseek = 3
        if mseek == 3:
            if isets1 == 1:
                thtpb, psipb, sigdy, sigdz = self.sensor_ir_kin(
                    vehicle, sbtl, vtel, dbtk
                )
                ththb, phihb = _sensor_ir_uthpb(psipb, thtpb)
                thb = _sensor_ir_thb(ththb, phihb)
                isets1 = 0
                epchac = time
            if skr_dyn == 1:
                (
                    mseek,
                    mguid,
                    thtpb,
                    psipb,
                    sigdy,
                    sigdz,
                    ehz,
                    ehy,
                    thb,
                    trcond,
                ) = self.sensor_ir_dyn(
                    vehicle, sbtl, dbtk, int_step, mseek, mguid, thb, trcond
                )
                timeac = time - epchac
                if timeac > dtimac:
                    fovyaw = store.get("fovyaw")
                    fovpitch = store.get("fovpitch")
                    if fabs(ehz) <= fovyaw and fabs(ehy) <= fovpitch:
                        mseek = 4
                    else:
                        trcond = 5
            else:
                thtpb, psipb, sigdy, sigdz = self.sensor_ir_kin(
                    vehicle, sbtl, vtel, dbtk
                )
                timeac = time - epchac
                if timeac > dtimac:
                    mseek = 4
        if mseek == 4:
            if skr_dyn == 1:
                (
                    mseek,
                    mguid,
                    thtpb,
                    psipb,
                    sigdy,
                    sigdz,
                    ehz,
                    ehy,
                    thb,
                    trcond,
                ) = self.sensor_ir_dyn(
                    vehicle, sbtl, dbtk, int_step, mseek, mguid, thb, trcond
                )
            else:
                thtpb, psipb, sigdy, sigdz = self.sensor_ir_kin(
                    vehicle, sbtl, vtel, dbtk
                )
            sigdpy = sigdy
            sigdpz = sigdz

        store.set("STEL", stel)
        store.set("VTEL", vtel)
        store.set("tgt_com_slot", tgt_com_slot)
        store.set("thtpb", thtpb)
        store.set("psipb", psipb)
        store.set("sigdpy", sigdpy)
        store.set("sigdpz", sigdpz)
        store.set("epchac", epchac)
        store.set("timeac", timeac)
        store.set("THB", thb)
        store.set("mseek", mseek)
        store.set("isets1", isets1)
        store.set("dbtk", dbtk)
        store.set("thtpbx", thtpb * DEG)
        store.set("psipbx", psipb * DEG)
        store.set("SBTL", sbtl)
        if "trcond" in store:
            store.set("trcond", trcond)
        if "mguid" in store:
            store.set("mguid", mguid)
