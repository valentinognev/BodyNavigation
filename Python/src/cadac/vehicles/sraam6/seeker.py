from math import acos, atan2, cos, sin, tan

import numpy as np

from cadac.constants import DEG
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr, polar_from_cart

SMALL = 1e-7

_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_ZEROS3 = (0.0, 0.0, 0.0)
_ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


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


class Sraam6Seeker:
    name = "seeker"

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

    def initialize(self, vehicle, ctx):
        pass

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
        woeb = tbl @ (_skew(utbl) @ vtbl) * dum1
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
        vtel = vehicle.store.get("VTEL")
        thtpb, psipb, sigdy, sigdz = self.seeker_kin(vehicle, sbtl, vtel, dbt)
        ehy = 0.0
        ehz = 0.0
        ththb, phihb = self.seeker_uthpb(psipb, thtpb)
        thb = self.seeker_thb(ththb, phihb)
        return mseek, mguid, thtpb, psipb, sigdy, sigdz, ehz, ehy, thb

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
