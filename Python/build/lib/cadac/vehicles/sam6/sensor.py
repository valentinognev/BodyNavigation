from math import acos, atan2, cos, fabs, log10, sin, sqrt, tan

import numpy as np

from cadac.constants import DEG, PI, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr, polar_from_cart, skew

SMALL = 1e-7
KBOLTZ = 1.38e-23


def _sign(variable):
    if variable < 0.0:
        return -1
    return 1


def _markov(sigma, bcor, time, int_step, value_saved):
    return 0.0


def _missile_index(ctx):
    combus = ctx.combus
    if not combus:
        return 0
    return sum(
        1
        for i, packet in enumerate(combus)
        if packet.type == "MISSILE6" and i < ctx.vehicle_slot
    )


def _pair_target(ctx, mtarget):
    type_name = "ROCKET5" if mtarget == 1 else "AIRCRAFT3"
    combus = ctx.combus
    if not combus:
        raise ValueError(f"no combus for mtarget {mtarget}")
    indices = [i for i, packet in enumerate(combus) if packet.type == type_name]
    if not indices:
        raise ValueError(f"no {type_name} packet")
    k = _missile_index(ctx)
    if k >= len(indices):
        raise ValueError(f"no {type_name} for missile index {k}")
    fst_tgt_slot = indices[0]
    tgt_slot = indices[k]
    packet = combus[tgt_slot]
    stel = np.asarray(packet.vars["SAEL"], dtype=float).copy()
    vtel = np.asarray(packet.vars["VAEL"], dtype=float).copy()
    dta = float(packet.vars.get("dta", 0.0))
    return stel, vtel, dta, tgt_slot, fst_tgt_slot


class Sam6Sensor:
    name = "sensor"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        plot = ("plot",)
        plot_scrn = ("plot", "scrn")
        for field in (
            Field("STEL", zeros3, "vec", "out", "sensor"),
            Field("VTEL", zeros3, "vec", "out", "sensor"),
            Field("tgt_slot", 0, "int", "out", "sensor"),
            Field("mseek", 0, "int", "data", "sensor"),
            Field("skr_dyn", 0, "int", "data", "sensor"),
            Field("isets1", 0, "int", "init", "sensor"),
            Field("epchac", 0.0, "real", "init", "sensor"),
            Field("fst_tgt_slot", 0, "int", "save", "sensor"),
            Field("mtarget", 0, "int", "data", "sensor"),
            Field("dbtk", 0.0, "real", "diag", "sensor", plot_scrn),
            Field("thtpb", 0.0, "real", "out", "sensor"),
            Field("psipb", 0.0, "real", "out", "sensor"),
            Field("SBTL", zeros3, "vec", "diag", "sensor"),
            Field("thtpbx", 0.0, "real", "diag", "sensor", plot),
            Field("psipbx", 0.0, "real", "diag", "sensor", plot),
            Field("dblind", 0.0, "real", "data", "sensor"),
            Field("ibreak", 0, "int", "init", "sensor"),
            Field("racq_ir", 0.0, "real", "data", "sensor"),
            Field("dtimac_ir", 0.0, "real", "data", "sensor"),
            Field("ehz", 0.0, "real", "diag", "sensor"),
            Field("ehy", 0.0, "real", "diag", "sensor"),
            Field("trtht", 0.0, "real", "data", "sensor"),
            Field("trthtd", 0.0, "real", "data", "sensor"),
            Field("trphid", 0.0, "real", "data", "sensor"),
            Field("trate", 0.0, "real", "data", "sensor"),
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
            Field("fovyaw_ir", 0.0, "real", "data", "sensor"),
            Field("fovpitch_ir", 0.0, "real", "data", "sensor"),
            Field("daim", 0.0, "real", "data", "sensor"),
            Field("BIASAI", zeros3, "vec", "data", "sensor"),
            Field("BIASSC", zeros3, "vec", "data", "sensor"),
            Field("RANDSC", zeros3, "vec", "data", "sensor"),
            Field("epy", 0.0, "real", "diag", "sensor", plot),
            Field("dta", 0.0, "real", "out", "sensor", plot),
            Field("epz", 0.0, "real", "diag", "sensor", plot),
            Field("ththb", 0.0, "real", "diag", "sensor"),
            Field("phihb", 0.0, "real", "diag", "sensor"),
            Field("TPB", zeros33, "mat", "init", "sensor"),
            Field("THB", zeros33, "mat", "init", "sensor"),
            Field("dvbtc", 0.0, "real", "diag", "sensor"),
            Field("EAHH", zeros3, "vec", "diag", "sensor"),
            Field("EPHH", zeros3, "vec", "diag", "sensor"),
            Field("EAPH", zeros3, "vec", "diag", "sensor"),
            Field("sigdy", 0.0, "real", "out", "sensor", plot),
            Field("sigdz", 0.0, "real", "out", "sensor", plot),
            Field("biaseh", 0.0, "real", "data", "sensor"),
            Field("randeh", 0.0, "real", "data", "sensor"),
            Field("timeac", 0.0, "real", "save", "sensor"),
            Field("racq_rf", 0.0, "real", "data", "sensor"),
            Field("dtimac_rf", 0.0, "real", "data", "sensor"),
            Field("temp_resx", 290.0, "real", "data", "sensor"),
            Field("biasaz", 0.0, "real", "data", "sensor"),
            Field("biasel", 0.0, "real", "data", "sensor"),
            Field("freqghz", 0.0, "real", "data", "sensor"),
            Field("rngegw", 0.0, "real", "data", "sensor"),
            Field("thta_3db", 0.0, "real", "data", "sensor"),
            Field("powrs", 0.0, "real", "data", "sensor"),
            Field("gainsdb", 0.0, "real", "data", "sensor"),
            Field("gainmdb", 0.0, "real", "data", "sensor"),
            Field("tgt_rcs", 0.0, "real", "data", "sensor"),
            Field("rlatmodb", 0.0, "real", "data", "sensor"),
            Field("rltotldb", 0.0, "real", "data", "sensor"),
            Field("dwltm", 0.0, "real", "data", "sensor"),
            Field("rnoisfgdb", 0.0, "real", "data", "sensor"),
            Field("plc5", 0.0, "real", "data", "sensor"),
            Field("plc4", 0.0, "real", "data", "sensor"),
            Field("plc3", 0.0, "real", "data", "sensor"),
            Field("plc2", 0.0, "real", "data", "sensor"),
            Field("plc1", 0.0, "real", "data", "sensor"),
            Field("plc0", 0.0, "real", "data", "sensor"),
            Field("biasgl1", 0.0, "real", "data", "sensor"),
            Field("biasgl2", 0.0, "real", "data", "sensor"),
            Field("biasgl3", 0.0, "real", "data", "sensor"),
            Field("randgl1", 0.0, "real", "data", "sensor"),
            Field("randgl2", 0.0, "real", "data", "sensor"),
            Field("randgl3", 0.0, "real", "data", "sensor"),
            Field("fovlim_rfx", 0.0, "real", "data", "sensor"),
            Field("forlim_rfx", 0.0, "real", "data", "sensor"),
            Field("gain_rf", 0.0, "real", "data", "sensor"),
            Field("aztbx", 0.0, "real", "diag", "sensor", plot),
            Field("eltbx", 0.0, "real", "diag", "sensor", plot),
            Field("dab", 0.0, "real", "out", "sensor", plot),
            Field("ddab", 0.0, "real", "out", "sensor", plot),
            Field("pwr_loss_db", 0.0, "real", "diag", "sensor"),
            Field("snr_db", 0.0, "real", "diag", "sensor", plot),
            Field("onax", 0.0, "real", "diag", "sensor", plot),
            Field("epaz_rf_saved", 0.0, "real", "save", "sensor"),
            Field("epel_rf_saved", 0.0, "real", "save", "sensor"),
            Field("range_rf_saved", 0.0, "real", "save", "sensor"),
            Field("rate_rf_saved", 0.0, "real", "save", "sensor"),
            Field("epsic", 0.0, "real", "diag", "sensor", plot),
            Field("ethtc", 0.0, "real", "diag", "sensor", plot),
            Field("mepsit", 0.0, "real", "diag", "sensor"),
            Field("methtt", 0.0, "real", "diag", "sensor"),
            Field("psisb", 0.0, "real", "state", "sensor", plot),
            Field("psisbd", 0.0, "real", "state", "sensor"),
            Field("thtsb", 0.0, "real", "state", "sensor", plot),
            Field("thtsbd", 0.0, "real", "state", "sensor"),
            Field("lamdqb", 0.0, "real", "out", "sensor", plot),
            Field("lamdrb", 0.0, "real", "out", "sensor", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mseek = store.get("mseek")
        if mseek == 0:
            return
        mtarget = store.get("mtarget")
        if mtarget not in (1, 2):
            raise ValueError(f"unknown mtarget {mtarget}")
        skr_type = mseek // 10
        skr_mode = mseek % 10
        if skr_type not in (1, 2):
            raise ValueError(f"unknown skr_type {skr_type}")

        skr_dyn = store.get("skr_dyn")
        racq_ir = store.get("racq_ir")
        dtimac_ir = store.get("dtimac_ir")
        fovyaw_ir = store.get("fovyaw_ir")
        fovpitch_ir = store.get("fovpitch_ir")
        racq_rf = store.get("racq_rf")
        dtimac_rf = store.get("dtimac_rf")
        forlim_rfx = store.get("forlim_rfx")
        isets1 = store.get("isets1")
        epchac = store.get("epchac")
        timeac = store.get("timeac")
        thb = np.asarray(store.get("THB"), dtype=float).copy()
        epsic = store.get("epsic")
        ethtc = store.get("ethtc")
        time = store.get("time")
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        trcond = store.get("trcond")
        mguide = store.get("mguide") if "mguide" in store else 0
        int_step = ctx.int_step

        ehz = 0.0
        ehy = 0.0
        thtpb = 0.0
        psipb = 0.0
        sigdy = 0.0
        sigdz = 0.0
        ththb = 0.0
        phihb = 0.0
        dab = 0.0
        ddab = 0.0
        aztbx = 0.0
        eltbx = 0.0
        lamdrb = 0.0
        lamdqb = 0.0

        stel, vtel, dta, tgt_slot, fst_tgt_slot = _pair_target(ctx, mtarget)
        sbtl = sbel - stel
        dbtk = float(np.linalg.norm(sbtl))

        if skr_type == 1:
            if skr_mode == 2:
                isets1 = 1
                if dbtk < racq_rf:
                    skr_mode = 3
            if skr_mode == 3:
                if isets1 == 1:
                    isets1 = 0
                    epchac = time
                    thtpb, psipb, sigdy, sigdz, lamdrb, lamdqb, ddab = self.sensor_kin(
                        vehicle, sbtl, vtel, dbtk
                    )
                if skr_dyn == 1:
                    lamdrb, lamdqb, dab, ddab, ethtc, epsic, aztbx, eltbx = (
                        self.sensor_rf_dyn(vehicle, sbtl, int_step)
                    )
                    timeac = time - epchac
                    if timeac > dtimac_rf:
                        if fabs(aztbx) <= forlim_rfx and fabs(eltbx) <= forlim_rfx:
                            skr_mode = 4
                        else:
                            trcond = 5
                else:
                    thtpb, psipb, sigdy, sigdz, lamdrb, lamdqb, ddab = self.sensor_kin(
                        vehicle, sbtl, vtel, dbtk
                    )
                    timeac = time - epchac
                    if timeac > dtimac_rf:
                        skr_mode = 4
            elif skr_mode == 4:
                if skr_dyn == 1:
                    lamdrb, lamdqb, dab, ddab, ethtc, epsic, aztbx, eltbx = (
                        self.sensor_rf_dyn(vehicle, sbtl, int_step)
                    )
                else:
                    thtpb, psipb, sigdy, sigdz, lamdrb, lamdqb, ddab = self.sensor_kin(
                        vehicle, sbtl, vtel, dbtk
                    )
            thtpbx = thtpb * DEG
            psipbx = psipb * DEG
        else:
            thtpbx = 0.0
            psipbx = 0.0

        if skr_type == 2:
            if skr_mode == 2:
                isets1 = 1
                if dbtk < racq_ir:
                    skr_mode = 3
            if skr_mode == 3:
                if isets1 == 1:
                    isets1 = 0
                    epchac = time
                    thtpb, psipb, sigdy, sigdz, lamdrb, lamdqb, ddab = self.sensor_kin(
                        vehicle, sbtl, vtel, dbtk
                    )
                    ththb, phihb = self.sensor_ir_uthpb(psipb, thtpb)
                    thb = self.sensor_ir_thb(ththb, phihb)
                if skr_dyn == 1:
                    mseek, mguide, thtpb, psipb, sigdy, sigdz, ehz, ehy, thb = (
                        self.sensor_ir_dyn(
                            vehicle, mseek, mguide, thb, sbtl, dbtk, int_step
                        )
                    )
                    timeac = time - epchac
                    if timeac > dtimac_ir:
                        if fabs(ehz) <= fovyaw_ir and fabs(ehy) <= fovpitch_ir:
                            skr_mode = 4
                        else:
                            trcond = 5
                else:
                    thtpb, psipb, sigdy, sigdz, lamdrb, lamdqb, ddab = self.sensor_kin(
                        vehicle, sbtl, vtel, dbtk
                    )
                    timeac = time - epchac
                    if timeac > dtimac_ir:
                        skr_mode = 4
            if skr_mode == 4:
                if skr_dyn == 1:
                    mseek, mguide, thtpb, psipb, sigdy, sigdz, ehz, ehy, thb = (
                        self.sensor_ir_dyn(
                            vehicle, mseek, mguide, thb, sbtl, dbtk, int_step
                        )
                    )
                else:
                    thtpb, psipb, sigdy, sigdz, lamdrb, lamdqb, ddab = self.sensor_kin(
                        vehicle, sbtl, vtel, dbtk
                    )
            thtpbx = thtpb * DEG
            psipbx = psipb * DEG

        mseek = 10 * skr_type + skr_mode

        store.set("STEL", stel)
        store.set("VTEL", vtel)
        store.set("tgt_slot", tgt_slot)
        store.set("trcond", trcond)
        store.set("mseek", mseek)
        store.set("dta", dta)
        store.set("isets1", isets1)
        store.set("epchac", epchac)
        store.set("fst_tgt_slot", fst_tgt_slot)
        store.set("timeac", timeac)
        store.set("dbtk", dbtk)
        store.set("thtpbx", thtpbx)
        store.set("psipbx", psipbx)
        store.set("SBTL", sbtl)
        store.set("ddab", ddab)
        store.set("lamdqb", lamdqb)
        store.set("lamdrb", lamdrb)
        store.set("epsic", epsic)
        store.set("ethtc", ethtc)
        store.set("aztbx", aztbx)
        store.set("eltbx", eltbx)
        store.set("dab", dab)
        if "mguide" in store:
            store.set("mguide", mguide)
        store.set("thtpb", thtpb)
        store.set("psipb", psipb)
        store.set("sigdy", sigdy)
        store.set("sigdz", sigdz)
        store.set("THB", thb)
        store.set("ehz", ehz)
        store.set("ehy", ehy)

    def sensor_kin(self, vehicle, sbtl, vtel, dbtk):
        store = vehicle.store
        tbl = np.asarray(store.get("TBL"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)
        stbl = sbtl * (-1.0)
        stbb = tbl @ stbl
        utbl = stbl / dbtk
        vtbl = vtel - vbel
        ddab = -fabs(float(utbl @ vtbl))
        woeb = tbl @ skew(utbl) @ vtbl / dbtk
        lamdqb = float(woeb[1])
        lamdrb = float(woeb[2])
        polar = polar_from_cart(stbb)
        psipb = float(polar[1])
        thtpb = float(polar[2])
        tpb = mat2tr(psipb, thtpb)
        woep = tpb @ woeb
        sigdy = float(woep[1])
        sigdz = float(woep[2])
        store.set("psisb", psipb)
        store.set("thtsb", thtpb)
        return thtpb, psipb, sigdy, sigdz, lamdrb, lamdqb, ddab

    def sensor_rf_dyn(self, vehicle, sbtl, int_step):
        store = vehicle.store
        temp_resx = store.get("temp_resx")
        biasaz = store.get("biasaz")
        biasel = store.get("biasel")
        freqghz = store.get("freqghz")
        rngegw = store.get("rngegw")
        thta_3db = store.get("thta_3db")
        powrs = store.get("powrs")
        gainsdb = store.get("gainsdb")
        gainmdb = store.get("gainmdb")
        tgt_rcs = store.get("tgt_rcs")
        rlatmodb = store.get("rlatmodb")
        rltotldb = store.get("rltotldb")
        dwltm = store.get("dwltm")
        rnoisfgdb = store.get("rnoisfgdb")
        plc5 = store.get("plc5")
        plc4 = store.get("plc4")
        plc3 = store.get("plc3")
        plc2 = store.get("plc2")
        plc1 = store.get("plc1")
        plc0 = store.get("plc0")
        gain_rf = store.get("gain_rf")
        psisb = store.get("psisb")
        psisbd = store.get("psisbd")
        thtsb = store.get("thtsb")
        thtsbd = store.get("thtsbd")
        epaz_saved = store.get("epaz_rf_saved")
        epel_saved = store.get("epel_rf_saved")
        range_saved = store.get("range_rf_saved")
        rate_saved = store.get("rate_rf_saved")
        time = store.get("time")
        tbl = np.asarray(store.get("TBL"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)
        vtel = np.asarray(store.get("VTEL"), dtype=float)
        wbecb = np.asarray(store.get("WBECB"), dtype=float)

        stbl = sbtl * (-1.0)
        stbb = tbl @ stbl
        polar = polar_from_cart(stbb)
        aztb = float(polar[1])
        eltb = float(polar[2])
        tpb = mat2tr(aztb, eltb)
        aztbx = aztb * DEG
        eltbx = eltb * DEG

        sotlc = self.sensor_rf_glint(vehicle)
        soblc = sotlc - sbtl
        dobc = float(np.linalg.norm(soblc))
        uoblc = soblc / dobc
        vtbl = vtel - vbel
        rdobc = float(uoblc @ vtbl)

        cpsisb = cos(psisb)
        spsisb = sin(psisb)
        cthtsb = cos(thtsb)
        sthtsb = sin(thtsb)
        tsb = np.array(
            [
                [cpsisb * cthtsb, spsisb, -cpsisb * sthtsb],
                [-spsisb * cthtsb, cpsisb, spsisb * sthtsb],
                [sthtsb, 0.0, cthtsb],
            ],
            dtype=float,
        )
        sobs = tsb @ tbl @ soblc
        ethtc = atan2(-float(sobs[2]), float(sobs[0]))
        epsic = atan2(float(sobs[1]), float(sobs[0]))

        onax = sqrt(ethtc * ethtc + epsic * epsic) * DEG
        k_noise = PI / 2.0
        wvelngth = (2.998e8) / (freqghz * 10.0e8)
        gains = 10.0 ** (gainsdb / 10.0)
        gainm = 10.0 ** (gainmdb / 10.0)
        rnoisfg = 10.0 ** (rnoisfgdb / 10.0)
        rlatmo = 10.0 ** (rlatmodb / 10.0)
        rltotl = 10.0 ** (rltotldb / 10.0)
        pwr_loss = (
            -(plc5 * 10.0e-11) * onax**5
            + (plc4 * 10.0e-9) * onax**4
            - (plc3 * 10.0e-7) * onax**3
            + (plc2 * 10.0e-5) * onax**2
            - (plc1 * 10.0e-3) * onax
            + plc0
        )
        if pwr_loss > 1:
            pwr_loss = 1
        powrt = powrs - pwr_loss
        ps = (
            powrt
            * gains
            * rlatmo
            * rlatmo
            * tgt_rcs
            * gainm
            * wvelngth
            * wvelngth
            / ((4.0 * PI) ** 3 * dobc**4 * rltotl)
        )
        pn = KBOLTZ * temp_resx * rnoisfg * (1.0 / dwltm)
        snr = ps / pn

        sigma_mp = sqrt(dwltm / int_step) * (thta_3db * RAD) / (k_noise * sqrt(snr))
        nepsi_mp = _markov(sigma_mp, 100.0, time, int_step, epaz_saved)
        netht_mp = _markov(sigma_mp, 100.0, time, int_step, epel_saved)
        mepsit = biasaz + nepsi_mp
        methtt = biasel + netht_mp

        lamdrs = gain_rf * (epsic + mepsit)
        lamdqs = gain_rf * (ethtc + methtt)
        woes = np.array([0.0, lamdqs, lamdrs], dtype=float)
        woeb = tsb.T @ woes
        lamdqb = float(woeb[1])
        lamdrb = float(woeb[2])

        sigma_range = sqrt(dwltm / int_step) * rngegw / (k_noise * sqrt(snr))
        eps_range = _markov(sigma_range, 10.0, time, int_step, range_saved)
        dab = dobc + eps_range

        vgw = 200 * wvelngth / 2
        sig_range_rate = sqrt(dwltm / int_step) * vgw / (k_noise * sqrt(snr))
        eps_range_rate = _markov(sig_range_rate, 10.0, time, int_step, rate_saved)
        ddab = rdobc + eps_range_rate

        pwr_loss_db = 10.0 * log10(pwr_loss)
        snr_db = 10.0 * log10(snr)

        wbes = tsb @ wbecb
        wbes2 = float(wbes[1])
        wbes3 = float(wbes[2])
        psisbd = lamdrs - wbes3
        thtsbd = (lamdqs - wbes2) / cpsisb
        psisbd_new = psisbd
        psisb = float(integrate(psisbd_new, psisbd, psisb, int_step))
        psisbd = psisbd_new
        thtsbd_new = thtsbd
        thtsb = float(integrate(thtsbd_new, thtsbd, thtsb, int_step))
        thtsbd = thtsbd_new

        woep = tpb @ woeb
        sigdy = float(woep[1])
        sigdz = float(woep[2])

        store.set("psisb", psisb)
        store.set("psisbd", psisbd)
        store.set("thtsb", thtsb)
        store.set("thtsbd", thtsbd)
        store.set("epaz_rf_saved", epaz_saved)
        store.set("epel_rf_saved", epel_saved)
        store.set("range_rf_saved", range_saved)
        store.set("rate_rf_saved", rate_saved)
        store.set("sigdy", sigdy)
        store.set("sigdz", sigdz)
        store.set("pwr_loss_db", pwr_loss_db)
        store.set("snr_db", snr_db)
        store.set("onax", onax)
        store.set("mepsit", mepsit)
        store.set("methtt", methtt)
        return lamdrb, lamdqb, dab, ddab, ethtc, epsic, aztbx, eltbx

    def sensor_rf_glint(self, vehicle):
        store = vehicle.store
        ttl = np.eye(3)
        biasgl = np.array(
            [store.get("biasgl1"), store.get("biasgl2"), store.get("biasgl3")],
            dtype=float,
        )
        randgl = np.array(
            [store.get("randgl1"), store.get("randgl2"), store.get("randgl3")],
            dtype=float,
        )
        sott = randgl + biasgl
        return ttl.T @ sott

    def sensor_ir_dyn(self, vehicle, mseek, mguide, thb, sbtl, dbtk, int_step):
        store = vehicle.store
        dblind = store.get("dblind")
        ibreak = store.get("ibreak")
        trtht = store.get("trtht")
        trthtd = store.get("trthtd")
        trphid = store.get("trphid")
        trate = store.get("trate")
        gk = store.get("gk")
        zetak = store.get("zetak")
        wnk = store.get("wnk")
        biast = store.get("biast")
        randt = store.get("randt")
        biasp = store.get("biasp")
        randp = store.get("randp")
        biaseh = store.get("biaseh")
        randeh = store.get("randeh")
        tpb = np.asarray(store.get("TPB"), dtype=float).copy()
        tbl = np.asarray(store.get("TBL"), dtype=float)
        ttl = np.eye(3)
        trcond = store.get("trcond")
        wbecb = np.asarray(store.get("WBECB"), dtype=float)
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
        sbth = thl @ sbtl
        sath = self.sensor_ir_aimp(thl, ttl, dbtk, vehicle)
        sabh = sath - sbth
        sabh1 = float(sabh[0])
        sabh2 = float(sabh[1])
        sabh3 = float(sabh[2])
        ey = atan2(-sabh3, sabh1)
        ez = atan2(sabh2, sabh1)
        ehy = ey + biaseh + randeh
        ehz = ez + biaseh + randeh
        eahh = np.array([0.0, ehz, -ehy], dtype=float)
        tbh = thb.T
        tph = tpb @ tbh
        thp = tph.T
        u1pp = np.array([1.0, 0.0, 0.0], dtype=float)
        u1hh = np.array([1.0, 0.0, 0.0], dtype=float)
        ephh = thp @ u1pp - u1hh
        eaph = eahh - ephh
        eapp = tph @ eaph
        epy = -float(eapp[2])
        epz = float(eapp[1])

        wsq = wnk * wnk
        gg = gk * wsq
        wlr1d_new = wlr2
        wlr1 = float(integrate(wlr1d_new, wlr1d, wlr1, int_step))
        wlr1d = wlr1d_new
        wlr2d_new = gg * epz - 2.0 * zetak * wnk * wlr1d - wsq * wlr1
        wlr2 = float(integrate(wlr2d_new, wlr2d, wlr2, int_step))
        wlr2d = wlr2d_new
        wlq1d_new = wlq2
        wlq1 = float(integrate(wlq1d_new, wlq1d, wlq1, int_step))
        wlq1d = wlq1d_new
        wlq2d_new = gg * epy - 2.0 * zetak * wnk * wlq1d - wsq * wlq1
        wlq2 = float(integrate(wlq2d_new, wlq2d, wlq2, int_step))
        wlq2d = wlq2d_new
        sigdz = wlr1
        sigdy = wlq1

        wbep = tpb @ wbecb
        wbep2 = float(wbep[1])
        wbep3 = float(wbep[2])
        wlrd_new = wlr1 - wbep3
        wlr = float(integrate(wlrd_new, wlrd, wlr, int_step))
        wlrd = wlrd_new
        psipb = wlr
        psipbd = wlrd
        wlqd_new = wlq1 - wbep2
        wlq = float(integrate(wlqd_new, wlqd, wlq, int_step))
        wlqd = wlqd_new
        thtpb = wlq
        thtpbd = wlqd
        tpb = mat2tr(psipb, thtpb)

        ththbc, phihbc = self.sensor_ir_uthpb(psipb, thtpb)
        ththb = ththbc + biast + randt
        phihb = phihbc + biasp + randp
        thb = self.sensor_ir_thb(ththb, phihb)

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
                mguide = 50
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
        store.set("trcond", trcond)
        store.set("epy", epy)
        store.set("epz", epz)
        store.set("ththb", ththb)
        store.set("phihb", phihb)
        store.set("TPB", tpb)
        store.set("EAHH", eahh)
        store.set("EPHH", ephh)
        store.set("EAPH", eaph)
        store.set("ibreak", ibreak)
        return mseek, mguide, thtpb, psipb, sigdy, sigdz, ehz, ehy, thb

    def sensor_ir_aimp(self, thl, ttl, dbtk, vehicle):
        store = vehicle.store
        daim = store.get("daim")
        biasai = np.asarray(store.get("BIASAI"), dtype=float)
        biassc = np.asarray(store.get("BIASSC"), dtype=float)
        randsc = np.asarray(store.get("RANDSC"), dtype=float)
        tht = thl @ ttl.T
        if dbtk < daim:
            return tht @ biasai
        return tht @ (biassc + randsc)

    def sensor_ir_uthpb(self, psipb, thtpb):
        ththb = acos(cos(thtpb) * cos(psipb))
        sinpsi = sin(psipb)
        tantht = tan(thtpb)
        if fabs(sinpsi) and fabs(tantht) < SMALL:
            phihb = 0.0
        else:
            phihb = atan2(sinpsi, tantht)
        return ththb, phihb

    def sensor_ir_thb(self, tht, phi):
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
