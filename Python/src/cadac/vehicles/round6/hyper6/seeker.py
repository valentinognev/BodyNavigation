"""HYPER6 vehicle RF seeker — port of Hyper::seeker (seeker.cpp).

Distinct from Hyper6RadarSeeker (ground radar track files).
"""

from math import acos, atan2, exp, fabs, log10, sqrt

import numpy as np

from cadac.constants import DEG, PI, RAD
from cadac.kernel.state import Field
from cadac.math.frames import angle, cadac_inverse, cadac_matmul, cart_from_pol, mat2tr, polar_from_cart
from cadac.stoch import gauss

KBOLTZ = 1.38e-23
_ZEROS3 = (0.0, 0.0, 0.0)
_ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _markov(sigma, bcor, time, int_step, value_saved):
    """CADAC utility markov(); returns (value, updated value_saved)."""
    value = gauss(0.0, sigma)
    if time == 0.0:
        value_saved = value
    elif bcor != 0.0:
        dum = exp(-bcor * int_step)
        dumsqrd = dum * dum
        value = value * sqrt(1.0 - dumsqrd) + value_saved * dum
        value_saved = value
    return value, value_saved


def _ellipse(cov23):
    """Matrix::ellipse for 2x2 covariance (major, minor, angle)."""
    a11 = float(cov23[0, 0])
    a22 = float(cov23[1, 1])
    a12 = float(cov23[0, 1])
    a1122 = a11 + a22
    dum1 = a1122 * a1122 - 4.0 * (a11 * a22 - a12 * a12)
    dum2 = sqrt(dum1) if dum1 >= 0.0 else 0.0
    ama = (a1122 + dum2) / 2.0
    ami = (a1122 - dum2) / 2.0
    phi = 0.0
    if ama != ami and (a11 - ama) != 0.0:
        dum1 = -a12 / (a11 - ama)
        ak1 = sqrt(1.0 / (1.0 + dum1 * dum1))
        dum = dum1 * ak1
        if fabs(dum) > 1.0:
            dum = 1.0 if dum >= 0.0 else -1.0
        phi = acos(dum)
    return ama, ami, phi


def _download_satellite(store, combus):
    """Subscribe satellite STII/VTII from combus id t{sat_num} (Hyper::seeker)."""
    sat_num = int(store.get("sat_num"))
    target_id = f"t{sat_num}"
    stii = np.zeros(3)
    vtii = np.zeros(3)
    tgt_com_slot = 0
    packets = list(combus) if combus is not None else []
    sat_count = 0
    for i, packet in enumerate(packets):
        if getattr(packet, "type", "") != "SAT3":
            continue
        sat_count += 1
        ident = getattr(packet, "id", None) or f"t{sat_count}"
        if ident != target_id:
            continue
        tgt_com_slot = i
        vars_ = packet.vars
        if "sbii" in vars_:
            stii = np.asarray(vars_["sbii"], dtype=float).copy()
        elif "STII" in vars_:
            stii = np.asarray(vars_["STII"], dtype=float).copy()
        if "vbii" in vars_:
            vtii = np.asarray(vars_["vbii"], dtype=float).copy()
        elif "VTII" in vars_:
            vtii = np.asarray(vars_["VTII"], dtype=float).copy()
        break
    return stii, vtii, tgt_com_slot


class Hyper6Seeker:
    name = "seeker"

    def __init__(self):
        self.PMAT = np.zeros((8, 8))
        self.QQ = np.zeros((8, 8))
        self.RR = np.zeros((4, 4))
        self.FF = np.zeros((8, 8))
        self.PHI = np.zeros((8, 8))
        self.GAMDT = np.zeros((8, 3))
        self.XH = np.zeros(8)

    def define(self, vehicle):
        store = vehicle.store
        plot = ("plot",)
        scrn_plot = ("scrn", "plot")
        com = ("com",)
        for field in (
            Field("STII", _ZEROS3, "vec", "", "combus"),
            Field("VTII", _ZEROS3, "vec", "", "combus"),
            Field("tgt_com_slot", 0, "int", "out", "combus"),
            Field("mseek", 0, "int", "data/diag", "seeker", com),
            Field("skr_dyn", 0, "int", "data", "seeker"),
            Field("isets1", 0, "int", "init", "seeker"),
            Field("epchac", 0.0, "real", "init", "seeker"),
            Field("ibreak", 0, "int", "init", "seeker"),
            Field("temp_resx", 290.0, "real", "data", "seeker"),
            Field("dblind", 0.0, "real", "data", "seeker"),
            Field("biasaz", 0.0, "real", "data", "seeker"),
            Field("biasel", 0.0, "real", "data", "seeker"),
            Field("freqghz", 0.0, "real", "data", "seeker"),
            Field("rngegw", 0.0, "real", "data", "seeker"),
            Field("thta_3db", 0.0, "real", "data", "seeker"),
            Field("powrs", 0.0, "real", "data", "seeker"),
            Field("gainsdb", 0.0, "real", "data", "seeker"),
            Field("gainmdb", 0.0, "real", "data", "seeker"),
            Field("tgt_rcs", 0.0, "real", "data", "seeker"),
            Field("rlatmodb", 0.0, "real", "data", "seeker"),
            Field("rltotldb", 0.0, "real", "data", "seeker"),
            Field("dwltm", 0.0, "real", "data", "seeker"),
            Field("rnoisfgd", 0.0, "real", "data", "seeker"),
            Field("plc5", 0.0, "real", "data", "seeker"),
            Field("plc4", 0.0, "real", "data", "seeker"),
            Field("plc3", 0.0, "real", "data", "seeker"),
            Field("plc2", 0.0, "real", "data", "seeker"),
            Field("plc1", 0.0, "real", "data", "seeker"),
            Field("plc0", 0.0, "real", "data", "seeker"),
            Field("biasgl1", 0.0, "real", "data", "seeker"),
            Field("biasgl2", 0.0, "real", "data", "seeker"),
            Field("biasgl3", 0.0, "real", "data", "seeker"),
            Field("randgl1", 0.0, "real", "data", "seeker"),
            Field("randgl2", 0.0, "real", "data", "seeker"),
            Field("randgl3", 0.0, "real", "data", "seeker"),
            Field("fovlimx", 0.0, "real", "data", "seeker"),
            Field("racq", 0.0, "real", "data", "seeker"),
            Field("dtimac", 0.0, "real", "data", "seeker"),
            Field("esfta", 0.0, "real", "data", "seeker"),
            Field("esfte", 0.0, "real", "data", "seeker"),
            Field("dbtk", 0.0, "real", "diag", "seeker"),
            Field("timeac", 0.0, "real", "save", "seeker"),
            Field("azabx", 0.0, "real", "out", "seeker", plot),
            Field("elabx", 0.0, "real", "out", "seeker", plot),
            Field("dab", 0.0, "real", "out", "seeker", scrn_plot),
            Field("ddab", 0.0, "real", "out", "seeker", scrn_plot),
            Field("pwr_loss_db", 0.0, "real", "diag", "seeker"),
            Field("snr_db", 0.0, "real", "diag", "seeker"),
            Field("onax", 0.0, "real", "diag", "seeker"),
            Field("epazt", 0.0, "real", "diag", "seeker"),
            Field("epelt", 0.0, "real", "diag", "seeker"),
            Field("epaz_glnt", 0.0, "real", "diag", "seeker"),
            Field("epel_glnt", 0.0, "real", "diag", "seeker"),
            Field("init_filter", 1, "int", "init", "seeker"),
            Field("epchup", 0.0, "real", "init", "seeker"),
            Field("ppos_skr", 0.0, "real", "data", "seeker"),
            Field("pvel_skr", 0.0, "real", "data", "seeker"),
            Field("psfct", 0.0, "real", "data", "seeker"),
            Field("qpos_skr", 0.0, "real", "data", "seeker"),
            Field("qvel_skr", 0.0, "real", "data", "seeker"),
            Field("qsfct", 0.0, "real", "data", "seeker"),
            Field("razab", 0.0, "real", "data", "seeker"),
            Field("relab", 0.0, "real", "data", "seeker"),
            Field("rdab", 0.0, "real", "data", "seeker"),
            Field("rddab", 0.0, "real", "data", "seeker"),
            Field("dtimkf", 0.0, "real", "data", "seeker"),
            Field("factp_skr", 0.0, "real", "data", "seeker"),
            Field("factq_skr", 0.0, "real", "data", "seeker"),
            Field("factr_skr", 0.0, "real", "data", "seeker"),
            Field("flag_out", 1, "int", "init", "seeker"),
            Field("mupdt", 0, "int", "diag", "seeker"),
            Field("esfcta", 0.0, "real", "data", "seeker"),
            Field("esfcte", 0.0, "real", "data", "seeker"),
            Field("STBIK", _ZEROS3, "vec", "out", "seeker"),
            Field("VTBIK", _ZEROS3, "vec", "out", "seeker"),
            Field("SIGPOS", _ZEROS3, "vec", "diag", "seeker"),
            Field("SIGVEL", _ZEROS3, "vec", "diag", "seeker"),
            Field("semi_major", 0.0, "real", "diag", "seeker", plot),
            Field("semi_minor", 0.0, "real", "diag", "seeker", plot),
            Field("ellipse_anglx", 0.0, "real", "diag", "seeker"),
            Field("ESTBI", _ZEROS3, "vec", "diag", "seeker"),
            Field("EVTBI", _ZEROS3, "vec", "diag", "seeker"),
            Field("eaz", 0.0, "real", "diag", "seeker", plot),
            Field("eel", 0.0, "real", "diag", "seeker", plot),
            Field("edab", 0.0, "real", "diag", "seeker", plot),
            Field("eddab", 0.0, "real", "diag", "seeker", plot),
            Field("SXH_SKR", _ZEROS3, "vec", "save", "seeker"),
            Field("VXH_SKR", _ZEROS3, "vec", "save", "seeker"),
            Field("SFH", _ZEROS3, "vec", "save", "seeker"),
            Field("PMAT1", _ZEROS33, "mat", "save", "seeker"),
            Field("PMAT2", _ZEROS33, "mat", "save", "seeker"),
            Field("PMAT3", _ZEROS33, "mat", "save", "seeker"),
            Field("PMAT4", _ZEROS33, "mat", "save", "seeker"),
            Field("PMAT5", _ZEROS33, "mat", "save", "seeker"),
            Field("PMAT6", _ZEROS33, "mat", "save", "seeker"),
            Field("PMAT7", _ZEROS33, "mat", "save", "seeker"),
            Field("PMAT8", _ZEROS33, "mat", "save", "seeker"),
            Field("dtim", 0.0, "real", "save", "seeker"),
            Field("epaz_saved", 0.0, "real", "save", "seeker"),
            Field("epel_saved", 0.0, "real", "save", "seeker"),
            Field("range_saved", 0.0, "real", "save", "seeker"),
            Field("rate_saved", 0.0, "real", "save", "seeker"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mseek = int(store.get("mseek"))
        if mseek not in (0, 2, 3, 4, 5):
            raise ValueError(f"unknown mseek {mseek}")
        skr_dyn = int(store.get("skr_dyn"))
        isets1 = int(store.get("isets1"))
        fovlimx = store.get("fovlimx")
        racq = store.get("racq")
        dtimac = store.get("dtimac")
        stbik = np.asarray(store.get("STBIK"), dtype=float).copy()
        vtbik = np.asarray(store.get("VTBIK"), dtype=float).copy()
        epchac = store.get("epchac")
        timeac = store.get("timeac")
        time = store.get("time")
        tbi = np.asarray(store.get("TBI"), dtype=float)
        sbii = np.asarray(store.get("SBII"), dtype=float)
        vbii = np.asarray(store.get("VBII"), dtype=float)
        mguide = int(store.get("mguide"))
        int_step = ctx.int_step

        stii, vtii, tgt_com_slot = _download_satellite(store, ctx.combus)
        sbti = sbii - stii
        dbtk = float(np.linalg.norm(sbti))

        azab = 0.0
        elab = 0.0
        dab = 0.0
        ddab = 0.0

        if mseek == 2:
            isets1 = 1
            if dbtk < racq:
                mseek = 3

        if mseek == 3:
            if isets1 == 1:
                isets1 = 0
                epchac = time
            if skr_dyn == 1:
                azab, elab, dab, ddab, mseek, mguide = self.seeker_rf(
                    vehicle, mseek, mguide, fovlimx, sbti, int_step
                )
                stbik, vtbik = self.seeker_filter(
                    vehicle, stbik, vtbik, azab, elab, dab, ddab, mseek, int_step
                )
                if fabs(azab) <= fovlimx * RAD and fabs(elab) <= fovlimx * RAD:
                    timeac = time - epchac
                    if timeac > dtimac:
                        mseek = 4
            else:
                azab, elab, dab, ddab = self.seeker_kin(vehicle, sbti, vtii, vbii, tbi)
                stbbk = cart_from_pol(dab, azab, elab)
                stbik = cadac_matmul(tbi.T.copy(), stbbk)
                vtbik = vtii - vbii
                timeac = time - epchac
                if timeac > dtimac:
                    mseek = 4

        if mseek == 4:
            if skr_dyn == 1:
                azab, elab, dab, ddab, mseek, mguide = self.seeker_rf(
                    vehicle, mseek, mguide, fovlimx, sbti, int_step
                )
                stbik, vtbik = self.seeker_filter(
                    vehicle, stbik, vtbik, azab, elab, dab, ddab, mseek, int_step
                )
            else:
                azab, elab, dab, ddab = self.seeker_kin(vehicle, sbti, vtii, vbii, tbi)
                stbbk = cart_from_pol(dab, azab, elab)
                stbik = cadac_matmul(tbi.T.copy(), stbbk)
                vtbik = vtii - vbii

        azabx = azab * DEG
        elabx = elab * DEG

        store.set("epchac", epchac)
        store.set("timeac", timeac)
        store.set("STII", stii)
        store.set("VTII", vtii)
        store.set("tgt_com_slot", tgt_com_slot)
        store.set("mguide", mguide)
        store.set("STBIK", stbik)
        store.set("VTBIK", vtbik)
        store.set("mseek", mseek)
        store.set("isets1", isets1)
        store.set("dbtk", dbtk)
        store.set("azabx", azabx)
        store.set("elabx", elabx)
        store.set("dab", dab)
        store.set("ddab", ddab)

    def seeker_kin(self, vehicle, sbti, vtii, vbii, tbi):
        stbi = np.asarray(sbti, dtype=float) * (-1.0)
        stbb = cadac_matmul(tbi, stbi)
        dab = float(np.linalg.norm(stbi))
        utbi = stbi * (1.0 / dab)
        vtbi = np.asarray(vtii, dtype=float) - np.asarray(vbii, dtype=float)
        ddab = float(utbi[0] * vtbi[0] + utbi[1] * vtbi[1] + utbi[2] * vtbi[2])
        polar = polar_from_cart(stbb)
        azob = float(polar[1])
        elob = float(polar[2])
        return azob, elob, dab, ddab

    def seeker_glint(self, vehicle):
        store = vehicle.store
        tti = np.eye(3)
        sott = np.array(
            [
                store.get("randgl1") + store.get("biasgl1"),
                store.get("randgl2") + store.get("biasgl2"),
                store.get("randgl3") + store.get("biasgl3"),
            ],
            dtype=float,
        )
        return cadac_matmul(tti.T.copy(), sott)

    def seeker_rf(self, vehicle, mseek, mguide, fovlimx, sbti, int_step):
        store = vehicle.store
        temp_resx = store.get("temp_resx")
        dblind = store.get("dblind")
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
        rnoisfgd = store.get("rnoisfgd")
        plc5 = store.get("plc5")
        plc4 = store.get("plc4")
        plc3 = store.get("plc3")
        plc2 = store.get("plc2")
        plc1 = store.get("plc1")
        plc0 = store.get("plc0")
        esfta = store.get("esfta")
        esfte = store.get("esfte")
        epaz_saved = store.get("epaz_saved")
        epel_saved = store.get("epel_saved")
        range_saved = store.get("range_saved")
        rate_saved = store.get("rate_saved")
        time = store.get("time")
        tbi = np.asarray(store.get("TBI"), dtype=float)
        vbii = np.asarray(store.get("VBII"), dtype=float)
        vtii = np.asarray(store.get("VTII"), dtype=float)
        trcode = store.get("trcode")

        soti = self.seeker_glint(vehicle)
        sobic = soti - np.asarray(sbti, dtype=float)
        sobbc = cadac_matmul(tbi, sobic)
        polar = polar_from_cart(sobbc)
        dobc = float(polar[0])
        azobc = float(polar[1])
        elobc = float(polar[2])
        uobic = sobic * (1.0 / dobc)

        stbi = np.asarray(sbti, dtype=float) * (-1.0)
        stbb = cadac_matmul(tbi, stbi)
        polar = polar_from_cart(stbb)
        dob = float(polar[0])
        azob = float(polar[1])
        elob = float(polar[2])
        uobi = stbi * (1.0 / dob)

        u1b = np.array([1.0, 0.0, 0.0], dtype=float)
        uobbc = cadac_matmul(tbi, uobic)
        ona = angle(u1b, uobbc)
        onax = ona * DEG

        k_noise = PI / 2.0
        wvelngth = (2.998e8) / (freqghz * 10.0e8)
        gains = 10.0 ** (gainsdb / 10.0)
        gainm = 10.0 ** (gainmdb / 10.0)
        rnoisfg = 10.0 ** (rnoisfgd / 10.0)
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
        if pwr_loss > 1.0:
            pwr_loss = 1.0
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

        vtbi = vtii - vbii
        rdobc = float(uobic[0] * vtbi[0] + uobic[1] * vtbi[1] + uobic[2] * vtbi[2])

        epaz_glnt = azobc - azob
        epel_glnt = elobc - elob
        sigma_mp = sqrt(dwltm / int_step) * (thta_3db * RAD) / (k_noise * sqrt(snr))
        epaz_mp, epaz_saved = _markov(sigma_mp, 100.0, time, int_step, epaz_saved)
        epel_mp, epel_saved = _markov(sigma_mp, 100.0, time, int_step, epel_saved)
        epazt = epaz_glnt + biasaz + epaz_mp
        epelt = epel_glnt + biasel + epel_mp
        azab = (1.0 + esfta) * azob + epazt
        elab = (1.0 + esfte) * elob + epelt

        sigma_range = sqrt(dwltm / int_step) * rngegw / (k_noise * sqrt(snr))
        eps_range, range_saved = _markov(sigma_range, 10.0, time, int_step, range_saved)
        dab = dobc + eps_range

        vgw = 200.0 * wvelngth / 2.0
        sig_range_rate = sqrt(dwltm / int_step) * vgw / (k_noise * sqrt(snr))
        eps_range_rate, rate_saved = _markov(
            sig_range_rate, 10.0, time, int_step, rate_saved
        )
        ddab = rdobc + eps_range_rate

        if mseek == 4:
            if fabs(azab) > fovlimx * RAD or fabs(elab) > fovlimx * RAD:
                trcode = 6.0
                mseek = 2
                mguide = 6
            if dab < dblind:
                mseek = 5

        store.set("epaz_saved", epaz_saved)
        store.set("epel_saved", epel_saved)
        store.set("range_saved", range_saved)
        store.set("rate_saved", rate_saved)
        store.set("trcode", trcode)
        store.set("pwr_loss_db", 10.0 * log10(pwr_loss) if pwr_loss > 0.0 else -1.0e30)
        store.set("snr_db", 10.0 * log10(snr) if snr > 0.0 else -1.0e30)
        store.set("onax", onax)
        store.set("epazt", epazt)
        store.set("epelt", epelt)
        store.set("epaz_glnt", epaz_glnt)
        store.set("epel_glnt", epel_glnt)
        return azab, elab, dab, ddab, mseek, mguide

    def seeker_filter(self, vehicle, stbik, vtbik, azab, elab, dab, ddab, mseek, int_step):
        store = vehicle.store
        init_filter = int(store.get("init_filter"))
        epchup = store.get("epchup")
        flag_out = int(store.get("flag_out"))
        mupdt = int(store.get("mupdt"))
        sigpos = np.asarray(store.get("SIGPOS"), dtype=float).copy()
        sigvel = np.asarray(store.get("SIGVEL"), dtype=float).copy()
        semi_major = store.get("semi_major")
        semi_minor = store.get("semi_minor")
        ellipse_anglx = store.get("ellipse_anglx")
        estbi = np.asarray(store.get("ESTBI"), dtype=float).copy()
        evtbi = np.asarray(store.get("EVTBI"), dtype=float).copy()
        eaz = store.get("eaz")
        eel = store.get("eel")
        edab = store.get("edab")
        eddab = store.get("eddab")
        ppos_skr = store.get("ppos_skr")
        pvel_skr = store.get("pvel_skr")
        psfct = store.get("psfct")
        qpos_skr = store.get("qpos_skr")
        qvel_skr = store.get("qvel_skr")
        qsfct = store.get("qsfct")
        razab = store.get("razab")
        relab = store.get("relab")
        rdab = store.get("rdab")
        rddab = store.get("rddab")
        dtimkf = store.get("dtimkf")
        factp_skr = store.get("factp_skr")
        factq_skr = store.get("factq_skr")
        factr_skr = store.get("factr_skr")
        esfcta = store.get("esfcta")
        esfcte = store.get("esfcte")
        time = store.get("time")
        grav = store.get("grav")
        tbi = np.asarray(store.get("TBI"), dtype=float)
        tdi = np.asarray(store.get("TDI"), dtype=float)
        sbii = np.asarray(store.get("SBII"), dtype=float)
        vbii = np.asarray(store.get("VBII"), dtype=float)
        stii = np.asarray(store.get("STII"), dtype=float)
        vtii = np.asarray(store.get("VTII"), dtype=float)
        vbiic = np.asarray(store.get("VBIIC"), dtype=float)
        sbiic = np.asarray(store.get("SBIIC"), dtype=float)
        tbic = np.asarray(store.get("TBIC"), dtype=float)
        fspcb = np.asarray(store.get("FSPCB"), dtype=float)
        sxh_skr = np.asarray(store.get("SXH_SKR"), dtype=float).copy()
        vxh_skr = np.asarray(store.get("VXH_SKR"), dtype=float).copy()
        sfh = np.asarray(store.get("SFH"), dtype=float).copy()
        dtim = store.get("dtim")

        xh = self.XH
        pmat = self.PMAT
        for m in range(3):
            xh[m] = sxh_skr[m]
            xh[m + 3] = vxh_skr[m]
        xh[6] = sfh[0]
        xh[7] = sfh[1]

        stbik = np.asarray(stbik, dtype=float).copy()
        vtbik = np.asarray(vtbik, dtype=float).copy()

        if mseek == 3 and init_filter:
            init_filter = 0
            sh = stii - sbiic
            vh = vtii - vbiic
            xh[:] = 0.0
            for i in range(3):
                xh[i] = sh[i]
                xh[i + 3] = vh[i]
            xh[6] = 1.0
            xh[7] = 1.0
            pmat[:, :] = 0.0
            for i in range(3):
                pmat[i, i] = (ppos_skr * (1.0 + factp_skr)) ** 2
                pmat[i + 3, i + 3] = (pvel_skr * (1.0 + factp_skr)) ** 2
            pmat[6, 6] = (psfct * (1.0 + factp_skr)) ** 2
            pmat[7, 7] = (psfct * (1.0 + factp_skr)) ** 2
            self.QQ[:, :] = 0.0
            for i in range(3):
                self.QQ[i, i] = (qpos_skr * (1.0 + factq_skr)) ** 2
                self.QQ[i + 3, i + 3] = (qvel_skr * (1.0 + factq_skr)) ** 2
            self.QQ[6, 6] = (qsfct * (1.0 + factq_skr)) ** 2
            self.QQ[7, 7] = (qsfct * (1.0 + factq_skr)) ** 2
            self.RR[:, :] = 0.0
            self.RR[0, 0] = (razab * (1.0 + factr_skr)) ** 2
            self.RR[1, 1] = (relab * (1.0 + factr_skr)) ** 2
            self.RR[2, 2] = (rdab * (1.0 + factr_skr)) ** 2
            self.RR[3, 3] = (rddab * (1.0 + factr_skr)) ** 2
            self.FF[:, :] = 0.0
            for i in range(3):
                self.FF[i, i + 3] = 1.0
            self.PHI = np.eye(8) + self.FF * int_step
            gg = np.zeros((8, 3))
            gg[3, 0] = 1.0
            gg[4, 1] = 1.0
            gg[5, 2] = 1.0
            self.GAMDT = gg * int_step

        if mseek == 4:
            dtim = time - epchup
            if dtim > dtimkf:
                mupdt = 1
            fspic = cadac_matmul(tbic.T.copy(), fspcb)
            gravd = np.array([0.0, 0.0, grav], dtype=float)
            xh = cadac_matmul(self.PHI, xh) - cadac_matmul(
                self.GAMDT, fspic + cadac_matmul(tdi.T.copy(), gravd)
            )
            mid = pmat + self.QQ * (int_step / 2.0)
            pmat = cadac_matmul(cadac_matmul(self.PHI, mid), self.PHI.T.copy()) + self.QQ * (
                int_step / 2.0
            )
            stbi = stii - sbii
            vtbi = vtii - vbii
            for i in range(3):
                if pmat[i, i] >= 0.0:
                    sigpos[i] = sqrt(pmat[i, i])
                elif flag_out == 1:
                    flag_out = 0
                if pmat[i + 3, i + 3] >= 0.0:
                    sigvel[i] = sqrt(pmat[i + 3, i + 3])
                elif flag_out == 1:
                    flag_out = 0

        if mseek == 4 and mupdt == 1:
            mupdt = 0
            epchup = time
            shi = xh[0:3].copy()
            vhi = xh[3:6].copy()
            shb = cadac_matmul(tbic, shi)
            polar = polar_from_cart(shb)
            dtbh = float(polar[0])
            azh = float(polar[1])
            elh = float(polar[2])
            shb0 = float(shb[0])
            shb1 = float(shb[1])
            shb2 = float(shb[2])
            dtb01 = sqrt(shb0 * shb0 + shb1 * shb1)
            shb02 = shb0 * shb0
            dsv = float(shi[0] * vhi[0] + shi[1] * vhi[1] + shi[2] * vhi[2])

            ha1 = (tbic[1, 0] * shb0 - tbic[0, 0] * shb1) / shb02
            ha2 = (tbic[1, 1] * shb0 - tbic[0, 1] * shb1) / shb02
            ha3 = (tbic[1, 2] * shb0 - tbic[0, 2] * shb1) / shb02
            he1 = (
                -tbic[2, 0] * dtb01**2 + shb2 * (tbic[0, 0] * shb0 + tbic[1, 0] * shb1)
            ) / dtb01**3
            he2 = (
                -tbic[2, 1] * dtb01**2 + shb2 * (tbic[0, 1] * shb0 + tbic[1, 1] * shb1)
            ) / dtb01**3
            he3 = (
                -tbic[2, 2] * dtb01**2 + shb2 * (tbic[0, 2] * shb0 + tbic[1, 2] * shb1)
            ) / dtb01**3
            ca7 = xh[6] / (1.0 + (shb1 / shb0) ** 2)
            ce8 = xh[7] / (1.0 + (-shb2 / dtb01) ** 2)

            hh = np.zeros((4, 8))
            hh[0, 0] = ca7 * ha1
            hh[0, 1] = ca7 * ha2
            hh[0, 2] = ca7 * ha3
            hh[1, 0] = ce8 * he1
            hh[1, 1] = ce8 * he2
            hh[1, 2] = ce8 * he3
            hh[2, 0] = xh[0] / dtbh
            hh[2, 1] = xh[1] / dtbh
            hh[2, 2] = xh[2] / dtbh
            dtbh2 = dtbh**2
            dtbh3 = dtbh**3
            hh[3, 0] = (dtbh2 * xh[3] - xh[0] * dsv) / dtbh3
            hh[3, 1] = (dtbh2 * xh[4] - xh[1] * dsv) / dtbh3
            hh[3, 2] = (dtbh2 * xh[5] - xh[2] * dsv) / dtbh3
            hh[3, 3] = hh[2, 0]
            hh[3, 4] = hh[2, 1]
            hh[3, 5] = hh[2, 2]
            hh[0, 6] = atan2(shb1, shb0)
            hh[1, 7] = atan2(-shb2, dtb01)

            innov = cadac_matmul(cadac_matmul(hh, pmat), hh.T.copy()) + self.RR
            gk = cadac_matmul(cadac_matmul(pmat, hh.T.copy()), cadac_inverse(innov))
            zh = np.array([azh, elh, dtbh, dsv / dtbh], dtype=float)
            zk = np.array([azab, elab, dab, ddab], dtype=float)
            ez = zk - zh
            xh = xh + cadac_matmul(gk, ez)
            for i in range(3):
                stbik[i] = xh[i]
                vtbik[i] = xh[i + 3]
            pmat = cadac_matmul(np.eye(8) - cadac_matmul(gk, hh), pmat)

            covpl = pmat[0:3, 0:3].copy()
            polar = polar_from_cart(shi)
            azal = float(polar[1])
            elal = float(polar[2])
            tai = mat2tr(azal, elal)
            covpa = cadac_matmul(tai, covpl)
            cov23 = np.array(
                [
                    [covpa[1, 1], covpa[1, 2]],
                    [covpa[1, 2], covpa[2, 2]],
                ],
                dtype=float,
            )
            semi_major, semi_minor, phi = _ellipse(cov23)
            ellipse_anglx = DEG * phi
            stbi = stii - sbii
            vtbi = vtii - vbii
            estbi = stbi - shi
            evtbi = vtbi - vhi
            stbb = cadac_matmul(tbi, stbi)
            polar = polar_from_cart(stbb)
            dabi = float(polar[0])
            azabi = float(polar[1])
            elabi = float(polar[2])
            utbi = stbi / dabi
            ddabi = float(utbi[0] * vtbi[0] + utbi[1] * vtbi[1] + utbi[2] * vtbi[2])
            eaz = azabi - azab
            eel = elabi - elab
            edab = dabi - dab
            eddab = ddabi - ddab

        for m in range(3):
            sxh_skr[m] = xh[m]
            vxh_skr[m] = xh[m + 3]
        sfh[0] = xh[6]
        sfh[1] = xh[7]
        self.XH = xh
        self.PMAT = pmat

        store.set("init_filter", init_filter)
        store.set("epchup", epchup)
        store.set("flag_out", flag_out)
        store.set("mupdt", mupdt)
        store.set("SXH_SKR", sxh_skr)
        store.set("VXH_SKR", vxh_skr)
        store.set("SFH", sfh)
        store.set("dtim", dtim)
        store.set("SIGPOS", sigpos)
        store.set("SIGVEL", sigvel)
        store.set("semi_major", semi_major)
        store.set("semi_minor", semi_minor)
        store.set("ellipse_anglx", ellipse_anglx)
        store.set("ESTBI", estbi)
        store.set("EVTBI", evtbi)
        store.set("eaz", eaz)
        store.set("eel", eel)
        store.set("edab", edab)
        store.set("eddab", eddab)
        return stbik, vtbik

    def terminate(self, vehicle, ctx):
        pass
