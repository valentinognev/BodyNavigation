from math import atan2, cos, exp, sin, sqrt, tan

import numpy as np

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr, polar_from_cart, skew

SMALL = 1.e-7


class Agm6Guidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        scrn_plot = ("scrn", "plot")
        for field in (
            Field("mguid", 0, "int", "data", "guidance"),
            Field("gnav", 0.0, "real", "data", "guidance"),
            Field("ancomx", 0.0, "real", "out", "guidance", scrn_plot),
            Field("alcomx", 0.0, "real", "out", "guidance", scrn_plot),
            Field("gn", 0.0, "real", "diag", "guidance"),
            Field("grav_bias", 0.0, "real", "data", "guidance"),
            Field("apny", 0.0, "real", "diag", "guidance"),
            Field("apnz", 0.0, "real", "diag", "guidance"),
            Field("adely", 0.0, "real", "diag", "guidance"),
            Field("adelz", 0.0, "real", "diag", "guidance"),
            Field("all", 0.0, "real", "diag", "guidance", plot),
            Field("ann", 0.0, "real", "diag", "guidance", plot),
            Field("epchta", 0.0, "real", "save", "guidance"),
            Field("tgo_tgt_acrft", 0.0, "real", "diag", "guidance"),
            Field("line_gain", 0.0, "real", "data", "guidance"),
            Field("nl_gain_fact", 0.0, "real", "data", "guidance"),
            Field("decrement", 0.0, "real", "data", "guidance"),
            Field("dtac", 0.0, "real", "diag", "guidance"),
            Field("VBEO", zeros3, "vec", "diag", "guidance"),
            Field("VBEF", zeros3, "vec", "diag", "guidance"),
            Field("init_guide_line", 1, "int", "init", "guidance"),
            Field("thtflx", 0.0, "real", "data", "guidance"),
            Field("SBTO", zeros3, "vec", "diag", "guidance"),
            Field("quad_pos", 0.0, "real", "data", "guidance"),
            Field("quad_vel", 0.0, "real", "data", "guidance"),
            Field("w1", 0.0, "real", "data", "guidance"),
            Field("w3", 0.0, "real", "data", "guidance"),
            Field("WOELC", zeros3, "vec", "out", "guidance"),
            Field("tgoc", 0.0, "real", "diag", "guidance"),
            Field("dtbc", 0.0, "real", "diag", "guidance"),
            Field("dvtbc", 0.0, "real", "diag", "guidance"),
            Field("psiobcx", 0.0, "real", "diag", "guidance", plot),
            Field("thtobcx", 0.0, "real", "diag", "guidance", plot),
            Field("UTBLC", zeros3, "vec", "out", "guidance"),
            Field("STELC", zeros3, "vec", "diag", "guidance"),
            Field("STBLC", zeros3, "vec", "diag", "guidance"),
            Field("STELM", zeros3, "vec", "save", "guidance"),
            Field("VTELC", zeros3, "vec", "save", "guidance"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mguid = store.get("mguid")
        mnav = store.get("mnav")
        if "launch_time" in store:
            launch_time = store.get("launch_time")
        else:
            launch_time = ctx.sim_time
        if mnav == 3:
            epchta = launch_time
            stelm = np.asarray(store.get("STCEL"), dtype=float).copy()
            vtelc = np.asarray(store.get("VTCEL"), dtype=float).copy()
            store.set("epchta", epchta)
            store.set("STELM", stelm)
            store.set("VTELC", vtelc)
        else:
            epchta = store.get("epchta")
            stelm = np.asarray(store.get("STELM"), dtype=float)
            vtelc = np.asarray(store.get("VTELC"), dtype=float)
        dtime = launch_time - epchta
        stelc = stelm + vtelc * dtime
        sbelc = np.asarray(store.get("SBELC"), dtype=float)
        stblc = stelc - sbelc
        store.set("STELC", stelc)
        store.set("STBLC", stblc)
        if mguid == 0:
            return
        # C++ runs mid and term as independent ifs; term overwrites ACBX last.
        guid_mid = mguid // 10
        guid_term = mguid % 10
        acbx = None
        if guid_mid == 2:
            sael = np.asarray(store.get("SAEL"), dtype=float)
            stalc = stelc - sael
            nl_gain_fact = store.get("nl_gain_fact")
            vbelc = np.asarray(store.get("VBELC"), dtype=float)
            acbx = self.guidance_mid_line(
                vehicle, stalc, stblc, vbelc, nl_gain_fact
            )
        if guid_mid == 3:
            acbx = self.guidance_mid_pronav(vehicle, stblc, vtelc)
        if guid_mid == 4:
            stblc = np.asarray(store.get("STEL"), dtype=float) - sbelc
            store.set("STBLC", stblc)
            acbx = self.guidance_mid_pronav(vehicle, stblc, vtelc)
        if guid_term == 5:
            acbx = self.guidance_term_pronav(vehicle)
        if guid_term == 6:
            acbx = self.guidance_term_comp(vehicle)
        if acbx is None:
            raise ValueError(f"unknown mguid {mguid}")
        all_ = float(acbx[1])
        ann = -float(acbx[2])
        aa = sqrt(all_ * all_ + ann * ann)
        gmax = store.get("gmax")
        if aa > gmax:
            aa = gmax
        if abs(ann) < SMALL and abs(all_) < SMALL:
            phi = 0.0
        else:
            phi = atan2(ann, all_)
        store.set("all", all_)
        store.set("ann", ann)
        store.set("alcomx", aa * cos(phi))
        store.set("ancomx", aa * sin(phi))

    def guidance_mid_line(self, vehicle, STALC, STBLC, VBELC, nl_gain_fact):
        # AGM6 C++ guidance_mid_line: LOS/LOA line guidance (Zipfel).
        # Gravity term in algv3 is inside line_gain (unlike SAM6/plane5).
        store = vehicle.store
        line_gain = store.get("line_gain")
        decrement = store.get("decrement")
        thtflx = store.get("thtflx")
        grav = store.get("grav")
        tblc = np.asarray(store.get("TBLC"), dtype=float)
        thtvlcx = store.get("thtvlcx")
        psivlcx = store.get("psivlcx")
        sbtl = np.asarray(store.get("SBTL"), dtype=float)
        stalc = np.asarray(STALC, dtype=float)
        stblc = np.asarray(STBLC, dtype=float)
        vbelc = np.asarray(VBELC, dtype=float)

        polar = polar_from_cart(stalc)
        dtac = float(polar[0])
        az_loa = float(polar[1])
        el_loa = float(polar[2])
        if thtflx == 0:
            tfl = mat2tr(az_loa, el_loa)
        else:
            tfl = mat2tr(az_loa, thtflx * RAD)

        polar = polar_from_cart(stblc)
        dtbc = float(polar[0])
        az_los = float(polar[1])
        el_los = float(polar[2])
        tol = mat2tr(az_los, el_los)

        tvl = mat2tr(psivlcx * RAD, thtvlcx * RAD)
        tbv = tblc @ tvl.T

        vbeo = tol @ vbelc
        vbef = tfl @ vbelc
        nl_gain = nl_gain_fact * (1.0 - exp(-dtbc / decrement))

        algv1 = grav * sin(thtvlcx * RAD) / AGRAV
        algv2 = line_gain * (-float(vbeo[1]) + nl_gain * float(vbef[1])) / AGRAV
        algv3 = (
            line_gain
            * ((-float(vbeo[2]) + nl_gain * float(vbef[2])) - grav * cos(thtvlcx * RAD))
            / AGRAV
        )
        acvx = np.array([algv1, algv2, algv3], dtype=float)
        acbx = tbv @ acvx
        sbto = tol @ sbtl

        store.set("dtac", dtac)
        store.set("dtbc", dtbc)
        store.set("VBEO", np.asarray(vbeo, dtype=float).reshape(3))
        store.set("VBEF", np.asarray(vbef, dtype=float).reshape(3))
        store.set("SBTO", np.asarray(sbto, dtype=float).reshape(3))
        return np.asarray(acbx, dtype=float).reshape(3)

    def guidance_mid_pronav(self, vehicle, STBLC, VTELC):
        store = vehicle.store
        gnav = store.get("gnav")
        grav_bias = store.get("grav_bias")
        grav = store.get("grav")
        tblc = np.asarray(store.get("TBLC"), dtype=float)
        vbelc = np.asarray(store.get("VBELC"), dtype=float)
        stblc = np.asarray(STBLC, dtype=float)
        vtelc = np.asarray(VTELC, dtype=float)
        dtbc = sqrt(
            float(stblc[0]) ** 2 + float(stblc[1]) ** 2 + float(stblc[2]) ** 2
        )
        utblc = stblc * (1.0 / dtbc)
        utbbc = tblc @ utblc
        polar = polar_from_cart(utbbc)
        psiobcx = float(polar[1]) * DEG
        thtobcx = float(polar[2]) * DEG
        vtblc = vtelc - vbelc
        dvtbc = abs(
            float(utblc[0] * vtblc[0] + utblc[1] * vtblc[1] + utblc[2] * vtblc[2])
        )
        tgoc = dtbc / dvtbc
        woelc = skew(utblc) @ vtblc * (1.0 / dtbc)
        grav_comp = np.array([0.0, 0.0, grav_bias * grav], dtype=float)
        acbx = tblc @ (
            (skew(woelc) @ utblc * gnav * dvtbc - grav_comp) * (1.0 / AGRAV)
        )
        store.set("WOELC", woelc)
        store.set("UTBLC", utblc)
        store.set("tgoc", tgoc)
        store.set("dtbc", dtbc)
        store.set("dvtbc", dvtbc)
        store.set("psiobcx", psiobcx)
        store.set("thtobcx", thtobcx)
        return np.asarray(acbx, dtype=float).reshape(3)

    def guidance_term_pronav(self, vehicle):
        # AGM6 C++ guidance_term_pronav: kinematic LOS-rate without compensation.
        # Uses thtpb (missile[279]), grav_bias in g's — not guidance_term_comp.
        store = vehicle.store
        gnav = store.get("gnav")
        grav_bias = store.get("grav_bias")
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)
        stel = np.asarray(store.get("STEL"), dtype=float)
        vtel = np.asarray(store.get("VTEL"), dtype=float)
        thtpb = store.get("thtpb")
        psipb = store.get("psipb")
        sigdpy = store.get("sigdpy")
        sigdpz = store.get("sigdpz")
        tblc = np.asarray(store.get("TBLC"), dtype=float)
        sbtl = sbel - stel
        dbt = sqrt(
            float(sbtl[0]) ** 2 + float(sbtl[1]) ** 2 + float(sbtl[2]) ** 2
        )
        dum = float(sbtl[0] * (vbel[0] - vtel[0])
                    + sbtl[1] * (vbel[1] - vtel[1])
                    + sbtl[2] * (vbel[2] - vtel[2]))
        dcvel = abs(dum / dbt)
        gravb = tblc @ np.array([0.0, 0.0, grav_bias], dtype=float)
        gn = gnav * dcvel
        apny = gn * sigdpz / (cos(psipb) * AGRAV)
        apnz = gn * (
            sigdpz * tan(thtpb) * tan(psipb) + sigdpy / cos(thtpb)
        ) / AGRAV
        all_ = apny - float(gravb[1])
        ann = apnz + float(gravb[2])
        store.set("gn", gn)
        store.set("apny", apny)
        store.set("apnz", apnz)
        return np.array([0.0, all_, -ann], dtype=float)

    def guidance_term_comp(self, vehicle):
        store = vehicle.store
        gnav = store.get("gnav")
        stel = np.asarray(store.get("STEL"), dtype=float)
        vtel = np.asarray(store.get("VTEL"), dtype=float)
        psipb = store.get("psipb")
        # C++ guidance_term_comp reads missile[281] (ththb), not missile[279] (thtpb).
        thtpb = store.get("ththb")
        sigdpy = store.get("sigdpy")
        sigdpz = store.get("sigdpz")
        tblc = np.asarray(store.get("TBLC"), dtype=float)
        fspcb = np.asarray(store.get("FSPCB"), dtype=float)
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)
        sbtl = sbel - stel
        dbt = sqrt(
            float(sbtl[0]) ** 2 + float(sbtl[1]) ** 2 + float(sbtl[2]) ** 2
        )
        dum = float(sbtl[0] * (vbel[0] - vtel[0])
                    + sbtl[1] * (vbel[1] - vtel[1])
                    + sbtl[2] * (vbel[2] - vtel[2]))
        dcvel = abs(dum / dbt)
        fspcb1 = float(fspcb[0])
        adely = fspcb1 * tan(psipb) / AGRAV
        adelz = fspcb1 * tan(thtpb) / (cos(psipb) * AGRAV)
        gravb = tblc @ np.array([0.0, 0.0, 1.0], dtype=float)
        gn = gnav * dcvel
        apny = gn * sigdpz / (cos(psipb) * AGRAV)
        apnz = gn * (
            sigdpz * tan(thtpb) * tan(psipb) + sigdpy / cos(thtpb)
        ) / AGRAV
        all_ = apny + adely - float(gravb[1])
        ann = apnz + adelz + float(gravb[2])
        store.set("gn", gn)
        store.set("apny", apny)
        store.set("apnz", apnz)
        store.set("adely", adely)
        store.set("adelz", adelz)
        return np.array([0.0, all_, -ann], dtype=float)

    def terminate(self, vehicle, ctx):
        pass
