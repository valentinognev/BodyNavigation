from math import asin, cos, exp, fabs, log, sin, sqrt, tan

import numpy as np

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.state import Field
from cadac.math.frames import (
    angle,
    cadac_inverse,
    cadac_matmul,
    mat2tr,
    polar_from_cart,
    skew,
)
from cadac.math.wgs84 import cad_in_geo84, cad_kepler


def _sign(variable):
    if variable < 0.0:
        return -1
    return 1


def _univec3(vec):
    v1 = float(vec[0])
    v2 = float(vec[1])
    v3 = float(vec[2])
    scale = sqrt(v1 * v1 + v2 * v2 + v3 * v3)
    if scale == 0.0:
        return np.zeros(3)
    return np.array([v1 / scale, v2 / scale, v3 / scale])


def _absolute(vec):
    return sqrt(
        float(vec[0]) * float(vec[0])
        + float(vec[1]) * float(vec[1])
        + float(vec[2]) * float(vec[2])
    )


def _dot(a, b):
    return (
        float(a[0]) * float(b[0])
        + float(a[1]) * float(b[1])
        + float(a[2]) * float(b[2])
    )


def _unit_cross(a, b):
    v1 = float(a[1]) * float(b[2]) - float(a[2]) * float(b[1])
    v2 = float(a[2]) * float(b[0]) - float(a[0]) * float(b[2])
    v3 = float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])
    scale = sqrt(v1 * v1 + v2 * v2 + v3 * v3)
    if scale == 0.0:
        raise ValueError("divide by zero in unit cross")
    return np.array([v1 / scale, v2 / scale, v3 / scale])


def _vec(store, name):
    return np.asarray(store.get(name), dtype=float).copy()


def _ltg_igrl_a1_a2(x):
    # C++ guidance_ltg_igrl; x==1 was exit(1). a1x*(2/3) is integer division.
    if x == 2:
        a1 = 1.0 / (1.0 - 0.5 * x * 1.001)
    else:
        a1 = 1.0 / (1.0 - 0.5 * x)
    if x == 1:
        raise ValueError("LTG Terminator: end-state cannot be reached")
    a2 = 1.0 / (1.0 - x)
    return a1, a2


class Hyper6Guidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        for field in (
            Field("mguide", 0, "int", "data", "guidance"),
            Field("line_gain", 0.0, "real", "data", "guidance"),
            Field("nl_gain_fact", 0.0, "real", "data", "guidance"),
            Field("decrement", 0.0, "real", "data", "guidance"),
            Field("wp_lonx", 0.0, "real", "data", "guidance"),
            Field("wp_latx", 0.0, "real", "data", "guidance"),
            Field("wp_alt", 0.0, "real", "data", "guidance"),
            Field("psifdx", 0.0, "real", "data", "guidance"),
            Field("thtfdx", 0.0, "real", "data", "guidance"),
            Field("point_gain", 0.0, "real", "data", "guidance"),
            Field("wp_sltrange", 999999.0, "real", "diag", "guidance"),
            Field("nl_gain", 0.0, "real", "diag", "guidance"),
            Field("VBEO", zeros3, "vec", "diag", "guidance"),
            Field("VBEF", zeros3, "vec", "diag", "guidance"),
            Field("wp_grdrange", 999999.0, "real", "diag", "guidance"),
            Field("SWBD", zeros3, "vec", "out", "guidance"),
            Field("rad_min", 0.0, "real", "diag", "guidance"),
            Field("rad_geometric", 0.0, "real", "diag", "guidance"),
            Field("wp_flag", 0, "int", "diag", "guidance"),
            Field("gnav", 0.0, "real", "data", "guidance"),
            Field("aycomx", 0.0, "real", "out", "guidance", plot),
            Field("azcomx", 0.0, "real", "out", "guidance", plot),
            Field("tgoc", 0.0, "real", "diag", "guidance", plot),
            Field("dtbc", 0.0, "real", "diag", "guidance"),
            Field("psiobcx", 0.0, "real", "diag", "guidance"),
            Field("thtobcx", 0.0, "real", "diag", "guidance"),
            Field("SBTHC", zeros3, "vec", "diag", "guidance", plot),
            Field("init_flag", 1, "int", "init", "guidance"),
            Field("time_ltg", 0.0, "real", "diag", "guidance"),
            Field("UTBC", zeros3, "vec", "out", "guidance", plot),
            Field("RBIAS", zeros3, "vec", "save", "guidance"),
            Field("beco_flag", 0, "int", "diag", "guidance"),
            Field("inisw_flag", 1, "int", "init", "guidance"),
            Field("skip_flag", 1, "int", "init", "guidance"),
            Field("ipas_flag", 1, "int", "init", "guidance"),
            Field("ipas2_flag", 1, "int", "init", "guidance"),
            Field("print_flag", 1, "int", "init", "guidance"),
            Field("ltg_count", 0, "int", "save", "guidance"),
            Field("ltg_step", 0.0, "real", "data", "guidance"),
            Field("dbi_desired", 0.0, "real", "data", "guidance"),
            Field("dvbi_desired", 0.0, "real", "data", "guidance"),
            Field("thtvdx_desired", 0.0, "real", "data", "guidance"),
            Field("num_stages", 0, "int", "data", "guidance"),
            Field("delay_ignition", 0.0, "real", "data", "guidance"),
            Field("amin", 0.0, "real", "data", "guidance"),
            Field("char_time1", 0.0, "real", "data", "guidance"),
            Field("char_time2", 0.0, "real", "data", "guidance"),
            Field("char_time3", 0.0, "real", "data", "guidance"),
            Field("exhaust_vel1", 0.0, "real", "data", "guidance"),
            Field("exhaust_vel2", 0.0, "real", "data", "guidance"),
            Field("exhaust_vel3", 0.0, "real", "data", "guidance"),
            Field("burnout_epoch1", 0.0, "real", "data", "guidance"),
            Field("burnout_epoch2", 0.0, "real", "data", "guidance"),
            Field("burnout_epoch3", 0.0, "real", "data", "guidance"),
            Field("lamd_limit", 0.0, "real", "data", "guidance"),
            Field("RGRAV", zeros3, "vec", "save", "guidance"),
            Field("RGO", zeros3, "vec", "save", "guidance"),
            Field("VGO", zeros3, "vec", "save", "guidance"),
            Field("SDII", zeros3, "vec", "save", "guidance"),
            Field("UD", zeros3, "vec", "save", "guidance"),
            Field("UY", zeros3, "vec", "save", "guidance"),
            Field("UZ", zeros3, "vec", "save", "guidance"),
            Field("vgom", 0.0, "real", "diag", "guidance"),
            Field("tgo", 0.0, "real", "save", "guidance"),
            Field("nst", 0, "int", "save", "guidance"),
            Field("ULAM", zeros3, "vec", "diag", "guidance"),
            Field("LAMD", zeros3, "vec", "diag", "guidance"),
            Field("isp_fuel", 0.0, "real", "out", "guidance"),
            Field("burntime", 0.0, "real", "out", "guidance"),
            Field("nstmax", 0, "int", "diag", "guidance"),
            Field("lamd", 0.0, "real", "diag", "guidance", plot),
            Field("dpd", 0.0, "real", "diag", "guidance"),
            Field("dbd", 0.0, "real", "diag", "guidance"),
            Field("gnavpn", 0.0, "real", "data", "guidance"),
            Field("gnavps", 0.0, "real", "data", "guidance"),
            Field("gs_flag", 1, "int", "init", "guidance"),
            Field("time_gs", 0.0, "real", "data", "guidance"),
            Field("num_burns", 0, "int", "data", "guidance"),
            Field("closing_rate", 0.0, "real", "data", "guidance"),
            Field("orbital_rate", 0.0, "real", "data", "guidance"),
            Field("satl1", 0.0, "real", "data", "guidance"),
            Field("satl2", 0.0, "real", "data", "guidance"),
            Field("satl3", 0.0, "real", "data", "guidance"),
            Field("dtime_gs", 0.0, "real", "save", "guidance"),
            Field("length_gs", 0.0, "real", "save", "guidance"),
            Field("para_gs", 0.0, "real", "save", "guidance"),
            Field("UB0AL", zeros3, "vec", "save", "guidance"),
            Field("epoch_gs", 0.0, "real", "save", "guidance"),
            Field("counter_gs", 1, "int", "save", "guidance"),
            Field("VBTLM", zeros3, "vec", "save", "guidance"),
            Field("DELTA_V", zeros3, "vec", "save", "guidance"),
            Field("burn_flag", 1, "int", "save", "guidance"),
            Field("SBTL", zeros3, "vec", "diag", "guidance", plot),
            Field("VBTL", zeros3, "vec", "diag", "guidance"),
            Field("delta_v", 0.0, "real", "diag", "guidance"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mguide = store.get("mguide")
        if mguide == 0:
            return
        wp_lonx = store.get("wp_lonx")
        wp_latx = store.get("wp_latx")
        wp_alt = store.get("wp_alt")
        alcomx = 0.0
        ancomx = 0.0
        phicomx = 0.0
        if mguide == 30:
            algv = self.guidance_line(
                vehicle,
                wp_lonx,
                wp_latx,
                wp_alt,
                store.get("psifdx"),
                store.get("thtfdx"),
            )
            alcomx = float(algv[1]) / store.get("grav")
        elif mguide == 3:
            algv = self.guidance_line(
                vehicle,
                wp_lonx,
                wp_latx,
                wp_alt,
                store.get("psifdx"),
                store.get("thtfdx"),
            )
            ancomx = -float(algv[2]) / store.get("grav")
        elif mguide == 33:
            algv = self.guidance_line(
                vehicle,
                wp_lonx,
                wp_latx,
                wp_alt,
                store.get("psifdx"),
                store.get("thtfdx"),
            )
            grav = store.get("grav")
            alcomx = float(algv[1]) / grav
            ancomx = -float(algv[2]) / grav
        elif mguide == 4:
            phicomx = self.guidance_arc(vehicle, wp_lonx, wp_latx, wp_alt)
        elif mguide == 5:
            self._execute_ltg(vehicle, ctx)
            return
        elif mguide == 6:
            accomx, utbbc = self.guidance_pronav(vehicle)
            self._store_terminal(store, float(accomx[1]), float(accomx[2]), utbbc)
            return
        elif mguide == 7:
            accomx, utbbc = self.guidance_AGL(vehicle)
            self._store_terminal(store, float(accomx[1]), float(accomx[2]), utbbc)
            return
        elif mguide == 8:
            utbc = self.guidance_glideslope(vehicle)
            self._store_terminal(store, 0.0, 0.0, utbc)
            return
        else:
            raise ValueError(f"unknown mguide {mguide}")
        store.set("alcomx", alcomx)
        store.set("ancomx", ancomx)
        store.set("phicomx", phicomx)
        store.set("aycomx", 0.0)
        store.set("azcomx", 0.0)

    def _store_terminal(self, store, aycomx, azcomx, utbc):
        store.set("alcomx", 0.0)
        store.set("ancomx", 0.0)
        store.set("phicomx", 0.0)
        store.set("aycomx", aycomx)
        store.set("azcomx", azcomx)
        store.set("UTBC", utbc)

    def _execute_ltg(self, vehicle, ctx):
        store = vehicle.store
        int_step = ctx.int_step
        init_flag = store.get("init_flag")
        time_ltg = store.get("time_ltg")
        ltg_count = store.get("ltg_count")
        utbc = _vec(store, "UTBC")
        if init_flag:
            init_flag = 0
            time_ltg = 0.0
        else:
            time_ltg = time_ltg + int_step
        ltg_count = ltg_count + 1
        ratio = int(store.get("ltg_step") / int_step)
        ltg_flag = ltg_count - (ltg_count // ratio) * ratio
        if ltg_flag == 0:
            utic = self.guidance_ltg(vehicle, int_step, time_ltg)
            utbc = cadac_matmul(_vec(store, "TBIC"), utic)
        store.set("init_flag", init_flag)
        store.set("time_ltg", time_ltg)
        store.set("ltg_count", ltg_count)
        self._store_terminal(store, 0.0, 0.0, utbc)

    def guidance_line(self, vehicle, wp_lonx, wp_latx, wp_alt, psifdx, thtfdx):
        store = vehicle.store
        line_gain = store.get("line_gain")
        nl_gain_fact = store.get("nl_gain_fact")
        decrement = store.get("decrement")
        time = store.get("time")
        grav = store.get("grav")
        sbiic = np.asarray(store.get("SBIIC"), dtype=float)
        vbecd = np.asarray(store.get("VBECD"), dtype=float)
        dvbec = store.get("dvbec")
        tdci = np.asarray(store.get("TDCI"), dtype=float)
        thtvdcx = store.get("thtvdcx")
        philimx = store.get("philimx")

        tfd = mat2tr(psifdx * RAD, thtfdx * RAD)
        swii = cad_in_geo84(wp_lonx * RAD, wp_latx * RAD, wp_alt, time)
        swbd = cadac_matmul(tdci, swii - sbiic)
        polar = polar_from_cart(swbd)
        wp_sltrange = float(polar[0])
        tod = mat2tr(float(polar[1]), float(polar[2]))
        swbg1 = float(swbd[0])
        swbg2 = float(swbd[1])
        wp_grdrange = sqrt(swbg1 * swbg1 + swbg2 * swbg2)
        vbeo = cadac_matmul(tod, vbecd)
        vbef = cadac_matmul(tfd, vbecd)
        nl_gain = nl_gain_fact * (1.0 - exp(-wp_sltrange / decrement))
        algv = np.array(
            [
                grav * sin(thtvdcx * RAD),
                line_gain * (-float(vbeo[1]) + nl_gain * float(vbef[1])),
                line_gain * (-float(vbeo[2]) + nl_gain * float(vbef[2]))
                - grav * cos(thtvdcx * RAD),
            ]
        )
        rad_min = dvbec * dvbec / (grav * tan(philimx * RAD))
        if wp_grdrange < 2.0 * rad_min:
            sh = np.array([swbg1, swbg2, 0.0])
            vh = np.array([float(vbecd[0]), float(vbecd[1]), 0.0])
            wp_flag = _sign(float(vh @ sh))
        else:
            wp_flag = 0
        store.set("wp_sltrange", wp_sltrange)
        store.set("nl_gain", nl_gain)
        store.set("VBEO", vbeo)
        store.set("VBEF", vbef)
        store.set("wp_grdrange", wp_grdrange)
        store.set("SWBD", swbd)
        store.set("rad_min", rad_min)
        store.set("wp_flag", wp_flag)
        return algv

    def guidance_arc(self, vehicle, wp_lonx, wp_latx, wp_alt):
        store = vehicle.store
        time = store.get("time")
        grav = store.get("grav")
        sbiic = np.asarray(store.get("SBIIC"), dtype=float)
        vbecd = np.asarray(store.get("VBECD"), dtype=float)
        dvbec = store.get("dvbec")
        tdci = np.asarray(store.get("TDCI"), dtype=float)
        philimx = store.get("philimx")

        swii = cad_in_geo84(wp_lonx * RAD, wp_latx * RAD, wp_alt, time)
        swbd = cadac_matmul(tdci, swii - sbiic)
        sh = np.array([float(swbd[0]), float(swbd[1]), 0.0])
        dwbh = sqrt(float(swbd[0]) * float(swbd[0]) + float(swbd[1]) * float(swbd[1]))
        vh = np.array([float(vbecd[0]), float(vbecd[1]), 0.0])
        uv = cadac_matmul(skew(vh), sh)
        psiwvx = DEG * angle(vh, sh)
        zz = np.array([0.0, 0.0, 1.0])
        psiwvx = psiwvx * _sign(float(uv @ zz))
        argument = 0.0
        if fabs(psiwvx) < 90.0:
            num = -2.0 * dvbec * dvbec * sin(psiwvx * RAD)
            denom = -grav * dwbh
            if denom != 0.0:
                argument = num / denom
            if fabs(argument) <= 1.0 and fabs(asin(argument)) < philimx * RAD:
                phicomx = DEG * asin(argument)
            else:
                phicomx = philimx * _sign(argument)
        else:
            phicomx = philimx * _sign(psiwvx)
        rad_geometric = 0.0
        if psiwvx != 0.0:
            rad_geometric = fabs(dwbh / (2.0 * sin(psiwvx * RAD)))
        rad_min = dvbec * dvbec / (grav * tan(philimx * RAD))
        if dwbh < 2.0 * rad_min:
            wp_flag = _sign(float(vh @ sh))
        else:
            wp_flag = 0
        store.set("wp_grdrange", dwbh)
        store.set("SWBD", swbd)
        store.set("rad_min", rad_min)
        store.set("rad_geometric", rad_geometric)
        store.set("wp_flag", wp_flag)
        return phicomx

    def guidance_ltg(self, vehicle, int_step, time_ltg):
        store = vehicle.store
        ltg_step = store.get("ltg_step")
        dbi_desired = store.get("dbi_desired")
        dvbi_desired = store.get("dvbi_desired")
        thtvdx_desired = store.get("thtvdx_desired")
        num_stages = store.get("num_stages")
        delay_ignition = store.get("delay_ignition")
        amin = store.get("amin")
        lamd_limit = store.get("lamd_limit")
        taun = np.array(
            [
                store.get("char_time1"),
                store.get("char_time2"),
                store.get("char_time3"),
            ],
            dtype=float,
        )
        vexn = np.array(
            [
                store.get("exhaust_vel1"),
                store.get("exhaust_vel2"),
                store.get("exhaust_vel3"),
            ],
            dtype=float,
        )
        botn = np.array(
            [
                0.0,
                store.get("burnout_epoch1"),
                store.get("burnout_epoch2"),
                store.get("burnout_epoch3"),
            ],
            dtype=float,
        )
        beco_flag = store.get("beco_flag")
        inisw_flag = store.get("inisw_flag")
        skip_flag = store.get("skip_flag")
        ipas2_flag = store.get("ipas2_flag")
        print_flag = store.get("print_flag")
        rbias = _vec(store, "RBIAS")
        rgrav = _vec(store, "RGRAV")
        rgo = _vec(store, "RGO")
        vgo = _vec(store, "VGO")
        sdii = _vec(store, "SDII")
        ud = _vec(store, "UD")
        uy = _vec(store, "UY")
        uz = _vec(store, "UZ")
        tgo = store.get("tgo")
        nst = store.get("nst")
        mprop = store.get("mprop")
        sbiic = _vec(store, "SBIIC")
        vbiic = _vec(store, "VBIIC")
        tbic = np.asarray(store.get("TBIC"), dtype=float).copy()
        fspcb = _vec(store, "FSPCB")

        abii = cadac_matmul(np.asarray(tbic.T, dtype=float).copy(), fspcb)
        amag1 = _absolute(abii)
        if inisw_flag:
            inisw_flag = 0
            spii = np.array(sbiic, copy=True)
            vpii = np.array(vbiic, copy=True)
            sdii, ud, uy, uz, vgo = self._ltg_crct(
                vehicle,
                vgo,
                dbi_desired,
                dvbi_desired,
                thtvdx_desired,
                spii,
                vpii,
                sbiic,
                vbiic,
            )
        else:
            vgo = vgo - abii * ltg_step
        vgom = _absolute(vgo)

        tgop, burnt, ligrl_n, tgon, l_igrl, nstmax, tgo, nst, taun = self._ltg_tgo(
            vehicle,
            tgo,
            nst,
            taun,
            vexn,
            botn,
            delay_ignition,
            vgom,
            amag1,
            amin,
            time_ltg,
            num_stages,
        )
        s_igrl, j_igrl, q_igrl, h_igrl, p_igrl, j_over_l, tlam, qprime = self._ltg_igrl(
            nst, nstmax, burnt, ligrl_n, tgon, taun, vexn, l_igrl, time_ltg
        )
        ulam, lamd, rgo, ipas2_flag, rgrav = self._ltg_trate(
            vehicle,
            ipas2_flag,
            vgo,
            s_igrl,
            q_igrl,
            j_over_l,
            lamd_limit,
            vgom,
            tgo,
            tgop,
            sdii,
            sbiic,
            vbiic,
            rbias,
            ud,
            uy,
            uz,
            rgrav,
            rgo,
        )
        tc = ulam + lamd * (time_ltg - tlam)
        utic = np.zeros(3)
        if skip_flag:
            skip_flag += 1
            if skip_flag == 10:
                skip_flag = 0
        else:
            utic = _univec3(tc)

        spii, vpii, rgrav, rbias = self._ltg_pdct(
            lamd,
            ulam,
            l_igrl,
            s_igrl,
            j_igrl,
            q_igrl,
            h_igrl,
            p_igrl,
            j_over_l,
            qprime,
            sbiic,
            vbiic,
            rgo,
            tgo,
        )
        sdii, ud, uy, uz, vgo = self._ltg_crct(
            vehicle,
            vgo,
            dbi_desired,
            dvbi_desired,
            thtvdx_desired,
            spii,
            vpii,
            sbiic,
            vbiic,
        )

        isp_fuel = vexn[nst - 1] / AGRAV
        burntime = botn[nst] - botn[nst - 1] - delay_ignition
        if burntime > 0.0:
            mprop = 3
        if tgo < 10.0 * int_step:
            beco_flag = 1
            mprop = 0
        if beco_flag and print_flag:
            print_flag = 0

        store.set("isp_fuel", isp_fuel)
        store.set("burntime", burntime)
        store.set("RBIAS", rbias)
        store.set("beco_flag", beco_flag)
        store.set("inisw_flag", inisw_flag)
        store.set("skip_flag", skip_flag)
        store.set("ipas2_flag", ipas2_flag)
        store.set("print_flag", print_flag)
        store.set("RGRAV", rgrav)
        store.set("RGO", rgo)
        store.set("VGO", vgo)
        store.set("SDII", sdii)
        store.set("UD", ud)
        store.set("UY", uy)
        store.set("UZ", uz)
        store.set("tgo", tgo)
        store.set("nst", nst)
        store.set("vgom", vgom)
        store.set("ULAM", ulam)
        store.set("LAMD", lamd)
        store.set("nstmax", nstmax)
        store.set("mprop", mprop)
        return utic

    def _ltg_tgo(
        self,
        vehicle,
        tgo,
        nst,
        taun,
        vexn,
        botn,
        delay_ignition,
        vgom,
        amag1,
        amin,
        time_ltg,
        num_stages,
    ):
        store = vehicle.store
        ipas_flag = store.get("ipas_flag")
        burnt = np.zeros(3)
        ligrl_n = np.zeros(3)
        tgon = np.zeros(3)
        if ipas_flag:
            nst = 1
        tgop = tgo
        tgo = 0.0
        l_igrl = 0.0
        nstmax = num_stages
        if time_ltg >= botn[nst]:
            nst += 1
        i = nst - 1
        while i < nstmax:
            if i == (nst - 1):
                taun[nst - 1] = taun[nst - 1] - (time_ltg - botn[nst - 1])
            if (amag1 >= amin) and (time_ltg > (botn[nst - 1] + delay_ignition)):
                taun[nst - 1] = vexn[nst - 1] * (1.0 / amag1)
            if i == (nst - 1):
                burnt[i] = botn[i + 1] - time_ltg
            else:
                burnt[i] = botn[i + 1] - botn[i]
            ligrl_n[i] = -vexn[i] * log(1.0 - burnt[i] / taun[i])
            l_igrl += ligrl_n[i]
            if l_igrl < vgom:
                tgo += burnt[i]
                tgon[i] = tgo
                i += 1
            else:
                i += 1
                break
        nstmax = i
        l_igrl = l_igrl - ligrl_n[i - 1]
        almx = vgom - l_igrl
        ligrl_n[i - 1] = almx
        burnt[i - 1] = taun[i - 1] * (1.0 - exp(-almx / vexn[i - 1]))
        tgo += burnt[i - 1]
        tgon[i - 1] = tgo
        l_igrl = vgom
        if ipas_flag:
            tgop = tgo
            ipas_flag = 0
        store.set("ipas_flag", ipas_flag)
        return tgop, burnt, ligrl_n, tgon, l_igrl, nstmax, tgo, nst, taun

    def _ltg_igrl(
        self, nst, nstmax, burnt, ligrl_n, tgon, taun, vexn, l_igrl, time_ltg
    ):
        ls_igrl = 0.0
        s_igrl = 0.0
        j_igrl = 0.0
        q_igrl = 0.0
        h_igrl = 0.0
        p_igrl = 0.0
        for i in range(nst - 1, nstmax):
            tb = float(burnt[i])
            tga = float(tgon[i])
            x = tb / float(taun[i])
            a1, a2 = _ltg_igrl_a1_a2(x)
            aa = float(vexn[i]) / float(taun[i])
            ll_igrl = float(ligrl_n[i])
            a1x = 4.0 * a1 - a2 - 3.0
            a2xsq = 2.0 * a2 - 4.0 * a1 + 2.0
            sa = (aa * tb * tb / 2.0) * (1.0 + a1x / 3.0 + a2xsq / 6.0)
            ja = (aa * tb * tb / 2.0) * (1.0 + a1x * (2 // 3) + a2xsq / 2.0)
            qa = (aa * tb * tb * tb / 6.0) * (1.0 + a1x / 2.0 + a2xsq * 0.3)
            pa = (aa * tb * tb * tb * tb / 12.0) * (1.0 + a1x * 0.6 + a2xsq * 0.4)
            if i != nst - 1:
                t1 = float(tgon[i - 1])
                ja = ja + t1 * ll_igrl
                pa = pa + 2.0 * t1 * qa + t1 * t1 * sa
                qa = qa + t1 * sa
            ha = ja * tga - qa
            sa = sa + ls_igrl * tb
            qa = qa + j_igrl * tb
            pa = pa + h_igrl * tb
            s_igrl = s_igrl + sa
            q_igrl = q_igrl + qa
            p_igrl = p_igrl + pa
            h_igrl = h_igrl + ha
            ls_igrl = ls_igrl + ll_igrl
            j_igrl = j_igrl + ja
        j_over_l = j_igrl / l_igrl
        tlam = time_ltg + j_over_l
        qprime = q_igrl - s_igrl * j_over_l
        return s_igrl, j_igrl, q_igrl, h_igrl, p_igrl, j_over_l, tlam, qprime

    def _ltg_trate(
        self,
        vehicle,
        ipas2_flag,
        vgo,
        s_igrl,
        q_igrl,
        j_over_l,
        lamd_limit,
        vgom,
        tgo,
        tgop,
        sdii,
        sbiic,
        vbiic,
        rbias,
        ud,
        uy,
        uz,
        rgrav,
        rgo,
    ):
        ulam = np.zeros(3)
        lamd = np.zeros(3)
        if vgom == 0.0:
            return ulam, lamd, rgo, ipas2_flag, rgrav
        ulam = _univec3(vgo)
        if ipas2_flag:
            ipas2_flag = 0
            rgo = ulam * s_igrl
        rgo, rgrav = self._ltg_rtgo(
            rgo, rgrav, tgo, tgop, sdii, sbiic, vbiic, rbias, ulam, ud, uy, uz, s_igrl
        )
        denom = q_igrl - s_igrl * j_over_l
        if denom != 0.0:
            lamd = (rgo - ulam * s_igrl) * (1.0 / denom)
        else:
            lamd = np.zeros(3)
        lamd_mag = _absolute(lamd)
        if lamd_mag >= lamd_limit:
            lamd = _univec3(lamd) * lamd_limit
        vehicle.store.set("lamd", _absolute(lamd))
        return ulam, lamd, rgo, ipas2_flag, rgrav

    def _ltg_rtgo(
        self,
        rgo,
        rgrav,
        tgo,
        tgop,
        sdii,
        sbiic,
        vbiic,
        rbias,
        ulam,
        ud,
        uy,
        uz,
        s_igrl,
    ):
        rgrav = rgrav * (tgo / tgop) * (tgo / tgop)
        rgo_local = sdii - (sbiic + vbiic * tgo + rgrav) - rbias
        rgoxy = ud * _dot(rgo_local, ud) + uy * _dot(rgo_local, uy)
        num = _dot(rgoxy, ulam)
        denom = _dot(ulam, uz)
        if denom == 0.0:
            return rgo, rgrav
        rgoz = (s_igrl - num) / denom
        return rgoxy + uz * rgoz, rgrav

    def _ltg_pdct(
        self,
        lamd,
        ulam,
        l_igrl,
        s_igrl,
        j_igrl,
        q_igrl,
        h_igrl,
        p_igrl,
        j_over_l,
        qprime,
        sbiic,
        vbiic,
        rgo,
        tgo,
    ):
        lmdsq = _dot(lamd, lamd)
        vthrust = ulam * (l_igrl - 0.5 * lmdsq * (h_igrl - j_igrl * j_over_l))
        rthrust = ulam * (
            s_igrl - 0.5 * lmdsq * (p_igrl - j_over_l * (q_igrl + qprime))
        ) + lamd * qprime
        rbias = rgo - rthrust
        sbiic1 = sbiic - rthrust * 0.1 - vthrust * (tgo / 30.0)
        vbiic1 = vbiic + rthrust * (1.2 / tgo) - vthrust * 0.1
        sbiic2, vbiic2, _flag = cad_kepler(sbiic1, vbiic1, tgo)
        vgrav = vbiic2 - vbiic1
        rgrav = sbiic2 - sbiic1 - vbiic1 * tgo
        spii = sbiic + vbiic * tgo + rgrav + rthrust
        vpii = vbiic + vgrav + vthrust
        return spii, vpii, rgrav, rbias

    def _ltg_crct(
        self,
        vehicle,
        vgo,
        dbi_desired,
        dvbi_desired,
        thtvdx_desired,
        spii,
        vpii,
        sbiic,
        vbiic,
    ):
        ud = _univec3(spii)
        sdii = ud * dbi_desired
        uy = _unit_cross(vbiic, sbiic)
        uz = _unit_cross(ud, uy)
        vdii = (
            ud * sin(thtvdx_desired * RAD) + uz * cos(thtvdx_desired * RAD)
        ) * dvbi_desired
        vgo = vgo - (vpii - vdii)
        dpi = _absolute(spii)
        ddi = _absolute(sdii)
        vehicle.store.set("dpd", dpi - ddi)
        vehicle.store.set("dbd", _absolute(sbiic) - ddi)
        return sdii, ud, uy, uz, vgo

    def guidance_pronav(self, vehicle):
        store = vehicle.store
        gnav = store.get("gnav")
        mseek = store.get("mseek")
        tbic = np.asarray(store.get("TBIC"), dtype=float).copy()
        if mseek > 3:
            stbic = _vec(store, "STBIK")
            vtbic = _vec(store, "VTBIK")
        else:
            stbic = _vec(store, "STCII") - _vec(store, "SBIIC")
            vtbic = _vec(store, "VTCII") - _vec(store, "VBIIC")
        dtbc = _absolute(stbic)
        utbic = _univec3(stbic)
        utbbc = cadac_matmul(tbic, utbic)
        polar = polar_from_cart(utbbc)
        dvtbc = _dot(utbic, vtbic)
        tgoc = fabs(dtbc / dvtbc)
        woiic = cadac_matmul(skew(utbic), vtbic) * (1.0 / dtbc)
        aapnb = cadac_matmul(cadac_matmul(tbic, skew(woiic)), utbic) * gnav * fabs(dvtbc)
        accomx = aapnb * (1.0 / AGRAV)
        stii = _vec(store, "STII")
        vtii = _vec(store, "VTII")
        uh1 = _univec3(stii)
        uh3 = _univec3(cadac_matmul(skew(stii), vtii))
        uh2 = cadac_matmul(skew(uh3), uh1)
        thi = np.vstack((uh1, uh2, uh3))
        sbthc = cadac_matmul(thi, stbic * (-1.0))
        store.set("tgoc", tgoc)
        store.set("dtbc", dtbc)
        store.set("psiobcx", float(polar[1]) * DEG)
        store.set("thtobcx", float(polar[2]) * DEG)
        store.set("SBTHC", sbthc)
        return accomx, utbbc

    def guidance_AGL(self, vehicle):
        store = vehicle.store
        gnavpn = store.get("gnavpn")
        gnavps = store.get("gnavps")
        tbic = np.asarray(store.get("TBIC"), dtype=float).copy()
        stbik = _vec(store, "STBIK")
        vtbik = _vec(store, "VTBIK")
        dtbc = _absolute(stbik)
        utbik = _univec3(stbik)
        utbbc = cadac_matmul(tbic, utbik)
        polar = polar_from_cart(utbbc)
        dvtbc = _dot(utbik, vtbik)
        tgoc = fabs(dtbc / dvtbc)
        acpuri = np.zeros(3)
        acpni = np.zeros(3)
        if tgoc != 0.0:
            acpuri = stbik * (gnavps / (tgoc * tgoc))
            acpni = vtbik * (gnavpn / tgoc)
        accomx = cadac_matmul(tbic, acpuri + acpni) * (1.0 / AGRAV)
        store.set("tgoc", tgoc)
        store.set("dtbc", dtbc)
        store.set("psiobcx", float(polar[1]) * DEG)
        store.set("thtobcx", float(polar[2]) * DEG)
        return accomx, utbbc

    def guidance_glideslope(self, vehicle):
        store = vehicle.store
        time = store.get("time")
        time_gs = store.get("time_gs")
        num_burns = store.get("num_burns")
        closing_rate = store.get("closing_rate")
        orbital_rate = store.get("orbital_rate")
        satl = np.array(
            [store.get("satl1"), store.get("satl2"), store.get("satl3")], dtype=float
        )
        sbiic = _vec(store, "SBIIC")
        vbiic = _vec(store, "VBIIC")
        stcii = _vec(store, "STCII")
        vtcii = _vec(store, "VTCII")
        tbic = np.asarray(store.get("TBIC"), dtype=float).copy()
        mseek = store.get("mseek")
        mprop = store.get("mprop")
        gs_flag = store.get("gs_flag")
        dtime_gs = store.get("dtime_gs")
        length_gs = store.get("length_gs")
        para_gs = store.get("para_gs")
        ub0al = _vec(store, "UB0AL")
        epoch_gs = store.get("epoch_gs")
        counter_gs = store.get("counter_gs")
        vbtlm = _vec(store, "VBTLM")
        delta_v = _vec(store, "DELTA_V")
        burn_flag = store.get("burn_flag")

        ul1i = _univec3(vtcii)
        ul3i = _univec3(stcii) * (-1.0)
        ul2i = cadac_matmul(skew(ul3i), ul1i)
        tli = np.vstack((ul1i, ul2i, ul3i))
        sbti = sbiic - stcii
        vbti = vbiic - vtcii
        sbtl = cadac_matmul(tli, sbti)
        vbtl = cadac_matmul(tli, vbti)
        if gs_flag:
            gs_flag = 0
            dtime_gs = time_gs / num_burns
            sb0al = sbtl - satl
            ub0al = _univec3(sb0al)
            length_gs = _absolute(sb0al)
            para_gs = (_dot(ub0al, vbtl) - closing_rate) / length_gs
            epoch_gs = time
            counter_gs = 0
        if counter_gs < num_burns and time >= (dtime_gs * counter_gs + epoch_gs):
            wt = orbital_rate * dtime_gs
            swt = sin(wt)
            cwt = cos(wt)
            phiss = np.array(
                [
                    [1.0, 0.0, 6.0 * (wt - swt)],
                    [0.0, cwt, 0.0],
                    [0.0, 0.0, 4.0 - 3.0 * cwt],
                ],
                dtype=float,
            )
            phisv = np.array(
                [
                    [4.0 * swt / orbital_rate - 3.0 * dtime_gs, 0.0, 2.0 * (1.0 - cwt) / orbital_rate],
                    [0.0, swt / orbital_rate, 0.0],
                    [-2.0 * (1.0 - cwt) / orbital_rate, 0.0, swt / orbital_rate],
                ],
                dtype=float,
            )
            dum = para_gs * dtime_gs * (counter_gs + 1)
            dlength = length_gs * exp(dum) + (closing_rate / para_gs) * (exp(dum) - 1.0)
            sbctl = satl + ub0al * dlength
            vbtlp = cadac_matmul(
                cadac_inverse(phisv), sbctl - cadac_matmul(phiss, sbtl)
            )
            delta_v = vbtlp - vbtl
            vbtlm = np.array(vbtl, copy=True)
            counter_gs = counter_gs + 1
            burn_flag = 1
        utb = cadac_matmul(
            cadac_matmul(tbic, np.asarray(tli.T, dtype=float).copy()),
            _univec3(delta_v),
        )
        ev = delta_v + vbtlm - vbtl
        delta_v_mag = _absolute(ev)
        if float(utb[0]) > 0.9:
            if delta_v_mag > fabs(closing_rate) and burn_flag:
                mprop = 4
            else:
                mprop = 0
                burn_flag = 0
        if mseek == 3:
            utb = cadac_matmul(tbic, _univec3(sbti) * (-1.0))
        store.set("dtime_gs", dtime_gs)
        store.set("length_gs", length_gs)
        store.set("para_gs", para_gs)
        store.set("UB0AL", ub0al)
        store.set("epoch_gs", epoch_gs)
        store.set("counter_gs", counter_gs)
        store.set("VBTLM", vbtlm)
        store.set("DELTA_V", delta_v)
        store.set("burn_flag", burn_flag)
        store.set("SBTL", sbtl)
        store.set("VBTL", vbtl)
        store.set("delta_v", delta_v_mag)
        store.set("gs_flag", gs_flag)
        store.set("mprop", mprop)
        return utb

    def terminate(self, vehicle, ctx):
        pass
