import math

import numpy as np

from cadac.constants import RAD
from cadac.kernel.state import Field
from cadac.math.frames import cadac_matmul
from cadac.math.wgs84 import cad_kepler


def _univec3(vec):
    v1 = float(vec[0])
    v2 = float(vec[1])
    v3 = float(vec[2])
    d = math.sqrt(v1 * v1 + v2 * v2 + v3 * v3)
    if d == 0.0:
        return np.zeros(3)
    return np.array([v1 / d, v2 / d, v3 / d], dtype=float)


def _absolute(vec):
    return math.sqrt(
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
    # C++ Matrix::operator% — unit vector cross product of two 3x1 vectors.
    v1 = float(a[1]) * float(b[2]) - float(a[2]) * float(b[1])
    v2 = float(a[2]) * float(b[0]) - float(a[0]) * float(b[2])
    v3 = float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])
    dv = math.sqrt(v1 * v1 + v2 * v2 + v3 * v3)
    if dv == 0.0:
        raise ValueError("divide by zero in unit cross")
    return np.array([v1 / dv, v2 / dv, v3 / dv], dtype=float)


def _ltg_igrl_a1_a2(x):
    # C++ guidance_ltg_igrl a1/a2; x==1 was exit(1).
    if x == 2:
        a1 = 1.0 / (1.0 - 0.5 * x * (1.001))
    else:
        a1 = 1.0 / (1.0 - 0.5 * x)
    if x == 1:
        raise ValueError("LTG Terminator: end-state cannot be reached")
    a2 = 1.0 / (1.0 - x)
    return a1, a2


class Rocket6Guidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        for field in (
            Field("mguide", 0, "int", "data", "guidance"),
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
            Field("UTIC", zeros3, "vec", "diag", "guidance"),
            Field("nstmax", 0, "int", "diag", "guidance"),
            Field("lamd", 0.0, "real", "diag", "guidance", plot),
            Field("dpd", 0.0, "real", "diag", "guidance", plot),
            Field("dbd", 0.0, "real", "diag", "guidance", plot),
            Field("ddb", 0.0, "real", "diag", "guidance", plot),
            Field("dvdb", 0.0, "real", "diag", "guidance", plot),
            Field("thtvddbx", 0.0, "real", "diag", "guidance", plot),
            Field("alphacomx", 0.0, "real", "out", "guidance"),
            Field("betacomx", 0.0, "real", "out", "guidance"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mguide = store.get("mguide")
        if mguide == 0:
            store.set("UTBC", np.zeros(3))
            store.set("UTIC", np.zeros(3))
            return
        if mguide != 5:
            raise ValueError(f"unknown mguide {mguide}")

        int_step = ctx.int_step
        ltg_step = store.get("ltg_step")
        init_flag = store.get("init_flag")
        time_ltg = store.get("time_ltg")
        utbc = np.asarray(store.get("UTBC"), dtype=float).copy()
        ltg_count = store.get("ltg_count")
        mprop = store.get("mprop")
        tbic = np.asarray(store.get("TBIC"), dtype=float)
        utic = np.zeros(3)

        if init_flag:
            init_flag = 0
            time_ltg = 0.0
        else:
            time_ltg += int_step
        if time_ltg > ltg_step * ltg_count:
            ltg_count += 1
            utic, mprop = self.guidance_ltg(vehicle, mprop, int_step, time_ltg)
            utbc = cadac_matmul(tbic, utic)

        store.set("init_flag", init_flag)
        store.set("time_ltg", time_ltg)
        store.set("ltg_count", ltg_count)
        store.set("mprop", mprop)
        store.set("UTBC", utbc)
        store.set("UTIC", utic)

    def guidance_ltg(self, vehicle, mprop, int_step, time_ltg):
        store = vehicle.store
        ltg_step = store.get("ltg_step")
        dbi_desired = store.get("dbi_desired")
        dvbi_desired = store.get("dvbi_desired")
        thtvdx_desired = store.get("thtvdx_desired")
        num_stages = store.get("num_stages")
        delay_ignition = store.get("delay_ignition")
        amin = store.get("amin")
        char_time1 = store.get("char_time1")
        char_time2 = store.get("char_time2")
        char_time3 = store.get("char_time3")
        exhaust_vel1 = store.get("exhaust_vel1")
        exhaust_vel2 = store.get("exhaust_vel2")
        exhaust_vel3 = store.get("exhaust_vel3")
        burnout_epoch1 = store.get("burnout_epoch1")
        burnout_epoch2 = store.get("burnout_epoch2")
        burnout_epoch3 = store.get("burnout_epoch3")
        lamd_limit = store.get("lamd_limit")
        beco_flag = store.get("beco_flag")
        inisw_flag = store.get("inisw_flag")
        skip_flag = store.get("skip_flag")
        ipas2_flag = store.get("ipas2_flag")
        print_flag = store.get("print_flag")
        rbias = np.asarray(store.get("RBIAS"), dtype=float).copy()
        rgrav = np.asarray(store.get("RGRAV"), dtype=float).copy()
        rgo = np.asarray(store.get("RGO"), dtype=float).copy()
        vgo = np.asarray(store.get("VGO"), dtype=float).copy()
        sdii = np.asarray(store.get("SDII"), dtype=float).copy()
        ud = np.asarray(store.get("UD"), dtype=float).copy()
        uy = np.asarray(store.get("UY"), dtype=float).copy()
        uz = np.asarray(store.get("UZ"), dtype=float).copy()
        tgo = store.get("tgo")
        nst = store.get("nst")
        dbi = store.get("dbi")
        dvbi = store.get("dvbi")
        thtvdx = store.get("thtvdx")
        fmassr = store.get("fmassr")
        vbiic = np.asarray(store.get("VBIIC"), dtype=float)
        sbiic = np.asarray(store.get("SBIIC"), dtype=float)
        tbic = np.asarray(store.get("TBIC"), dtype=float)
        fspcb = np.asarray(store.get("FSPCB"), dtype=float)

        utic = np.zeros(3)
        abii = cadac_matmul(tbic.T.copy(), fspcb)
        amag1 = _absolute(abii)

        spii = np.zeros(3)
        vpii = np.zeros(3)
        if inisw_flag:
            inisw_flag = 0
            spii = np.array(sbiic, dtype=float, copy=True)
            vpii = np.array(vbiic, dtype=float, copy=True)
            sdii, ud, uy, uz, _vmiss, vgo = self._crct(
                vehicle,
                sdii,
                ud,
                uy,
                uz,
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
        taun = np.array([char_time1, char_time2, char_time3], dtype=float)
        vexn = np.array([exhaust_vel1, exhaust_vel2, exhaust_vel3], dtype=float)
        botn = np.array(
            [0.0, burnout_epoch1, burnout_epoch2, burnout_epoch3], dtype=float
        )

        (
            tgop,
            burnt,
            ligrl_n,
            tgon,
            l_igrl,
            nstmax,
            tgo,
            nst,
            taun,
        ) = self._tgo(
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
            vehicle,
        )
        (
            s_igrl,
            j_igrl,
            q_igrl,
            h_igrl,
            p_igrl,
            j_over_l,
            tlam,
            qprime,
        ) = self._igrl(
            nst,
            nstmax,
            burnt,
            ligrl_n,
            tgon,
            taun,
            vexn,
            l_igrl,
            time_ltg,
        )
        ulam, lamd, rgo, ipas2_flag, rgrav = self._trate(
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
            vehicle,
        )

        tc = ulam + lamd * (time_ltg - tlam)
        if skip_flag:
            skip_flag += 1
            if skip_flag == 10:
                skip_flag = 0
        else:
            utic = _univec3(tc)

        spii, vpii, rgrav, rbias = self._pdct(
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
        sdii, ud, uy, uz, _vmiss, vgo = self._crct(
            vehicle,
            sdii,
            ud,
            uy,
            uz,
            vgo,
            dbi_desired,
            dvbi_desired,
            thtvdx_desired,
            spii,
            vpii,
            sbiic,
            vbiic,
        )

        if fmassr > 0:
            mprop = 4
        if tgo < 10 * int_step:
            beco_flag = 1
            mprop = 0

        ddb = 0.0
        dvdb = 0.0
        thtvddbx = 0.0
        if beco_flag and print_flag:
            print_flag = 0
            ddb = dbi_desired - dbi
            dvdb = dvbi_desired - dvbi
            thtvddbx = thtvdx_desired - thtvdx

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
        store.set("ddb", ddb)
        store.set("dvdb", dvdb)
        store.set("thtvddbx", thtvddbx)
        return utic, mprop

    def _tgo(
        self,
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
        vehicle,
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
            dum1 = taun[i]
            dum2 = burnt[i]
            ligrl_n[i] = -vexn[i] * math.log(1.0 - dum2 / dum1)
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
        dum3 = vexn[i - 1]
        burnt[i - 1] = taun[i - 1] * (1.0 - math.exp(-almx / dum3))
        tgo += burnt[i - 1]
        tgon[i - 1] = tgo
        l_igrl = vgom
        if ipas_flag:
            tgop = tgo
            ipas_flag = 0
        store.set("ipas_flag", ipas_flag)
        return tgop, burnt, ligrl_n, tgon, l_igrl, nstmax, tgo, nst, taun

    def _igrl(
        self,
        nst,
        nstmax,
        burnt,
        ligrl_n,
        tgon,
        taun,
        vexn,
        l_igrl,
        time_ltg,
    ):
        ls_igrl = 0.0
        s_igrl = 0.0
        j_igrl = 0.0
        q_igrl = 0.0
        h_igrl = 0.0
        p_igrl = 0.0
        for i in range(nst - 1, nstmax):
            tb = burnt[i]
            tga = tgon[i]
            dummy = taun[i]
            x = tb / dummy
            a1, a2 = _ltg_igrl_a1_a2(x)
            dummo = taun[i]
            aa = vexn[i] / dummo
            ll_igrl = ligrl_n[i]
            a1x = 4.0 * a1 - a2 - 3.0
            a2xsq = 2.0 * a2 - 4.0 * a1 + 2.0
            sa = (aa * tb * tb / 2.0) * (1.0 + a1x / 3.0 + a2xsq / 6.0)
            ha = (aa * tb * tb * tb / 3.0) * (1.0 + a1x * 0.75 + a2xsq * 0.6)
            # C++ a1x*(2/3) is integer division → 0
            ja = (aa * tb * tb / 2.0) * (1.0 + a1x * (2 // 3) + a2xsq / 2.0)
            qa = (aa * tb * tb * tb / 6.0) * (1.0 + a1x / 2.0 + a2xsq * 0.3)
            pa = (aa * tb * tb * tb * tb / 12.0) * (1.0 + a1x * 0.6 + a2xsq * 0.4)
            if i != nst - 1:
                t1 = tgon[i - 1]
                ha = ha + 2.0 * t1 * ja + t1 * t1 * ll_igrl
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

    def _trate(
        self,
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
        vehicle,
    ):
        ulam = np.zeros(3)
        lamd = np.zeros(3)
        if vgom == 0:
            vehicle.store.set("lamd", 0.0)
            return ulam, lamd, rgo, ipas2_flag, rgrav
        ulam = _univec3(vgo)
        lamd = np.zeros(3)
        if ipas2_flag:
            ipas2_flag = 0
            rgo = ulam * s_igrl
        rgo, rgrav = self._trate_rtgo(
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
        )
        denom = q_igrl - s_igrl * j_over_l
        if denom != 0:
            lamd = (rgo - ulam * s_igrl) * (1.0 / denom)
        else:
            lamd = np.zeros(3)
        lamd_mag = _absolute(lamd)
        if lamd_mag >= lamd_limit:
            ulmd = _univec3(lamd)
            lamd = ulmd * lamd_limit
        lamd_mag = _absolute(lamd)
        vehicle.store.set("lamd", lamd_mag)
        return ulam, lamd, rgo, ipas2_flag, rgrav

    def _trate_rtgo(
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
        rgox = _dot(rgo_local, ud)
        rgoy = _dot(rgo_local, uy)
        rgoxy = ud * rgox + uy * rgoy
        num = _dot(rgoxy, ulam)
        denom = _dot(ulam, uz)
        if denom == 0:
            return rgo, rgrav
        rgoz = (s_igrl - num) / denom
        rgo = rgoxy + uz * rgoz
        return rgo, rgrav

    def _pdct(
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

    def _crct(
        self,
        vehicle,
        sdii,
        ud,
        uy,
        uz,
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
            ud * math.sin(thtvdx_desired * RAD)
            + uz * math.cos(thtvdx_desired * RAD)
        ) * dvbi_desired
        vmiss = vpii - vdii
        vgo = vgo - vmiss
        dpi = _absolute(spii)
        ddi = _absolute(sdii)
        dpd = dpi - ddi
        dbi_now = _absolute(sbiic)
        dbd = dbi_now - ddi
        vehicle.store.set("dpd", dpd)
        vehicle.store.set("dbd", dbd)
        return sdii, ud, uy, uz, vmiss, vgo
