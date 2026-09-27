from math import cos, exp, fabs, log, sin, sqrt
from types import SimpleNamespace

import numpy as np

from cadac.constants import AGRAV, DEG, RAD, REARTH
from cadac.kernel.state import Field
from cadac.math.frames import cadac_inverse, cadac_matmul, mat2tr, polar_from_cart, skew
from cadac.tables.lookup import Datadeck
from cadac.vehicles.round6.hyper6.vehicle import Hyper6

RTOL = 1e-12
INT_STEP = 0.01
TBIC = mat2tr(0.3, -0.2)
SBIIC = np.array([REARTH + 50e3, 1.0e5, 2.0e5])
VBIIC = np.array([400.0, 2500.0, 200.0])
FSPCB = np.array([34.0, 0.1, -0.2])
STBIK = np.array([12000.0, -3500.0, 800.0])
VTBIK = np.array([-180.0, 40.0, 25.0])
STCII = np.array([7005000.0, 98000.0, 201000.0])
VTCII = np.array([80.0, 7480.0, 40.0])
STII = np.array([7.1e6, 0.0, 0.0])
VTII = np.array([0.0, 7500.0, 0.0])
SATL = np.array([100.0, -50.0, 20.0])
DBI_DESIRED = 6470e3
DVBI_DESIRED = 6600.0
THTVDX_DESIRED = 1.0
DELAY_IGNITION = 0.1
CHAR_TIME = (81.9, 112.2, 100.0)
EXHAUST_VEL = (2795.0, 2785.0, 2700.0)
BURNOUT_EPOCH = (51.5, 126.0, 200.0)


def _execute_module(veh, name, method="execute", ctx=None):
    if ctx is None:
        ctx = SimpleNamespace(int_step=INT_STEP)
    for module in veh.modules:
        if module.name == name:
            getattr(module, method)(veh, ctx)
            return
    raise AssertionError(f"module {name!r} not on vehicle")


def _close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0.0)


def _ensure(store, name, value, ftype):
    if name not in store:
        store.define(Field(name, value, ftype, "data", "test"))
    else:
        store.set(name, value)


def _hyper6_vehicle(mguide, wp=(0.1, 0.05, 15000.0)):
    deck = Datadeck.from_tables([])
    veh = Hyper6("hyper6", aero_deck=deck, prop_deck=deck, events=[])
    veh.define()
    store = veh.store
    lon, lat, alt = wp
    store.set("mguide", mguide)
    store.set("wp_lonx", lon)
    store.set("wp_latx", lat)
    store.set("wp_alt", alt)
    store.set("time", 10.0)
    store.set("grav", 9.81)
    store.set("ltg_step", INT_STEP)
    store.set("num_stages", 2)
    store.set("dbi_desired", DBI_DESIRED)
    store.set("dvbi_desired", DVBI_DESIRED)
    store.set("thtvdx_desired", THTVDX_DESIRED)
    store.set("delay_ignition", DELAY_IGNITION)
    store.set("amin", 3.0)
    store.set("lamd_limit", 0.01)
    store.set("char_time1", CHAR_TIME[0])
    store.set("char_time2", CHAR_TIME[1])
    store.set("char_time3", CHAR_TIME[2])
    store.set("exhaust_vel1", EXHAUST_VEL[0])
    store.set("exhaust_vel2", EXHAUST_VEL[1])
    store.set("exhaust_vel3", EXHAUST_VEL[2])
    store.set("burnout_epoch1", BURNOUT_EPOCH[0])
    store.set("burnout_epoch2", BURNOUT_EPOCH[1])
    store.set("burnout_epoch3", BURNOUT_EPOCH[2])
    store.set("SBIIC", SBIIC)
    store.set("VBIIC", VBIIC)
    store.set("TBIC", TBIC)
    store.set("FSPCB", FSPCB)
    store.set("mprop", 0)
    store.set("gnav", 3.0)
    store.set("gnavpn", 4.0)
    store.set("gnavps", 1.5)
    store.set("time_gs", 200.0)
    store.set("num_burns", 4)
    store.set("closing_rate", -30.0)
    store.set("orbital_rate", 0.0011)
    store.set("satl1", float(SATL[0]))
    store.set("satl2", float(SATL[1]))
    store.set("satl3", float(SATL[2]))
    store.set("alcomx", 7.0)
    store.set("ancomx", 7.0)
    store.set("phicomx", 7.0)
    _ensure(store, "mseek", 4, "int")
    _ensure(store, "STBIK", STBIK, "vec")
    _ensure(store, "VTBIK", VTBIK, "vec")
    _ensure(store, "STCII", STCII, "vec")
    _ensure(store, "VTCII", VTCII, "vec")
    _ensure(store, "STII", STII, "vec")
    _ensure(store, "VTII", VTII, "vec")
    return veh


def _abs3(vec):
    return sqrt(float(vec[0]) ** 2 + float(vec[1]) ** 2 + float(vec[2]) ** 2)


def _unit(vec):
    scale = _abs3(vec)
    if scale == 0.0:
        return np.zeros(3)
    return np.array(
        [float(vec[0]) / scale, float(vec[1]) / scale, float(vec[2]) / scale]
    )


def _dot(a, b):
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _unit_cross(a, b):
    v1 = float(a[1]) * float(b[2]) - float(a[2]) * float(b[1])
    v2 = float(a[2]) * float(b[0]) - float(a[0]) * float(b[2])
    v3 = float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])
    scale = sqrt(v1 * v1 + v2 * v2 + v3 * v3)
    if scale == 0.0:
        raise ValueError("divide by zero in unit cross")
    return np.array([v1 / scale, v2 / scale, v3 / scale])


def _cpp_clock(store, int_step):
    """Hyper::guidance mguide==5 clock (guidance.cpp)."""
    init_flag = store.get("init_flag")
    time_ltg = store.get("time_ltg")
    ltg_count = store.get("ltg_count")
    if init_flag:
        init_flag = 0
        time_ltg = 0.0
    else:
        time_ltg = time_ltg + int_step
    ltg_count = ltg_count + 1
    ratio = int(store.get("ltg_step") / int_step)
    ltg_flag = ltg_count - (ltg_count // ratio) * ratio
    return init_flag, time_ltg, ltg_count, ltg_flag == 0


def _igrl_a1_a2(x):
    if x == 2:
        a1 = 1.0 / (1.0 - 0.5 * x * 1.001)
    else:
        a1 = 1.0 / (1.0 - 0.5 * x)
    if x == 1:
        raise ValueError("LTG Terminator: end-state cannot be reached")
    a2 = 1.0 / (1.0 - x)
    return a1, a2


def _cpp_ltg_utbc(store, int_step, time_ltg):
    """One Hyper::guidance_ltg call plus UTBC=TBIC*UTIC. Reads store, does not write it."""
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
    inisw_flag = store.get("inisw_flag")
    skip_flag = store.get("skip_flag")
    ipas2_flag = store.get("ipas2_flag")
    ipas_flag = store.get("ipas_flag")
    vgo = np.array(store.get("VGO"), dtype=float, copy=True)
    rgrav = np.array(store.get("RGRAV"), dtype=float, copy=True)
    rgo = np.array(store.get("RGO"), dtype=float, copy=True)
    sdii = np.array(store.get("SDII"), dtype=float, copy=True)
    ud = np.array(store.get("UD"), dtype=float, copy=True)
    uy = np.array(store.get("UY"), dtype=float, copy=True)
    uz = np.array(store.get("UZ"), dtype=float, copy=True)
    rbias = np.array(store.get("RBIAS"), dtype=float, copy=True)
    tgo = store.get("tgo")
    nst = store.get("nst")
    sbiic = np.array(store.get("SBIIC"), dtype=float, copy=True)
    vbiic = np.array(store.get("VBIIC"), dtype=float, copy=True)
    tbic = np.array(store.get("TBIC"), dtype=float, copy=True)
    fspcb = np.array(store.get("FSPCB"), dtype=float, copy=True)
    mprop = store.get("mprop")

    abii = cadac_matmul(np.array(tbic.T, dtype=float, copy=True), fspcb)
    amag1 = _abs3(abii)
    if inisw_flag:
        spii = np.array(sbiic, copy=True)
        vpii = np.array(vbiic, copy=True)
        sdii, ud, uy, uz, vgo = _cpp_crct(
            vgo, dbi_desired, dvbi_desired, thtvdx_desired, spii, vpii, sbiic, vbiic
        )
    else:
        vgo = vgo - abii * ltg_step
    vgom = _abs3(vgo)

    if ipas_flag:
        nst = 1
    tgop = tgo
    tgo = 0.0
    l_igrl = 0.0
    nstmax = num_stages
    if time_ltg >= botn[nst]:
        nst += 1
    burnt = np.zeros(3)
    ligrl_n = np.zeros(3)
    tgon = np.zeros(3)
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

    s_igrl, j_igrl, q_igrl, h_igrl, p_igrl, j_over_l, tlam, qprime = _cpp_igrl(
        nst, nstmax, burnt, ligrl_n, tgon, taun, vexn, l_igrl, time_ltg
    )
    ulam = np.zeros(3)
    lamd = np.zeros(3)
    if vgom != 0.0:
        ulam = _unit(vgo)
        if ipas2_flag:
            rgo = ulam * s_igrl
        rgo, rgrav = _cpp_rtgo(
            rgo, rgrav, tgo, tgop, sdii, sbiic, vbiic, rbias, ulam, ud, uy, uz, s_igrl
        )
        denom = q_igrl - s_igrl * j_over_l
        if denom != 0.0:
            lamd = (rgo - ulam * s_igrl) * (1.0 / denom)
        else:
            lamd = np.zeros(3)
        lamd_mag = _abs3(lamd)
        if lamd_mag >= lamd_limit:
            lamd = _unit(lamd) * lamd_limit
    tc = ulam + lamd * (time_ltg - tlam)
    utic = np.zeros(3)
    if not skip_flag:
        utic = _unit(tc)
    utbc = np.array(store.get("UTBC"), dtype=float, copy=True)
    if not skip_flag:
        utbc = cadac_matmul(tbic, utic)
    burntime = botn[nst] - botn[nst - 1] - delay_ignition
    if burntime > 0.0:
        mprop = 3
    if tgo < 10.0 * int_step:
        mprop = 0
    isp_fuel = vexn[nst - 1] / AGRAV
    return {
        "UTBC": utbc,
        "mprop": mprop,
        "isp_fuel": isp_fuel,
        "burntime": burntime,
        "nst": nst,
    }


def _cpp_igrl(nst, nstmax, burnt, ligrl_n, tgon, taun, vexn, l_igrl, time_ltg):
    ls_igrl = 0.0
    s_igrl = 0.0
    j_igrl = 0.0
    q_igrl = 0.0
    h_igrl = 0.0
    p_igrl = 0.0
    for i in range(nst - 1, nstmax):
        tb = burnt[i]
        tga = tgon[i]
        x = tb / taun[i]
        a1, a2 = _igrl_a1_a2(x)
        aa = vexn[i] / taun[i]
        ll_igrl = ligrl_n[i]
        a1x = 4.0 * a1 - a2 - 3.0
        a2xsq = 2.0 * a2 - 4.0 * a1 + 2.0
        sa = (aa * tb * tb / 2.0) * (1.0 + a1x / 3.0 + a2xsq / 6.0)
        ja = (aa * tb * tb / 2.0) * (1.0 + a1x * (2 // 3) + a2xsq / 2.0)
        qa = (aa * tb * tb * tb / 6.0) * (1.0 + a1x / 2.0 + a2xsq * 0.3)
        pa = (aa * tb * tb * tb * tb / 12.0) * (1.0 + a1x * 0.6 + a2xsq * 0.4)
        if i != nst - 1:
            t1 = tgon[i - 1]
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


def _cpp_rtgo(rgo, rgrav, tgo, tgop, sdii, sbiic, vbiic, rbias, ulam, ud, uy, uz, s_igrl):
    rgrav = rgrav * (tgo / tgop) * (tgo / tgop)
    rgo_local = sdii - (sbiic + vbiic * tgo + rgrav) - rbias
    rgoxy = ud * _dot(rgo_local, ud) + uy * _dot(rgo_local, uy)
    num = _dot(rgoxy, ulam)
    denom = _dot(ulam, uz)
    if denom == 0.0:
        return rgo, rgrav
    rgoz = (s_igrl - num) / denom
    return rgoxy + uz * rgoz, rgrav


def _cpp_crct(vgo, dbi_desired, dvbi_desired, thtvdx_desired, spii, vpii, sbiic, vbiic):
    ud = _unit(spii)
    sdii = ud * dbi_desired
    uy = _unit_cross(vbiic, sbiic)
    uz = _unit_cross(ud, uy)
    vdii = (
        ud * sin(thtvdx_desired * RAD) + uz * cos(thtvdx_desired * RAD)
    ) * dvbi_desired
    vgo = vgo - (vpii - vdii)
    return sdii, ud, uy, uz, vgo


def _cpp_pronav(store):
    """Hyper::guidance_pronav acceleration and UTBBC (guidance.cpp)."""
    gnav = store.get("gnav")
    mseek = store.get("mseek")
    tbic = np.array(store.get("TBIC"), dtype=float, copy=True)
    if mseek > 3:
        stbic = np.array(store.get("STBIK"), dtype=float, copy=True)
        vtbic = np.array(store.get("VTBIK"), dtype=float, copy=True)
    else:
        stbic = np.array(store.get("STCII"), dtype=float) - np.array(
            store.get("SBIIC"), dtype=float
        )
        vtbic = np.array(store.get("VTCII"), dtype=float) - np.array(
            store.get("VBIIC"), dtype=float
        )
    dtbc = _abs3(stbic)
    utbic = _unit(stbic)
    utbbc = cadac_matmul(tbic, utbic)
    polar = polar_from_cart(utbbc)
    dvtbc = _dot(utbic, vtbic)
    tgoc = fabs(dtbc / dvtbc)
    woiic = cadac_matmul(skew(utbic), vtbic) * (1.0 / dtbc)
    aapnb = (
        cadac_matmul(cadac_matmul(tbic, skew(woiic)), utbic) * gnav * fabs(dvtbc)
    )
    accomx = aapnb * (1.0 / AGRAV)
    stii = np.array(store.get("STII"), dtype=float, copy=True)
    vtii = np.array(store.get("VTII"), dtype=float, copy=True)
    uh1 = _unit(stii)
    uh3 = _unit(cadac_matmul(skew(stii), vtii))
    uh2 = cadac_matmul(skew(uh3), uh1)
    thi = np.vstack([uh1, uh2, uh3])
    sbthc = cadac_matmul(thi, stbic * (-1.0))
    return {
        "aycomx": float(accomx[1]),
        "azcomx": float(accomx[2]),
        "UTBC": utbbc,
        "tgoc": tgoc,
        "dtbc": dtbc,
        "psiobcx": float(polar[1]) * DEG,
        "thtobcx": float(polar[2]) * DEG,
        "SBTHC": sbthc,
    }


def _cpp_agl(store):
    """Hyper::guidance_AGL acceleration and UTBBC (guidance.cpp)."""
    gnavpn = store.get("gnavpn")
    gnavps = store.get("gnavps")
    tbic = np.array(store.get("TBIC"), dtype=float, copy=True)
    stbik = np.array(store.get("STBIK"), dtype=float, copy=True)
    vtbik = np.array(store.get("VTBIK"), dtype=float, copy=True)
    dtbc = _abs3(stbik)
    utbik = _unit(stbik)
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
    return {
        "aycomx": float(accomx[1]),
        "azcomx": float(accomx[2]),
        "UTBC": utbbc,
        "tgoc": tgoc,
        "dtbc": dtbc,
        "psiobcx": float(polar[1]) * DEG,
        "thtobcx": float(polar[2]) * DEG,
    }


def _cpp_level_frame(stcii, vtcii):
    ul1 = _unit(vtcii)
    ul3 = _unit(stcii) * (-1.0)
    ul2 = cadac_matmul(skew(ul3), ul1)
    return np.vstack([ul1, ul2, ul3])


def _cpp_glideslope(store):
    """Hyper::guidance_glideslope UTBC (guidance.cpp). Reads store, does not write it."""
    time = store.get("time")
    time_gs = store.get("time_gs")
    num_burns = store.get("num_burns")
    closing_rate = store.get("closing_rate")
    orbital_rate = store.get("orbital_rate")
    satl = np.array(
        [store.get("satl1"), store.get("satl2"), store.get("satl3")], dtype=float
    )
    sbiic = np.array(store.get("SBIIC"), dtype=float, copy=True)
    vbiic = np.array(store.get("VBIIC"), dtype=float, copy=True)
    stcii = np.array(store.get("STCII"), dtype=float, copy=True)
    vtcii = np.array(store.get("VTCII"), dtype=float, copy=True)
    tbic = np.array(store.get("TBIC"), dtype=float, copy=True)
    mseek = store.get("mseek")
    mprop = store.get("mprop")
    gs_flag = store.get("gs_flag")
    dtime_gs = store.get("dtime_gs")
    length_gs = store.get("length_gs")
    para_gs = store.get("para_gs")
    ub0al = np.array(store.get("UB0AL"), dtype=float, copy=True)
    epoch_gs = store.get("epoch_gs")
    counter_gs = store.get("counter_gs")
    vbtlm = np.array(store.get("VBTLM"), dtype=float, copy=True)
    delta_v = np.array(store.get("DELTA_V"), dtype=float, copy=True)
    burn_flag = store.get("burn_flag")

    tli = _cpp_level_frame(stcii, vtcii)
    sbti = sbiic - stcii
    vbti = vbiic - vtcii
    sbtl = cadac_matmul(tli, sbti)
    vbtl = cadac_matmul(tli, vbti)
    if gs_flag:
        dtime_gs = time_gs / num_burns
        sb0al = sbtl - satl
        ub0al = _unit(sb0al)
        length_gs = _abs3(sb0al)
        para_gs = (_dot(ub0al, vbtl) - closing_rate) / length_gs
        epoch_gs = time
        counter_gs = 0
    if counter_gs < num_burns and time >= (dtime_gs * counter_gs + epoch_gs):
        w = orbital_rate
        wt = orbital_rate * dtime_gs
        swt = sin(wt)
        cwt = cos(wt)
        phiss = np.array(
            [[1.0, 0.0, 6.0 * (wt - swt)], [0.0, cwt, 0.0], [0.0, 0.0, 4.0 - 3.0 * cwt]],
            dtype=float,
        )
        phisv = np.array(
            [
                [4.0 * swt / w - 3.0 * dtime_gs, 0.0, 2.0 * (1.0 - cwt) / w],
                [0.0, swt / w, 0.0],
                [-2.0 * (1.0 - cwt) / w, 0.0, swt / w],
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
        cadac_matmul(tbic, np.array(tli.T, dtype=float, copy=True)), _unit(delta_v)
    )
    ev = delta_v + vbtlm - vbtl
    if float(utb[0]) > 0.9:
        if _abs3(ev) > fabs(closing_rate) and burn_flag:
            mprop = 4
        else:
            mprop = 0
    if mseek == 3:
        utb = cadac_matmul(tbic, _unit(sbti) * (-1.0))
    return {"UTBC": utb, "mprop": mprop, "gs_flag": 0 if store.get("gs_flag") else gs_flag}


def test_hyper6_ltg_advances_clock():
    veh = _hyper6_vehicle(mguide=5)
    _execute_module(veh, "guidance")
    _execute_module(veh, "guidance")
    assert veh.store.get("ltg_count") == 2


def test_hyper6_pronav_and_agl_issue_accel():
    for mguide in (6, 7):
        veh = _hyper6_vehicle(mguide=mguide)
        _execute_module(veh, "guidance")
        assert veh.store.get("aycomx") != 0.0 or veh.store.get("azcomx") != 0.0


def test_hyper6_glideslope_writes_utbc():
    veh = _hyper6_vehicle(mguide=8)
    _execute_module(veh, "guidance")
    assert np.linalg.norm(np.asarray(veh.store.get("UTBC"))) > 0


def test_hyper6_ltg_clock_and_utbc_match_cpp():
    veh = _hyper6_vehicle(mguide=5)
    store = veh.store
    store.set("skip_flag", 0)
    init_flag, time_ltg, ltg_count, due = _cpp_clock(store, INT_STEP)
    assert due
    expected = _cpp_ltg_utbc(store, INT_STEP, time_ltg)
    _execute_module(veh, "guidance")
    assert store.get("ltg_count") == ltg_count
    assert store.get("init_flag") == init_flag
    _close(store.get("time_ltg"), time_ltg)
    _close(store.get("UTBC"), expected["UTBC"])
    assert store.get("mprop") == expected["mprop"]
    _close(store.get("isp_fuel"), expected["isp_fuel"])
    _close(store.get("burntime"), expected["burntime"])
    assert np.linalg.norm(np.asarray(store.get("UTBC"))) > 0.0


def test_hyper6_ltg_ratio_holds_previous_utbc():
    veh = _hyper6_vehicle(mguide=5)
    store = veh.store
    store.set("ltg_step", 0.05)
    sentinel = np.array([0.2, 0.3, 0.4])
    store.set("UTBC", sentinel)
    _init, time_ltg, ltg_count, due = _cpp_clock(store, INT_STEP)
    assert due is False
    assert ltg_count == 1
    _execute_module(veh, "guidance")
    assert store.get("ltg_count") == ltg_count
    _close(store.get("time_ltg"), time_ltg)
    _close(store.get("UTBC"), sentinel)


def test_hyper6_pronav_matches_cpp():
    veh = _hyper6_vehicle(mguide=6)
    expected = _cpp_pronav(veh.store)
    assert expected["aycomx"] != 0.0 or expected["azcomx"] != 0.0
    _execute_module(veh, "guidance")
    _close(veh.store.get("aycomx"), expected["aycomx"])
    _close(veh.store.get("azcomx"), expected["azcomx"])
    _close(veh.store.get("UTBC"), expected["UTBC"])
    _close(veh.store.get("tgoc"), expected["tgoc"])
    _close(veh.store.get("dtbc"), expected["dtbc"])
    _close(veh.store.get("psiobcx"), expected["psiobcx"])
    _close(veh.store.get("thtobcx"), expected["thtobcx"])
    _close(veh.store.get("SBTHC"), expected["SBTHC"])
    _close(veh.store.get("alcomx"), 0.0)
    _close(veh.store.get("ancomx"), 0.0)
    _close(veh.store.get("phicomx"), 0.0)


def test_hyper6_pronav_datalink_matches_cpp():
    veh = _hyper6_vehicle(mguide=6)
    veh.store.set("mseek", 2)
    expected = _cpp_pronav(veh.store)
    assert expected["aycomx"] != 0.0 or expected["azcomx"] != 0.0
    _execute_module(veh, "guidance")
    _close(veh.store.get("aycomx"), expected["aycomx"])
    _close(veh.store.get("azcomx"), expected["azcomx"])
    _close(veh.store.get("UTBC"), expected["UTBC"])


def test_hyper6_agl_matches_cpp():
    veh = _hyper6_vehicle(mguide=7)
    expected = _cpp_agl(veh.store)
    assert expected["aycomx"] != 0.0 or expected["azcomx"] != 0.0
    _execute_module(veh, "guidance")
    _close(veh.store.get("aycomx"), expected["aycomx"])
    _close(veh.store.get("azcomx"), expected["azcomx"])
    _close(veh.store.get("UTBC"), expected["UTBC"])
    _close(veh.store.get("tgoc"), expected["tgoc"])
    _close(veh.store.get("dtbc"), expected["dtbc"])
    _close(veh.store.get("psiobcx"), expected["psiobcx"])
    _close(veh.store.get("thtobcx"), expected["thtobcx"])


def test_hyper6_glideslope_utbc_matches_cpp():
    veh = _hyper6_vehicle(mguide=8)
    expected = _cpp_glideslope(veh.store)
    assert np.linalg.norm(expected["UTBC"]) > 0.0
    _execute_module(veh, "guidance")
    _close(veh.store.get("UTBC"), expected["UTBC"])
    assert veh.store.get("gs_flag") == 0
    assert veh.store.get("mprop") == expected["mprop"]
    _close(veh.store.get("aycomx"), 0.0)
    _close(veh.store.get("azcomx"), 0.0)


def test_hyper6_glideslope_mseek3_points_utbc():
    veh = _hyper6_vehicle(mguide=8)
    veh.store.set("mseek", 3)
    expected = _cpp_glideslope(veh.store)
    _execute_module(veh, "guidance")
    _close(veh.store.get("UTBC"), expected["UTBC"])


def test_hyper6_unknown_mguide_still_raises():
    for mguide in (-1, 99):
        veh = _hyper6_vehicle(mguide=mguide)
        try:
            _execute_module(veh, "guidance")
        except ValueError as exc:
            assert "unknown mguide" in str(exc)
        else:
            raise AssertionError(f"mguide {mguide} did not raise")
