import math

import numpy as np

from cadac.constants import DEG, EPS, PI
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field
from cadac.math.frames import mat3tr

SMALL = 1e-7


def _sign(variable):
    if variable < 0.0:
        return -1
    return 1


class Sam6Kinematics:
    name = "kinematics"

    def define(self, vehicle):
        store = vehicle.store
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        scrn_plot_com = ("scrn", "plot", "com")
        scrn_plot = ("scrn", "plot")
        plot = ("plot",)
        scrn = ("scrn",)
        for field in (
            Field("time", 0.0, "real", "exec", "kinematics", scrn_plot_com),
            Field("event_time", 0.0, "real", "exec", "kinematics"),
            Field("int_step_new", 0.0, "real", "exec", "kinematics"),
            Field("out_step_fact", 0.0, "real", "exec", "kinematics"),
            Field("stop", 0, "int", "exec", "kinematics"),
            Field("lconv", 0, "int", "exec", "kinematics", plot),
            Field("launch_delay", 99999.0, "real", "exec", "kinematics"),
            Field("launch_epoch", 0.0, "real", "exec", "kinematics"),
            Field("launch_time", 0.0, "real", "exec", "kinematics"),
            Field("msl_time", 0.0, "real", "exec", "kinematics", plot),
            Field("ck", 50.0, "real", "data", "kinematics"),
            Field("q0d", 0.0, "real", "state", "kinematics"),
            Field("q0", 0.0, "real", "state", "kinematics"),
            Field("q1d", 0.0, "real", "state", "kinematics"),
            Field("q1", 0.0, "real", "state", "kinematics"),
            Field("q2d", 0.0, "real", "state", "kinematics"),
            Field("q2", 0.0, "real", "state", "kinematics"),
            Field("q3d", 0.0, "real", "state", "kinematics"),
            Field("q3", 0.0, "real", "state", "kinematics"),
            Field("TBL", zeros33, "mat", "out", "kinematics"),
            Field("psibl", 0.0, "real", "diag", "kinematics"),
            Field("thtbl", 0.0, "real", "diag", "kinematics"),
            Field("phibl", 0.0, "real", "diag", "kinematics"),
            Field("psiblx", 0.0, "real", "in/diag", "kinematics", scrn_plot),
            Field("thtblx", 0.0, "real", "in/diag", "kinematics", scrn_plot),
            Field("phiblx", 0.0, "real", "in/diag", "kinematics", scrn_plot),
            Field("alppx", 0.0, "real", "out", "kinematics", plot),
            Field("phipx", 0.0, "real", "out", "kinematics", plot),
            Field("alpp", 0.0, "real", "out", "kinematics"),
            Field("phip", 0.0, "real", "out", "kinematics"),
            Field("alphax", 0.0, "real", "diag", "kinematics", scrn_plot),
            Field("betax", 0.0, "real", "diag", "kinematics", scrn_plot),
            Field("ortho_error", 0.0, "real", "diag", "kinematics", scrn),
            Field("etbl", 0.0, "real", "diag", "kinematics"),
            Field("TLB", zeros33, "mat", "diag", "kinematics"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        launch_delay = store.get("launch_delay")
        psiblx = store.get("psiblx")
        thtblx = store.get("thtblx")
        phiblx = store.get("phiblx")
        spsi = math.sin(psiblx / (2.0 * DEG))
        cpsi = math.cos(psiblx / (2.0 * DEG))
        stht = math.sin(thtblx / (2.0 * DEG))
        ctht = math.cos(thtblx / (2.0 * DEG))
        sphi = math.sin(phiblx / (2.0 * DEG))
        cphi = math.cos(phiblx / (2.0 * DEG))
        q0 = cpsi * ctht * cphi + spsi * stht * sphi
        q1 = cpsi * ctht * sphi - spsi * stht * cphi
        q2 = cpsi * stht * cphi + spsi * ctht * sphi
        q3 = -cpsi * stht * sphi + spsi * ctht * cphi
        tbl = mat3tr(psiblx / DEG, thtblx / DEG, phiblx / DEG)
        store.set("time", ctx.sim_time)
        store.set("int_step_new", ctx.int_step)
        store.set("launch_epoch", launch_delay)
        store.set("q0", q0)
        store.set("q1", q1)
        store.set("q2", q2)
        store.set("q3", q3)
        store.set("TBL", tbl)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        int_step_new = store.get("int_step_new")
        out_step_fact = store.get("out_step_fact")
        ck = store.get("ck")
        launch_epoch = store.get("launch_epoch")
        vbeb = store.get("VBEB")
        pp = store.get("pp")
        qq = store.get("qq")
        rr = store.get("rr")
        q0d = store.get("q0d")
        q0 = store.get("q0")
        q1d = store.get("q1d")
        q1 = store.get("q1")
        q2d = store.get("q2d")
        q2 = store.get("q2")
        q3d = store.get("q3d")
        q3 = store.get("q3")

        time = ctx.sim_time
        launch_time = time - launch_epoch
        ctx.int_step = int_step_new
        ctx.out_fact = out_step_fact
        int_step = ctx.int_step

        lnch_delay = _radar_lnch_delay(ctx)
        del_time = time - lnch_delay
        msl_time = 0.0
        if del_time >= 0.0:
            msl_time = del_time

        ortho_error = 1.0 - (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        names = store.names()
        trcond = store.get("trcond") if "trcond" in names else None
        if trcond is not None and "trortho" in names:
            if math.fabs(ortho_error) > store.get("trortho"):
                trcond = 1

        new_q0d = 0.5 * (-pp * q1 - qq * q2 - rr * q3) + ck * ortho_error * q0
        new_q1d = 0.5 * (pp * q0 + rr * q2 - qq * q3) + ck * ortho_error * q1
        new_q2d = 0.5 * (qq * q0 - rr * q1 + pp * q3) + ck * ortho_error * q2
        new_q3d = 0.5 * (rr * q0 + qq * q1 - pp * q2) + ck * ortho_error * q3
        q0 = integrate(new_q0d, q0d, q0, int_step)
        q1 = integrate(new_q1d, q1d, q1, int_step)
        q2 = integrate(new_q2d, q2d, q2, int_step)
        q3 = integrate(new_q3d, q3d, q3, int_step)
        q0d = new_q0d
        q1d = new_q1d
        q2d = new_q2d
        q3d = new_q3d

        tbl = np.zeros((3, 3))
        tbl[0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
        tbl[0, 1] = 2.0 * (q1 * q2 + q0 * q3)
        tbl[0, 2] = 2.0 * (q1 * q3 - q0 * q2)
        tbl[1, 0] = 2.0 * (q1 * q2 - q0 * q3)
        tbl[1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
        tbl[1, 2] = 2.0 * (q2 * q3 + q0 * q1)
        tbl[2, 0] = 2.0 * (q1 * q3 + q0 * q2)
        tbl[2, 1] = 2.0 * (q2 * q3 - q0 * q1)
        tbl[2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

        tlb = tbl.T.copy()
        ubl = tlb @ tbl
        e1 = ubl[0, 0] - 1.0
        e2 = ubl[1, 1] - 1.0
        e3 = ubl[2, 2] - 1.0
        etbl = math.sqrt(e1 * e1 + e2 * e2 + e3 * e3)

        tbl13 = tbl[0, 2]
        tbl11 = tbl[0, 0]
        tbl33 = tbl[2, 2]
        tbl12 = tbl[0, 1]
        tbl23 = tbl[1, 2]
        if math.fabs(tbl13) < 1.0:
            thtbl = math.asin(-tbl13)
            cthtbl = math.cos(thtbl)
        else:
            thtbl = PI / 2.0 * _sign(-tbl13)
            cthtbl = EPS
        cpsi = tbl11 / cthtbl
        if math.fabs(cpsi) >= 1.0:
            cpsi = (1.0 - EPS) * _sign(cpsi)
        psibl = math.acos(cpsi) * _sign(tbl12)
        cphi = tbl33 / cthtbl
        if math.fabs(cphi) >= 1.0:
            cphi = (1.0 - EPS) * _sign(cphi)
        phibl = math.acos(cphi) * _sign(tbl23)
        psiblx = DEG * psibl
        thtblx = DEG * thtbl
        phiblx = DEG * phibl

        vbeb1 = vbeb[0]
        vbeb2 = vbeb[1]
        vbeb3 = vbeb[2]
        alpha = math.atan2(vbeb3, vbeb1)
        dvbe = float(np.linalg.norm(vbeb))
        beta = math.asin(vbeb2 / dvbe)
        dum = vbeb1 / dvbe
        if math.fabs(dum) >= 1.0:
            dum = (1.0 - EPS) * _sign(dum)
        alpp = math.acos(dum)
        if math.fabs(vbeb2) < EPS and math.fabs(vbeb3) < EPS:
            phip = 0.0
        elif math.fabs(vbeb2) < SMALL:
            phip = math.atan2(SMALL, vbeb3)
        else:
            phip = math.atan2(vbeb2, vbeb3)
        alphax = alpha * DEG
        betax = beta * DEG
        alppx = alpp * DEG
        phipx = phip * DEG

        if trcond is not None and "tralp" in names:
            if alpp > store.get("tralp"):
                trcond = 2

        store.set("q0d", q0d)
        store.set("q0", q0)
        store.set("q1d", q1d)
        store.set("q1", q1)
        store.set("q2d", q2d)
        store.set("q2", q2)
        store.set("q3d", q3d)
        store.set("q3", q3)
        store.set("time", time)
        store.set("event_time", ctx.event_time)
        store.set("int_step_new", int_step_new)
        store.set("launch_time", launch_time)
        store.set("msl_time", msl_time)
        store.set("TBL", tbl)
        store.set("alppx", alppx)
        store.set("phipx", phipx)
        store.set("alpp", alpp)
        store.set("phip", phip)
        store.set("psibl", psibl)
        store.set("thtbl", thtbl)
        store.set("phibl", phibl)
        store.set("psiblx", psiblx)
        store.set("thtblx", thtblx)
        store.set("phiblx", phiblx)
        store.set("alphax", alphax)
        store.set("betax", betax)
        store.set("ortho_error", ortho_error)
        store.set("etbl", etbl)
        store.set("TLB", tlb)
        if trcond is not None:
            store.set("trcond", trcond)

    def terminate(self, vehicle, ctx):
        pass


def _radar_lnch_delay(ctx):
    combus = ctx.combus
    if not combus:
        return 0.0
    radar = None
    for packet in combus:
        if packet.type == "RADAR0":
            radar = packet
    if radar is None:
        return 0.0
    k = sum(
        1
        for i, packet in enumerate(combus)
        if packet.type == "MISSILE6" and i < ctx.vehicle_slot
    )
    name = f"lnch_delay_m{k + 1}"
    return float(radar.vars.get(name, 0.0))
