import numpy as np

from cadac.constants import AGRAV, DEG
from cadac.kernel.state import Field
from cadac.math.frames import cadac_sign


def rcs_prop(command, limiter):
    output = command
    if abs(command) > limiter:
        output = limiter * cadac_sign(command)
    return output


def rcs_schmitt(input_new, previous, dead_zone, hysteresis):
    trend = cadac_sign(input_new - previous)
    side = cadac_sign(previous)
    trigger = (dead_zone * side + hysteresis * trend) / 2.0
    if previous >= trigger and side == 1:
        return 1
    if previous <= trigger and side == -1:
        return -1
    return 0


class Sam6Rcs:
    name = "rcs"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("mrcs_moment", 0, "int", "data", "rcs"),
            Field("mrcs_force", 0, "int", "data", "rcs"),
            Field("dead_zone", 0.0, "real", "data", "rcs"),
            Field("hysteresis", 0.0, "real", "data", "rcs"),
            Field("rcs_tau", 0.0, "real", "data", "rcs"),
            Field("roll_mom_max", 0.0, "real", "data", "rcs"),
            Field("pitch_mom_max", 0.0, "real", "data", "rcs"),
            Field("yaw_mom_max", 0.0, "real", "data", "rcs"),
            Field("rcs_zeta", 0.0, "real", "data", "rcs"),
            Field("rcs_freq", 0.0, "real", "data", "rcs"),
            Field("rcs_arm", 0.0, "real", "data", "rcs"),
            Field("roll_save", 0.0, "real", "save", "rcs"),
            Field("pitch_save", 0.0, "real", "save", "rcs"),
            Field("yaw_save", 0.0, "real", "save", "rcs"),
            Field("FMRCS", zeros3, "vec", "out", "rcs"),
            Field("rcs_time", 0.0, "real", "save", "rcs"),
            Field("rcs_isp", 0.0, "real", "data", "rcs"),
            Field("rcs_fmass", 0.0, "real", "out", "rcs"),
            Field("rate_gain_rcs", 0.0, "real", "data", "rcs"),
            Field("phibdcomx", 0.0, "real", "data", "rcs"),
            Field("thtbdcomx", 0.0, "real", "data", "rcs"),
            Field("psibdcomx", 0.0, "real", "data", "rcs"),
            Field("e_roll", 0.0, "real", "diag", "rcs"),
            Field("e_pitch", 0.0, "real", "diag", "rcs"),
            Field("e_yaw", 0.0, "real", "diag", "rcs"),
            Field("o_roll", 0, "int", "save", "rcs"),
            Field("o_pitch", 0, "int", "save", "rcs"),
            Field("o_yaw", 0, "int", "save", "rcs"),
            Field("roll_count", 0, "int", "save", "rcs"),
            Field("pitch_count", 0, "int", "save", "rcs"),
            Field("yaw_count", 0, "int", "save", "rcs"),
            Field("acc_gain", 0.0, "real", "data", "rcs"),
            Field("rcs_thrust", 0.0, "real", "data", "rcs"),
            Field("FARCS", zeros3, "vec", "out", "rcs"),
            Field("e_right", 0.0, "real", "diag", "rcs"),
            Field("e_down", 0.0, "real", "diag", "rcs"),
            Field("o_right", 0, "int", "save", "rcs"),
            Field("o_down", 0, "int", "save", "rcs"),
            Field("right_save", 0.0, "real", "save", "rcs"),
            Field("down_save", 0.0, "real", "save", "rcs"),
            Field("right_count", 0, "int", "save", "rcs"),
            Field("down_count", 0, "int", "save", "rcs"),
            Field("factdead_zone", 0.0, "real", "save", "rcs"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mrcs_moment = store.get("mrcs_moment")
        mrcs_force = store.get("mrcs_force")
        if mrcs_moment == 0 and mrcs_force == 0:
            store.set("FMRCS", np.zeros(3))
            store.set("FARCS", np.zeros(3))
            return

        rcs_type = mrcs_moment // 10
        rcs_mode = mrcs_moment % 10
        need_moment = rcs_type in (1, 2)
        need_force = mrcs_force in (1, 2)
        if not need_moment and not need_force:
            store.set("FMRCS", np.zeros(3))
            store.set("FARCS", np.zeros(3))
            return

        dead_zone = store.get("dead_zone")
        hysteresis = store.get("hysteresis")
        rcs_tau = store.get("rcs_tau")
        roll_mom_max = store.get("roll_mom_max")
        pitch_mom_max = store.get("pitch_mom_max")
        yaw_mom_max = store.get("yaw_mom_max")
        rcs_zeta = store.get("rcs_zeta")
        rcs_freq = store.get("rcs_freq")
        rcs_arm = store.get("rcs_arm")
        rcs_isp = store.get("rcs_isp")
        rate_gain_rcs = store.get("rate_gain_rcs")
        phibdcomx = store.get("phibdcomx")
        thtbdcomx = store.get("thtbdcomx")
        psibdcomx = store.get("psibdcomx")
        acc_gain = store.get("acc_gain")
        rcs_thrust = store.get("rcs_thrust")

        roll_save = store.get("roll_save")
        pitch_save = store.get("pitch_save")
        yaw_save = store.get("yaw_save")
        rcs_time = store.get("rcs_time")
        o_roll = store.get("o_roll")
        o_pitch = store.get("o_pitch")
        o_yaw = store.get("o_yaw")
        roll_count = store.get("roll_count")
        pitch_count = store.get("pitch_count")
        yaw_count = store.get("yaw_count")
        o_right = store.get("o_right")
        o_down = store.get("o_down")
        right_save = store.get("right_save")
        down_save = store.get("down_save")
        right_count = store.get("right_count")
        down_count = store.get("down_count")

        e_roll = 0.0
        e_pitch = 0.0
        e_yaw = 0.0
        e_right = 0.0
        e_down = 0.0
        fmrcs = [0.0, 0.0, 0.0]
        farcs = [0.0, 0.0, 0.0]
        rcs_fmass = 0.0
        pgain = 0.0
        ppcx = qqcx = rrcx = 0.0
        phiblcx = thtblcx = psibdcx = 0.0
        alphacx = betacx = 0.0
        ancomx = alcomx = 0.0
        alphacomx = betacomx = 0.0
        ai11 = ai33 = xcg = 0.0
        utbc = (0.0, 0.0, 0.0)
        fspcb = (0.0, 0.0, 0.0)

        if need_moment:
            wbecb = store.get("WBECB")
            ppcx = float(wbecb[0])
            qqcx = float(wbecb[1])
            rrcx = float(wbecb[2])
            phiblcx = store.get("phiblcx")
            thtblcx = store.get("thtblcx")
            psibdcx = store.get("psibdcx")
            alphacx = store.get("alphacx")
            betacx = store.get("betacx")
            utbc = store.get("UTBC")
            alphacomx = store.get("alphacomx")
            betacomx = store.get("betacomx")
            ai11 = store.get("ai11")
            ai33 = store.get("ai33")

        if need_force:
            fspcb = store.get("FSPCB")
            ancomx = store.get("ancomx")
            alcomx = store.get("alcomx")
            xcg = store.get("xcg")

        if rcs_type == 1:
            rgain_roll = 2.0 * rcs_zeta * rcs_freq * ai11
            rgain_pitch = 2.0 * rcs_zeta * rcs_freq * ai33
            rgain_yaw = 2.0 * rcs_zeta * rcs_freq * ai33
            if rcs_zeta:
                pgain = rcs_freq / (2.0 * rcs_zeta)

            e_roll = rgain_roll * (pgain * (phibdcomx - phiblcx) - ppcx)
            fmrcs[0] = rcs_prop(e_roll, roll_mom_max)

            if rcs_mode == 1:
                e_pitch = rgain_pitch * (pgain * (thtbdcomx - thtblcx) - qqcx)
                e_yaw = rgain_yaw * (pgain * (psibdcomx - psibdcx) - rrcx)
            elif rcs_mode == 2:
                e_pitch = rgain_pitch * (pgain * (-utbc[2]) * DEG - qqcx)
                e_yaw = rgain_yaw * (pgain * (utbc[1]) * DEG - rrcx)
            elif rcs_mode == 4:
                e_pitch = rate_gain_rcs * (-qqcx)
                e_yaw = rate_gain_rcs * (-rrcx)

            fmrcs[1] = rcs_prop(e_pitch, pitch_mom_max)
            fmrcs[2] = rcs_prop(e_yaw, yaw_mom_max)

        if mrcs_force == 1:
            ay = float(fspcb[1])
            e_right = acc_gain * (alcomx * AGRAV - ay)
            farcs[1] = rcs_prop(e_right, rcs_thrust)

            az = float(fspcb[2])
            e_down = -acc_gain * (ancomx * AGRAV + az)
            farcs[2] = rcs_prop(e_down, rcs_thrust)

        if rcs_type == 2:
            e_roll = phibdcomx - (rcs_tau * ppcx + phiblcx)
            o_roll_save = o_roll
            o_roll = rcs_schmitt(e_roll, roll_save, dead_zone, hysteresis)
            roll_save = e_roll
            if o_roll != o_roll_save:
                roll_count += 1

            if rcs_mode == 1:
                e_pitch = thtbdcomx - (rcs_tau * qqcx + thtblcx)
                e_yaw = psibdcomx - (rcs_tau * rrcx + psibdcx)
            if rcs_mode == 3:
                e_pitch = alphacomx - (rcs_tau * qqcx + alphacx)
                e_yaw = -betacomx - (rcs_tau * rrcx - betacx)
            elif rcs_mode == 2:
                e_pitch = -rcs_tau * qqcx - utbc[2] * DEG
                e_yaw = -rcs_tau * rrcx + utbc[1] * DEG
            elif rcs_mode == 4:
                e_pitch = -rcs_tau * qqcx
                e_yaw = -rcs_tau * rrcx

            o_pitch_save = o_pitch
            o_pitch = rcs_schmitt(e_pitch, pitch_save, dead_zone, hysteresis)
            pitch_save = e_pitch
            if o_pitch != o_pitch_save:
                pitch_count += 1

            o_yaw_save = o_yaw
            o_yaw = rcs_schmitt(e_yaw, yaw_save, dead_zone, hysteresis)
            yaw_save = e_yaw
            if o_yaw != o_yaw_save:
                yaw_count += 1

            fmrcs[0] = o_roll * roll_mom_max
            fmrcs[1] = o_pitch * pitch_mom_max
            fmrcs[2] = o_yaw * yaw_mom_max

        if mrcs_force == 2:
            ay = float(fspcb[1])
            e_right = acc_gain * (alcomx * AGRAV - ay)

            o_right_save = o_right
            o_right = rcs_schmitt(e_right, right_save, dead_zone, hysteresis)
            right_save = e_right
            if o_right != o_right_save:
                right_count += 1

            az = float(fspcb[2])
            e_down = -acc_gain * (ancomx * AGRAV + az)
            o_down_save = o_down
            o_down = rcs_schmitt(e_down, down_save, dead_zone, hysteresis)
            down_save = e_down
            if o_down != o_down_save:
                down_count += 1

            farcs[0] = 0.0
            farcs[1] = o_right * rcs_thrust
            farcs[2] = o_down * rcs_thrust

            dx = rcs_arm - xcg
            fmrcs[1] = farcs[2] * dx
            fmrcs[2] = -farcs[1] * dx

            int_step = ctx.int_step
            if o_down:
                rcs_time += int_step
            if o_right:
                rcs_time += int_step
            rcs_fmass = rcs_thrust * rcs_time / (rcs_isp * AGRAV)

        store.set("FMRCS", fmrcs)
        store.set("rcs_fmass", rcs_fmass)
        store.set("FARCS", farcs)
        store.set("roll_save", roll_save)
        store.set("pitch_save", pitch_save)
        store.set("yaw_save", yaw_save)
        store.set("rcs_time", rcs_time)
        store.set("o_roll", o_roll)
        store.set("o_pitch", o_pitch)
        store.set("o_yaw", o_yaw)
        store.set("roll_count", roll_count)
        store.set("pitch_count", pitch_count)
        store.set("yaw_count", yaw_count)
        store.set("o_right", o_right)
        store.set("o_down", o_down)
        store.set("right_save", right_save)
        store.set("down_save", down_save)
        store.set("right_count", right_count)
        store.set("down_count", down_count)
        store.set("e_roll", e_roll)
        store.set("e_pitch", e_pitch)
        store.set("e_yaw", e_yaw)
        store.set("e_right", e_right)
        store.set("e_down", e_down)

    def terminate(self, vehicle, ctx):
        pass
