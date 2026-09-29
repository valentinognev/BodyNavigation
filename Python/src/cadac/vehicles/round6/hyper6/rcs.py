"""HYPER6 RCS thrusters — port of Hyper::rcs (rcs.cpp)."""

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


class Hyper6Rcs:
    name = "rcs"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
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
            Field("roll_save", 0.0, "real", "save", "rcs"),
            Field("pitch_save", 0.0, "real", "save", "rcs"),
            Field("yaw_save", 0.0, "real", "save", "rcs"),
            Field("FMRCS", zeros3, "vec", "out", "rcs"),
            Field("phibdcomx", 0.0, "real", "data", "rcs"),
            Field("thtbdcomx", 0.0, "real", "data", "rcs"),
            Field("psibdcomx", 0.0, "real", "data", "rcs"),
            Field("e_roll", 0.0, "real", "diag", "rcs"),
            Field("e_pitch", 0.0, "real", "diag", "rcs"),
            Field("e_yaw", 0.0, "real", "diag", "rcs"),
            Field("o_roll", 0, "int", "save", "rcs"),
            Field("o_pitch", 0, "int", "save", "rcs"),
            Field("o_yaw", 0, "int", "save", "rcs"),
            Field("roll_count", 0, "int", "save", "rcs", plot),
            Field("pitch_count", 0, "int", "save", "rcs", plot),
            Field("yaw_count", 0, "int", "save", "rcs", plot),
            Field("side_force_max", 0.0, "real", "data", "rcs"),
            Field("FARCS", zeros3, "vec", "out", "rcs"),
            Field("e_right", 0.0, "real", "diag", "rcs"),
            Field("e_down", 0.0, "real", "diag", "rcs"),
            Field("o_right", 0, "int", "save", "rcs"),
            Field("o_down", 0, "int", "save", "rcs"),
            Field("right_save", 0.0, "real", "save", "rcs"),
            Field("down_save", 0.0, "real", "save", "rcs"),
            Field("rcs_minit_flag", 1, "int", "init", "rcs"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mrcs_moment = store.get("mrcs_moment")
        mrcs_force = store.get("mrcs_force")
        if mrcs_force not in (0, 1):
            raise ValueError(f"unknown mrcs_force {mrcs_force}")
        if mrcs_moment not in (0, 10, 11, 12, 20, 21, 22):
            raise ValueError(f"unknown mrcs_moment {mrcs_moment}")

        rcs_type = mrcs_moment // 10
        rcs_mode = mrcs_moment % 10

        e_roll = 0.0
        e_pitch = 0.0
        e_yaw = 0.0
        e_right = 0.0
        e_down = 0.0
        fmrcs = [0.0, 0.0, 0.0]
        farcs = [0.0, 0.0, 0.0]

        phibdcomx = store.get("phibdcomx")
        thtbdcomx = store.get("thtbdcomx")
        psibdcomx = store.get("psibdcomx")
        rcs_minit_flag = store.get("rcs_minit_flag")
        minit = store.get("minit") if "minit" in store else 0
        beco_flag = store.get("beco_flag") if "beco_flag" in store else 0
        psivdcx = store.get("psivdcx") if "psivdcx" in store else 0.0

        # Automatic mission planning: hold initial yaw heading as command.
        if minit and rcs_mode == 1 and (rcs_minit_flag or beco_flag):
            rcs_minit_flag = 0
            psibdcomx = psivdcx

        if rcs_type == 1:
            rcs_zeta = store.get("rcs_zeta")
            rcs_freq = store.get("rcs_freq")
            roll_mom_max = store.get("roll_mom_max")
            pitch_mom_max = store.get("pitch_mom_max")
            yaw_mom_max = store.get("yaw_mom_max")
            ppcx = store.get("ppcx")
            qqcx = store.get("qqcx")
            rrcx = store.get("rrcx")
            phibdcx = store.get("phibdcx")
            thtbdcx = store.get("thtbdcx")
            psibdcx = store.get("psibdcx")
            utbc = store.get("UTBC")
            ibbb = store.get("IBBB")

            rgain_roll = 2.0 * rcs_zeta * rcs_freq * float(ibbb[0, 0])
            rgain_pitch = 2.0 * rcs_zeta * rcs_freq * float(ibbb[1, 1])
            rgain_yaw = 2.0 * rcs_zeta * rcs_freq * float(ibbb[2, 2])
            pgain = rcs_freq / (2.0 * rcs_zeta)

            e_roll = rgain_roll * (pgain * (phibdcomx - phibdcx) - ppcx)
            fmrcs[0] = rcs_prop(e_roll, roll_mom_max)

            if rcs_mode == 1:
                e_pitch = rgain_pitch * (pgain * (thtbdcomx - thtbdcx) - qqcx)
                e_yaw = rgain_yaw * (pgain * (psibdcomx - psibdcx) - rrcx)
            elif rcs_mode == 2:
                e_pitch = rgain_pitch * (pgain * (-utbc[2]) * DEG - qqcx)
                e_yaw = rgain_yaw * (pgain * (utbc[1]) * DEG - rrcx)

            fmrcs[1] = rcs_prop(e_pitch, pitch_mom_max)
            fmrcs[2] = rcs_prop(e_yaw, yaw_mom_max)

        if mrcs_force == 1:
            vmass = store.get("vmass")
            side_force_max = store.get("side_force_max")
            aycomx = store.get("aycomx")
            azcomx = store.get("azcomx")

            e_right = vmass * aycomx * AGRAV
            farcs[1] = rcs_prop(e_right, side_force_max)

            e_down = vmass * azcomx * AGRAV
            farcs[2] = rcs_prop(e_down, side_force_max)

        if rcs_type == 2:
            dead_zone = store.get("dead_zone")
            hysteresis = store.get("hysteresis")
            rcs_tau = store.get("rcs_tau")
            roll_mom_max = store.get("roll_mom_max")
            pitch_mom_max = store.get("pitch_mom_max")
            yaw_mom_max = store.get("yaw_mom_max")
            ppcx = store.get("ppcx")
            qqcx = store.get("qqcx")
            rrcx = store.get("rrcx")
            phibdcx = store.get("phibdcx")
            thtbdcx = store.get("thtbdcx")
            psibdcx = store.get("psibdcx")
            utbc = store.get("UTBC")
            roll_save = store.get("roll_save")
            pitch_save = store.get("pitch_save")
            yaw_save = store.get("yaw_save")
            o_roll = store.get("o_roll")
            o_pitch = store.get("o_pitch")
            o_yaw = store.get("o_yaw")
            roll_count = store.get("roll_count")
            pitch_count = store.get("pitch_count")
            yaw_count = store.get("yaw_count")

            e_roll = phibdcomx - (rcs_tau * ppcx + phibdcx)
            o_roll_save = o_roll
            o_roll = rcs_schmitt(e_roll, roll_save, dead_zone, hysteresis)
            roll_save = e_roll
            if o_roll != o_roll_save:
                roll_count += 1

            if rcs_mode == 1:
                e_pitch = thtbdcomx - (rcs_tau * qqcx + thtbdcx)
                e_yaw = psibdcomx - (rcs_tau * rrcx + psibdcx)
            elif rcs_mode == 2:
                e_pitch = -rcs_tau * qqcx - utbc[2] * DEG
                e_yaw = -rcs_tau * rrcx + utbc[1] * DEG

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

            store.set("roll_save", roll_save)
            store.set("pitch_save", pitch_save)
            store.set("yaw_save", yaw_save)
            store.set("o_roll", o_roll)
            store.set("o_pitch", o_pitch)
            store.set("o_yaw", o_yaw)
            store.set("roll_count", roll_count)
            store.set("pitch_count", pitch_count)
            store.set("yaw_count", yaw_count)

        store.set("FMRCS", fmrcs)
        store.set("FARCS", farcs)
        store.set("psibdcomx", psibdcomx)
        store.set("rcs_minit_flag", rcs_minit_flag)
        store.set("e_roll", e_roll)
        store.set("e_pitch", e_pitch)
        store.set("e_yaw", e_yaw)
        store.set("e_right", e_right)
        store.set("e_down", e_down)

    def terminate(self, vehicle, ctx):
        pass
