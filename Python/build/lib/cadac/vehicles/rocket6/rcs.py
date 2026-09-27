from cadac.constants import DEG
from cadac.kernel.state import Field


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def rcs_schmitt(input_new, previous, dead_zone, hysteresis):
    trend = _sign(input_new - previous)
    side = _sign(previous)
    trigger = (dead_zone * side + hysteresis * trend) / 2.0
    if previous >= trigger and side == 1:
        return 1
    if previous <= trigger and side == -1:
        return -1
    return 0


def rcs_prop(command, limiter):
    output = command
    if abs(command) > limiter:
        output = limiter * _sign(command)
    return output


class Rocket6Rcs:
    name = "rcs"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        scrn_plot = ("scrn", "plot")
        for field in (
            Field("mrcs_moment", 0, "int", "data", "rcs"),
            Field("mrcs_force", 0, "int", "data", "rcs"),
            Field("dead_zone", 0.0, "real", "data", "rcs", plot),
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
            Field("roll_count", 0, "int", "save", "rcs", scrn_plot),
            Field("pitch_count", 0, "int", "save", "rcs", scrn_plot),
            Field("yaw_count", 0, "int", "save", "rcs", scrn_plot),
            Field("acc_gain", 0.0, "real", "data", "rcs"),
            Field("side_force_max", 0.0, "real", "data", "rcs"),
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
        mrcs_force = store.get("mrcs_force")
        if mrcs_force != 0:
            raise ValueError(f"unknown mrcs_force {mrcs_force}")
        mrcs_moment = store.get("mrcs_moment")
        if mrcs_moment not in (0, 20, 21, 22):
            raise ValueError(f"unknown mrcs_moment {mrcs_moment}")

        rcs_type = mrcs_moment // 10
        rcs_mode = mrcs_moment % 10

        e_roll = 0.0
        e_pitch = 0.0
        e_yaw = 0.0
        fmrcs = [0.0, 0.0, 0.0]
        farcs = [0.0, 0.0, 0.0]

        if rcs_type == 2:
            dead_zone = store.get("dead_zone")
            hysteresis = store.get("hysteresis")
            rcs_tau = store.get("rcs_tau")
            roll_mom_max = store.get("roll_mom_max")
            pitch_mom_max = store.get("pitch_mom_max")
            yaw_mom_max = store.get("yaw_mom_max")
            phibdcomx = store.get("phibdcomx")
            thtbdcomx = store.get("thtbdcomx")
            psibdcomx = store.get("psibdcomx")
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
        store.set("e_roll", e_roll)
        store.set("e_pitch", e_pitch)
        store.set("e_yaw", e_yaw)

    def terminate(self, vehicle, ctx):
        pass
