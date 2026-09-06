import numpy as np

from cadac.kernel.state import Field


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
        if mrcs_moment != 0 or mrcs_force != 0:
            raise ValueError(
                f"mrcs_moment={mrcs_moment!r} mrcs_force={mrcs_force!r} "
                "not supported in this slice"
            )
        store.set("FMRCS", np.zeros(3))
        store.set("FARCS", np.zeros(3))

    def terminate(self, vehicle, ctx):
        pass
