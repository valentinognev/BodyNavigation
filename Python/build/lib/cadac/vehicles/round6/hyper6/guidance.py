from cadac.kernel.state import Field


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
        mguide = vehicle.store.get("mguide")
        if mguide == 0:
            return
        raise ValueError(f"unknown mguide {mguide}")

    def terminate(self, vehicle, ctx):
        pass
