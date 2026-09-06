import numpy as np

from cadac.constants import DEG
from cadac.kernel.state import Field
from cadac.math.frames import polar_from_cart


def _skew(vec):
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


class Aim5Seeker:
    name = "seeker"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("scrn", "plot")
        for field in (
            Field("acft_com_slot", 0, "int", "out", "combus"),
            Field("VTEL", zeros3, "vec", "out", "combus"),
            Field("psivlx_acft", 0.0, "real", "out", "combus"),
            Field("thtvlx_acft", 0.0, "real", "out", "combus"),
            Field("tgt_num", 1, "int", "data", "combus"),
            Field("mseek", 0, "int", "data", "seeker"),
            Field("dta", 0.0, "real", "out", "seeker", plot),
            Field("dvta", 0.0, "real", "out", "seeker"),
            Field("tgo_aim", 0.0, "real", "diag", "seeker"),
            Field("los_azx", 0.0, "real", "diag", "seeker"),
            Field("los_elx", 0.0, "real", "diag", "seeker"),
            Field("sigdy", 0.0, "real", "diag", "seeker"),
            Field("sigdz", 0.0, "real", "diag", "seeker"),
            Field("UTAA", zeros3, "vec", "out", "seeker"),
            Field("WOEA", zeros3, "vec", "out", "seeker"),
            Field("STAL", zeros3, "vec", "out", "seeker"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        tgt_num = store.get("tgt_num")
        mseek = store.get("mseek")
        if mseek not in {0, 1}:
            raise ValueError(f"unsupported mseek={mseek}")

        aircraft = [
            (i, packet)
            for i, packet in enumerate(ctx.combus)
            if packet.type == "AIRCRAFT3"
        ]
        idx = tgt_num - 1
        packet = None
        if 0 <= idx < len(aircraft):
            acft_com_slot, packet = aircraft[idx]
            store.set("acft_com_slot", acft_com_slot)
            store.set("VTEL", packet.vars["VBEL"])
            store.set("psivlx_acft", packet.vars["psivlx"])
            store.set("thtvlx_acft", packet.vars["thtvlx"])

        if mseek == 0:
            return
        if packet is None:
            raise ValueError(
                f"no AIRCRAFT3 at tgt_num={tgt_num}"
            )

        stel = packet.vars["SBEL"]
        vtel = packet.vars["VBEL"]
        sbel = store.get("SBEL")
        vbel = store.get("VBEL")
        tbl = store.get("TBL")
        stal = stel - sbel
        dta = float(np.linalg.norm(stal))
        utal = stal / dta
        utaa = tbl @ utal
        polar = polar_from_cart(utaa)
        los_azx = polar[1] * DEG
        los_elx = polar[2] * DEG
        vtael = vtel - vbel
        dvta = float(utal @ vtael)
        tgo_aim = dta / abs(dvta)
        woea = tbl @ (_skew(utal) @ vtael) * (1.0 / dta)
        store.set("STAL", stal)
        store.set("dta", dta)
        store.set("UTAA", utaa)
        store.set("los_azx", los_azx)
        store.set("los_elx", los_elx)
        store.set("dvta", dvta)
        store.set("tgo_aim", tgo_aim)
        store.set("WOEA", woea)
        store.set("sigdy", woea[1])
        store.set("sigdz", woea[2])

    def terminate(self, vehicle, ctx):
        pass
