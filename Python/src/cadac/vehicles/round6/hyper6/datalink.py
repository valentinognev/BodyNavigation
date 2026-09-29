"""HYPER6 datalink — port of Hyper::datalink (datalink.cpp)."""

import numpy as np

from cadac.constants import EPS
from cadac.kernel.state import Field

_ZEROS3 = (0.0, 0.0, 0.0)


class Hyper6Datalink:
    name = "datalink"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("sat_num", 0, "int", "data", "combus"),
            Field("mnav", 0, "int", "out", "datalink"),
            Field("STCII", _ZEROS3, "vec", "out", "datalink"),
            Field("VTCII", _ZEROS3, "vec", "out", "datalink"),
            Field("tgt_pos", 0.0, "real", "save", "datalink"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mnav = 0
        sat_num = int(store.get("sat_num"))
        stcii = np.array(store.get("STCII"), dtype=float, copy=True)
        vtcii = np.array(store.get("VTCII"), dtype=float, copy=True)
        tgt_pos = float(store.get("tgt_pos"))

        packets = list(ctx.combus) if ctx.combus is not None else []
        for packet in packets:
            ident = getattr(packet, "id", None) or getattr(packet, "name", "")
            if ident == "r1":
                vars_ = packet.vars
                stcii = np.array(vars_[f"stcii{sat_num}"], dtype=float, copy=True)
                vtcii = np.array(vars_[f"vtcii{sat_num}"], dtype=float, copy=True)
            tgt_pos_new = float(np.linalg.norm(stcii))
            if abs(tgt_pos - tgt_pos_new) > EPS:
                mnav = 3
                tgt_pos = tgt_pos_new

        store.set("tgt_pos", tgt_pos)
        store.set("mnav", mnav)
        store.set("STCII", stcii)
        store.set("VTCII", vtcii)

    def terminate(self, vehicle, ctx):
        pass
