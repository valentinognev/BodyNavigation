import numpy as np

from cadac.constants import EPS
from cadac.kernel.state import Field


class Agm6Datalink:
    name = "datalink"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("mnav", 0, "int", "out", "datalink"),
            Field("STCEL", zeros3, "vec", "out", "datalink"),
            Field("VTCEL", zeros3, "vec", "out", "datalink"),
            Field("tgt_pos", 0.0, "real", "save", "datalink"),
            Field("SAEL", zeros3, "vec", "out", "datalink"),
            Field("VAEL", zeros3, "vec", "out", "datalink"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mnav = 0
        stcel = np.array(store.get("STCEL"), dtype=float, copy=True)
        vtcel = np.array(store.get("VTCEL"), dtype=float, copy=True)
        sael = np.array(store.get("SAEL"), dtype=float, copy=True)
        vael = np.array(store.get("VAEL"), dtype=float, copy=True)
        tgt_pos = store.get("tgt_pos")
        tgt_num = store.get("tgt_num")
        aircraft = None
        for packet in ctx.combus or ():
            if packet is not None and packet.type == "AIRCRAFT3":
                aircraft = packet
                break
        if aircraft is not None:
            vars_ = aircraft.vars
            stcel = np.array(vars_[f"STCEL{tgt_num}"], dtype=float, copy=True)
            vtcel = np.array(vars_[f"VTCEL{tgt_num}"], dtype=float, copy=True)
            sael = np.array(vars_["SAEL"], dtype=float, copy=True)
            vael = np.array(vars_["VAEL"], dtype=float, copy=True)
            tgt_pos_new = float(np.linalg.norm(stcel))
            if abs(tgt_pos - tgt_pos_new) > EPS:
                mnav = 3
                tgt_pos = tgt_pos_new
        store.set("tgt_pos", tgt_pos)
        store.set("mnav", mnav)
        store.set("STCEL", stcel)
        store.set("VTCEL", vtcel)
        store.set("SAEL", sael)
        store.set("VAEL", vael)

    def terminate(self, vehicle, ctx):
        pass
