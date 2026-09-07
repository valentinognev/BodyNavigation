import numpy as np

from cadac.kernel.state import Field

_ZEROS3 = (0.0, 0.0, 0.0)


class Sraam6TargetGuidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("msl_num", 0, "int", "data", "guidance"),
            Field("tgt_option", 0, "int", "data", "guidance"),
            Field("guid_gain", 0.0, "real", "data", "guidance"),
            Field("ACOML", _ZEROS3, "vec", "out", "guidance"),
            Field("gturn", 0.0, "real", "data", "guidance"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        msl_num = store.get("msl_num")
        tgt_option = store.get("tgt_option")
        guid_gain = store.get("guid_gain")
        gturn = store.get("gturn")
        grav = store.get("grav")
        tvl = np.asarray(store.get("TVL"), dtype=float)
        sael = np.asarray(store.get("SAEL"), dtype=float)
        vael = np.asarray(store.get("VAEL"), dtype=float)

        if tgt_option not in (0, 1, 2):
            raise ValueError(f"unknown tgt_option {tgt_option}")

        acoml = np.zeros(3)
        if tgt_option == 0:
            acoml = np.array([0.0, 0.0, -grav])
        if tgt_option == 1:
            acomv = np.array([0.0, gturn * grav, -grav])
            acoml = tvl.T @ acomv
        if tgt_option == 2:
            missiles = [
                packet
                for packet in (ctx.combus or ())
                if packet.type == "MISSILE6"
            ]
            if msl_num >= 1:
                idx = msl_num - 1
                if idx < len(missiles):
                    packet = missiles[idx]
                    mseek = int(packet.vars["mseek"])
                    if mseek % 10 == 4:
                        sbel = np.asarray(packet.vars["SBEL"], dtype=float)
                        vbel = np.asarray(packet.vars["VBEL"], dtype=float)
                        sabl = sael - sbel
                        dab = float(np.linalg.norm(sabl))
                        dum = float(np.linalg.norm(np.cross(vael, vbel)))
                        gain = guid_gain * dum / dab
                        uvbel = vbel * (1.0 / float(np.linalg.norm(vbel)))
                        uvael = vael * (1.0 / float(np.linalg.norm(vael)))
                        epsl = np.cross(uvael, uvbel)
                        acoml = np.cross(epsl, uvael) * gain
                        acoml = acoml + np.array([0.0, 0.0, -grav])

        store.set("ACOML", acoml)

    def terminate(self, vehicle, ctx):
        pass
