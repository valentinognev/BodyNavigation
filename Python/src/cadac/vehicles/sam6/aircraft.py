import numpy as np

from cadac.kernel.state import Field


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


def _first_missile6(ctx):
    combus = ctx.combus or ()
    for packet in combus:
        if packet.type == "MISSILE6":
            return packet
    raise ValueError("no MISSILE6 packet")


class Sam6AircraftGuidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("acft_option", 0, "int", "data", "guidance"),
            Field("guid_gain", 0.0, "real", "data", "guidance"),
            Field("ACOML", zeros3, "vec", "out", "guidance"),
            Field("gturn", 0.0, "real", "data", "guidance"),
            Field("man_start", 0.0, "real", "data", "guidance"),
            Field("man_stop", 0.0, "real", "data", "guidance"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        acft_option = store.get("acft_option")
        if acft_option not in (0, 1, 2):
            raise ValueError(f"acft_option={acft_option!r} not supported")
        time = store.get("time")
        grav = store.get("grav")
        man_start = store.get("man_start")
        man_stop = store.get("man_stop")
        in_window = man_start <= time < man_stop
        if acft_option == 0 or not in_window:
            store.set("ACOML", np.array([0.0, 0.0, -grav], dtype=float))
            return
        if acft_option == 1:
            tvl = np.asarray(store.get("TVL"), dtype=float)
            acomv = np.array(
                [0.0, store.get("gturn") * grav, -grav], dtype=float
            )
            store.set("ACOML", tvl.T @ acomv)
            return
        packet = _first_missile6(ctx)
        stel = np.asarray(packet.vars["SBEL"], dtype=float)
        vtel = np.asarray(packet.vars["VBEL"], dtype=float)
        sael = np.asarray(store.get("SAEL"), dtype=float)
        vael = np.asarray(store.get("VAEL"), dtype=float)
        satl = sael - stel
        dab = np.linalg.norm(satl)
        dum = np.linalg.norm(_skew(vael) @ vtel)
        gain = store.get("guid_gain") * dum / dab
        uvtel = vtel / np.linalg.norm(vtel)
        uvael = vael / np.linalg.norm(vael)
        epsl = _skew(uvael) @ uvtel
        acoml = _skew(epsl) @ uvael * gain
        acoml = acoml + np.array([0.0, 0.0, -grav], dtype=float)
        store.set("ACOML", acoml)

    def terminate(self, vehicle, ctx):
        pass
