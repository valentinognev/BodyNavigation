from math import acos, cos, sin, sqrt

from cadac.constants import EPS, RAD, REARTH
from cadac.kernel.state import Field

BIG = 1e10


def _angle(vec1, vec2):
    scalar = float(vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2])
    abs1 = sqrt(float(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2))
    abs2 = sqrt(float(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2))
    dum = abs1 * abs2
    if dum > EPS:
        argument = scalar / dum
    else:
        argument = 1.0
    if argument > 1.0:
        argument = 1.0
    if argument < -1.0:
        argument = -1.0
    return acos(argument)


class Cruise5Targeting:
    name = "targeting"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("mtargeting", 0, "int", "data", "targeting", ("scrn", "plot")),
            Field("del_radius", 0.0, "real", "data", "targeting"),
            Field("clost_tgt_slot", 0, "int", "out", "targeting"),
            Field("tgtng_sat_slot", 0, "int", "out", "targeting"),
        ):
            if field.name not in store:
                store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def targeting_satellite(self, vehicle, combus):
        store = vehicle.store
        del_radius = store.get("del_radius")
        sbii = store.get("sbii")
        radius = REARTH + del_radius
        stii = (0.0, 0.0, 0.0)
        for packet in combus:
            if packet.type == "TARGET3":
                stii = packet.vars["sbii"]
                break
        visibility = []
        for i, packet in enumerate(combus):
            if packet.type != "SATELLITE3":
                continue
            ssii = packet.vars["sbii"]
            dsi = sqrt(float(ssii[0] ** 2 + ssii[1] ** 2 + ssii[2] ** 2))
            grazing_angle = acos(radius / dsi)
            satellite_missile_angle = _angle(sbii, ssii)
            if satellite_missile_angle < grazing_angle:
                tracking = 1
            else:
                radius_crit = 0.0
                dum = cos(satellite_missile_angle - grazing_angle)
                if abs(dum) > EPS:
                    radius_crit = radius / dum
                dbi = sqrt(float(sbii[0] ** 2 + sbii[1] ** 2 + sbii[2] ** 2))
                if dbi > radius_crit:
                    tracking = 1
                else:
                    tracking = 0
            visibility.append({"tracking": tracking, "vehicle_slot": i})
            # C++ uses angle(SBII, STII), not SSII.
            satellite_target_angle = _angle(sbii, stii)
            if satellite_target_angle > grazing_angle:
                visibility[-1]["tracking"] = 0
        return visibility

    def targeting_grnd_ranges(self, vehicle, combus):
        store = vehicle.store
        lon_c = store.get("lonx") * RAD
        lat_c = store.get("latx") * RAD
        slots = []
        ranges = []
        for i, packet in enumerate(combus):
            if packet.type != "TARGET3":
                continue
            lon_t = packet.vars["lonx"] * RAD
            lat_t = packet.vars["latx"] * RAD
            dum = sin(lat_t) * sin(lat_c) + cos(lat_t) * cos(lat_c) * cos(
                lon_t - lon_c
            )
            slots.append(i)
            ranges.append(REARTH * acos(dum))
        return slots, ranges

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mtargeting = store.get("mtargeting")
        if mtargeting == 0:
            return
        if mtargeting != 1:
            raise ValueError(f"unknown mtargeting {mtargeting}")

        combus = ctx.combus
        visibility = self.targeting_satellite(vehicle, combus)

        clost_tgt_slot = 0
        tgtng_sat_slot = 0
        wp_lonx = 0.0
        wp_latx = 0.0
        wp_alt = 0.0
        satellite_found = False
        for vis in visibility:
            if not satellite_found and vis["tracking"]:
                satellite_found = True
                tgtng_sat_slot = vis["vehicle_slot"]
                slots, ranges = self.targeting_grnd_ranges(vehicle, combus)
                range_m = BIG
                for slot, new_range in zip(slots, ranges):
                    if new_range < range_m:
                        range_m = new_range
                        clost_tgt_slot = slot

        if satellite_found:
            data_t = combus[clost_tgt_slot].vars
            wp_lonx = data_t["lonx"]
            wp_latx = data_t["latx"]
            wp_alt = data_t["alt"]

        store.set("clost_tgt_slot", clost_tgt_slot)
        store.set("tgtng_sat_slot", tgtng_sat_slot)
        store.set("wp_lonx", wp_lonx)
        store.set("wp_latx", wp_latx)
        store.set("wp_alt", wp_alt)

    def terminate(self, vehicle, ctx):
        pass
