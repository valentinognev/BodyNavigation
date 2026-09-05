from math import sqrt

from cadac.kernel.state import Field


class Plane5Intercept:
    name = "intercept"

    def define(self, vehicle):
        vehicle.store.define(Field("stop_run", 0, "int", "data", "intercept"))

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        stop_run = store.get("stop_run")
        time = store.get("time")
        psivlx = store.get("psivlx")
        thtvlx = store.get("thtvlx")
        mguidance = store.get("mguidance")
        swbl = store.get("SWBL")
        wp_flag = store.get("wp_flag")
        write = store.get("write")
        if write:
            if mguidance == 30 or mguidance == 40:
                if wp_flag == -1:
                    swbl1 = swbl[0]
                    swbl2 = swbl[1]
                    dwbh = sqrt(swbl1 * swbl1 + swbl2 * swbl2)
                    write = 0
                    if stop_run == 1:
                        vehicle.health = 0
                        ctx.combus[ctx.vehicle_slot].status = 0
            if mguidance == 33:
                if wp_flag == -1:
                    swbl1 = swbl[0]
                    swbl2 = swbl[1]
                    swbl3 = swbl[2]
                    dwbh = sqrt(swbl1 * swbl1 + swbl2 * swbl2 + swbl3 * swbl3)
                    write = 0
                    if stop_run == 1:
                        vehicle.health = 0
                        ctx.combus[ctx.vehicle_slot].status = 0
        store.set("write", write)

    def terminate(self, vehicle, ctx):
        pass
