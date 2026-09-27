from cadac.constants import DEG, RAD
from cadac.kernel.state import Field
from cadac.math.frames import mat2tr, polar_from_cart


class Aim5Intercept:
    name = "intercept"

    def define(self, vehicle):
        store = vehicle.store
        store.define(Field("aspazx", 0.0, "real", "diag", "intercept"))
        store.define(Field("aspelx", 0.0, "real", "diag", "intercept"))

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        time = store.get("time")
        vbel = store.get("VBEL")
        acft_com_slot = store.get("acft_com_slot")
        vtel = store.get("VTEL")
        psivlx_acft = store.get("psivlx_acft")
        thtvlx_acft = store.get("thtvlx_acft")
        dta = store.get("dta")
        dvta = store.get("dvta")
        stal = store.get("STAL")
        if dta < 500 and dvta > 0:
            vtael = vtel - vbel
            ttl = mat2tr(psivlx_acft * RAD, thtvlx_acft * RAD)
            polar = polar_from_cart(ttl @ vtael)
            store.set("aspazx", polar[1] * DEG)
            store.set("aspelx", polar[2] * DEG)
            vehicle.health = 0
            ctx.combus[ctx.vehicle_slot].status = 0
            ctx.combus[acft_com_slot].status = 0

    def terminate(self, vehicle, ctx):
        pass
