from math import cos

from cadac.constants import RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field

FOOT = 3.280834
NT = 4.448


class Plane6Propulsion:
    name = "propulsion"

    def __init__(self, deck):
        self.deck = deck

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("mprop", 0, "int", "data", "propulsion"),
            Field("vmachcom", 0.0, "real", "data", "propulsion"),
            Field("throttle", 0.0, "real", "data", "propulsion", ("scrn", "plot")),
            Field("gmach", 0.0, "real", "data", "propulsion"),
            Field("thrustf", 0.0, "real", "save", "propulsion"),
            Field("thrust", 0.0, "real", "out", "propulsion", ("scrn", "plot")),
            Field("thrust_req", 0.0, "real", "diag", "propulsion"),
            Field("mfreeze_prop", 0, "int", "save", "propulsion"),
            Field("powerd", 0.0, "real", "state", "propulsion"),
            Field("power", 0.0, "real", "state", "propulsion", ("plot",)),
            Field("power_com", 0.0, "real", "diag", "propulsion"),
            Field("tpower", 0.0, "real", "diag", "propulsion"),
            Field("idle", 0.0, "real", "diag", "propulsion", ("plot",)),
            Field("mil", 0.0, "real", "diag", "propulsion", ("plot",)),
            Field("max", 0.0, "real", "diag", "propulsion", ("plot",)),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        dt = ctx.int_step
        mprop = store.get("mprop")
        vmachcom = store.get("vmachcom")
        throttle = store.get("throttle")
        gmach = store.get("gmach")
        thrustf = store.get("thrustf")
        mfreeze_prop = store.get("mfreeze_prop")
        vmach = store.get("vmach")
        pdynmc = store.get("pdynmc")
        alphax = store.get("alphax")
        refa = store.get("refa")
        cdrag = store.get("cdrag")

        thrust = 0.0
        thrust_req = 0.0
        if mprop == 1:
            thrust = self.propulsion_thrust(vehicle, throttle, dt)
        elif mprop == 2:
            emach = vmachcom - vmach
            throttle = gmach * emach
            if throttle < 0:
                throttle = 0
            if throttle > 0.77:
                throttle = 0.77
            thrust = self.propulsion_thrust(vehicle, throttle, dt)
            thrust_req = cdrag * pdynmc * refa / cos(alphax * RAD)
        else:
            thrust = 0

        if "mfreeze" in store.names():
            mfreeze = store.get("mfreeze")
            if mfreeze == 0:
                mfreeze_prop = 0
            else:
                if mfreeze != mfreeze_prop:
                    mfreeze_prop = mfreeze
                    thrustf = thrust
                thrust = thrustf

        store.set("thrustf", thrustf)
        store.set("mfreeze_prop", mfreeze_prop)
        store.set("thrust", thrust)
        store.set("thrust_req", thrust_req)
        store.set("throttle", throttle)

    def propulsion_thrust(self, vehicle, throttle, int_step):
        store = vehicle.store
        powerd = store.get("powerd")
        power = store.get("power")
        vmach = store.get("vmach")
        hbe = store.get("hbe")

        if throttle <= 0.77:
            power_com = 64.94 * throttle
        else:
            power_com = 217.38 * throttle - 117.38

        if power_com <= 50:
            tpower = 1
        else:
            tpower = 0.2

        powerd_new = (power_com - power) / tpower
        power = integrate(powerd_new, powerd, power, int_step)
        powerd = powerd_new

        hbe_ft = hbe * FOOT
        idle_lb = self.deck.look_up("idle_vs_mach_alt", vmach, hbe_ft)
        idle = idle_lb * NT
        mil_lb = self.deck.look_up("mil_vs_mach_alt", vmach, hbe_ft)
        mil = mil_lb * NT
        max_lb = self.deck.look_up("max_vs_mach_alt", vmach, hbe_ft)
        max_thr = max_lb * NT

        if power < 50:
            thrust = idle + power * 0.02 * (mil - idle)
        else:
            thrust = mil + (power - 50) * 0.02 * (max_thr - mil)

        store.set("powerd", powerd)
        store.set("power", power)
        store.set("power_com", power_com)
        store.set("tpower", tpower)
        store.set("idle", idle)
        store.set("mil", mil)
        store.set("max", max_thr)
        return thrust

    def terminate(self, vehicle, ctx):
        pass
