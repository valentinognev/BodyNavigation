import numpy as np

from cadac.constants import DEG, WEII3
from cadac.eom.round3 import Round3Environment, Round3Newton
from cadac.eom.round6 import _cad_in_orb
from cadac.kernel.events import EventEngine
from cadac.kernel.state import Field, StateStore
from cadac.math.earth import cadtei, cadtge
from cadac.math.frames import cadac_matmul, mat2tr, polar_from_cart
from cadac.math.wgs84 import cad_geo84_in


class Hyper6SatelliteForces:
    """CADAC ``Satellite::forces``: thrust along 1V as specific force."""

    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        if "FSPV" not in store:
            store.define(Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces"))
        store.define(Field("sat_thrust", 0.0, "real", "data", "forces"))
        store.define(Field("sat_mass", 100.0, "real", "data", "forces"))

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        sat_thrust = store.get("sat_thrust")
        sat_mass = store.get("sat_mass")
        store.set("FSPV", np.array([sat_thrust / sat_mass, 0.0, 0.0]))

    def terminate(self, vehicle, ctx):
        pass


class Hyper6SatelliteNewton(Round3Newton):
    """Round3 newton plus HYPER6 orbital-element init (``minit == 1``)."""

    def define(self, vehicle):
        super().define(vehicle)
        store = vehicle.store
        extras = (
            Field("minit", 0, "int", "data", "newton"),
            Field("semi", 0.0, "real", "data", "newton"),
            Field("ecc", 0.0, "real", "data", "newton"),
            Field("inclx", 0.0, "real", "data", "newton"),
            Field("lon_anodex", 0.0, "real", "data", "newton"),
            Field("arg_perix", 0.0, "real", "data", "newton"),
            Field("true_anomx", 0.0, "real", "data", "newton"),
        )
        for field in extras:
            if field.name not in store:
                store.define(field)

    def initialize(self, vehicle, ctx):
        if vehicle.store.get("minit") == 1:
            self._init_from_orbit(vehicle, ctx)
            return
        super().initialize(vehicle, ctx)

    def _init_from_orbit(self, vehicle, ctx):
        store = vehicle.store
        sbii, vbii, _flag = _cad_in_orb(
            store.get("semi"),
            store.get("ecc"),
            store.get("inclx"),
            store.get("lon_anodex"),
            store.get("arg_perix"),
            store.get("true_anomx"),
        )
        weii = np.zeros((3, 3))
        weii[0, 1] = -WEII3
        weii[1, 0] = WEII3
        lon, lat, alt = cad_geo84_in(sbii, ctx.sim_time)
        tge = cadtge(lon, lat)
        tei = cadtei(ctx.sim_time)
        tig = cadac_matmul(tei.T, tge.T)
        vbeg = cadac_matmul(tig.T, vbii - cadac_matmul(weii, sbii))
        polar = polar_from_cart(vbeg)
        psivg = float(polar[1])
        thtvg = float(polar[2])
        store.set("weii", weii)
        store.set("tge", tge)
        store.set("tig", tig)
        store.set("vbeg", vbeg)
        store.set("tgv", mat2tr(psivg, thtvg).T)
        store.set("dvbe", float(polar[0]))
        store.set("psivg", psivg)
        store.set("thtvg", thtvg)
        store.set("psivgx", psivg * DEG)
        store.set("thtvgx", thtvg * DEG)
        store.set("lonx", float(lon) * DEG)
        store.set("latx", float(lat) * DEG)
        store.set("alt", float(alt))
        store.set("sb0ii", np.array(sbii, dtype=float, copy=True))
        store.set("sbii", np.array(sbii, dtype=float, copy=True))
        store.set("vbii", np.array(vbii, dtype=float, copy=True))


class Hyper6Satellite:
    type = "SAT3"

    def __init__(self, name, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Round3Environment(),
            Hyper6SatelliteForces(),
            Hyper6SatelliteNewton(),
        ]

    def define(self):
        for module in self.modules:
            module.define(self)
        vbii = self.store.field("vbii")
        if "com" not in vbii.outputs:
            vbii.outputs = (*vbii.outputs, "com")
        self.com_names = [
            name
            for name in self.store.names()
            if "com" in self.store.field(name).outputs
        ]
