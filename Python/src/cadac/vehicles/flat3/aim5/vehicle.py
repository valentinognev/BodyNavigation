from cadac.eom.flat3 import Flat3Environment, Flat3Kinematics, Flat3Newton
from cadac.kernel.events import EventEngine
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat3.aim5.aero import Aim5Aero
from cadac.vehicles.flat3.aim5.control import Aim5Control
from cadac.vehicles.flat3.aim5.forces import Aim5Forces
from cadac.vehicles.flat3.aim5.guidance import Aim5Guidance
from cadac.vehicles.flat3.aim5.intercept import Aim5Intercept
from cadac.vehicles.flat3.aim5.propulsion import Aim5Propulsion
from cadac.vehicles.flat3.aim5.seeker import Aim5Seeker
from cadac.vehicles.flat3.aim5.target import Aim5Target

_COM_EXTRA = ("SBEL", "VBEL", "psivlx", "thtvlx")


def _sael_fields():
    return (
        Field("sael1", 0.0, "real", "init", "newton"),
        Field("sael2", 0.0, "real", "init", "newton"),
        Field("sael3", 0.0, "real", "init", "newton"),
        Field("dvae", 0.0, "real", "init", "newton"),
    )


class Aim5Flat3Newton(Flat3Newton):
    def define(self, vehicle):
        super().define(vehicle)
        store = vehicle.store
        for field in _sael_fields():
            if field.name not in store:
                store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        if "sael1" in store:
            store.set("sbel1", store.get("sael1"))
        if "sael2" in store:
            store.set("sbel2", store.get("sael2"))
        if "sael3" in store:
            store.set("sbel3", store.get("sael3"))
        if "dvae" in store:
            store.set("dvbe", store.get("dvae"))
        super().initialize(vehicle, ctx)


def _define_aim5_vehicle(vehicle):
    store = vehicle.store
    orig_define = store.define

    def define_skip_if_exists(field):
        if field.name not in store:
            orig_define(field)

    store.define = define_skip_if_exists
    try:
        for module in vehicle.modules:
            module.define(vehicle)
        for field in _sael_fields():
            store.define(field)
    finally:
        store.define = orig_define
    names = store.names()
    flagged = [
        name
        for name in names
        if "com" in store.field(name).outputs
    ]
    extra = [name for name in _COM_EXTRA if name not in flagged]
    vehicle.com_names = flagged + extra


class Aim5:
    type = "AIM5"

    def __init__(self, name, aero_deck, prop_deck, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Aim5Target(),
            Flat3Environment(),
            Flat3Kinematics(),
            Aim5Aero(aero_deck),
            Aim5Propulsion(prop_deck),
            Aim5Seeker(),
            Aim5Guidance(),
            Aim5Control(),
            Aim5Forces(),
            Aim5Flat3Newton(),
            Aim5Intercept(),
        ]

    def define(self):
        _define_aim5_vehicle(self)
