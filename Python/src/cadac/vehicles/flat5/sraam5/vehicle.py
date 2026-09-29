from cadac.kernel.events import EventEngine
from cadac.kernel.state import StateStore
from cadac.vehicles.flat5.sraam5.aerodynamics import Sraam5Aerodynamics
from cadac.vehicles.flat5.sraam5.ai_radar import Sraam5AiRadar
from cadac.vehicles.flat5.sraam5.control import Sraam5Control
from cadac.vehicles.flat5.sraam5.environment import Sraam5Environment
from cadac.vehicles.flat5.sraam5.forces import Sraam5Forces
from cadac.vehicles.flat5.sraam5.guidance import Sraam5Guidance
from cadac.vehicles.flat5.sraam5.ins import Sraam5Ins
from cadac.vehicles.flat5.sraam5.intercept import Sraam5Intercept
from cadac.vehicles.flat5.sraam5.newton import Sraam5Newton
from cadac.vehicles.flat5.sraam5.propulsion import Sraam5Propulsion
from cadac.vehicles.flat5.sraam5.rotations import Sraam5Rotations
from cadac.vehicles.flat5.sraam5.seeker import Sraam5Seeker
from cadac.vehicles.flat5.sraam5.target import Sraam5Target


class Sraam5:
    type = "SRAAM5"

    def __init__(self, name, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Sraam5Target(),
            Sraam5Environment(),
            Sraam5Seeker(),
            Sraam5AiRadar(),
            Sraam5Ins(),
            Sraam5Guidance(),
            Sraam5Control(),
            Sraam5Aerodynamics(),
            Sraam5Propulsion(),
            Sraam5Forces(),
            Sraam5Newton(),
            Sraam5Rotations(),
            Sraam5Intercept(),
        ]

    def define(self):
        store = self.store
        orig_define = store.define

        def define_skip_if_exists(field):
            if field.name not in store:
                orig_define(field)

        store.define = define_skip_if_exists
        try:
            for module in self.modules:
                module.define(self)
        finally:
            store.define = orig_define
        self.com_names = [
            name
            for name in self.store.names()
            if "com" in self.store.field(name).outputs
        ]
