import math
from math import atan2, sqrt

import numpy as np

from cadac.constants import AGRAV, DEG, EPS, RAD
from cadac.eom.flat3 import (
    Flat3AircraftEnvironment,
    Flat3AircraftNewton,
    Flat3Kinematics,
)
from cadac.kernel.events import EventEngine
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cart_from_pol

_ZEROS3 = (0.0, 0.0, 0.0)
_MINIT_OK = frozenset({12, 13, 14, 15, 21, 22, 23, 24})


def _sign(variable):
    if variable < 0.0:
        return -1
    return 1


def _matcar(dbe, azbel, elbel):
    """Fortran UTL3 MATCAR: cartesian from polar (speed, heading, flight-path)."""
    return np.asarray(cart_from_pol(dbe, azbel, elbel), dtype=float)


class Sraam6TargetMinit:
    """Fortran G1I MINIT LAR/CIRCLE auto-geometry (codes 12–15, 21–24)."""

    name = "target"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("minit", 0, "int", "data", "target"),
            Field("an1c", 0.0, "real", "data", "target"),
            Field("dvt1e", 0.0, "real", "data", "target"),
            Field("ht1e", 0.0, "real", "data", "target"),
            Field("tauhx", 0.0, "real", "data", "target"),
            Field("dvt2e", 0.0, "real", "data", "target"),
            Field("an2c", 0.0, "real", "data", "target"),
            Field("sighx", 0.0, "real", "data", "target"),
            Field("wloadt2", 0.0, "real", "data", "target"),
            Field("clat2", 0.0, "real", "data", "target"),
            Field("ht2e", 0.0, "real", "data", "target"),
            Field("rhl", 0.0, "real", "data", "target"),
            Field("alamhx", 0.0, "real", "diag", "target"),
            Field("amuhx", 0.0, "real", "data", "target"),
            Field("rcomb", 0.0, "real", "data", "target"),
            Field("alpt2x", 0.0, "real", "data", "target"),
            Field("phibt2x", 0.0, "real", "data", "target"),
            Field("ax2c", 0.0, "real", "data", "target"),
            Field("psit1lx", 0.0, "real", "init/diag", "target"),
            Field("thtt1lx", 0.0, "real", "init/diag", "target"),
            Field("psit2lx", 0.0, "real", "init/diag", "target"),
            Field("thtt2lx", 0.0, "real", "init/diag", "target"),
            Field("phit1lcx", 0.0, "real", "data", "target"),
            Field("phit2lcx", 0.0, "real", "data", "target"),
            Field("anuhx", 0.0, "real", "diag", "target"),
            Field("psilar4x", 0.0, "real", "diag", "target"),
            Field("ST1EL", _ZEROS3, "vec", "state", "target"),
            Field("ST2EL", _ZEROS3, "vec", "state", "target"),
            Field("VT1EL", _ZEROS3, "vec", "state", "target"),
            Field("VT2EL", _ZEROS3, "vec", "state", "target"),
            Field("SBEL", _ZEROS3, "vec", "state", "target"),
            Field("VBEL", _ZEROS3, "vec", "state", "target"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        minit = int(store.get("minit"))
        if minit == 0:
            return
        if minit not in _MINIT_OK:
            raise ValueError(f"unsupported minit {minit}")

        an1c = float(store.get("an1c"))
        dvt1e = float(store.get("dvt1e"))
        ht1e = float(store.get("ht1e"))
        tauhx = float(store.get("tauhx"))
        dvt2e = float(store.get("dvt2e"))
        an2c = float(store.get("an2c"))
        sighx = float(store.get("sighx"))
        wloadt2 = float(store.get("wloadt2"))
        clat2 = float(store.get("clat2"))
        ht2e = float(store.get("ht2e"))
        rhl = float(store.get("rhl"))
        alamhx = float(store.get("alamhx"))
        amuhx = float(store.get("amuhx"))
        rcomb = float(store.get("rcomb"))
        alpt2x = float(store.get("alpt2x"))
        psit1lx = float(store.get("psit1lx"))
        thtt1lx = float(store.get("thtt1lx"))
        psit2lx = float(store.get("psit2lx"))
        thtt2lx = float(store.get("thtt2lx"))
        phit1lcx = float(store.get("phit1lcx"))
        phit2lcx = float(store.get("phit2lcx"))
        anuhx = float(store.get("anuhx"))
        psilar4x = float(store.get("psilar4x"))
        phit1lx = phit1lcx
        phit2lx = phit2lcx

        mins = int(minit / 100)
        minc = int((minit - mins * 100) / 10)
        minm = minit - mins * 100 - minc * 10

        st2el = np.asarray(store.get("ST2EL"), dtype=float).copy()

        # Shooter alpha (Fortran: before ST2EL assignment; uses prior ST2EL(3))
        if alpt2x == 0.0:
            rhot2 = 1.225 * (1.0 + float(st2el[2]) / 41900.0) ** 4
            alpt2x = 2.0 * an2c / (rhot2 * dvt2e**2) * wloadt2 / clat2

        # Shooter position
        if minc == 1:
            st2el = np.array([0.0, 0.0, -ht2e])
        elif minc == 2:
            if minm == 1:
                st2el = np.array(
                    [
                        rhl * math.cos(tauhx / DEG),
                        rhl * math.sin(tauhx / DEG),
                        -ht2e,
                    ]
                )
            elif minm >= 2:
                st2el = np.array(
                    [
                        -rcomb * math.sin(amuhx / DEG),
                        rcomb * (1.0 - math.cos(amuhx / DEG)),
                        -ht2e,
                    ]
                )

        # Shooter angles — LAR-2
        if minc == 1 and minm == 2:
            psit2lx = 0.0
            thtt2lx = 0.0
            phit2lx = 0.0

        # Shooter angles — LAR-3/4/5
        if minc == 1 and minm >= 3:
            phit2l = math.atan(an2c)
            sphit2 = math.sin(phit2l)
            phit2lx = DEG * phit2l
            psit2lx = -DEG * math.atan(sphit2 * math.tan(alpt2x / DEG))
            thtt2lx = 0.0
            phit2lcx = phit2lx

        # LAR-4 PSILAR4 diagnostic
        if minc == 1 and minm == 4:
            gh = an2c * math.sin(phit2l)
            rht = dvt2e**2 / (gh * AGRAV)
            sigh = (alpt2x + alamhx) / DEG
            denom2 = rhl**2 / 4.0 + rht**2 - rhl * rht * math.sin(sigh)
            denom = math.sqrt(denom2)
            argu = (rht - rhl / 2.0 * math.sin(sigh)) / denom
            if abs(argu) <= 1.0:
                angl1 = math.acos((rht - rhl / 2.0 * math.sin(sigh)) / denom)
            else:
                angl1 = 3.1416
            if rht <= denom:
                angl2 = math.acos(rht / denom)
            else:
                angl2 = -1.5708
            alpt2 = alpt2x / DEG
            con1 = rht * math.sin(alpt2)
            con2 = rhl - 2.0 * rht * math.cos(alpt2)
            conx = con1 / con2
            angxi = math.atan(conx)
            con3 = 2.0 * math.sin(angxi)
            con4 = con3 / math.sin(alpt2)
            if abs(con4) > 1.0:
                con4 = math.copysign(1.0, con4)
            angzeta = math.cos(con4)
            deltax = (1.5708 + angxi - angzeta) * DEG
            if alamhx <= 70.0:
                psilar4 = angl1 - angl2 - alpt2
                psilar4x = psilar4 * DEG
            else:
                psilar4x = alamhx + deltax

        # Target-centered LAR-1 / UK circles
        if minc == 2:
            thtt2lx = 0.0
            if minm == 1:
                phit2lx = 0.0
                psit2lx = -180.0 + tauhx - sighx
                alamhx = sighx - alpt2x
            elif minm == 2:
                phit2lx = DEG * math.atan(dvt2e**2 / (AGRAV * rcomb))
                an2c = 1.0 / math.cos(phit2lx / DEG)
                psit2lx = -amuhx
                sighx = amuhx / 2.0
                tauhx = 180.0 - sighx
            elif minm >= 3:
                phit2lx = -DEG * math.atan(dvt2e**2 / (AGRAV * rcomb))
                an2c = 1.0 / math.cos(phit2lx / DEG)
                psit2lx = -180.0 - amuhx
                sighx = amuhx / 2.0
                tauhx = -sighx
            phit2lcx = phit2lx

        # Second alpha pass (Fortran: only if still zero — circle intent)
        if alpt2x == 0.0:
            rhot2 = 1.225 * (1.0 + float(st2el[2]) / 41900.0) ** 4
            alpt2x = 2.0 * an2c / (rhot2 * dvt2e**2) * wloadt2 / clat2

        vt2el = _matcar(dvt2e, psit2lx / DEG, thtt2lx / DEG)

        # Target position
        if minc == 1:
            st1el = np.array(
                [
                    rhl * math.cos(alamhx / DEG),
                    rhl * math.sin(alamhx / DEG),
                    -ht1e,
                ]
            )
        else:
            st1el = np.array([0.0, 0.0, -ht1e])
            if minm >= 2:
                rhl = 2.0 * rcomb * abs(math.sin(amuhx / (DEG * 2.0)))

        # Target angles — shooter-centered LAR-2..5
        if minc == 1:
            thtt1lx = 0.0
            if minm == 2:
                psit1lx = 180.0 + alamhx - tauhx
                phit1lx = 0.0
                sighx = alamhx
            elif minm == 3:
                psit1lx = alpt2x + 2.0 * alamhx - 180.0
                phit1lx = -DEG * math.atan(an1c)
                tauhx = -(alpt2x + alamhx)
                sighx = -tauhx
            elif minm == 4:
                psit1lx = 180.0 - alpt2x
                phit1lx = DEG * math.atan(an1c)
                tauhx = alpt2x + alamhx
                sighx = tauhx
            elif minm == 5:
                psit1lx = 2.0 * alamhx + alpt2x
                phit1lx = DEG * math.atan(an1c)
                tauhx = 180.0 - (alpt2x + alamhx)
                sighx = 180.0 - tauhx

        # Target angles — target-centered LAR-1 / circles
        if minc == 2:
            psit1lx = 0.0
            thtt1lx = 0.0
            if minm >= 2:
                an1c = an2c
                phit1lx = DEG * math.atan(an1c)
            if minm == 4:
                phit1lx = -phit1lx

        phit1lcx = phit1lx
        anuhx = psit1lx - psit2lx

        vt1el = _matcar(dvt1e, psit1lx / DEG, thtt1lx / DEG)

        store.set("ST2EL", st2el)
        store.set("ST1EL", st1el)
        store.set("VT2EL", vt2el)
        store.set("VT1EL", vt1el)
        store.set("alpt2x", alpt2x)
        store.set("alamhx", alamhx)
        store.set("tauhx", tauhx)
        store.set("sighx", sighx)
        store.set("psit2lx", psit2lx)
        store.set("thtt2lx", thtt2lx)
        store.set("psit1lx", psit1lx)
        store.set("thtt1lx", thtt1lx)
        store.set("phit1lcx", phit1lcx)
        store.set("phit2lcx", phit2lcx)
        store.set("anuhx", anuhx)
        store.set("psilar4x", psilar4x)
        store.set("an1c", an1c)
        store.set("an2c", an2c)
        store.set("rhl", rhl)
        store.set("dvt1e", dvt1e)
        store.set("dvt2e", dvt2e)
        store.set("SBEL", st2el.copy())
        store.set("VBEL", vt2el.copy())

        # Wire Flat3 TARGET3 newton init when those fields exist.
        if "sael1" in store:
            store.set("sael1", float(st1el[0]))
            store.set("sael2", float(st1el[1]))
            store.set("sael3", float(st1el[2]))
            store.set("psialx", psit1lx)
            store.set("thtalx", thtt1lx)
            store.set("dvae", dvt1e)

    def execute(self, vehicle, ctx):
        pass

    def terminate(self, vehicle, ctx):
        pass


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


class Sraam6TargetControl:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("phiav", 0.0, "real", "state", "control"),
            Field("phiavd", 0.0, "real", "state", "control"),
            Field("tphi", 0.0, "real", "data", "control"),
            Field("philimx", 120.0, "real", "data", "control"),
            Field("phiavx", 0.0, "real", "out", "control", ("com",)),
            Field("phiavcx", 0.0, "real", "diag", "control"),
            Field("anx", 0.0, "real", "state", "control", ("com",)),
            Field("anxd", 0.0, "real", "state", "control"),
            Field("tanx", 0.0, "real", "data", "control"),
            Field("alplimx", 40.0, "real", "data", "control"),
            Field("ancomx", 0.0, "real", "diag", "control"),
            Field("clalpha", 0.0523, "real", "data", "control"),
            Field("wingloading", 3247.0, "real", "data", "control"),
            Field("phiavout", 0.0, "real", "out", "control"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        tphi = store.get("tphi")
        philimx = store.get("philimx")
        tanx = store.get("tanx")
        alplimx = store.get("alplimx")
        clalpha = store.get("clalpha")
        wingloading = store.get("wingloading")
        grav = store.get("grav")
        pdynmc = store.get("pdynmc")
        tvl = np.asarray(store.get("TVL"), dtype=float)
        tgt_option = store.get("tgt_option")
        acoml = np.asarray(store.get("ACOML"), dtype=float)
        phiav = store.get("phiav")
        phiavd = store.get("phiavd")
        anx = store.get("anx")
        anxd = store.get("anxd")
        int_step = ctx.int_step

        acomv = tvl @ acoml
        acoma2 = float(acomv[1])
        acoma3 = float(acomv[2])
        if abs(acoma2) < EPS and abs(acoma3) < EPS:
            phiavc = 0.0
        else:
            phiavc = atan2(acoma2, -acoma3)
        phiavcx = phiavc * DEG

        if tphi:
            phiavd_new = (phiavc - phiav) / tphi
            phiav = integrate(phiavd_new, phiavd, phiav, int_step)
            phiavd = phiavd_new
        else:
            phiav = phiavc

        phiavx = phiav * DEG
        if abs(phiavx) >= philimx:
            phiavx = philimx * _sign(phiavx)

        phiavout = phiavx * RAD

        ancomx = sqrt(acoma2 * acoma2 + acoma3 * acoma3) / grav

        if tanx:
            anxd_new = (ancomx - anx) / tanx
            anx = integrate(anxd_new, anxd, anx, int_step)
            anxd = anxd_new
        else:
            anx = ancomx
        if tgt_option > 0:
            anlimx = pdynmc * clalpha * alplimx / (wingloading * grav)
            if abs(anx) >= anlimx:
                anx = anlimx * _sign(anx)

        store.set("phiav", phiav)
        store.set("phiavd", phiavd)
        store.set("anx", anx)
        store.set("anxd", anxd)
        store.set("phiavout", phiavout)
        store.set("phiavx", phiavx)
        store.set("phiavcx", phiavcx)
        store.set("ancomx", ancomx)

    def terminate(self, vehicle, ctx):
        pass


class Sraam6TargetForces:
    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("FSPA", _ZEROS3, "vec", "out", "forces"),
            Field("acc_longx", 0.0, "real", "data", "forces"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        acc_longx = store.get("acc_longx")
        grav = store.get("grav")
        anx = store.get("anx")
        acoma1 = acc_longx * grav
        acoma2 = 0.0
        acoma3 = -anx * grav
        store.set("FSPA", np.array([acoma1, acoma2, acoma3]))

    def terminate(self, vehicle, ctx):
        pass


class Sraam6Target:
    type = "TARGET3"

    def __init__(self, name, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Sraam6TargetMinit(),
            Flat3AircraftEnvironment(),
            Flat3Kinematics(),
            Flat3AircraftNewton(),
            Sraam6TargetGuidance(),
            Sraam6TargetControl(),
            Sraam6TargetForces(),
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
