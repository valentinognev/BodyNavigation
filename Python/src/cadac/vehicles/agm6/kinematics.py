import math

import numpy as np

from cadac.constants import DEG, EPS
from cadac.eom.flat6 import Flat6Kinematics

SMALL = 1e-7


def _sign(variable):
    if variable < 0.0:
        return -1
    return 1


class Agm6Kinematics(Flat6Kinematics):
    name = "kinematics"

    def execute(self, vehicle, ctx):
        super().execute(vehicle, ctx)
        store = vehicle.store
        tbl = store.get("TBL")
        vbeb = np.asarray(store.get("VBEB"), dtype=float)
        # C++ Flat6::kinematics loads flat6[72] VAELS (smoothed wind), not
        # plotted VAEL=VAELS+Dryden (flat6[74]).
        wind = store.get("VAELS") if "VAELS" in store.names() else store.get("VAEL")
        vael = np.asarray(wind, dtype=float)
        dvba = store.get("dvba")
        vbab = vbeb - tbl @ vael
        vbab1 = float(vbab[0])
        vbab2 = float(vbab[1])
        vbab3 = float(vbab[2])
        alpha = math.atan2(vbab3, vbab1)
        beta = math.asin(vbab2 / dvba)
        dum = vbab1 / dvba
        if math.fabs(dum) >= 1.0:
            dum = (1.0 - EPS) * _sign(dum)
        alpp = math.acos(dum)
        if math.fabs(vbab2) < EPS and math.fabs(vbab3) < EPS:
            phip = 0.0
        elif math.fabs(vbab2) < SMALL:
            phip = math.atan2(SMALL, vbab3)
        else:
            phip = math.atan2(vbab2, vbab3)
        alphax = alpha * DEG
        betax = beta * DEG
        alppx = alpp * DEG
        phipx = phip * DEG
        store.set("alphax", alphax)
        store.set("betax", betax)
        store.set("alppx", alppx)
        store.set("phipx", phipx)
        store.set("alpp", alpp)
        store.set("phip", phip)
        if "tralp" in store.names() and alpp > store.get("tralp"):
            if "trcond" in store.names():
                store.set("trcond", 5)
