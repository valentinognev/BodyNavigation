import math

from cadac.constants import R


def iso62(alt_m, dvbe):
    if alt_m < 11000.0:
        k = 288.15 - 0.0065 * alt_m
        press = 101325.0 * (k / 288.15) ** 5.2559
    else:
        k = 216.0
        press = 22630.0 * math.exp(-0.00015769 * (alt_m - 11000.0))
    rho = press / (R * k)
    vsound = math.sqrt(1.4 * R * k)
    mach = abs(dvbe / vsound)
    pdynmc = 0.5 * rho * dvbe ** 2
    return {
        "k": k,
        "press": press,
        "rho": rho,
        "vsound": vsound,
        "mach": mach,
        "pdynmc": pdynmc,
    }
