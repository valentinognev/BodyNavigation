from math import exp

from cadac.constants import R
from cadac.env.iso62 import iso62


def test_below_tropopause():
    out = iso62(3000.0, 250.0)
    k = 288.15 - 0.0065 * 3000.0
    press = 101325.0 * (k / 288.15) ** 5.2559
    rho = press / (R * k)
    vsound = (1.4 * R * k) ** 0.5
    assert abs(out["press"] - press) < 1e-12
    assert abs(out["rho"] - rho) < 1e-12
    assert abs(out["mach"] - abs(250.0 / vsound)) < 1e-12


def test_stratosphere_20000():
    alt = 20000.0
    dvbe = 250.0
    out = iso62(alt, dvbe)
    k = 216.0
    press = 22630.0 * exp(-0.00015769 * (alt - 11000.0))
    rho = press / (R * k)
    vsound = (1.4 * R * k) ** 0.5
    mach = abs(dvbe / vsound)
    pdynmc = 0.5 * rho * dvbe ** 2
    assert abs(out["k"] - k) < 1e-12
    assert abs(out["press"] - press) < 1e-12
    assert abs(out["rho"] - rho) < 1e-12
    assert abs(out["vsound"] - vsound) < 1e-12
    assert abs(out["mach"] - mach) < 1e-12
    assert abs(out["pdynmc"] - pdynmc) < 1e-12
