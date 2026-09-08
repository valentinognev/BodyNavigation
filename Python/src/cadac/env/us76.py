import math


def atmosphere76(balt: float) -> tuple[float, float, float]:
    rearth = 6369.0  # radius of the earth - km
    gmr = 34.163195  # gas constant
    rhosl = 1.22500  # sea level density - kg/m^3
    pressl = 101325  # sea level pressure - Pa
    tempksl = 288.15  # sea level temperature - dK

    htab = [0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.852]  # altitude
    ttab = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]  # temperture
    ptab = [1.0, 2.233611e-1, 5.403295e-2, 8.5666784e-3, 1.0945601e-3,
            6.6063531e-4, 3.9046834e-5, 3.68501e-6]  # pressure
    gtab = [-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0]  # temperture gradient

    delta = 0

    # convert geometric (m) to geopotential altitude (km)
    alt = balt / 1000
    # C++: if(alt<84.852) table; else vacuum + last-layer tempk.
    if alt >= 84.852:
        return 0.0, 0.0, 186.946
    h = alt * rearth / (alt + rearth)

    # binary search determines altitude table entry i below actual altitude
    i = 0  # offset of first value in table
    j = 7  # offset of last value in table
    while True:
        k = (i + j) // 2  # integer division
        if h < htab[k]:
            j = k
        else:
            i = k
        if j <= (i + 1):
            break

    # normalized temperature 'theta' from table look-up and gradient interpolation
    tgrad = gtab[i]
    tbase = ttab[i]
    deltah = h - htab[i]
    tlocal = tbase + tgrad * deltah
    theta = tlocal / ttab[0]

    # normalized pressure from hydrostatic equations
    if tgrad == 0:
        delta = ptab[i] * math.exp(-gmr * deltah / tbase)
    else:
        delta = ptab[i] * math.pow((tbase / tlocal), (gmr / tgrad))

    # normalized density
    sigma = delta / theta

    # output
    rho = rhosl * sigma
    press = pressl * delta
    tempk = tempksl * theta
    return rho, press, tempk
