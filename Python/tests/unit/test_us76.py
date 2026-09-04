from cadac.env.us76 import atmosphere76


def test_sea_level():
    rho, press, tempk = atmosphere76(0.0)
    assert abs(rho - 1.225) < 1e-12
    assert abs(press - 101325) < 1e-12
    assert abs(tempk - 288.15) < 1e-12


def test_11km_geometric():
    rho, press, tempk = atmosphere76(11000.0)
    assert abs(rho - 0.36480011506536575) < 1e-12
    assert abs(press - 22699.830123704545) < 1e-12
    assert abs(tempk - 216.77327586206891) < 1e-12
