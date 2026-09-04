def integrate(dydx_new, dydx, y, dt):
    return y + (dydx_new + dydx) * dt / 2
