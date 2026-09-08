from typing import Any


def integrate(dydx_new: Any, dydx: Any, y: Any, dt: float) -> Any:
    return y + (dydx_new + dydx) * dt / 2
