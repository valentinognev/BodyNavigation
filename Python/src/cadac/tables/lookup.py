from dataclasses import dataclass

import numpy as np

from cadac.constants import EPS


def _expected_shape(dim: int, x1, x2, x3) -> tuple[int, ...]:
    if dim == 1:
        return (len(x1),)
    if dim == 2:
        if x2 is None:
            raise ValueError("2D table requires x2")
        return (len(x1), len(x2))
    if dim == 3:
        if x2 is None or x3 is None:
            raise ValueError("3D table requires x2 and x3")
        return (len(x1), len(x2), len(x3))
    raise ValueError(f"dim must be 1, 2, or 3, got {dim}")


@dataclass
class Table:
    name: str
    dim: int
    x1: np.ndarray
    x2: np.ndarray | None
    x3: np.ndarray | None
    values: np.ndarray

    def __post_init__(self) -> None:
        self.x1 = np.asarray(self.x1, dtype=float)
        self.x2 = None if self.x2 is None else np.asarray(self.x2, dtype=float)
        self.x3 = None if self.x3 is None else np.asarray(self.x3, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        expected = _expected_shape(self.dim, self.x1, self.x2, self.x3)
        if self.values.shape != expected:
            raise ValueError(
                f"table {self.name!r}: values shape {self.values.shape} "
                f"!= {expected}"
            )


class Datadeck:
    def __init__(self, tables: dict[str, Table]) -> None:
        self._tables = tables

    @classmethod
    def from_tables(cls, tables: list[Table]) -> "Datadeck":
        return cls({t.name: t for t in tables})

    def table(self, name: str) -> Table:
        return self._tables[name]

    def find_index(self, max: int, value: float, breakpoints) -> int:
        if value >= breakpoints[max]:
            return max
        if value <= breakpoints[0]:
            return 0
        index = 0
        while index <= max:
            mid = (index + max) // 2
            if value < breakpoints[mid]:
                max = mid - 1
            elif value > breakpoints[mid]:
                index = mid + 1
            else:
                return mid
        return max

    def look_up(self, name: str, x1: float) -> float:
        table = self._tables[name]
        n = len(table.x1)
        loc = self.find_index(n - 1, x1, table.x1)
        if loc == n - 1:
            return float(table.values[-1])
        return self._interpolate_1d(table, loc, x1)

    def _interpolate_1d(self, table: Table, loc: int, val: float) -> float:
        ind1 = loc
        ind2 = loc + 1
        diff = val - table.x1[ind1]
        dx = table.x1[ind2] - table.x1[ind1]
        dy = table.values[ind2] - table.values[ind1]
        dumx = 0.0
        if dx > EPS:
            dumx = diff / dx
        dy = dumx * dy
        return float(table.values[ind1] + dy)
