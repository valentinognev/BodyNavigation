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

    def look_up(
        self,
        name: str,
        x1: float,
        x2: float | None = None,
        x3: float | None = None,
    ) -> float:
        table = self._tables[name]
        if x2 is None:
            n = len(table.x1)
            loc = self.find_index(n - 1, x1, table.x1)
            if loc == n - 1:
                return float(table.values[-1])
            return self._interpolate_1d(table, loc, x1)
        n1 = len(table.x1)
        n2 = len(table.x2)
        loc1 = self.find_index(n1 - 1, x1, table.x1)
        loc2 = self.find_index(n2 - 1, x2, table.x2)
        if x3 is None:
            return self._interpolate_2d(
                table, loc1, loc1 + 1, loc2, loc2 + 1, x1, x2
            )
        n3 = len(table.x3)
        loc3 = self.find_index(n3 - 1, x3, table.x3)
        return self._interpolate_3d(
            table,
            loc1,
            loc1 + 1,
            loc2,
            loc2 + 1,
            loc3,
            loc3 + 1,
            x1,
            x2,
            x3,
        )

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

    def _interpolate_2d(
        self,
        table: Table,
        ind10: int,
        ind11: int,
        ind20: int,
        ind21: int,
        value1: float,
        value2: float,
    ) -> float:
        dx1 = 0.0
        dx2 = 0.0
        dumx1 = 0.0
        dumx2 = 0.0
        var1_dim = len(table.x1)
        var2_dim = len(table.x2)
        diff1 = value1 - table.x1[ind10]
        diff2 = value2 - table.x2[ind20]
        if ind10 == var1_dim - 1:
            ind11 = ind10
        else:
            dx1 = float(table.x1[ind11] - table.x1[ind10])
        if ind20 == var2_dim - 1:
            ind21 = ind20
        else:
            dx2 = float(table.x2[ind21] - table.x2[ind20])
        if dx1 > EPS:
            dumx1 = diff1 / dx1
        if dx2 > EPS:
            dumx2 = diff2 / dx2
        y11 = table.values[ind10, ind20]
        y12 = table.values[ind10, ind21]
        y21 = table.values[ind11, ind20]
        y22 = table.values[ind11, ind21]
        y1 = dumx1 * (y21 - y11) + y11
        y2 = dumx1 * (y22 - y12) + y12
        return float(dumx2 * (y2 - y1) + y1)

    def _interpolate_3d(
        self,
        table: Table,
        ind10: int,
        ind11: int,
        ind20: int,
        ind21: int,
        ind30: int,
        ind31: int,
        value1: float,
        value2: float,
        value3: float,
    ) -> float:
        dx1 = 0.0
        dx2 = 0.0
        dx3 = 0.0
        dumx1 = 0.0
        dumx2 = 0.0
        dumx3 = 0.0
        var1_dim = len(table.x1)
        var2_dim = len(table.x2)
        var3_dim = len(table.x3)
        diff1 = value1 - table.x1[ind10]
        diff2 = value2 - table.x2[ind20]
        diff3 = value3 - table.x3[ind30]
        if ind10 == var1_dim - 1:
            ind11 = ind10
        else:
            dx1 = float(table.x1[ind11] - table.x1[ind10])
        if ind20 == var2_dim - 1:
            ind21 = ind20
        else:
            dx2 = float(table.x2[ind21] - table.x2[ind20])
        if ind30 == var3_dim - 1:
            ind31 = ind30
        else:
            dx3 = float(table.x3[ind31] - table.x3[ind30])
        if dx1 > EPS:
            dumx1 = diff1 / dx1
        if dx2 > EPS:
            dumx2 = diff2 / dx2
        if dx3 > EPS:
            dumx3 = diff3 / dx3
        y11 = table.values[ind10, ind20, ind30]
        y12 = table.values[ind11, ind20, ind30]
        y31 = table.values[ind10, ind20, ind31]
        y32 = table.values[ind11, ind20, ind31]
        y1 = dumx1 * (y12 - y11) + y11
        y3 = dumx1 * (y32 - y31) + y31
        y21 = dumx3 * (y3 - y1) + y1
        y11 = table.values[ind10, ind21, ind30]
        y12 = table.values[ind11, ind21, ind30]
        y31 = table.values[ind10, ind21, ind31]
        y32 = table.values[ind11, ind21, ind31]
        y1 = dumx1 * (y12 - y11) + y11
        y3 = dumx1 * (y32 - y31) + y31
        y22 = dumx3 * (y3 - y1) + y1
        return float(dumx2 * (y22 - y21) + y21)
