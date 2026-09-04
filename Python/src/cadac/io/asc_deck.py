from pathlib import Path

from cadac.tables.lookup import Table


def parse_asc_deck(path) -> tuple[str, list[Table]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    title = ""
    tables: list[Table] = []
    i = 0
    n = len(lines)
    while i < n:
        parts = lines[i].split()
        if not parts:
            i += 1
            continue
        keyword = parts[0]
        if keyword == "TITLE":
            title = lines[i].split(None, 1)[1].strip() if len(parts) > 1 else ""
            i += 1
            continue
        if keyword == "1DIM":
            table, i = _parse_1dim(lines, i)
            tables.append(table)
            continue
        if keyword == "2DIM":
            table, i = _parse_2dim(lines, i)
            tables.append(table)
            continue
        if keyword == "3DIM":
            raise NotImplementedError(f"{keyword} tables are not supported yet")
        i += 1
    return title, tables


def _parse_1dim(lines: list[str], i: int) -> tuple[Table, int]:
    name = lines[i].split()[1]
    i += 1
    while i < len(lines) and not lines[i].split():
        i += 1
    count = int(lines[i].split()[1])
    i += 1
    x1: list[float] = []
    values: list[float] = []
    while len(x1) < count:
        row = lines[i].split()
        i += 1
        if not row:
            continue
        x1.append(float(row[0]))
        values.append(float(row[1]))
    return Table(name=name, dim=1, x1=x1, x2=None, x3=None, values=values), i


def _skip_blank(lines: list[str], i: int) -> int:
    while i < len(lines) and not lines[i].split():
        i += 1
    return i


def _read_axis_counts(lines: list[str], i: int, ndim: int) -> tuple[list[int], int]:
    needed = ndim * 2
    got: list[str] = []
    while len(got) < needed:
        i = _skip_blank(lines, i)
        row = lines[i].split()
        i += 1
        for tok in row:
            got.append(tok)
            if len(got) == needed:
                break
    dims = [int(got[j * 2 + 1]) for j in range(ndim)]
    return dims, i


class _TokenCursor:
    def __init__(self, lines: list[str], i: int) -> None:
        self.lines = lines
        self.i = i
        self._buf: list[str] = []

    def next_float(self) -> float:
        while not self._buf:
            self.i = _skip_blank(self.lines, self.i)
            self._buf = self.lines[self.i].split()
            self.i += 1
        return float(self._buf.pop(0))


def _parse_2dim(lines: list[str], i: int) -> tuple[Table, int]:
    name = lines[i].split()[1]
    i += 1
    dims, i = _read_axis_counts(lines, i, 2)
    nx1, nx2 = dims
    var_dim = [nx1, nx2, 1]
    num_rows = max(var_dim)
    x1 = [0.0] * nx1
    x2 = [0.0] * nx2
    data = [0.0] * (nx1 * nx2)
    cur = _TokenCursor(lines, i)
    for tt in range(num_rows):
        if tt < var_dim[0]:
            x1[tt] = cur.next_float()
        if tt < var_dim[1] and var_dim[1] != 1:
            x2[tt] = cur.next_float()
        if tt < var_dim[0]:
            offset = tt * var_dim[1] * var_dim[2]
            for ttt in range(var_dim[1] * var_dim[2]):
                data[offset + ttt] = cur.next_float()
    values = [data[r * nx2 : (r + 1) * nx2] for r in range(nx1)]
    return Table(name=name, dim=2, x1=x1, x2=x2, x3=None, values=values), cur.i
