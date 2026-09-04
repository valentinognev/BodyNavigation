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
        if keyword in ("2DIM", "3DIM"):
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
