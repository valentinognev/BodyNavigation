from pathlib import Path

PLOT_COLUMNS = (
    "time",
    "FSPV1",
    "FSPV2",
    "FSPV3",
    "pdynmc",
    "mach",
    "lonx",
    "latx",
    "alt",
    "dvbe",
    "psivgx",
    "thtvgx",
    "SBEG1",
    "SBEG2",
    "SBEG3",
    "VBEG1",
    "VBEG2",
    "VBEG3",
    "throttle",
    "mass",
    "thrust",
    "fmassr",
    "cl_ov_cd",
)

_VEC_COMPONENT = {
    "FSPV1": ("FSPV", 0),
    "FSPV2": ("FSPV", 1),
    "FSPV3": ("FSPV", 2),
    "SBEG1": ("sbeg", 0),
    "SBEG2": ("sbeg", 1),
    "SBEG3": ("sbeg", 2),
    "VBEG1": ("vbeg", 0),
    "VBEG2": ("vbeg", 1),
    "VBEG3": ("vbeg", 2),
}


def flagged_plot_columns(store):
    columns = []
    for name in store.names():
        field = store.field(name)
        if "plot" not in field.outputs:
            continue
        if field.type == "vec":
            columns.extend([f"{name}1", f"{name}2", f"{name}3"])
        elif field.type != "mat":
            columns.append(name)
    return columns


def plot_row(store, columns=None):
    if columns is None:
        columns = PLOT_COLUMNS
    row = {}
    for column in columns:
        spec = _VEC_COMPONENT.get(column)
        if spec is not None:
            name, index = spec
            row[column] = float(store.get(name)[index])
            continue
        if column and column[-1] in "123":
            name = column[:-1]
            if name in store.names() and store.field(name).type == "vec":
                row[column] = float(store.get(name)[int(column[-1]) - 1])
                continue
        value = store.get(column)
        row[column] = int(value) if type(value) is int else float(value)
    return row


def write_plot_csv(path, title: str, columns: list[str], rows: list[list[float]]) -> None:
    path = Path(path)
    n = len(columns)
    lines = [title, f"0  0 {n}", ",".join(columns) + ","]
    for row in rows:
        lines.append(",".join(str(value) for value in row) + ",")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
