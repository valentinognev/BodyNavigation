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


def plot_row(store):
    row = {}
    for column in PLOT_COLUMNS:
        spec = _VEC_COMPONENT.get(column)
        if spec is None:
            value = store.get(column)
            row[column] = int(value) if type(value) is int else float(value)
        else:
            name, index = spec
            row[column] = float(store.get(name)[index])
    return row


def write_plot_csv(path, title: str, columns: list[str], rows: list[list[float]]) -> None:
    path = Path(path)
    n = len(columns)
    lines = [title, f"0  0 {n}", ",".join(columns) + ","]
    for row in rows:
        lines.append(",".join(str(value) for value in row) + ",")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
