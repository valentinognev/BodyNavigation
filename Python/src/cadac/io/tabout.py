"""CADAC tabout.asc writers (Aim/Plane::tabout_banner + tabout_data)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# C++ tabout: label_length=14, field width 15, eight across (k>7).
_LABEL_LEN = 14
_FIELD_WIDTH = 15
_ACROSS = 8

# C++ hardcodes program vehicle names in banners (PLANE → FALCON5).
_BANNER_VEHICLE = {
    "PLANE": "FALCON5",
    "AIM5": "AIM5",
    "CRUISE3": "CRUISE3",
    "HYPER5": "HYPER5",
    "HYPER6": "HYPER6",
    "MISSILE6": "MISSILE6",
    "ROTOR": "ROTOR",
}


def _truncate_label(name: str) -> str:
    return name[:_LABEL_LEN]


def scrn_stems(store) -> list[str]:
    """Module-variable stems flagged for scrn (vectors keep one stem)."""
    stems = []
    for name in store.names():
        field = store.field(name)
        if "scrn" not in field.outputs:
            continue
        if field.type == "mat":
            continue
        stems.append(name)
    return stems


def _is_vector_stem(store, name: str) -> bool:
    field = store.field(name)
    if field.type == "vec":
        return True
    buff = _truncate_label(name)
    return bool(buff) and buff[0].isupper()


def banner_vehicle_label(vehicle_type: str) -> str:
    return _BANNER_VEHICLE.get(vehicle_type, vehicle_type)


def write_tabout_banner(
    stream,
    title: str,
    vehicle_type: str,
    store,
    *,
    when: datetime | None = None,
    nmonte: int = 0,
    nmc: int = 0,
) -> None:
    """Write CADAC tabout_banner header (title/date, Vehicle line, scrn labels)."""
    when = when or datetime.now()
    # C++ __DATE__ / __TIME__: "Mmm dd yyyy" (space-padded day) and HH:MM:SS
    date_s = f"{when.strftime('%b')} {when.day:2d} {when.year}"
    time_s = when.strftime("%H:%M:%S")
    stream.write(f"\n{title}   {date_s} {time_s}\n")
    if nmonte:
        stream.write(f" MONTE Run # {nmc + 1}\n")
    stream.write(f"\n Vehicle: {banner_vehicle_label(vehicle_type)} \n")
    k = 0
    for name in scrn_stems(store):
        buff = _truncate_label(name)
        if _is_vector_stem(store, name):
            for i in range(1, 4):
                pad = _FIELD_WIDTH - len(buff)
                stream.write(buff)
                stream.write(f"{i:<{pad}}" if pad > 0 else str(i))
                k += 1
                if k > 7:
                    k = 0
                    stream.write("\n")
        else:
            stream.write(f"{buff:<{_FIELD_WIDTH}}")
            k += 1
            if k > 7:
                k = 0
                stream.write("\n")
    stream.write("\n\n")


def collect_tabout_values(store) -> list:
    """Flatten scrn-flagged values like tabout_data (ints stay int, vecs→3)."""
    values = []
    for name in scrn_stems(store):
        field = store.field(name)
        if field.type == "int":
            values.append(int(store.get(name)))
        elif _is_vector_stem(store, name):
            vec = store.get(name)
            values.extend(float(vec[i]) for i in range(3))
        else:
            values.append(float(store.get(name)))
    return values


def write_tabout_data(stream, vehicle_name: str, store) -> None:
    """Append one CADAC tabout.asc data block (name + eight-across values)."""
    stream.write(f"{vehicle_name}\n")
    k = 0
    for value in collect_tabout_values(store):
        stream.write(f"{value:<{_FIELD_WIDTH}}")
        k += 1
        if k > 7:
            k = 0
            stream.write("\n")
    stream.write("\n")


def open_tabout_stream(out_dir: Path):
    """Open tabout.asc for writing (C++ ofstream ftabout)."""
    path = Path(out_dir) / "tabout.asc"
    stream = path.open("w", encoding="utf-8", newline="\n")
    return path, stream
