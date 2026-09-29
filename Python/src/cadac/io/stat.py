"""CADAC stati.asc / stat.asc writers (Missile::plot_banner + stat_data + merge_stat_files)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

_LABEL_LEN = 13
_FIELD_WIDTH = 16
_ACROSS = 5


def _truncate_label(name: str) -> str:
    return name[:_LABEL_LEN]


def plot_stems(store) -> list[str]:
    """Module-variable stems flagged for plot (vectors keep one stem)."""
    stems = []
    for name in store.names():
        field = store.field(name)
        if "plot" not in field.outputs:
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


def nvariables(store) -> int:
    """CADAC nvariables = stems + 2×vector_count."""
    stems = plot_stems(store)
    m = sum(1 for name in stems if _is_vector_stem(store, name))
    return len(stems) + 2 * m


def write_stat_banner(
    stream,
    title: str,
    vehicle_name: str,
    store,
    *,
    when: datetime | None = None,
) -> int:
    """Write CADAC plot_banner header used for stati.asc. Returns nvariables."""
    when = when or datetime.now()
    date_s = f"{when.strftime('%b')} {when.day:2d} {when.year}"
    time_s = when.strftime("%H:%M:%S")
    nvar = nvariables(store)
    stream.write(f"1{title} '{vehicle_name} ' {date_s} {time_s}\n")
    stream.write(f"  0  0 {nvar}\n")
    k = 0
    for name in plot_stems(store):
        buff = _truncate_label(name)
        if _is_vector_stem(store, name):
            for i in range(1, 4):
                # C++: width(len)<<stem; width(16-len)<<i;  (ios::left)
                pad = _FIELD_WIDTH - len(buff)
                stream.write(buff)
                stream.write(f"{i:<{pad}}" if pad > 0 else str(i))
                k += 1
                if k > 4:
                    k = 0
                    stream.write("\n")
        else:
            stream.write(f"{buff:<{_FIELD_WIDTH}}")
            k += 1
            if k > 4:
                k = 0
                stream.write("\n")
    if nvar % _ACROSS:
        stream.write("\n")
    return nvar


def collect_stat_values(store) -> list[float]:
    """Flatten plot-flagged values like Missile::stat_data (ints→float, vecs→3)."""
    values: list[float] = []
    for name in plot_stems(store):
        field = store.field(name)
        if field.type == "int":
            values.append(float(store.get(name)))
        elif _is_vector_stem(store, name):
            vec = store.get(name)
            values.extend(float(vec[i]) for i in range(3))
        else:
            values.append(float(store.get(name)))
    return values


def write_stat_data(stream, store, nmc: int, vehicle_slot: int) -> None:
    """Append one CADAC stati.asc data block ending with |MC#||object#|."""
    values = collect_stat_values(store)
    k = 0
    for value in values:
        if k > 4:
            k = 0
            stream.write("\n")
        stream.write(f"{value:<{_FIELD_WIDTH}}")
        k += 1
    # C++ writes MC# and object# without the five-across wrap check
    stream.write(f"{float(nmc + 1):<{_FIELD_WIDTH}}")
    stream.write(f"{float(vehicle_slot + 1):<{_FIELD_WIDTH}}")
    stream.write("\n")


def merge_stat_files(
    stat_paths: list[Path],
    out_path: Path,
    title: str,
    *,
    when: datetime | None = None,
) -> None:
    """Merge stati.asc files onto stat.asc (C++ merge_stat_files)."""
    if not stat_paths:
        return
    when = when or datetime.now()
    date_s = f"{when.strftime('%b')} {when.day:2d} {when.year}"
    time_s = when.strftime("%H:%M:%S")
    first = Path(stat_paths[0])
    lines0 = first.read_text(encoding="utf-8").splitlines()
    num_labels = 0
    if len(lines0) >= 2:
        parts = lines0[1].split()
        if len(parts) >= 3:
            num_labels = int(parts[2])
    num_label_lines = num_labels // 5
    if num_labels % 5 > 0:
        num_label_lines += 1
    strip_lines = num_label_lines + 2

    out_lines: list[str] = []
    first_time = True
    for line in lines0:
        if first_time:
            first_time = False
            out_lines.append(f"1{title}  {date_s} {time_s}")
        elif line:
            out_lines.append(line)

    for path in stat_paths[1:]:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        for line in lines[strip_lines:]:
            if line:
                out_lines.append(line)

    Path(out_path).write_text("\n".join(out_lines) + "\n", encoding="utf-8", newline="\n")


def open_stat_streams(
    out_dir: Path,
    title: str,
    vehicles: list,
    *,
    when: datetime | None = None,
) -> tuple[list[Path | None], list]:
    """
    Open stati.asc for each MISSILE6 (filename uses 1-based vehicle slot).
    Returns (path_per_slot, open_file_per_slot); non-missiles are None.
    """
    paths: list[Path | None] = [None] * len(vehicles)
    streams: list = [None] * len(vehicles)
    for slot, vehicle in enumerate(vehicles):
        if getattr(vehicle, "type", None) != "MISSILE6":
            continue
        path = out_dir / f"stat{slot + 1}.asc"
        stream = path.open("w", encoding="utf-8", newline="\n")
        write_stat_banner(
            stream, title, getattr(vehicle, "name", ""), vehicle.store, when=when
        )
        stream.flush()
        paths[slot] = path
        streams[slot] = stream
    return paths, streams
