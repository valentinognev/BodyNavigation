"""CADAC ploti.asc writers and plot.asc merge (Missile::plot_data + merge_plot_files)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from cadac.io.stat import collect_stat_values, write_stat_banner

_FIELD_WIDTH = 16
_ACROSS = 5

# C++ opens ploti.asc only for the primary vehicle type of each program.
PLOT_ASC_TYPES = frozenset(
    {
        "MISSILE6",
        "AIM5",
        "ROTOR",
        "PLANE",
        "PLANE6",
        "CRUISE3",
        "HYPER5",
        "HYPER6",
    }
)


def write_plot_data(stream, store, *, merge: bool = False) -> None:
    """Append one CADAC ploti.asc data block; merge replaces time with -1.0."""
    values = collect_stat_values(store)
    if merge and values:
        values[0] = -1.0
    k = 0
    for value in values:
        if k > 4:
            k = 0
            stream.write("\n")
        stream.write(f"{value:<{_FIELD_WIDTH}}")
        k += 1
    stream.write("\n")


def merge_plot_files(
    plot_paths: list[Path],
    out_path: Path,
    title: str,
    *,
    when: datetime | None = None,
) -> None:
    """Merge ploti.asc files onto plot.asc (C++ merge_plot_files)."""
    if not plot_paths:
        return
    when = when or datetime.now()
    date_s = f"{when.strftime('%b')} {when.day:2d} {when.year}"
    time_s = when.strftime("%H:%M:%S")
    first = Path(plot_paths[0])
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

    for path in plot_paths[1:]:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        for line in lines[strip_lines:]:
            if line:
                out_lines.append(line)

    Path(out_path).write_text("\n".join(out_lines) + "\n", encoding="utf-8", newline="\n")


def open_plot_streams(
    out_dir: Path,
    title: str,
    vehicles: list,
    *,
    when: datetime | None = None,
) -> tuple[list[Path | None], list]:
    """
    Open ploti.asc for each primary vehicle (filename uses 1-based vehicle slot).
    Returns (path_per_slot, open_file_per_slot); non-plot types are None.
    """
    paths: list[Path | None] = [None] * len(vehicles)
    streams: list = [None] * len(vehicles)
    for slot, vehicle in enumerate(vehicles):
        if getattr(vehicle, "type", None) not in PLOT_ASC_TYPES:
            continue
        path = out_dir / f"plot{slot + 1}.asc"
        stream = path.open("w", encoding="utf-8", newline="\n")
        write_stat_banner(
            stream, title, getattr(vehicle, "name", ""), vehicle.store, when=when
        )
        stream.flush()
        paths[slot] = path
        streams[slot] = stream
    return paths, streams
