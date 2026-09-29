"""CADAC doc.asc writers (Vehicle::document + document_input)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_SEP = (
    "---------------------------------------------------------------------------------------------------------------------"
)
_SEP10 = (
    "----------------------------------------------------------------------------------------------------------------------"
)
_STAR = "*" * 117


@dataclass
class DocEntry:
    """One non-empty module-variable slot for doc.asc / document_input."""

    name: str
    type: str
    definition: str
    module: str
    role: str
    outputs: str
    loc: int = 0


def entries_from_store(store) -> list[DocEntry]:
    """Build Document[]-like entries from a StateStore (no empty slots)."""
    entries: list[DocEntry] = []
    for loc, name in enumerate(store.names()):
        field = store.field(name)
        out = ",".join(field.outputs) if field.outputs else ""
        entries.append(
            DocEntry(
                name=name,
                type=field.type,
                definition="",
                module=field.module,
                role=field.role,
                outputs=out,
                loc=loc,
            )
        )
    return entries


def write_doc_header(
    stream, vehicle_type: str, title: str, *, when: datetime | None = None
) -> None:
    """Write CADAC Vehicle::document banner for one vehicle type."""
    when = when or datetime.now()
    date_s = f"{when.strftime('%b')} {when.day:2d} {when.year}"
    time_s = when.strftime("%H:%M:%S")
    stream.write(f"{_STAR}\n")
    label = f" {vehicle_type} "
    pad = max(0, (117 - len(label)) // 2)
    stream.write("*" * pad + label + "*" * (117 - pad - len(label)) + "\n")
    stream.write(f"{_STAR}\n")
    stream.write(f"\n*** {title}   {date_s} {time_s} ***\n\n")


def write_doc_table(stream, section_title: str, entries: list[DocEntry]) -> None:
    """Write one module-variable array table (LOC/NAME/DEF/MODULE/PURPOSE/OUTPUT)."""
    stream.write(f"\n\n                                       {section_title} \n\n")
    stream.write(f"{_SEP}\n")
    stream.write(
        "|LOC|        NAME       |                    DEFINITION                       "
        "|   MODULE   | PURPOSE |    OUTPUT    |\n"
    )
    stream.write(f"{_SEP}\n")
    for i, entry in enumerate(entries):
        # C++: error flag, width(4)<<loc, name (+ ' int ' for ints),
        # width(54)<<def, width(13)<<mod, width(10)<<role, out
        stream.write(" ")
        stream.write(f"{entry.loc:<4}")
        if entry.type == "int":
            stream.write(f"{entry.name:<15}")
            stream.write(f"{' int ':<5}")
        else:
            stream.write(f"{entry.name:<20}")
        stream.write(f"{entry.definition:<54}")
        stream.write(f"{entry.module:<13}")
        stream.write(f"{entry.role:<10}")
        stream.write(f"{entry.outputs}\n")
        if (i + 1) % 10 == 0:
            stream.write(f"{_SEP10}\n")


def write_vehicle_doc(
    stream,
    vehicle_type: str,
    title: str,
    store,
    *,
    when: datetime | None = None,
) -> list[DocEntry]:
    """Document one vehicle type to doc.asc. Returns DocEntry list for document_input."""
    entries = entries_from_store(store)
    write_doc_header(stream, vehicle_type, title, when=when)
    write_doc_table(stream, f"{vehicle_type} Module-Variable Array", entries)
    return entries


def open_doc_stream(out_dir: Path):
    """Open doc.asc for writing (C++ ofstream fdoc)."""
    path = Path(out_dir) / "doc.asc"
    stream = path.open("w", encoding="utf-8", newline="\n")
    return path, stream


def document_input(out_dir: Path, doc_entries: list[DocEntry]) -> None:
    """Annotate local input.asc with module-variable definitions (C++ document_input).

    Copies input.asc → input_copy.asc, then rewrites input.asc inserting
    `//` definition comments after each module-variable value. No-op when
    input.asc is absent (Python scenarios are JSONC).
    """
    out_dir = Path(out_dir)
    input_path = out_dir / "input.asc"
    if not input_path.is_file():
        return

    by_name = {e.name: e for e in doc_entries}
    text = input_path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    (out_dir / "input_copy.asc").write_text(text, encoding="utf-8")

    lines = text.splitlines()
    out_lines: list[str] = []
    past_vehicles = False
    in_object = False

    for raw in lines:
        stripped = raw.strip()
        if not past_vehicles:
            out_lines.append(raw)
            if "VEHICLES" in raw:
                past_vehicles = True
            continue

        if not stripped:
            out_lines.append(raw)
            continue

        tokens = stripped.split()
        first = tokens[0]

        if first.startswith("//") or (first and not first[0].isalnum()):
            indent = "\t\t\t" if in_object else ""
            out_lines.append(indent + stripped if in_object else raw)
            continue

        if first == "STOP":
            out_lines.append(raw)
            break

        if first == "END":
            out_lines.append("\tEND")
            in_object = False
            continue

        if first in ("IF", "ENDIF"):
            out_lines.append("\t\t\t" + stripped)
            continue

        if first in ("AERO_DECK", "PROP_DECK", "WEATHER_DECK"):
            out_lines.append("\t\t\t" + stripped)
            continue

        # Vehicle type header (PLANE6 name, MISSILE6 m1, …)
        if (
            first.isupper()
            and any(ch.isdigit() for ch in first)
            and first not in by_name
        ):
            out_lines.append("\t" + stripped)
            in_object = True
            continue

        if in_object and len(tokens) >= 2:
            name, numerical = tokens[0], tokens[1]
            entry = by_name.get(name)
            if entry is not None:
                type_prefix = "'int' " if entry.type == "int" else ""
                out_lines.append(
                    f"\t\t\t{name}  {numerical}    //{type_prefix}{entry.definition}"
                    f"  module {entry.module}"
                )
            else:
                out_lines.append(
                    f"\t\t\t{name}  {numerical}   //*** <<< Check spelling"
                )
            continue

        out_lines.append(raw)

    input_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
