import json
from pathlib import Path

from cadac.io.asc_deck import parse_asc_deck
from cadac.tables.lookup import Table


def _table_to_dict(table: Table) -> dict:
    raw = {
        "name": table.name,
        "dim": table.dim,
        "x1": table.x1.tolist(),
        "values": table.values.tolist(),
    }
    if table.x2 is not None:
        raw["x2"] = table.x2.tolist()
    if table.x3 is not None:
        raw["x3"] = table.x3.tolist()
    return raw


def deck_asc_to_jsonc(src, dst) -> None:
    title, tables = parse_asc_deck(src)
    payload = {"title": title, "tables": [_table_to_dict(t) for t in tables]}
    Path(dst).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
