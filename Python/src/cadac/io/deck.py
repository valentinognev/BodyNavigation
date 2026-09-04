from pathlib import Path

from cadac.io.jsonc import loads
from cadac.tables.lookup import Table


def _table_from_mapping(raw: dict) -> Table:
    return Table(
        name=raw["name"],
        dim=int(raw["dim"]),
        x1=raw["x1"],
        x2=raw.get("x2"),
        x3=raw.get("x3"),
        values=raw["values"],
    )


def load_deck(path) -> list[Table]:
    text = Path(path).read_text(encoding="utf-8")
    data = loads(text)
    return [_table_from_mapping(t) for t in data["tables"]]
