from dataclasses import asdict, dataclass
import json
from pathlib import Path

KINDS = ("vehicle", "module", "mode", "kernel", "e2e", "harvest")
STATUSES = ("ported", "stubbed", "missing", "deferred", "diverged")


@dataclass(frozen=True)
class InventoryRow:
    program: str
    kind: str
    name: str
    cpp: str
    python: str | None
    status: str
    note: str

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind {self.kind!r}")
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r}")


def dump_inventory(path: Path, rows: list[InventoryRow]) -> None:
    path.write_text(
        json.dumps([asdict(r) for r in rows], indent=2) + "\n",
        encoding="utf-8",
    )


def load_inventory(path: Path) -> list[InventoryRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [InventoryRow(**item) for item in data]
