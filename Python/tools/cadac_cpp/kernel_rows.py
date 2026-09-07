from cadac_cpp.schema import InventoryRow

_KERNEL = (
    ("events", "ported", "cadac.kernel.events"),
    ("look_up", "ported", "cadac.tables.lookup"),
    ("integrate", "ported", "CADAC stored-slope"),
    ("combus", "ported", "cadac.kernel.executive"),
    ("atmosphere76", "ported", ""),
    ("iso62", "ported", "HYPER3"),
    ("GAUSS", "ported", "stores mean, no sample"),
    ("RAYL", "ported", "stores mean"),
    ("MARKOV", "ported", "stores 0"),
    ("markov_noise", "deferred", "live Monte Carlo"),
    ("nmonte", "deferred", ""),
    ("plot", "ported", "slot 0 only"),
    ("csv", "ported", ""),
    ("merge", "deferred", "CADAC Studio"),
    ("scrn", "deferred", ""),
    ("tabout", "deferred", ""),
    ("doc", "deferred", ""),
    ("traj", "deferred", ""),
    ("comscrn", "deferred", ""),
)


def kernel_rows() -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for name, status, note in _KERNEL:
        python = note if note.startswith("cadac.") else None
        rows.append(
            InventoryRow(
                program="kernel",
                kind="kernel",
                name=name,
                cpp="",
                python=python,
                status=status,
                note=note,
            )
        )
    return rows
