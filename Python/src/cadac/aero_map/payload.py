from dataclasses import dataclass
import math

_MDT_COEFFS = ("cn", "cm", "ca", "cl", "cd")
_AID_COEFFS = (("CL", "cl"), ("CD", "cd"), ("Cm", "cm"))
_REF_KEYS = ("sref", "lref", "xcg")


@dataclass
class AeroPayload:
    source: str  # "misdc" | "aid" | "file"
    solver: str  # "mdt" | "datcom" | "tornado" | "avl" | "flow5" | "cadac_deck"
    axes: dict  # {"mach": list[float], "alpha": list[float], "beta": list[float]}
    tables: dict[str, list]  # "cn"|"cm"|"ca"|"cl"|"cd" → 2D [n_mach][n_alpha]
    ref: dict  # sref, lref, xcg floats optional


def _finite(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _ref_from_mapping(data: dict) -> dict:
    ref = {}
    for key in _REF_KEYS:
        if key not in data:
            continue
        number = _finite(data[key])
        if number is not None:
            ref[key] = number
    return ref


def _ref_from_rows(rows: list[dict]) -> dict:
    ref = {}
    for key in _REF_KEYS:
        for row in rows:
            if key not in row:
                continue
            number = _finite(row[key])
            if number is not None:
                ref[key] = number
                break
    return ref


def _dense_tables(tables: dict[str, list]) -> dict[str, list]:
    dense = {}
    for name, grid in tables.items():
        if any(cell is None for row in grid for cell in row):
            continue
        if grid:
            dense[name] = grid
    return dense


def _drop_empty_machs(
    machs: list[float], tables: dict[str, list]
) -> tuple[list[float], dict[str, list]]:
    keep = [
        i
        for i, _mach in enumerate(machs)
        if any(cell is not None for grid in tables.values() for cell in grid[i])
    ]
    if len(keep) == len(machs):
        return machs, tables
    return (
        [machs[i] for i in keep],
        {name: [grid[i] for i in keep] for name, grid in tables.items()},
    )


def _drop_empty_alphas(
    alphas: list[float], tables: dict[str, list]
) -> tuple[list[float], dict[str, list]]:
    keep = [
        j
        for j, _alpha in enumerate(alphas)
        if any(row[j] is not None for grid in tables.values() for row in grid)
    ]
    if len(keep) == len(alphas):
        return alphas, tables
    return (
        [alphas[j] for j in keep],
        {name: [[row[j] for j in keep] for row in grid] for name, grid in tables.items()},
    )


def payload_from_mdt_rows(rows: list[dict]) -> AeroPayload:
    """MISDC parse_for006 rows with keys alpha, mach, cn, cm, ca (optional cl, cd, beta)."""
    by_ma: dict[tuple[float, float], dict] = {}
    betas: set[float] = set()
    for row in rows:
        mach = _finite(row.get("mach"))
        alpha = _finite(row.get("alpha"))
        if mach is None or alpha is None:
            continue
        by_ma[(mach, alpha)] = row
        beta = _finite(row["beta"]) if "beta" in row else None
        if beta is not None:
            betas.add(beta)

    machs = sorted({m for m, _a in by_ma})
    alphas = sorted({a for _m, a in by_ma})

    tables: dict[str, list] = {}
    for name in _MDT_COEFFS:
        grid = []
        any_finite = False
        for mach in machs:
            row = []
            for alpha in alphas:
                cell = by_ma.get((mach, alpha), {})
                value = _finite(cell[name]) if name in cell else None
                if value is not None:
                    any_finite = True
                row.append(value)
            grid.append(row)
        if any_finite:
            tables[name] = grid

    machs, tables = _drop_empty_machs(machs, tables)
    alphas, tables = _drop_empty_alphas(alphas, tables)
    return AeroPayload(
        source="misdc",
        solver="mdt",
        axes={"mach": machs, "alpha": alphas, "beta": sorted(betas)},
        tables=_dense_tables(tables),
        ref=_ref_from_rows(rows),
    )


def _aid_machs(coeff: dict) -> list[float]:
    if "MACH" in coeff:
        raw = coeff["MACH"]
    elif "mach" in coeff:
        raw = coeff["mach"]
    else:
        return [0.3]
    if isinstance(raw, (list, tuple)):
        machs = [_finite(m) for m in raw]
        return [m for m in machs if m is not None]
    number = _finite(raw)
    return [number] if number is not None else []


def payload_from_aid(coeff: dict) -> AeroPayload:
    alphas_raw = coeff.get("alpha") or []
    by_alpha: dict[float, dict[str, float | None]] = {}
    for i, raw_alpha in enumerate(alphas_raw):
        alpha = _finite(raw_alpha)
        if alpha is None:
            continue
        values: dict[str, float | None] = {}
        for src, dest in _AID_COEFFS:
            series = coeff.get(src) or []
            values[dest] = _finite(series[i]) if i < len(series) else None
        by_alpha[alpha] = values

    alphas = sorted(by_alpha)
    tables: dict[str, list] = {}
    for _src, dest in _AID_COEFFS:
        row = [by_alpha[alpha][dest] for alpha in alphas]
        if any(cell is not None for cell in row):
            tables[dest] = [row]

    machs = _aid_machs(coeff)[:1]
    if not machs:
        tables = {}
    else:
        machs, tables = _drop_empty_machs(machs, tables)
    alphas, tables = _drop_empty_alphas(alphas, tables)

    solver = coeff["solver"] if "solver" in coeff else "datcom"
    return AeroPayload(
        source="aid",
        solver=solver,
        axes={"mach": machs, "alpha": alphas, "beta": []},
        tables=_dense_tables(tables),
        ref=_ref_from_mapping(coeff),
    )
