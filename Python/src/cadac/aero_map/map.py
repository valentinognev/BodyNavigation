from copy import deepcopy
from dataclasses import dataclass

from cadac.aero_map.payload import AeroPayload
from cadac.aero_map.schema import required_tables
from cadac.aero_map.tables import (
    aim5_from_payload,
    cruise5_from_payload,
    missile6_from_payload,
    plane_from_payload,
    rocket6_from_payload,
    sam6_from_payload,
)

_DEFAULT_TITLE = "aero deck"

_BUILDERS = {
    ("sraam6", "MISSILE6"): missile6_from_payload,
    ("agm6", "MISSILE6"): missile6_from_payload,
    ("aim5", "AIM5"): aim5_from_payload,
    (None, "PLANE"): plane_from_payload,
    ("cruise5", "CRUISE3"): cruise5_from_payload,
    ("sam6", "MISSILE6"): sam6_from_payload,
}

_MAPPED_NAMES = {
    ("sraam6", "MISSILE6"): frozenset(
        {"cn0_vs_mach_alpha", "clm0_vs_mach_alpha", "ca0_vs_mach"}
    ),
    ("agm6", "MISSILE6"): frozenset(
        {"cn0_vs_mach_alpha", "clm0_vs_mach_alpha", "ca0_vs_mach"}
    ),
    ("aim5", "AIM5"): frozenset(
        {
            "cl_aim_vs_alpha_mach",
            "cd_aim_on_vs_alpha_mach",
            "cd_aim_off_vs_alpha_mach",
        }
    ),
    (None, "PLANE"): frozenset(
        {
            "cl_30MAC_vs_mach_alphax",
            "cd_30MAC_vs_mach_alphax",
            "cl_35MAC_vs_mach_alphax",
            "cd_35MAC_vs_mach_alphax",
            "cl_40MAC_vs_mach_alphax",
            "cd_40MAC_vs_mach_alphax",
        }
    ),
    ("cruise5", "CRUISE3"): frozenset(
        {"cd0_vs_mach", "cl0_vs_mach", "cla_vs_mach"}
    ),
    ("sam6", "MISSILE6"): frozenset(
        {
            "cn0_vs_mach,betax,alphax",
            "clm0_vs_mach,betax,alphax",
            "ca0_vs_mach,betax,alphax",
        }
    ),
}

_ROCKET6 = ("rocket6", "HYPER6")


def _rocket_names(slv: int) -> tuple[str, ...]:
    return tuple(
        name.replace("slv1", f"slv{slv}")
        for name in required_tables("rocket6", "HYPER6")
    )


def _rocket_mapped_names(slv: int) -> frozenset[str]:
    return frozenset(
        {
            f"ca0slv{slv}_vs_mach",
            f"cn0slv{slv}_vs_mach_alpha",
            f"clm0slv{slv}_vs_mach_alpha",
        }
    )


@dataclass
class PreviewRow:
    name: str
    status: str  # "mapped" | "merged" | "missing"


@dataclass
class MapResult:
    rows: list[PreviewRow]
    deck: dict
    can_confirm: bool


def _noop(template: dict | None) -> MapResult:
    if template is None:
        deck: dict = {"title": _DEFAULT_TITLE, "tables": []}
    else:
        deck = deepcopy(template)
    return MapResult(rows=[], deck=deck, can_confirm=True)


def _template_tables(template: dict | None) -> dict[str, dict]:
    if not template:
        return {}
    return {t["name"]: t for t in template.get("tables") or []}


def _title(template: dict | None) -> str:
    if template is not None and "title" in template:
        return template["title"]
    return _DEFAULT_TITLE


def map_payload(
    family: str | None,
    vtype: str,
    payload: AeroPayload,
    template: dict | None,
    slv: int = 1,
) -> MapResult:
    if (family, vtype) == _ROCKET6:
        names = _rocket_names(slv)
        mapped = rocket6_from_payload(payload, slv)
        mapped_names = _rocket_mapped_names(slv)
    else:
        names = required_tables(family, vtype)
        builder = _BUILDERS.get((family, vtype))
        mapped = builder(payload) if builder is not None else {}
        mapped_names = _MAPPED_NAMES.get((family, vtype), frozenset())
    if not names:
        return _noop(template)
    by_name = _template_tables(template)
    rows: list[PreviewRow] = []
    tables: list[dict] = []
    for name in names:
        if name in mapped_names:
            built = mapped.get(name)
            if built is not None:
                rows.append(PreviewRow(name=name, status="mapped"))
                tables.append(built)
            else:
                rows.append(PreviewRow(name=name, status="missing"))
            continue
        source = by_name.get(name)
        if source is not None:
            rows.append(PreviewRow(name=name, status="merged"))
            tables.append(deepcopy(source))
        else:
            rows.append(PreviewRow(name=name, status="missing"))

    present = {t["name"] for t in tables if "name" in t}
    present.update(row.name for row in rows)
    for source in (template.get("tables") or []) if template else []:
        name = source.get("name")
        if name and name not in present:
            tables.append(deepcopy(source))
            present.add(name)

    can_confirm = all(row.status != "missing" for row in rows)
    return MapResult(
        rows=rows,
        deck={"title": _title(template), "tables": tables},
        can_confirm=can_confirm,
    )
