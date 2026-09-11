from cadac.aero_map.payload import AeroPayload

_SRAAM_AGM_CN = "cn0_vs_mach_alpha"
_SRAAM_AGM_CM = "clm0_vs_mach_alpha"
_SRAAM_AGM_CA = "ca0_vs_mach"
_AIM5_CL = "cl_aim_vs_alpha_mach"
_AIM5_CD_ON = "cd_aim_on_vs_alpha_mach"
_AIM5_CD_OFF = "cd_aim_off_vs_alpha_mach"
_SAM_CN = "cn0_vs_mach,betax,alphax"
_SAM_CM = "clm0_vs_mach,betax,alphax"
_SAM_CA = "ca0_vs_mach,betax,alphax"
_CRUISE_CD0 = "cd0_vs_mach"
_CRUISE_CL0 = "cl0_vs_mach"
_CRUISE_CLA = "cla_vs_mach"
_PLANE_MACS = ("30MAC", "35MAC", "40MAC")


def table_1d(name: str, x1: list, values: list) -> dict:
    return {"name": name, "dim": 1, "x1": list(x1), "values": list(values)}


def table_2d(name: str, x1: list, x2: list, values: list) -> dict:
    return {
        "name": name,
        "dim": 2,
        "x1": list(x1),
        "x2": list(x2),
        "values": [list(row) for row in values],
    }


def table_3d(name: str, x1: list, x2: list, x3: list, values: list) -> dict:
    return {
        "name": name,
        "dim": 3,
        "x1": list(x1),
        "x2": list(x2),
        "x3": list(x3),
        "values": [[[cell for cell in row] for row in plane] for plane in values],
    }


def transpose(grid: list) -> list:
    if not grid:
        return []
    return [list(col) for col in zip(*grid)]


def ca0_values(alphas: list, ca_grid: list) -> list:
    if not alphas:
        return []
    if 0.0 in alphas:
        j = alphas.index(0.0)
        return [row[j] for row in ca_grid]
    n = len(alphas)
    return [sum(row) / n for row in ca_grid]


def missile6_from_payload(payload: AeroPayload) -> dict[str, dict]:
    machs = payload.axes["mach"]
    alphas = payload.axes["alpha"]
    src = payload.tables
    tables: dict[str, dict] = {}
    if "cn" in src:
        tables[_SRAAM_AGM_CN] = table_2d(_SRAAM_AGM_CN, machs, alphas, src["cn"])
    if "cm" in src:
        tables[_SRAAM_AGM_CM] = table_2d(_SRAAM_AGM_CM, machs, alphas, src["cm"])
    if "ca" in src and alphas:
        tables[_SRAAM_AGM_CA] = table_1d(
            _SRAAM_AGM_CA, machs, ca0_values(alphas, src["ca"])
        )
    return tables


def aim5_from_payload(payload: AeroPayload) -> dict[str, dict]:
    machs = payload.axes["mach"]
    alphas = payload.axes["alpha"]
    src = payload.tables
    tables: dict[str, dict] = {}
    cl_grid = src["cl"] if "cl" in src else src.get("cn")
    if cl_grid is not None:
        tables[_AIM5_CL] = table_2d(_AIM5_CL, alphas, machs, transpose(cl_grid))
    cd_grid = src["cd"] if "cd" in src else src.get("ca")
    if cd_grid is not None:
        cd_t = transpose(cd_grid)
        tables[_AIM5_CD_ON] = table_2d(_AIM5_CD_ON, alphas, machs, cd_t)
        tables[_AIM5_CD_OFF] = table_2d(_AIM5_CD_OFF, alphas, machs, cd_t)
    return tables


def plane_from_payload(payload: AeroPayload) -> dict[str, dict]:
    machs = payload.axes["mach"]
    alphas = payload.axes["alpha"]
    src = payload.tables
    tables: dict[str, dict] = {}
    if "cl" in src:
        for mac in _PLANE_MACS:
            name = f"cl_{mac}_vs_mach_alphax"
            tables[name] = table_2d(name, machs, alphas, src["cl"])
    if "cd" in src:
        for mac in _PLANE_MACS:
            name = f"cd_{mac}_vs_mach_alphax"
            tables[name] = table_2d(name, machs, alphas, src["cd"])
    return tables


def _nearest_alpha_index(alphas: list) -> int:
    return min(range(len(alphas)), key=lambda i: (abs(alphas[i]), i))


def _two_nearest_alpha_indices(alphas: list) -> tuple[int, int] | None:
    if len(alphas) < 2:
        return None
    i, j = sorted(range(len(alphas)), key=lambda k: (abs(alphas[k]), k))[:2]
    if abs(alphas[j] - alphas[i]) > 0:
        return i, j
    return None


def cruise5_from_payload(payload: AeroPayload) -> dict[str, dict]:
    machs = payload.axes["mach"]
    alphas = payload.axes["alpha"]
    src = payload.tables
    tables: dict[str, dict] = {}
    if not alphas:
        return tables
    j0 = _nearest_alpha_index(alphas)
    if "cd" in src:
        tables[_CRUISE_CD0] = table_1d(
            _CRUISE_CD0, machs, [row[j0] for row in src["cd"]]
        )
    if "cl" in src:
        tables[_CRUISE_CL0] = table_1d(
            _CRUISE_CL0, machs, [row[j0] for row in src["cl"]]
        )
        pair = _two_nearest_alpha_indices(alphas)
        if pair is not None:
            i, j = pair
            da = alphas[j] - alphas[i]
            tables[_CRUISE_CLA] = table_1d(
                _CRUISE_CLA,
                machs,
                [(row[j] - row[i]) / da for row in src["cl"]],
            )
    return tables


def _beta_axis(payload: AeroPayload) -> list:
    betas = list(payload.axes.get("beta") or [])
    if 0.0 in betas:
        return [0.0]
    if betas:
        return [betas[0]]
    return [0.0]


def _wrap_beta_slice(grid: list) -> list:
    return [[list(row)] for row in grid]


def sam6_from_payload(payload: AeroPayload) -> dict[str, dict]:
    machs = payload.axes["mach"]
    alphas = payload.axes["alpha"]
    betas = _beta_axis(payload)
    src = payload.tables
    tables: dict[str, dict] = {}
    if "cn" in src:
        tables[_SAM_CN] = table_3d(
            _SAM_CN, machs, betas, alphas, _wrap_beta_slice(src["cn"])
        )
    if "cm" in src:
        tables[_SAM_CM] = table_3d(
            _SAM_CM, machs, betas, alphas, _wrap_beta_slice(src["cm"])
        )
    if "ca" in src:
        tables[_SAM_CA] = table_3d(
            _SAM_CA, machs, betas, alphas, _wrap_beta_slice(src["ca"])
        )
    return tables


def rocket6_from_payload(payload: AeroPayload, slv: int = 1) -> dict[str, dict]:
    machs = payload.axes["mach"]
    alphas = payload.axes["alpha"]
    src = payload.tables
    cn_name = f"cn0slv{slv}_vs_mach_alpha"
    cm_name = f"clm0slv{slv}_vs_mach_alpha"
    ca_name = f"ca0slv{slv}_vs_mach"
    tables: dict[str, dict] = {}
    if "cn" in src:
        tables[cn_name] = table_2d(cn_name, machs, alphas, src["cn"])
    if "cm" in src:
        tables[cm_name] = table_2d(cm_name, machs, alphas, src["cm"])
    if "ca" in src and alphas:
        tables[ca_name] = table_1d(
            ca_name, machs, ca0_values(alphas, src["ca"])
        )
    return tables
