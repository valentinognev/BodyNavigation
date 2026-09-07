from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarvestRow:
    jsonc: str
    cpp_dir: str
    asc_name: str
    golden: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


HARVEST_ROWS = [
    HarvestRow(
        "Python/cases/hyper3/input_climb.jsonc",
        "CADAC_Simulations/HYPER3_250114/HYPER3",
        "input_climb.asc",
        "Python/tests/e2e/goldens/hyper3/plot1.csv",
    ),
    HarvestRow(
        "Python/cases/falcon5/input_turning_to_IP.jsonc",
        "CADAC_Simulations/FALCON5_250116/FALCON5",
        "input_turning_to_IP.asc",
        "Python/tests/e2e/goldens/falcon5/plot.csv",
    ),
    HarvestRow(
        "Python/cases/falcon6/input_gamma.jsonc",
        "CADAC_Simulations/FALCON6_250201/FALCON6",
        "input_gamma.asc",
        "Python/tests/e2e/goldens/falcon6/plot.csv",
    ),
    HarvestRow(
        "Python/cases/hyper5/input.jsonc",
        "CADAC_Simulations/HYPER5_250113/HYPER5",
        "input_Demo_4_7_pro_nav.asc",
        "Python/tests/e2e/goldens/hyper5/plot.csv",
    ),
    HarvestRow(
        "Python/cases/hyper6/input_climb.jsonc",
        "CADAC_Simulations/HYPER6_250125/HYPER6",
        "input_climb.asc",
        "Python/tests/e2e/goldens/hyper6/plot.csv",
    ),
    HarvestRow(
        "Python/cases/aim5/input_hori.jsonc",
        "CADAC_Simulations/AIM5_250114/AIM5",
        "input_hori.asc",
        "Python/tests/e2e/goldens/aim5/plot.csv",
    ),
    HarvestRow(
        "Python/cases/cruise5/input_1.jsonc",
        "CADAC_Simulations/CRUISE5_250115/CRUISE5",
        "input_1.asc",
        "Python/tests/e2e/goldens/cruise5/plot.csv",
    ),
    HarvestRow(
        "Python/cases/magsix/input.jsonc",
        "CADAC_Simulations/MAGSIX_231111/MAGSIX",
        "input_attitudeMR1.asc",
        "Python/tests/e2e/goldens/magsix/plot.csv",
    ),
    HarvestRow(
        "Python/cases/magsix/input_trajectoryMR1.jsonc",
        "CADAC_Simulations/MAGSIX_231111/MAGSIX",
        "input_trajectoryMR1.asc",
        "Python/tests/e2e/goldens/magsix/trajectory/plot.csv",
    ),
    HarvestRow(
        "Python/cases/rocket6/input.jsonc",
        "CADAC_Simulations/ROCKET6_250122/ROCKET6",
        "input_insertion.asc",
        "Python/tests/e2e/goldens/rocket6/plot.csv",
    ),
    HarvestRow(
        "Python/cases/sam6/input_SAM_autopilot.jsonc",
        "CADAC_Simulations/SAM6_250217/SAM6",
        "input_SAM_autopilot.asc",
        "Python/tests/e2e/goldens/sam6/plot.csv",
    ),
    HarvestRow(
        "Python/cases/sraam6/input_1v1.jsonc",
        "CADAC_Simulations/SRAAM6_250130/SRAAM6",
        "input_1v1.asc",
        "Python/tests/e2e/goldens/sraam6/plot.csv",
    ),
    HarvestRow(
        "Python/cases/agm6/input_freeflight.jsonc",
        "CADAC_Simulations/AGM6_250217/AGM6",
        "input_3_1 AGM6 Free Flight.asc",
        "Python/tests/e2e/goldens/agm6/plot.csv",
    ),
    HarvestRow(
        "Python/cases/agm6/input_testcase.jsonc",
        "CADAC_Simulations/AGM6_250217/AGM6",
        "input_2_1 AGM6 Test Case.asc",
        "Python/tests/e2e/goldens/agm6/test_case_plot.csv",
    ),
]
