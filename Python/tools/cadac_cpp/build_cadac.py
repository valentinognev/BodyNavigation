import os
import subprocess
from pathlib import Path

from cadac_cpp.extract_cpp import PROGRAM_DIRS
from cadac_cpp.harvest_table import repo_root


def build_program(program: str) -> Path:
    root = repo_root()
    tools = Path(__file__).resolve().parent
    build = tools / "build" / program
    build.mkdir(parents=True, exist_ok=True)
    binary = build / program.lower()
    cadac_dir = Path(os.path.relpath(root / PROGRAM_DIRS[program], tools))
    build_rel = Path(os.path.relpath(build, tools))
    binary_rel = Path(os.path.relpath(binary, tools))
    subprocess.run(
        [
            "make", "-f", "Makefile.cadac",
            "COMPAT=compat.hpp",
            f"CADAC_DIR={cadac_dir}",
            f"PROGRAM={program.lower()}",
            f"BUILD={build_rel}",
            str(binary_rel),
        ],
        cwd=tools,
        check=True,
    )
    return binary
