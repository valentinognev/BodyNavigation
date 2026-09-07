import re
import subprocess
from pathlib import Path

from cadac_cpp.extract_cpp import PROGRAM_DIRS
from cadac_cpp.harvest_table import HARVEST_ROWS, repo_root

_MONTE_INT = re.compile(r"\b(MONTE|nmonte|NMONTE)\s+\d+")
_OPTIONS = re.compile(r"\bOPTIONS\b")
_Y_MONTE = re.compile(r"\by_monte\b", re.IGNORECASE)
_N_CSV = re.compile(r"\bn_csv\b", re.IGNORECASE)
_Y_CSV = re.compile(r"\by_csv\b", re.IGNORECASE)

_TIMEOUTS_S = {
    "HYPER3": 120,
    "CRUISE5": 900,
    "ROCKET6": 600,
}
_DEFAULT_TIMEOUT_S = 300
_CADAC_IO_NAMES = (
    "input.asc",
    "input_copy.asc",
    "doc.asc",
    "plot.csv",
    "plot1.csv",
    "plot2.csv",
    "traj.csv",
)


def snapshot_cadac_io(cwd: Path) -> dict[str, bytes]:
    snap = {}
    for name in _CADAC_IO_NAMES:
        path = cwd / name
        if path.is_file():
            snap[name] = path.read_bytes()
    return snap


def restore_cadac_io(cwd: Path, snap: dict[str, bytes]) -> None:
    for name, data in snap.items():
        (cwd / name).write_bytes(data)


def find_plot_csv(cwd: Path) -> Path:
    plot1 = cwd / "plot1.csv"
    if plot1.is_file():
        return plot1
    plot = cwd / "plot.csv"
    if plot.is_file():
        return plot
    raise FileNotFoundError(f"no plot1.csv or plot.csv in {cwd}")


def force_monte_off(text: str) -> str:
    out_lines = []
    for line in text.splitlines(keepends=True):
        new = _MONTE_INT.sub(lambda match: f"{match.group(1)} 0", line)
        if _OPTIONS.search(new):
            new = _Y_MONTE.sub("n_monte", new)
        out_lines.append(new)
    return "".join(out_lines)


def force_csv_on(text: str) -> str:
    out_lines = []
    for line in text.splitlines(keepends=True):
        if not _OPTIONS.search(line):
            out_lines.append(line)
            continue
        new = _N_CSV.sub("y_csv", line)
        if not _Y_CSV.search(new):
            nl = "\r\n" if new.endswith("\r\n") else "\n" if new.endswith("\n") else ""
            body = new[:-len(nl)] if nl else new
            new = body.rstrip() + " y_csv" + nl
        out_lines.append(new)
    return "".join(out_lines)


def csv_from_plot_asc(asc_path: Path) -> Path:
    lines = asc_path.read_text(encoding="utf-8", errors="replace").splitlines()
    dest = asc_path.with_suffix(".csv")
    out = []
    vars_n = 0
    j = 0
    l = 0
    buf = []
    for line in lines:
        if j == 0:
            out.append(line)
        elif j == 1:
            nums = [int(tok) for tok in line.split() if tok]
            vars_n = nums[2]
            out.append(line)
        else:
            for tok in line.split():
                buf.append(tok + ",")
            if vars_n and ((l - 1) * 5) // vars_n >= 1:
                out.append("".join(buf))
                buf = []
                l = 1
        j += 1
        l += 1
    if buf:
        out.append("".join(buf))
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    return dest


def _program_key(row) -> str:
    return next(k for k, v in PROGRAM_DIRS.items() if v == row.cpp_dir)


def _timeout_s(program: str) -> int:
    return _TIMEOUTS_S.get(program, _DEFAULT_TIMEOUT_S)


def harvest_row(row, timeout_s=600):
    root = repo_root()
    cwd = root / row.cpp_dir
    path = cwd / "input.asc"
    snap = snapshot_cadac_io(cwd)
    source = (cwd / row.asc_name).read_text(encoding="utf-8", errors="replace")
    try:
        path.write_text(force_csv_on(force_monte_off(source)), encoding="utf-8")
        from cadac_cpp.build_cadac import build_program
        key = _program_key(row)
        binary = build_program(key)
        for name in ("plot.csv", "plot1.csv", "plot.asc", "plot1.asc"):
            (cwd / name).unlink(missing_ok=True)
        proc = subprocess.run([str(binary)], cwd=cwd, check=False, timeout=timeout_s)
        try:
            plot = find_plot_csv(cwd)
        except FileNotFoundError:
            asc = cwd / "plot1.asc" if (cwd / "plot1.asc").is_file() else cwd / "plot.asc"
            if asc.is_file():
                plot = csv_from_plot_asc(asc)
            elif proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, [str(binary)])
            else:
                raise
        if row.golden.endswith("hyper3/plot1.csv"):
            dest = root / "Python/tests/e2e/goldens/hyper3/plot1.gpp.csv"
        else:
            dest = root / row.golden
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(plot.read_bytes())
        return dest
    finally:
        restore_cadac_io(cwd, snap)
        if "input.asc" not in snap:
            path.unlink(missing_ok=True)


def harvest_all(skip_failed=True):
    from cadac_cpp.build_cadac import build_program

    results = {}
    for row in HARVEST_ROWS:
        program = _program_key(row)
        print(f"harvest_all: {program} {row.asc_name}", flush=True)
        try:
            build_program(program)
        except subprocess.CalledProcessError:
            results[row.golden] = "build_failed"
            print(f"harvest_all: {program} -> build_failed", flush=True)
            if not skip_failed:
                raise
            continue
        try:
            harvest_row(row, timeout_s=_timeout_s(program))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            results[row.golden] = "run_failed"
            print(f"harvest_all: {program} -> run_failed", flush=True)
            if not skip_failed:
                raise
            continue
        results[row.golden] = "ok"
        print(f"harvest_all: {program} -> ok", flush=True)
    return results
