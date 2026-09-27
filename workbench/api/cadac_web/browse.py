from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from cadac_web.paths import CASES_ROOT, CasePathError, resolve_case

_MISSING = object()


def browse_start(program: str | None, stem: str | None) -> Path:
    if program and stem:
        try:
            path = resolve_case(program, stem)
        except CasePathError:
            return CASES_ROOT()
        if path.is_file():
            return path.parent
    return CASES_ROOT()


def ask_open_path(start: Path | None) -> str | None:
    directory = start if start is not None else Path.home()
    for opener in (_zenity, _kdialog, _tkinter):
        result = opener(directory)
        if result is not _MISSING:
            return result
    return None


def _zenity(directory: Path):
    if shutil.which("zenity") is None:
        return _MISSING
    try:
        proc = subprocess.run(
            ["zenity", "--file-selection", f"--filename={directory}/"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return _MISSING
    if proc.returncode == 0:
        path = proc.stdout.strip()
        return path or None
    if proc.returncode == 1:
        return None
    return _MISSING


def _kdialog(directory: Path):
    if shutil.which("kdialog") is None:
        return _MISSING
    try:
        proc = subprocess.run(
            ["kdialog", "--getopenfilename", str(directory)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return _MISSING
    if proc.returncode == 0:
        path = proc.stdout.strip()
        return path or None
    if proc.returncode == 1:
        return None
    return _MISSING


def _tkinter(directory: Path):
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return _MISSING
    root = tk.Tk()
    root.withdraw()
    try:
        chosen = filedialog.askopenfilename(initialdir=str(directory))
    except Exception:
        return _MISSING
    finally:
        root.destroy()
    if not chosen:
        return None
    return chosen
