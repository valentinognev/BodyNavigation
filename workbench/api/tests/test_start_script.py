from pathlib import Path


def _start_sh() -> Path:
    return Path(__file__).resolve().parents[2] / "start.sh"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_start_sh_ports():
    text = _start_sh().read_text(encoding="utf-8")
    assert "8001" in text
    assert "5174" in text


def test_start_sh_kills_before_starting():
    text = _start_sh().read_text(encoding="utf-8")
    body = "\n".join(
        ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    )
    kill_at = body.index("kill.sh")
    assert kill_at < body.index("start_api")
    assert "already running" not in body


def test_root_start_sh_delegates_to_workbench():
    text = (_repo_root() / "start.sh").read_text(encoding="utf-8")
    assert "workbench/start.sh" in text


def test_root_kill_sh_delegates_to_workbench():
    text = (_repo_root() / "kill.sh").read_text(encoding="utf-8")
    assert "workbench/kill.sh" in text


def test_root_scripts_are_unix_lf():
    for name in ("start.sh", "kill.sh"):
        raw = (_repo_root() / name).read_bytes()
        assert b"\r" not in raw, f"{name} has CRLF line endings"


def test_start_sh_bootstraps_api_venv():
    text = _start_sh().read_text(encoding="utf-8")
    body = "\n".join(
        ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    )
    assert "-m venv" in body
    assert "pip install -e ." in body
