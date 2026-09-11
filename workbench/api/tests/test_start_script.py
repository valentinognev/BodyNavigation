from pathlib import Path


def _start_sh() -> Path:
    return Path(__file__).resolve().parents[2] / "start.sh"


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
