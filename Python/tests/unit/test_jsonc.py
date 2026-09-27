from pathlib import Path

from cadac.io.jsonc import leading_comment, leading_comment_prefix, load, loads

_CASES = Path(__file__).resolve().parents[2] / "cases"


def _scenario_files():
    for path in sorted(_CASES.rglob("*.jsonc")):
        try:
            data = loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        title = data.get("title")
        if not isinstance(title, str) or not title:
            continue
        if not (data.get("vehicles") or data.get("modules")):
            continue
        yield path


def test_catalog_scenarios_have_overview_comment():
    missing = []
    for path in _scenario_files():
        comment = leading_comment(path.read_text(encoding="utf-8"))
        if comment is None or len(comment) < 40:
            missing.append(str(path.relative_to(_CASES)))
    assert missing == []


def test_leading_comment_block_before_root():
    text = "/* AIM5, 5-DOF flat.\n   Two vehicles. */\n{\n  \"title\": \"t\"\n}\n"
    assert leading_comment(text) == "AIM5, 5-DOF flat.\nTwo vehicles."
    assert leading_comment_prefix(text) == "/* AIM5, 5-DOF flat.\n   Two vehicles. */\n"


def test_leading_comment_absent_when_file_starts_with_value():
    assert leading_comment('{ "a": 1 }\n') is None
    assert leading_comment_prefix('{ "a": 1 }\n') == ""


def test_leading_comment_ignores_comment_inside_value():
    text = '{ "a": 1, /* skip */ "b": 2 }\n'
    assert leading_comment(text) is None


def test_leading_line_comments_before_root():
    text = "// line one\n// line two\n{ \"a\": 1 }\n"
    assert leading_comment(text) == "line one\nline two"
    assert leading_comment_prefix(text) == "// line one\n// line two\n"


def test_loads_line_comment_after_value():
    text = '{ "lonx": -80.55, // Vehicle longitude - deg\n  "alt": 3000 }'
    assert loads(text) == {"lonx": -80.55, "alt": 3000}


def test_loads_block_comment():
    assert loads('{ "a": 1, /* skip */ "b": 2 }') == {"a": 1, "b": 2}


def test_load_reads_path(tmp_path):
    p = tmp_path / "v.jsonc"
    p.write_text(
        '{ "lonx": -80.55, // Vehicle longitude - deg\n  "alt": 3000 }\n',
        encoding="utf-8",
        newline="\n",
    )
    assert load(p) == {"lonx": -80.55, "alt": 3000}
