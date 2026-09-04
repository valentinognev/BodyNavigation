from cadac.io.jsonc import load, loads


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
