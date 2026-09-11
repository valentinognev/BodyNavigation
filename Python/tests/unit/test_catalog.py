import json
from pathlib import Path

from cadac.io import catalog
from cadac.io.catalog import (
    EXTRA_SOURCES,
    SKIP_STEMS,
    classify_asc,
    dest_jsonc,
    family_for,
    iter_asc_jobs,
    repo_root,
    run_catalog,
    translate_one,
    write_report,
)
from cadac.io.scenario import load_scenario

ROOT = Path(__file__).resolve().parents[2].parent  # BodyNavigation
HYPER3 = ROOT / "CADAC_Simulations/HYPER3_250114/HYPER3"
SEC104 = ROOT / "CADAC_Simulations/HYPER6 Input Problems for Sec 10_4"


def test_skip_stems():
    assert "readme" in SKIP_STEMS
    assert classify_asc(HYPER3 / "readme.asc") == "skip"
    assert classify_asc(HYPER3 / "input_copy.asc") == "skip"


def test_scenario_climb():
    assert classify_asc(HYPER3 / "input_climb.asc") == "scenario"


def test_deck_aero():
    assert classify_asc(HYPER3 / "ghame3_aero_deck.asc") == "deck"


def test_sec104_is_scenario():
    assert classify_asc(SEC104 / "10_1_1_input_aero.asc") == "scenario"


def test_family_map():
    assert family_for("AGM6") == "agm6"
    assert family_for("HYPER3") is None
    assert family_for("ROCKET6") == "rocket6"


def test_dest_hyper6_sec104():
    root = repo_root()
    src = root / "CADAC_Simulations/HYPER6 Input Problems for Sec 10_4/10_1_1_input_aero.asc"
    dst = dest_jsonc("HYPER6", src)
    assert dst == root / "Python/cases/hyper6/10_1_1_input_aero.jsonc"


def test_translate_one_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "cases_dir", lambda program: tmp_path)
    src = HYPER3 / "input_climb.asc"
    dest = dest_jsonc("HYPER3", src)
    dest.write_text("keep-me")
    original = dest.read_text()
    status, _err = translate_one("HYPER3", src)
    assert status == "exists"
    assert dest.read_text() == original


def test_iter_includes_sec104_and_agm_extra():
    jobs = iter_asc_jobs()
    stems = {(p, s.name) for p, s in jobs}
    assert EXTRA_SOURCES == [
        ("HYPER6", "CADAC_Simulations/HYPER6 Input Problems for Sec 10_4"),
        ("AGM6", "CADAC_Simulations/AGM6_250217/Additional input Files"),
    ]
    assert ("HYPER6", "10_1_1_input_aero.asc") in stems
    assert ("AGM6", "input_3_2 AGM6 Free Flight.asc") in stems or any(
        p == "AGM6" and "Free Flight" in s.name for p, s in jobs
    )


def test_iter_excludes_readme():
    jobs = iter_asc_jobs()
    assert not any(s.stem.lower() in SKIP_STEMS for _p, s in jobs)
    assert not any(s.stem.lower() == "readme" for _p, s in jobs)


def test_translate_one_captures_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "cases_dir", lambda program: tmp_path)

    def boom(*_a, **_k):
        raise ValueError("boom")

    monkeypatch.setattr(catalog, "translate_scenario_asc", boom)
    src = HYPER3 / "input_climb.asc"
    status, err = translate_one("HYPER3", src)
    assert status == "failed"
    assert err == "ValueError: boom"
    assert not dest_jsonc("HYPER3", src).exists()


def test_run_catalog_writes_missing_sec104(tmp_path, monkeypatch):
    # If 10_1_1 jsonc missing in real cases dir, run_catalog translated list contains it.
    climb = repo_root() / "Python/cases/hyper3/input_climb.jsonc"
    original = climb.read_text()
    report = run_catalog()
    assert "failed" in report and "translated" in report and "skipped" in report
    dest = repo_root() / "Python/cases/hyper6/10_1_1_input_aero.jsonc"
    assert dest.is_file()
    rel = "Python/cases/hyper6/10_1_1_input_aero.jsonc"
    assert rel in report["translated"] or rel in report["skipped"]
    # existing e2e case still present
    assert climb.is_file()
    assert climb.read_text() == original


def test_sec104_load_scenario():
    load_scenario(repo_root() / "Python/cases/hyper6/10_1_1_input_aero.jsonc")


def test_write_report(tmp_path):
    dest = tmp_path / "catalog-report.json"
    payload = {"translated": ["Python/cases/x.jsonc"], "skipped": [], "failed": []}
    out = write_report(payload, dest)
    assert out == dest
    assert json.loads(dest.read_text()) == payload
