from pathlib import Path

from cadac_cpp.field_help import (
    build_glossary,
    descriptions_in,
    render_variable_help_ts,
    shared_descriptions,
)

ROOT = Path(__file__).resolve().parents[3]


def test_real_init_keeps_the_definition_sentence():
    text = 'cruise[35].init("alphax",0,"Angle of attack - deg","aerodynamics","data","");'
    assert descriptions_in(text)["alphax"] == "Angle of attack - deg"


def test_int_init_skips_the_type_token():
    text = (
        'cruise[10].init("mprop","int",0,'
        '"=0:none; =1:fixed-throttle; =2:auto-throttle","propulsion","data","");'
    )
    assert descriptions_in(text)["mprop"] == "=0:none; =1:fixed-throttle; =2:auto-throttle"


def test_vector_init_skips_the_three_components():
    text = 'round3[10].init("FSPV",0,0,0,"Specific force in V-coord - m/s^2","forces","out","plot");'
    assert descriptions_in(text)["FSPV"] == "Specific force in V-coord - m/s^2"


def test_matrix_init_skips_nine_components():
    text = (
        'round3[0].init("TBV",1,0,0, 0,1,0, 0,0,1,'
        '"TM of body wrt velocity coord","kinematics","out","");'
    )
    assert descriptions_in(text)["TBV"] == "TM of body wrt velocity coord"


def test_duplicate_name_keeps_the_most_common_sentence():
    text = """
a.init("mass",0,"Vehicle mass - kg","propulsion","out","");
a.init("mass",0,"Mass - kg","propulsion","out","");
a.init("mass",0,"Vehicle mass - kg","propulsion","out","");
"""
    assert descriptions_in(text)["mass"] == "Vehicle mass - kg"


def test_tied_sentences_keep_the_shorter_one():
    text = """
a.init("mass",0,"Vehicle mass - kg","propulsion","out","");
a.init("mass",0,"Mass - kg","propulsion","out","");
"""
    assert descriptions_in(text)["mass"] == "Mass - kg"


def test_commented_init_is_ignored():
    text = '// cruise[35].init("alphax",0,"Angle of attack - deg","aerodynamics","data","");'
    assert descriptions_in(text) == {}


def test_shared_sentence_is_the_most_common_across_programs():
    glossary = {
        "hyper3": {"mprop": "throttle modes"},
        "falcon5": {"mprop": "Mode switch - ND"},
        "falcon6": {"mprop": "Mode switch - ND"},
    }
    assert shared_descriptions(glossary)["mprop"] == "Mode switch - ND"


def test_render_exports_program_tables_and_the_shared_fallback():
    text = render_variable_help_ts({"hyper3": {"alphax": "Angle of attack - deg"}})
    assert "export const variableHelp" in text
    assert 'alphax: "Angle of attack - deg"' in text
    assert "export const sharedHelp" in text


def test_committed_variable_help_matches_cadac():
    glossary = build_glossary(ROOT)
    path = ROOT / "workbench/web/src/forms/variableHelp.ts"
    assert path.read_text(encoding="utf-8") == render_variable_help_ts(glossary)


def test_hyper3_alphax_and_falcon5_mprop_come_from_that_program():
    glossary = build_glossary(ROOT)
    assert glossary["hyper3"]["alphax"] == "Angle of attack - deg"
    assert glossary["hyper3"]["mprop"] == "=0:none; =1:fixed-throttle; =2:auto-throttle"
    assert glossary["falcon5"]["mprop"] == "Mode switch - ND"
    assert glossary["hyper3"]["alt"] == "Vehicle altitude - m"
