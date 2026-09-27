from pathlib import Path

from cadac_cpp.field_choices import (
    FAMILIES,
    build_choices,
    comment_mode_labels,
    enumerated_options,
    merge_param_choices,
    render_field_choices_ts,
)

ROOT = Path(__file__).resolve().parents[3]


def test_enumerated_sentence_keeps_each_integer_and_its_label():
    options = enumerated_options("=0:none; =1:fixed-throttle; =2:auto-throttle")
    assert options == [(0, "none"), (1, "fixed-throttle"), (2, "auto-throttle")]


def test_one_marker_is_not_a_closed_set():
    assert enumerated_options("Update epoch, =0: forcing first update") is None


def test_increment_for_more_stays_open():
    assert enumerated_options("=0:Unfreeze; =1:Freeze; increment for more") is None


def test_mode_codes_without_labels_use_the_integer():
    assert merge_param_choices([0, 3, 30], None) == [(0, "0"), (3, "3"), (30, "30")]


def test_comment_table_names_each_mode():
    text = """
//	   mguide = 30 line-guidance lateral, with maut 35
//		      = 03 line-guidance in pitch
//			  =  4 arc-guidance lateral
//mguidance:
//			= 30 line-guidance lateral, with mcontrol 46
//			*11: Heading and flight path angle control
// * Waypoint guidance: mguid=30
//mguide=5
//     maut= 0 No control
//        >= 1 Roll controller
// mrpop = 3 Constant thrust rocket under LTG control
"""
    labels = comment_mode_labels(text)
    assert labels["mguide"][30] == "line-guidance lateral, with maut 35"
    assert labels["mguide"][3] == "line-guidance in pitch"
    assert labels["mguide"][4] == "arc-guidance lateral"
    assert 5 not in labels["mguide"]
    assert labels["mguidance"][30] == "line-guidance lateral, with mcontrol 46"
    assert labels["mguidance"][11] == "Heading and flight path angle control"
    assert labels["mguid"][30] == "Waypoint guidance"
    assert labels["maut"][0] == "No control"
    assert labels["maut"][1] == "Roll controller"
    assert labels["mprop"][3] == "Constant thrust rocket under LTG control"


def test_branch_comment_names_a_mode_the_table_skipped():
    text = """
//returning,if no guidance
if(mguide==0){
    return;
}
//lateral line guidance
if(mguide==30){
}
"""
    labels = comment_mode_labels(text)
    assert labels["mguide"][0] == "returning,if no guidance"
    assert labels["mguide"][30] == "lateral line guidance"


def test_title_above_a_caveat_is_the_mode_name():
    text = """
//midcourse pro-nav guidance against target coordinates from datalink (set 'mnav=3')
if(guid_mid==3)
    return;
//midcourse pro-nav guidance against true target (no datalink)
//However, missile coordinates are corrupted by INS errors
if(guid_mid==4){
}
"""
    labels = comment_mode_labels(text)["guid_mid"]
    assert labels[3].startswith("midcourse pro-nav guidance against target")
    assert labels[4].startswith("midcourse pro-nav guidance against true target")


def test_comment_after_the_comparison_names_the_mode():
    text = """
if(mroll==0||mroll==1)
    //roll feedback for right side up
else if(mroll==2)
    //roll feedback for inverted flight
"""
    labels = comment_mode_labels(text)["mroll"]
    assert labels[0] == "roll feedback for right side up"
    assert labels[1] == "roll feedback for right side up"
    assert labels[2] == "roll feedback for inverted flight"


def test_table_text_wins_over_a_later_branch_comment():
    text = """
//	   mguide = 30 line-guidance lateral
//lateral line guidance
if(mguide==30){
}
"""
    assert comment_mode_labels(text)["mguide"][30] == "line-guidance lateral"


def test_comment_fills_only_codes_the_sentence_left_blank():
    sentence = enumerated_options("=0:none; =1:fixed-throttle")
    assert merge_param_choices([0, 1, 30], sentence, {0: "no thrusting", 30: "line guidance"}) == [
        (0, "none"),
        (1, "fixed-throttle"),
        (30, "line guidance"),
    ]


def test_a_single_mode_code_is_not_a_dropdown():
    assert merge_param_choices([0], None) is None


def test_sentence_labels_cover_codes_the_parser_also_found():
    sentence = enumerated_options("=0:none; =1:fixed-throttle; =2:auto-throttle")
    assert merge_param_choices([0, 1, 2], sentence) == sentence


def test_families_are_the_registered_set():
    assert FAMILIES == (
        "aim5",
        "cruise5",
        "magsix",
        "rocket6",
        "sam6",
        "sraam6",
        "agm6",
    )


def test_hyper6_mguide_uses_the_cadac_table():
    labels = dict(build_choices(ROOT)["params"]["hyper6"]["mguide"])
    assert labels[30] == "line-guidance lateral, with maut 35"
    assert labels[3] == "line-guidance in pitch"
    assert "glideslope" in labels[8]
    assert labels[0] == "returning,if no guidance"


def test_hyper3_types_modules_and_mprop_come_from_cadac():
    choices = build_choices(ROOT)
    assert choices["types"]["hyper3"] == ["CRUISE3"]
    assert "newton" in choices["modules"]["hyper3"]
    assert choices["params"]["hyper3"]["mprop"] == [
        (0, "none"),
        (1, "fixed-throttle"),
        (2, "auto-throttle"),
    ]
    assert "mfreeze" not in choices["params"]["hyper3"]


def test_render_exports_families_types_modules_and_params():
    text = render_field_choices_ts(
        {
            "types": {"hyper3": ["CRUISE3"]},
            "modules": {"hyper3": ["newton"]},
            "params": {"hyper3": {"mprop": [(0, "none"), (1, "fixed-throttle")]}},
        }
    )
    assert "export const FAMILIES" in text
    assert '"aim5"' in text
    assert "export const vehicleTypes" in text
    assert "CRUISE3" in text
    assert "export const moduleNames" in text
    assert "newton" in text
    assert "export const paramChoices" in text
    assert "fixed-throttle" in text
    assert 'export const PHASES = ["def", "init", "exec", "term"]' in text


def test_committed_field_choices_match_cadac():
    path = ROOT / "workbench/web/src/forms/fieldChoices.ts"
    assert path.read_text(encoding="utf-8") == render_field_choices_ts(build_choices(ROOT))
