import json
from pathlib import Path

import pytest

from trial_matcher.split_criteria import (
    EMPTY_TEXT,
    FLATTENED_WRAPPER,
    INLINE_NUMBERED,
    NO_BULLETS,
    NO_HEADERS,
    TEXT_BEFORE_HEADER,
    split_criteria,
    summarize_notes,
)

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "eligibility_texts.json").read_text())


def real(name):
    return split_criteria(FIXTURES[name]["text"])


def kinds(result):
    return [c.kind for c in result.criteria]


def texts(result):
    return [c.text for c in result.criteria]


def counts(result):
    return {k: kinds(result).count(k) for k in ("inclusion", "exclusion", "unknown")}


# --- real eligibility texts from the API ---------------------------------


def test_real_plain_bullets():
    r = real("plain_bullets")
    assert counts(r) == {"inclusion": 6, "exclusion": 7, "unknown": 0}
    assert r.notes == frozenset()
    assert texts(r)[0] == "Diagnosis of liver metastases from various primary tumors"
    assert [c.position for c in r.criteria] == list(range(1, 14))  # 1-based, across sections


def test_real_numbered_list_under_key_header():
    r = real("numbered_key_header")  # "Key Inclusion Criteria:" + "1. ..." items
    assert counts(r) == {"inclusion": 6, "exclusion": 5, "unknown": 0}
    assert texts(r)[0].startswith("Eastern Cooperative Oncology Group (ECOG)")  # "1. " removed


def test_real_header_in_prose_form():
    r = real("prose_header")  # "Criteria for patient inclusion are:" / "The exclusion criteria are:"
    assert counts(r) == {"inclusion": 5, "exclusion": 9, "unknown": 0}


def test_real_group_label_header_without_colon():
    r = real("group_header")  # "Cohort A (Breast Cohort) Inclusion Criteria"
    assert counts(r) == {"inclusion": 10, "exclusion": 5, "unknown": 0}


def test_real_no_bullets_uses_blank_lines():
    r = real("blank_line_paragraphs")
    assert counts(r) == {"inclusion": 8, "exclusion": 11, "unknown": 0}
    assert NO_BULLETS in r.notes
    assert texts(r)[1] == "Aged 18-75 years;"


def test_real_run_on_text_stays_whole_but_headers_still_split_kinds():
    r = real("run_on_no_separators")  # no bullets or blank lines inside a section
    assert kinds(r) == ["inclusion", "exclusion"]
    assert texts(r)[0].startswith("Diagnosed with breast cancer Aged 18 years or older")
    assert texts(r)[1].startswith("Diagnosis of another type of cancer")  # text after "✔ Exclusion Criteria:"


def test_real_header_with_text_on_the_same_line():
    r = real("header_with_text_on_line")
    assert kinds(r) == ["inclusion", "exclusion"]
    assert texts(r)[0].startswith("The inclusion criteria include")


def test_real_inline_numbers_and_stray_dash():
    r = real("inline_numbers_and_junk")  # "1\. a; 2. b; 3. c ..." on one line, lone "\-" line
    assert INLINE_NUMBERED in r.notes
    assert counts(r) == {"inclusion": 5, "exclusion": 8, "unknown": 0}
    assert texts(r)[:2] == ["Patients with advanced cancer;", "Life expectancy >12 weeks;"]
    assert all(t not in ("-", "\\-") for t in texts(r))


def test_real_inline_numbers_when_first_marker_looks_like_a_bullet():
    r = real("inline_numbers_one_line")
    assert INLINE_NUMBERED in r.notes
    assert counts(r) == {"inclusion": 9, "exclusion": 21, "unknown": 0}
    assert texts(r)[0] == "Age 18-75 years, regardless of sex;"


def test_real_wrapper_bullet_is_flattened():
    r = real("wrapper_bullet")  # "* Subjects who meet all ...:" with the real list indented below
    assert FLATTENED_WRAPPER in r.notes
    assert counts(r) == {"inclusion": 19, "exclusion": 32, "unknown": 0}
    assert texts(r)[0].startswith("Subjects who meet all the following criteria")
    assert texts(r)[1].startswith("Subjects or their legal representative")


def test_real_no_inclusion_exclusion_header_falls_back_to_unknown():
    r = real("no_inclusion_header")  # only "Eligibility Criteria:" + bullets
    assert NO_HEADERS in r.notes
    assert counts(r) == {"inclusion": 0, "exclusion": 0, "unknown": 6}
    assert any("Meet one of the following risk criteria:\n- " in t for t in texts(r))  # nested kept


def test_real_nested_bullets_with_one_parent():
    r = real("nested_bullets")  # "* Group I" with indented children
    assert counts(r) == {"inclusion": 5, "exclusion": 2, "unknown": 0}


# --- headers (small synthetic inputs) ------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        "Inclusion Criteria:",
        "INCLUSION CRITERIA",
        "inclusion criteria:",
        "**Inclusion Criteria:**",
        "## Inclusion Criteria",
        "* Inclusion Criteria:",
        "Key Inclusion Criteria:",
        "Main inclusion criteria",
        "Inclusion Criteria (Phase 2):",
        "Inclusion Criteria for Cohort A:",
        "Inclusion Criteria：",
        "Inclusion:",
        "The inclusion criteria are:",
    ],
)
def test_header_variants_are_recognised(header):
    r = split_criteria(f"{header}\n\n* Age 18 or older\n\nExclusion Criteria:\n\n* Pregnant")
    assert kinds(r) == ["inclusion", "exclusion"]
    assert texts(r) == ["Age 18 or older", "Pregnant"]


def test_a_criterion_that_mentions_exclusion_is_not_a_header():
    r = split_criteria("Inclusion Criteria:\n\n* Exclusion of patients with prior therapy is not required\n* Age 18+")
    assert kinds(r) == ["inclusion", "inclusion"]


def test_a_sentence_mentioning_inclusion_criteria_is_not_a_header():
    r = split_criteria("Inclusion Criteria:\n\n* Patient must meet all inclusion criteria of the parent study\n* Age 18+")
    assert kinds(r) == ["inclusion", "inclusion"]


def test_text_before_first_header_is_unknown():
    r = split_criteria("Study population: adults.\n\nInclusion Criteria:\n\n* Age 18+")
    assert kinds(r) == ["unknown", "inclusion"]
    assert TEXT_BEFORE_HEADER in r.notes


def test_repeated_headers_for_cohorts():
    r = split_criteria("Inclusion Criteria:\n* A\nExclusion Criteria:\n* B\nInclusion Criteria for Part 2:\n* C")
    assert kinds(r) == ["inclusion", "exclusion", "inclusion"]
    assert [c.position for c in r.criteria] == [1, 2, 3]


def test_inclusion_only_text():
    r = split_criteria("Inclusion Criteria:\n\n* Age 18+\n* ECOG 0-1")
    assert kinds(r) == ["inclusion", "inclusion"]


# --- bullets, wrapping, nesting (small synthetic inputs) -----------------


@pytest.mark.parametrize(
    "line_a, line_b",
    [("- one", "- two"), ("* one", "* two"), ("• one", "• two"), ("1. one", "2. two"),
     ("1) one", "2) two"), ("(1) one", "(2) two"), ("a. one", "b. two")],
)
def test_bullet_and_numbering_styles(line_a, line_b):
    r = split_criteria(f"Inclusion Criteria:\n{line_a}\n{line_b}")
    assert texts(r) == ["one", "two"]


def test_wrapped_lines_are_glued_to_their_item():
    # No wrapped lines turned up in ~19K real texts, so this one is synthetic.
    text = "Inclusion Criteria:\n* Histologically confirmed\n  breast cancer, stage II\n  or III\n* Age 18+"
    assert texts(split_criteria(text)) == ["Histologically confirmed breast cancer, stage II or III", "Age 18+"]


def test_nested_sub_bullets_stay_with_their_parent():
    text = "Inclusion Criteria:\n* Histologically confirmed breast cancer\n  * HER2-negative\n  * ER-positive\n* Age 18+"
    r = split_criteria(text)
    assert texts(r) == ["Histologically confirmed breast cancer\n- HER2-negative\n- ER-positive", "Age 18+"]


def test_letters_under_numbers_are_nested():
    text = "Inclusion Criteria:\n1. Adequate organ function:\na. ANC >= 1.5\nb. Platelets >= 100\n2. Age 18+"
    assert texts(split_criteria(text)) == [
        "Adequate organ function:\n- ANC >= 1.5\n- Platelets >= 100",
        "Age 18+",
    ]


def test_wrapped_line_inside_a_nested_item():
    text = "Inclusion Criteria:\n* Parent\n  * child that\n    keeps going\n* Next"
    assert texts(split_criteria(text)) == ["Parent\n- child that keeps going", "Next"]


def test_sub_bullets_dedent_back_to_top_level():
    text = "Inclusion Criteria:\n* A\n  * a1\n* B\n* C"
    assert texts(split_criteria(text)) == ["A\n- a1", "B", "C"]


def test_numbers_inside_a_criterion_are_not_bullets():
    r = split_criteria("Inclusion Criteria:\n* ECOG 0-1\n* 18 years or older\n* Hemoglobin 9.5 g/dL")
    assert texts(r) == ["ECOG 0-1", "18 years or older", "Hemoglobin 9.5 g/dL"]


def test_markdown_escapes_are_removed():
    r = split_criteria("Inclusion Criteria:\n\n* INR \\< 1.5\\*\n* Age \\>= 18")
    assert texts(r) == ["INR < 1.5*", "Age >= 18"]


def test_junk_items_are_dropped():
    r = split_criteria("Inclusion Criteria:\n\n* Age 18+\n* -\n* <!-- -->\n* ECOG 0-1")
    assert texts(r) == ["Age 18+", "ECOG 0-1"]


def test_whitespace_is_normalised_and_tabs_are_handled():
    r = split_criteria("Inclusion Criteria:\r\n\r\n*   Age    18+\r\n\t* sub item\r\n")
    assert texts(r) == ["Age 18+\n- sub item"]


def test_inline_numbering_needs_three_items_in_order():
    r = split_criteria("Inclusion Criteria:\n\nAge 18+ (see 1) and 2) below)")
    assert INLINE_NUMBERED not in r.notes
    assert len(r.criteria) == 1


# --- fallbacks and bookkeeping ------------------------------------------


@pytest.mark.parametrize("text", [None, "", "   \n\n  "])
def test_empty_text_gives_no_criteria(text):
    r = split_criteria(text)
    assert r.criteria == [] and r.notes == frozenset({EMPTY_TEXT})


def test_no_headers_at_all_is_unknown_but_still_split():
    r = split_criteria("* Age 18+\n* ECOG 0-1")
    assert kinds(r) == ["unknown", "unknown"]
    assert NO_HEADERS in r.notes


def test_plain_sentence_is_a_single_unknown_item():
    r = split_criteria("Adults with cancer.")
    assert texts(r) == ["Adults with cancer."] and kinds(r) == ["unknown"]


def test_summarize_notes_counts_fallbacks():
    results = [split_criteria("* a\n* b"), split_criteria("Inclusion Criteria:\n* a"), split_criteria(None)]
    counts_ = summarize_notes(results)
    assert counts_["trials"] == 3
    assert counts_[NO_HEADERS] == 1
    assert counts_[EMPTY_TEXT] == 1
