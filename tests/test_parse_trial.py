import json
from datetime import date
from pathlib import Path

import pytest

from trial_matcher.parse_trial import parse_age_years, parse_trial

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def trials():
    """Five real trials, chosen for their age formats. Keyed by NCT id."""
    studies = json.loads((FIXTURES / "studies_parse_cases.json").read_text())["studies"]
    return {t.nct_id: t for t in map(parse_trial, studies)}


# --- age parsing ---------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("18 Years", 18.0),
        ("1 Year", 1.0),
        ("9 Months", 0.75),
        ("1 Month", 1 / 12),
        ("30 Days", 30 / 365.25),
        ("1 Day", 1 / 365.25),
        ("6 Weeks", 42 / 365.25),
        ("12 Hours", 12 / (365.25 * 24)),
        ("  75 years ", 75.0),
    ],
)
def test_age_units_convert_to_years(text, expected):
    assert parse_age_years(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", [None, "", "N/A", "n/a", "  N/A  "])
def test_age_no_limit_is_none(text):
    assert parse_age_years(text) is None


@pytest.mark.parametrize("text", ["adult", "18", "18 Decades", "Years 18"])
def test_age_garbage_raises(text):
    with pytest.raises(ValueError):
        parse_age_years(text)


# --- whole-trial parsing on real records ----------------------------------


def test_years_with_no_max_age(trials):
    t = trials["NCT06647862"]
    assert t.min_age_years == 18.0
    assert t.max_age_years is None  # field absent in the API = "N/A"
    assert t.phase == ["PHASE3"]
    assert t.conditions == ["Chronic Myelomonocytic Leukemia"]
    assert t.interventions[:2] == ["IMM01", "Azacitidine"]
    assert t.status == "RECRUITING"
    assert t.study_type == "INTERVENTIONAL"
    assert t.sex == "ALL"
    assert t.healthy_volunteers is False
    assert t.eligibility_text.startswith("Inclusion Criteria:")
    assert t.last_update_posted == date(2025, 11, 25)


def test_min_and_max_age_in_years(trials):
    t = trials["NCT07118566"]
    assert (t.min_age_years, t.max_age_years) == (18.0, 75.0)


def test_months_min_age(trials):
    t = trials["NCT07358260"]
    assert t.min_age_years == pytest.approx(0.75)
    assert t.max_age_years == 30.0


def test_single_month_min_age(trials):
    assert trials["NCT07825935"].min_age_years == pytest.approx(1 / 12)


def test_no_age_limits_at_all(trials):
    t = trials["NCT07850154"]
    assert t.min_age_years is None and t.max_age_years is None


def test_raw_json_is_kept(trials):
    raw = trials["NCT06647862"].raw_json
    assert raw["protocolSection"]["identificationModule"]["nctId"] == "NCT06647862"


def test_missing_id_or_title_raises():
    with pytest.raises(ValueError):
        parse_trial({"protocolSection": {"identificationModule": {"nctId": "NCT1"}}})
    with pytest.raises(ValueError):
        parse_trial({})


def test_sparse_study_gets_defaults():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT1", "briefTitle": "T"},
        }
    }
    t = parse_trial(study)
    assert t.conditions == [] and t.interventions == [] and t.phase == []
    assert t.eligibility_text is None and t.last_update_posted is None


def test_month_only_update_date():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT1", "briefTitle": "T"},
            "statusModule": {"lastUpdatePostDateStruct": {"date": "2025-03"}},
        }
    }
    assert parse_trial(study).last_update_posted == date(2025, 3, 1)
