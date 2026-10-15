"""Turn one raw ClinicalTrials.gov study (API v2 JSON) into a flat record for the trials table."""

import re
from dataclasses import dataclass, field
from datetime import date

# Years per unit. The API uses Years, Months, Weeks, Days and Hours.
_YEARS_PER_UNIT = {
    "year": 1.0,
    "month": 1 / 12,
    "week": 7 / 365.25,
    "day": 1 / 365.25,
    "hour": 1 / (365.25 * 24),
}
_AGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+?)s?\s*$")


@dataclass(frozen=True)
class ParsedTrial:
    nct_id: str
    title: str
    status: str
    study_type: str
    phase: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    sex: str | None = None
    min_age_years: float | None = None
    max_age_years: float | None = None
    healthy_volunteers: bool | None = None
    eligibility_text: str | None = None
    last_update_posted: date | None = None
    raw_json: dict = field(default_factory=dict)


def parse_age_years(value: str | None) -> float | None:
    """'18 Years' -> 18.0, '9 Months' -> 0.75, '30 Days' -> ~0.082.

    None, '' and 'N/A' mean "no limit" and give None. Anything else raises ValueError.
    """
    if value is None or value.strip().upper() in ("", "N/A"):
        return None
    match = _AGE_RE.match(value)
    if not match or match.group(2).lower() not in _YEARS_PER_UNIT:
        raise ValueError(f"unrecognised age: {value!r}")
    return float(match.group(1)) * _YEARS_PER_UNIT[match.group(2).lower()]


def parse_trial(study: dict) -> ParsedTrial:
    """Raises ValueError if the study lacks an NCT id or a title."""
    protocol = study.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    eligibility = protocol.get("eligibilityModule", {})

    nct_id = ident.get("nctId")
    title = ident.get("briefTitle")
    if not nct_id or not title:
        raise ValueError(f"study missing nctId or briefTitle: {nct_id!r}")

    interventions = protocol.get("armsInterventionsModule", {}).get("interventions", [])
    posted = status.get("lastUpdatePostDateStruct", {}).get("date")

    return ParsedTrial(
        nct_id=nct_id,
        title=title,
        status=status.get("overallStatus", ""),
        study_type=design.get("studyType", ""),
        phase=design.get("phases", []),
        conditions=protocol.get("conditionsModule", {}).get("conditions", []),
        interventions=[i["name"] for i in interventions if i.get("name")],
        sex=eligibility.get("sex"),
        min_age_years=parse_age_years(eligibility.get("minimumAge")),
        max_age_years=parse_age_years(eligibility.get("maximumAge")),
        healthy_volunteers=eligibility.get("healthyVolunteers"),
        eligibility_text=eligibility.get("eligibilityCriteria"),
        last_update_posted=_parse_date(posted),
        raw_json=study,
    )


def _parse_date(value: str | None) -> date | None:
    """The API gives 'YYYY-MM-DD', or just 'YYYY-MM' for some date fields."""
    if not value:
        return None
    if len(value) == 7:
        value += "-01"
    return date.fromisoformat(value)
