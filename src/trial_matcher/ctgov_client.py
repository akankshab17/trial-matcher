"""Client for the ClinicalTrials.gov API v2: paging, retries, polite delay."""

import time
from collections.abc import Callable, Iterator

import httpx

STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"

# Oncology, interventional, still open to patients.
DEFAULT_QUERY = {
    "query.cond": "cancer",
    "filter.overallStatus": "RECRUITING|NOT_YET_RECRUITING",
    "aggFilters": "studyType:int",
}

# Ask only for what we store; drops locations, results, etc. (~4 KB/trial vs ~100 KB).
DEFAULT_FIELDS = [
    "NCTId",
    "BriefTitle",
    "OverallStatus",
    "Phase",
    "StudyType",
    "Condition",
    "InterventionName",
    "Sex",
    "MinimumAge",
    "MaximumAge",
    "HealthyVolunteers",
    "EligibilityCriteria",
    "LastUpdatePostDate",
]

RETRY_STATUSES = {429, 500, 502, 503, 504}


class CTGovError(Exception):
    """The API kept failing after all retries."""


class CTGovClient:
    def __init__(
        self,
        http: httpx.Client | None = None,
        query: dict[str, str] | None = None,
        page_size: int = 1000,
        delay: float = 0.5,
        max_retries: int = 5,
        backoff: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._http = http or httpx.Client(timeout=60)
        self.query = dict(DEFAULT_QUERY if query is None else query)
        self.page_size = page_size
        self.delay = delay
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep

    def _get(self, params: dict[str, str | int]) -> dict:
        """GET with retries on 429/5xx/network errors (exponential backoff)."""
        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            wait = self.backoff * 2**attempt
            try:
                resp = self._http.get(STUDIES_URL, params=params)
            except httpx.TransportError as exc:
                last_error = f"network error: {exc}"
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code not in RETRY_STATUSES:
                    raise CTGovError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                last_error = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    wait = max(wait, float(retry_after))
            if attempt < self.max_retries:
                self._sleep(wait)
        raise CTGovError(f"gave up after {self.max_retries + 1} attempts: {last_error}")

    def count(self) -> int:
        """How many trials match the query (one tiny request)."""
        params = {**self.query, "countTotal": "true", "pageSize": 1, "fields": "NCTId"}
        return self._get(params)["totalCount"]

    def fetch_page(self, page_token: str | None = None) -> dict:
        params: dict[str, str | int] = {
            **self.query,
            "pageSize": self.page_size,
            "fields": ",".join(DEFAULT_FIELDS),
        }
        if page_token:
            params["pageToken"] = page_token
        return self._get(params)

    def iter_pages(self, max_trials: int | None = None) -> Iterator[list[dict]]:
        """Yield one list of raw study dicts per API page, stopping at max_trials or the last page."""
        if max_trials is not None and max_trials <= 0:
            return
        yielded = 0
        token = None
        while True:
            data = self.fetch_page(token)
            page, token = data.get("studies", []), data.get("nextPageToken")
            if max_trials is not None:
                page = page[: max_trials - yielded]
            yielded += len(page)
            if page:
                yield page
            if not token or (max_trials is not None and yielded >= max_trials):
                return
            self._sleep(self.delay)

    def iter_studies(self, max_trials: int | None = None) -> Iterator[dict]:
        """Yield raw study dicts one by one (same paging as iter_pages)."""
        for page in self.iter_pages(max_trials):
            yield from page
