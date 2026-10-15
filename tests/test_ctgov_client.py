import json
from pathlib import Path

import httpx
import pytest

from trial_matcher.ctgov_client import CTGovClient, CTGovError

FIXTURES = Path(__file__).parent / "fixtures"
PAGE_1 = json.loads((FIXTURES / "studies_page_1.json").read_text())  # real, has nextPageToken
PAGE_2 = json.loads((FIXTURES / "studies_page_2.json").read_text())
LAST_PAGE = {k: v for k, v in PAGE_2.items() if k != "nextPageToken"}


def make_client(handler, **kwargs):
    """Client wired to a fake server; records sleeps instead of waiting."""
    sleeps = []
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = CTGovClient(http=http, sleep=sleeps.append, **kwargs)
    return client, sleeps


def paging_handler(requests):
    def handler(request):
        requests.append(request)
        token = request.url.params.get("pageToken")
        if token is None:
            return httpx.Response(200, json=PAGE_1)
        assert token == PAGE_1["nextPageToken"]
        return httpx.Response(200, json=LAST_PAGE)

    return handler


def test_follows_page_tokens_until_the_end():
    requests = []
    client, _ = make_client(paging_handler(requests), delay=0.5)
    studies = list(client.iter_studies())
    assert len(studies) == len(PAGE_1["studies"]) + len(LAST_PAGE["studies"])
    assert len(requests) == 2


def test_polite_delay_between_pages_only():
    client, sleeps = make_client(paging_handler([]), delay=0.5)
    list(client.iter_studies())
    assert sleeps == [0.5]  # once between two pages, not after the last


def test_max_trials_stops_early_without_extra_requests():
    requests = []
    client, _ = make_client(paging_handler(requests))
    studies = list(client.iter_studies(max_trials=1))
    assert len(studies) == 1
    assert len(requests) == 1


def test_request_carries_query_and_fields():
    requests = []
    client, _ = make_client(paging_handler(requests), page_size=250)
    next(client.iter_studies())
    params = requests[0].url.params
    assert params["query.cond"] == "cancer"
    assert params["filter.overallStatus"] == "RECRUITING|NOT_YET_RECRUITING"
    assert params["aggFilters"] == "studyType:int"
    assert params["pageSize"] == "250"
    assert "EligibilityCriteria" in params["fields"]


def test_count():
    def handler(request):
        assert request.url.params["countTotal"] == "true"
        return httpx.Response(200, json={"totalCount": 19456, "studies": []})

    client, _ = make_client(handler)
    assert client.count() == 19456


def test_retries_then_succeeds_with_backoff():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503)
        return httpx.Response(200, json=LAST_PAGE)

    client, sleeps = make_client(handler, backoff=1.0)
    assert len(list(client.iter_studies())) == len(LAST_PAGE["studies"])
    assert sleeps == [1.0, 2.0]  # exponential


def test_retries_on_network_error():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json=LAST_PAGE)

    client, _ = make_client(handler)
    assert list(client.iter_studies())


def test_429_honours_retry_after():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "30"})
        return httpx.Response(200, json=LAST_PAGE)

    client, sleeps = make_client(handler, backoff=1.0)
    list(client.iter_studies())
    assert sleeps == [30.0]


def test_gives_up_after_max_retries():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500)

    client, _ = make_client(handler, max_retries=2)
    with pytest.raises(CTGovError):
        list(client.iter_studies())
    assert len(calls) == 3  # first try + 2 retries


def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, text="bad query")

    client, sleeps = make_client(handler)
    with pytest.raises(CTGovError, match="400"):
        list(client.iter_studies())
    assert len(calls) == 1 and sleeps == []


def test_iter_pages_yields_one_list_per_page():
    client, _ = make_client(paging_handler([]))
    pages = list(client.iter_pages())
    assert [len(p) for p in pages] == [len(PAGE_1["studies"]), len(LAST_PAGE["studies"])]


def test_iter_pages_trims_the_last_page_to_max_trials():
    client, _ = make_client(paging_handler([]))
    pages = list(client.iter_pages(max_trials=len(PAGE_1["studies"]) + 1))
    assert sum(len(p) for p in pages) == len(PAGE_1["studies"]) + 1


def test_max_trials_zero_makes_no_requests():
    requests = []
    client, _ = make_client(paging_handler(requests))
    assert list(client.iter_pages(max_trials=0)) == []
    assert requests == []
