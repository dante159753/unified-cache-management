import json
from http.client import HTTPException
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request

import a2_scheduler.github_client as github_client
import pytest
from a2_scheduler.github_client import GitHubClient, GitHubUnavailable

TOKEN = "test-token"
REPOSITORY = "ModelEngine-Group/unified-cache-management"
API_URL = "https://api.github.com"


@pytest.fixture(autouse=True)
def retry_sleeps(monkeypatch):
    delays = []
    monkeypatch.setattr(github_client.time, "sleep", delays.append)
    return delays


def runs_url(status):
    return "{}/repos/{}/actions/runs?status={}&per_page=100".format(
        API_URL, REPOSITORY, status
    )


def jobs_url(run_id):
    return "{}/repos/{}/actions/runs/{}/jobs?per_page=30".format(
        API_URL, REPOSITORY, run_id
    )


class FakeResponse:
    def __init__(self, payload, link=None):
        self._body = BytesIO(json.dumps(payload).encode("utf-8"))
        self.headers = {} if link is None else {"Link": link}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._body.close()

    def read(self):
        return self._body.read()


class FakeOpener:
    def __init__(self, runs, jobs):
        self._runs = runs
        self._jobs = jobs
        self.request_urls = []

    def open(self, request, timeout):
        url = request.full_url
        self.request_urls.append(url)
        if "/actions/runs?" in url:
            status = url.split("status=", 1)[1].split("&", 1)[0]
            return FakeResponse({"workflow_runs": self._runs[status]})
        run_id = int(url.split("/actions/runs/", 1)[1].split("/", 1)[0])
        return FakeResponse({"jobs": self._jobs[run_id]})


class ErrorOpener:
    def __init__(self, status):
        self._status = status

    def open(self, request, timeout):
        raise HTTPError(request.full_url, self._status, "unavailable", {}, None)


class HttpExceptionOpener:
    def open(self, request, timeout):
        raise HTTPException("invalid response")


class PageOpener:
    def __init__(self, pages):
        self._pages = pages
        self.request_urls = []

    def open(self, request, timeout):
        url = request.full_url
        self.request_urls.append(url)
        payload, link = self._pages[url]
        return FakeResponse(payload, link)


class RedirectTarget:
    url = "https://attacker.example/capture"

    def __init__(self):
        self.authorization_headers = []

    def open(self, request, timeout):
        self.authorization_headers.append(request.get_header("Authorization"))
        return FakeResponse({"workflow_runs": []})


class RedirectingOpener:
    def __init__(self, redirect_handler, target):
        self._redirect_handler = redirect_handler
        self._target = target

    def open(self, request, timeout):
        if self._redirect_handler is None:
            redirected_request = Request(
                self._target.url, headers=dict(request.header_items())
            )
            return self._target.open(redirected_request, timeout)
        self._redirect_handler.add_parent(self._target)
        return self._redirect_handler.http_error_302(
            request,
            None,
            302,
            "Found",
            {"Location": self._target.url},
        )


def test_finds_queued_a2_job_inside_in_progress_run():
    opener = FakeOpener(
        runs={"queued": [], "in_progress": [{"id": 42}]},
        jobs={
            42: [
                {
                    "id": 99,
                    "name": "test-e2e-pc-a2",
                    "status": "queued",
                    "labels": ["self-hosted", "npu", "a2"],
                }
            ]
        },
    )
    jobs = GitHubClient(REPOSITORY, TOKEN, opener).list_target_jobs()
    assert [(job.id, job.status) for job in jobs] == [(99, "queued")]


def test_api_error_is_unknown_not_empty():
    client = GitHubClient(REPOSITORY, TOKEN, ErrorOpener(503))
    with pytest.raises(GitHubUnavailable):
        client.list_target_jobs()


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_retries_transient_error_on_current_page(status, retry_sleeps):
    second_page = "{}&page=2".format(runs_url("queued"))
    opener = PageOpener(
        {
            runs_url("queued"): (
                {"workflow_runs": []},
                '<{}>; rel="next"'.format(second_page),
            ),
            second_page: ({"workflow_runs": []}, None),
            runs_url("in_progress"): ({"workflow_runs": []}, None),
        }
    )
    original_open = opener.open
    attempts = []

    def flaky_open(request, timeout):
        attempts.append(request.full_url)
        if request.full_url == second_page and attempts.count(second_page) < 3:
            raise HTTPError(request.full_url, status, "unavailable", {}, None)
        return original_open(request, timeout)

    opener.open = flaky_open
    assert GitHubClient(REPOSITORY, TOKEN, opener).list_target_jobs() == []
    assert attempts == [runs_url("queued")] + [second_page] * 3 + [
        runs_url("in_progress")
    ]
    assert retry_sleeps == [1, 2]


@pytest.mark.parametrize("status", [302, 401, 403, 404, 429, 502])
def test_http_failure_has_bounded_retries_and_safe_details(status, retry_sleeps):
    attempts = []

    class FailingOpener:
        def open(self, request, timeout):
            attempts.append(request.full_url)
            raise HTTPError(request.full_url, status, TOKEN, {}, None)

    expected_attempts = 3 if status == 502 else 1
    with pytest.raises(GitHubUnavailable) as captured:
        GitHubClient(REPOSITORY, TOKEN, FailingOpener()).list_target_jobs()
    assert len(attempts) == expected_attempts
    assert retry_sleeps == ([1, 2] if status == 502 else [])
    assert "HTTP {}".format(status) in str(captured.value)
    assert "/actions/runs" in str(captured.value)
    assert TOKEN not in str(captured.value)


@pytest.mark.parametrize(
    "error", [URLError(TOKEN), TimeoutError(TOKEN), HTTPException(TOKEN)]
)
def test_transport_failure_is_retried_and_remains_unknown(error, retry_sleeps):
    attempts = []

    class FailingOpener:
        def open(self, request, timeout):
            attempts.append(request.full_url)
            raise error

    with pytest.raises(GitHubUnavailable) as captured:
        GitHubClient(REPOSITORY, TOKEN, FailingOpener()).list_target_jobs()
    assert len(attempts) == 3
    assert retry_sleeps == [1, 2]
    assert TOKEN not in str(captured.value)


def test_http_exception_is_unknown_not_empty():
    client = GitHubClient(REPOSITORY, TOKEN, HttpExceptionOpener())
    with pytest.raises(GitHubUnavailable):
        client.list_target_jobs()


def test_default_opener_rejects_redirect_before_sending_authorization(monkeypatch):
    target = RedirectTarget()

    def fake_build_opener(*handlers):
        redirect_handler = handlers[0] if handlers else None
        return RedirectingOpener(redirect_handler, target)

    monkeypatch.setattr(github_client, "build_opener", fake_build_opener)

    with pytest.raises(GitHubUnavailable):
        GitHubClient(REPOSITORY, TOKEN).list_target_jobs()

    assert target.authorization_headers == []


def test_rejects_unsafe_pagination_url_without_sending_request():
    unsafe_url = "https://attacker.example/pagination"
    opener = PageOpener(
        {
            runs_url("queued"): (
                {"workflow_runs": []},
                '<{}>; rel="next"'.format(unsafe_url),
            ),
            unsafe_url: ({"workflow_runs": []}, None),
            runs_url("in_progress"): ({"workflow_runs": []}, None),
        }
    )

    with pytest.raises(GitHubUnavailable):
        GitHubClient(REPOSITORY, TOKEN, opener).list_target_jobs()

    assert unsafe_url not in opener.request_urls


def test_follows_safe_github_pagination():
    second_page = "{}&page=2".format(runs_url("queued"))
    second_jobs_page = "{}&page=2".format(jobs_url(41))
    opener = PageOpener(
        {
            runs_url("queued"): (
                {"workflow_runs": [{"id": 41}]},
                '<{}>; rel="next"'.format(second_page),
            ),
            second_page: ({"workflow_runs": [{"id": 42}]}, None),
            runs_url("in_progress"): ({"workflow_runs": []}, None),
            jobs_url(41): (
                {"jobs": []},
                '<{}>; rel="next"'.format(
                    second_jobs_page.replace(
                        "/repos/" + REPOSITORY, "/repositories/1017089821"
                    )
                ),
            ),
            second_jobs_page: (
                {
                    "jobs": [
                        {
                            "id": 91,
                            "status": "queued",
                            "labels": ["self-hosted", "npu", "a2"],
                        }
                    ]
                },
                None,
            ),
            jobs_url(42): (
                {
                    "jobs": [
                        {
                            "id": 92,
                            "status": "queued",
                            "labels": ["self-hosted", "npu", "a2"],
                        }
                    ]
                },
                None,
            ),
        }
    )

    jobs = GitHubClient(REPOSITORY, TOKEN, opener).list_target_jobs()

    assert [job.id for job in jobs] == [91, 92]
    assert second_jobs_page in opener.request_urls


@pytest.mark.parametrize(
    "next_url",
    [
        "https://attacker.example/repositories/1017089821/actions/runs?status=queued&page=2",
        "https://api.github.com/repositories/not-a-number/actions/runs?page=2",
        "https://api.github.com/repositories/1017089821/actions/runs/42/jobs?page=2",
        "https://user@api.github.com/repositories/1017089821/actions/runs?page=2",
        "https://api.github.com:8443/repositories/1017089821/actions/runs?page=2",
    ],
)
def test_rejects_pagination_links_outside_current_endpoint(next_url):
    opener = PageOpener(
        {
            runs_url("queued"): (
                {"workflow_runs": []},
                '<{}>; rel="next"'.format(next_url),
            )
        }
    )
    with pytest.raises(GitHubUnavailable):
        GitHubClient(REPOSITORY, TOKEN, opener).list_target_jobs()
    assert opener.request_urls == [runs_url("queued")]


def test_failed_later_jobs_page_does_not_return_partial_queue(retry_sleeps):
    second_page = "{}&page=2".format(jobs_url(42))
    opener = PageOpener(
        {
            runs_url("queued"): ({"workflow_runs": [{"id": 42}]}, None),
            runs_url("in_progress"): ({"workflow_runs": []}, None),
            jobs_url(42): (
                {
                    "jobs": [
                        {
                            "id": 99,
                            "status": "queued",
                            "labels": ["self-hosted", "npu", "a2"],
                        }
                    ]
                },
                '<{}>; rel="next"'.format(second_page),
            ),
        }
    )
    original_open = opener.open

    def failing_open(request, timeout):
        if request.full_url == second_page:
            raise HTTPError(request.full_url, 502, "unavailable", {}, None)
        return original_open(request, timeout)

    opener.open = failing_open
    with pytest.raises(GitHubUnavailable):
        GitHubClient(REPOSITORY, TOKEN, opener).list_target_jobs()
    assert retry_sleeps == [1, 2]


def test_queries_duplicate_run_id_once():
    opener = FakeOpener(
        runs={"queued": [{"id": 42}], "in_progress": [{"id": 42}]},
        jobs={
            42: [
                {
                    "id": 99,
                    "status": "queued",
                    "labels": ["self-hosted", "npu", "a2"],
                }
            ]
        },
    )

    jobs = GitHubClient(REPOSITORY, TOKEN, opener).list_target_jobs()

    assert [job.id for job in jobs] == [99]
    assert opener.request_urls.count(jobs_url(42)) == 1


def test_excludes_jobs_with_nonmatching_labels_or_status():
    opener = FakeOpener(
        runs={"queued": [{"id": 42}], "in_progress": []},
        jobs={
            42: [
                {
                    "id": 99,
                    "status": "queued",
                    "labels": ["self-hosted", "npu", "a2"],
                },
                {
                    "id": 100,
                    "status": "queued",
                    "labels": ["self-hosted", "npu"],
                },
                {
                    "id": 101,
                    "status": "completed",
                    "labels": ["self-hosted", "npu", "a2"],
                },
            ]
        },
    )

    jobs = GitHubClient(REPOSITORY, TOKEN, opener).list_target_jobs()

    assert [job.id for job in jobs] == [99]
