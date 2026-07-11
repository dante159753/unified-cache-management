import json
from http.client import HTTPException
from io import BytesIO
from urllib.error import HTTPError

import pytest
from a2_scheduler.github_client import GitHubClient, GitHubUnavailable

TOKEN = "test-token"
REPOSITORY = "ModelEngine-Group/unified-cache-management"
API_URL = "https://api.github.com"


def runs_url(status):
    return "{}/repos/{}/actions/runs?status={}&per_page=100".format(
        API_URL, REPOSITORY, status
    )


def jobs_url(run_id):
    return "{}/repos/{}/actions/runs/{}/jobs?per_page=100".format(
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


def test_http_exception_is_unknown_not_empty():
    client = GitHubClient(REPOSITORY, TOKEN, HttpExceptionOpener())
    with pytest.raises(GitHubUnavailable):
        client.list_target_jobs()


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
    opener = PageOpener(
        {
            runs_url("queued"): (
                {"workflow_runs": [{"id": 41}]},
                '<{}>; rel="next"'.format(second_page),
            ),
            second_page: ({"workflow_runs": [{"id": 42}]}, None),
            runs_url("in_progress"): ({"workflow_runs": []}, None),
            jobs_url(41): (
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
