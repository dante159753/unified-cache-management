import json
from http.client import HTTPException
from io import BytesIO
from urllib.error import HTTPError

import pytest
from a2_scheduler.github_client import GitHubClient, GitHubUnavailable

TOKEN = "test-token"
REPOSITORY = "ModelEngine-Group/unified-cache-management"


class FakeResponse:
    def __init__(self, payload):
        self._body = BytesIO(json.dumps(payload).encode("utf-8"))

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

    def open(self, request, timeout):
        url = request.full_url
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
