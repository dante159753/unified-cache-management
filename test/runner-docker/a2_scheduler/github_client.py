import json
import socket
import time
from http.client import HTTPException
from typing import FrozenSet, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .models import Job

_API_URL = "https://api.github.com"
_ACTIONS_PATH = "/repos/ModelEngine-Group/unified-cache-management/actions/"
_DEFAULT_REQUIRED_LABELS = frozenset(("self-hosted", "npu", "a2"))
_TIMEOUT_SECONDS = 30
_MAX_REQUEST_ATTEMPTS = 3
_RETRYABLE_HTTP_CODES = frozenset((500, 502, 503, 504))


class GitHubUnavailable(RuntimeError):
    pass


class _RejectRedirectHandler(HTTPRedirectHandler):
    def http_error_302(self, request, response, code, message, headers):
        raise HTTPError(request.full_url, code, message, headers, response)

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


class GitHubClient:
    def __init__(
        self,
        repository: str,
        token: str,
        opener=None,
        required_labels: FrozenSet[str] = _DEFAULT_REQUIRED_LABELS,
    ) -> None:
        self._repository = repository
        self._token = token
        self._opener = opener or build_opener(_RejectRedirectHandler())
        self._required_labels = frozenset(required_labels)

    def list_target_jobs(self) -> List[Job]:
        try:
            run_ids = set()
            for status in ("queued", "in_progress"):
                for run in self._pages(
                    "/repos/{}/actions/runs?status={}&per_page=100".format(
                        self._repository, status
                    ),
                    "workflow_runs",
                ):
                    run_ids.add(self._integer(run.get("id"), "run id"))

            jobs = []
            for run_id in sorted(run_ids):
                for job in self._pages(
                    "/repos/{}/actions/runs/{}/jobs?per_page=30".format(
                        self._repository, run_id
                    ),
                    "jobs",
                ):
                    target_job = self._target_job(job, run_id)
                    if target_job is not None:
                        jobs.append(target_job)
            return jobs
        except (
            HTTPException,
            HTTPError,
            URLError,
            socket.timeout,
            TimeoutError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            raise GitHubUnavailable("GitHub API is unavailable") from error

    def _pages(self, path: str, collection: str) -> List[dict]:
        url = urljoin(_API_URL, path)
        visited_urls = set()
        items = []
        while url is not None:
            if url in visited_urls:
                raise ValueError("pagination loop")
            visited_urls.add(url)
            document, link = self._request_json(url)
            page_items = document.get(collection)
            if not isinstance(page_items, list) or not all(
                isinstance(item, dict) for item in page_items
            ):
                raise ValueError("invalid GitHub response")
            items.extend(page_items)
            url = self._next_url(link, url)
        return items

    def _request_json(self, url: str) -> Tuple[dict, Optional[str]]:
        request = Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": "Bearer {}".format(self._token),
                "User-Agent": "ucm-a2-scheduler",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        for attempt in range(1, _MAX_REQUEST_ATTEMPTS + 1):
            try:
                with self._opener.open(request, timeout=_TIMEOUT_SECONDS) as response:
                    body = response.read()
                    headers = getattr(response, "headers", None)
                    link = headers.get("Link") if headers is not None else None
                break
            except HTTPError as error:
                error.close()
                if (
                    error.code not in _RETRYABLE_HTTP_CODES
                    or attempt == _MAX_REQUEST_ATTEMPTS
                ):
                    raise GitHubUnavailable(
                        "GitHub API HTTP {} at {} after {} attempt(s)".format(
                            error.code, urlparse(url).path, attempt
                        )
                    ) from error
            except (HTTPException, URLError, OSError) as error:
                if attempt == _MAX_REQUEST_ATTEMPTS:
                    raise GitHubUnavailable(
                        "GitHub API {} at {} after {} attempt(s)".format(
                            type(error).__name__, urlparse(url).path, attempt
                        )
                    ) from error
            time.sleep(attempt)
        document = json.loads(body.decode("utf-8"))
        if not isinstance(document, dict):
            raise ValueError("invalid GitHub response")
        return document, link

    def _target_job(self, document: dict, run_id: int) -> Optional[Job]:
        job_id = self._integer(document.get("id"), "job id")
        status = document.get("status")
        labels = document.get("labels")
        if (
            not isinstance(status, str)
            or not isinstance(labels, list)
            or not all(isinstance(label, str) for label in labels)
        ):
            raise ValueError("invalid GitHub job")
        job_labels = frozenset(labels)
        if status not in ("queued", "in_progress"):
            return None
        if not self._required_labels.issubset(job_labels):
            return None
        return Job(id=job_id, status=status, labels=job_labels, run_id=run_id)

    @staticmethod
    def _integer(value, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("{} must be a non-negative integer".format(name))
        return value

    @staticmethod
    def _next_url(link: Optional[str], current_url: str) -> Optional[str]:
        if link is None:
            return None
        if not isinstance(link, str):
            raise ValueError("invalid pagination link")
        for part in link.split(","):
            fields = [field.strip() for field in part.split(";")]
            if (
                len(fields) < 2
                or not fields[0].startswith("<")
                or not fields[0].endswith(">")
            ):
                raise ValueError("invalid pagination link")
            if 'rel="next"' in fields[1:]:
                next_url = urljoin(_API_URL, fields[0][1:-1])
                parsed_url = urlparse(next_url)
                current_path = urlparse(current_url).path
                path_parts = parsed_url.path.split("/", 3)
                # GitHub emits numeric repository URLs; retain the original endpoint.
                if (
                    len(path_parts) == 4
                    and path_parts[1] == "repositories"
                    and path_parts[2].isascii()
                    and path_parts[2].isdigit()
                    and path_parts[3]
                    == "actions/" + current_path.split("/actions/", 1)[-1]
                ):
                    parsed_url = parsed_url._replace(path=current_path)
                if (
                    parsed_url.scheme != "https"
                    or parsed_url.netloc != "api.github.com"
                    or not parsed_url.path.startswith(_ACTIONS_PATH)
                    or parsed_url.path != current_path
                ):
                    raise ValueError("unsafe pagination link")
                return parsed_url.geturl()
        return None
