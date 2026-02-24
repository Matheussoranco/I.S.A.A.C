"""GitHubConnector — GitHub REST API v3 integration.

Requires ``GITHUB_TOKEN`` environment variable.  Uses httpx for HTTP
requests against the GitHub REST API.
"""

from __future__ import annotations

import logging
from typing import Any

from isaac.skills.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

_API_BASE = "https://api.github.com"


class GitHubConnector(BaseConnector):
    """Interact with GitHub repositories via the REST API v3."""

    name = "github"
    description = (
        "List repos, read files, create/list issues, and search code on GitHub. "
        "Requires GITHUB_TOKEN."
    )
    requires_env: list[str] = ["GITHUB_TOKEN"]

    def _headers(self) -> dict[str, str]:
        import os
        token = os.environ.get("GITHUB_TOKEN", "")
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a GitHub operation.

        Parameters
        ----------
        action : str
            One of ``list_repos``, ``read_file``, ``create_issue``,
            ``list_issues``, ``search_code``.
        user : str
            GitHub username (for ``list_repos``).
        repo : str
            Repository in ``owner/name`` format.
        path : str
            File path within the repo (for ``read_file``).
        branch : str
            Branch name (default ``main``).
        title : str
            Issue title (for ``create_issue``).
        body : str
            Issue body (for ``create_issue``).
        query : str
            Search query (for ``search_code``).
        """
        action: str = kwargs.get("action", "")

        handlers = {
            "list_repos": self._list_repos,
            "read_file": self._read_file,
            "create_issue": self._create_issue,
            "list_issues": self._list_issues,
            "search_code": self._search_code,
        }

        handler = handlers.get(action)
        if handler is None:
            return {"error": f"Unknown action: {action}"}
        try:
            return handler(**kwargs)
        except Exception as exc:
            logger.error("GitHub %s failed: %s", action, exc)
            return {"error": str(exc)}

    def _list_repos(self, **kwargs: Any) -> dict[str, Any]:
        import httpx  # type: ignore[import-untyped]
        user = kwargs.get("user", "")
        url = f"{_API_BASE}/users/{user}/repos" if user else f"{_API_BASE}/user/repos"
        resp = httpx.get(url, headers=self._headers(), timeout=15)
        resp.raise_for_status()
        repos = [
            {"name": r["full_name"], "description": r.get("description", ""), "url": r["html_url"]}
            for r in resp.json()[:30]
        ]
        return {"repos": repos}

    def _read_file(self, **kwargs: Any) -> dict[str, Any]:
        import httpx
        repo = kwargs.get("repo", "")
        path = kwargs.get("path", "")
        branch = kwargs.get("branch", "main")
        url = f"{_API_BASE}/repos/{repo}/contents/{path}?ref={branch}"
        resp = httpx.get(url, headers=self._headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        import base64
        content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
        return {"repo": repo, "path": path, "content": content, "sha": data.get("sha", "")}

    def _create_issue(self, **kwargs: Any) -> dict[str, Any]:
        import httpx
        repo = kwargs.get("repo", "")
        title = kwargs.get("title", "")
        body = kwargs.get("body", "")
        url = f"{_API_BASE}/repos/{repo}/issues"
        resp = httpx.post(
            url, headers=self._headers(), json={"title": title, "body": body}, timeout=15,
        )
        resp.raise_for_status()
        issue = resp.json()
        return {"number": issue["number"], "url": issue["html_url"], "title": issue["title"]}

    def _list_issues(self, **kwargs: Any) -> dict[str, Any]:
        import httpx
        repo = kwargs.get("repo", "")
        url = f"{_API_BASE}/repos/{repo}/issues?state=open&per_page=20"
        resp = httpx.get(url, headers=self._headers(), timeout=15)
        resp.raise_for_status()
        issues = [
            {"number": i["number"], "title": i["title"], "url": i["html_url"]}
            for i in resp.json()[:20]
        ]
        return {"repo": repo, "issues": issues}

    def _search_code(self, **kwargs: Any) -> dict[str, Any]:
        import httpx
        query = kwargs.get("query", "")
        url = f"{_API_BASE}/search/code?q={query}&per_page=10"
        resp = httpx.get(url, headers=self._headers(), timeout=15)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        results = [
            {"path": i["path"], "repo": i["repository"]["full_name"], "url": i["html_url"]}
            for i in items[:10]
        ]
        return {"query": query, "results": results}
