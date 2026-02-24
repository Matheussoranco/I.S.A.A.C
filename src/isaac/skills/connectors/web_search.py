"""WebSearchConnector — DuckDuckGo-based web search (no API key required).

Falls back to httpx scraping if the ``duckduckgo_search`` package is
unavailable or fails.
"""

from __future__ import annotations

import logging
from typing import Any

from isaac.skills.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class WebSearchConnector(BaseConnector):
    """Search the web via DuckDuckGo (no API key required)."""

    name = "web_search"
    description = "Search the web using DuckDuckGo and return titles, URLs, and snippets."
    requires_env: list[str] = []

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a web search.

        Parameters
        ----------
        query : str
            The search query.
        max_results : int
            Maximum number of results to return (default 5).
        """
        query: str = kwargs.get("query", "")
        max_results: int = int(kwargs.get("max_results", 5))

        if not query:
            return {"error": "No query provided.", "results": []}

        # Try duckduckgo-search first
        try:
            return self._search_ddg(query, max_results)
        except Exception as exc:
            logger.warning("DDG search failed: %s — trying httpx fallback.", exc)

        # Fallback: httpx scraping
        try:
            return self._search_httpx(query, max_results)
        except Exception as exc:
            logger.error("Web search fallback also failed: %s", exc)
            return {"error": str(exc), "results": []}

    def _search_ddg(self, query: str, max_results: int) -> dict[str, Any]:
        """Search via duckduckgo-search package."""
        from duckduckgo_search import DDGS  # type: ignore[import-untyped]

        results: list[dict[str, str]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                })
        return {"query": query, "results": results}

    def _search_httpx(self, query: str, max_results: int) -> dict[str, Any]:
        """Fallback search by scraping DuckDuckGo HTML."""
        import httpx  # type: ignore[import-untyped]

        url = "https://html.duckduckgo.com/html/"
        resp = httpx.post(url, data={"q": query}, timeout=10, follow_redirects=True)
        resp.raise_for_status()

        results: list[dict[str, str]] = []
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]
            soup = BeautifulSoup(resp.text, "html.parser")
            for link in soup.select(".result__a")[:max_results]:
                href = link.get("href", "")
                title = link.get_text(strip=True)
                snippet_el = link.find_next(".result__snippet")
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                results.append({"title": title, "url": href, "snippet": snippet})
        except ImportError:
            # No BeautifulSoup — return raw text
            results.append({"title": "Raw HTML", "url": url, "snippet": resp.text[:500]})

        return {"query": query, "results": results}
