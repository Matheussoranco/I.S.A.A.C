"""Search Tool — Web search via DuckDuckGo (no API key required).

Falls back to a user-provided Searx instance if configured.
"""

from __future__ import annotations

import logging
from typing import Any

from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(IsaacTool):
    """Search the web using DuckDuckGo (or Searx fallback)."""

    name = "web_search"
    description = "Search the web using DuckDuckGo. No API key required."
    risk_level = 2
    requires_approval = False
    sandbox_required = False

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Run a web search.

        Parameters
        ----------
        query:
            The search query string.
        max_results:
            Maximum number of results (default 5).
        """
        query: str = kwargs.get("query", "")
        if not query:
            return ToolResult(success=False, error="Missing 'query' parameter.")

        max_results: int = int(kwargs.get("max_results", 5))

        result = await self._ddg_search(query, max_results)
        if result.success:
            return result

        logger.warning("DuckDuckGo search failed, trying Searx fallback.")
        return await self._searx_search(query, max_results)

    # ------------------------------------------------------------------
    # DuckDuckGo backend
    # ------------------------------------------------------------------

    async def _ddg_search(self, query: str, max_results: int) -> ToolResult:
        """Search via the ``ddgs`` (formerly ``duckduckgo-search``) package."""
        try:
            try:
                from ddgs import DDGS  # type: ignore[import-untyped]  # new name
            except ImportError:
                from duckduckgo_search import DDGS  # type: ignore[import-untyped]  # old name
        except ImportError:
            return ToolResult(
                success=False,
                error="ddgs package is not installed. Run: pip install ddgs",
            )

        try:
            results: list[dict[str, str]] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", r.get("link", "")),
                            "snippet": r.get("body", ""),
                        }
                    )

            if not results:
                return ToolResult(success=True, output="No results found.")

            lines: list[str] = []
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r['title']}")
                lines.append(f"   {r['url']}")
                lines.append(f"   {r['snippet']}")
                lines.append("")

            return ToolResult(success=True, output="\n".join(lines))
        except Exception as exc:
            logger.error("DuckDuckGo search error: %s", exc)
            return ToolResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Searx fallback
    # ------------------------------------------------------------------

    async def _searx_search(self, query: str, max_results: int) -> ToolResult:
        """Search via a self-hosted Searx instance."""
        try:
            from isaac.config.settings import get_settings

            settings = get_settings()
            searx_url: str = getattr(settings, "searx_url", "")
        except Exception:
            searx_url = ""

        if not searx_url:
            return ToolResult(
                success=False,
                error="DuckDuckGo and Searx both unavailable. Install duckduckgo-search or set SEARX_URL.",
            )

        try:
            import urllib.request
            import urllib.parse
            import json

            params = urllib.parse.urlencode(
                {"q": query, "format": "json", "categories": "general"}
            )
            url = f"{searx_url.rstrip('/')}/search?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "ISAAC/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            results = data.get("results", [])[:max_results]
            if not results:
                return ToolResult(success=True, output="No results found.")

            lines: list[str] = []
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r.get('title', '')}")
                lines.append(f"   {r.get('url', '')}")
                lines.append(f"   {r.get('content', '')}")
                lines.append("")

            return ToolResult(success=True, output="\n".join(lines))
        except Exception as exc:
            logger.error("Searx search error: %s", exc)
            return ToolResult(success=False, error=str(exc))
