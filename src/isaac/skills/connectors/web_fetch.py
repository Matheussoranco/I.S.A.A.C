"""WebFetchConnector — fetch and extract readable text from a URL.

Uses httpx for HTTP requests and BeautifulSoup for content extraction.
Strips scripts, styles, and returns clean readable text (max 10k chars).
"""

from __future__ import annotations

import logging
from typing import Any

from isaac.skills.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class WebFetchConnector(BaseConnector):
    """Fetch a web page and extract its readable text content."""

    name = "web_fetch"
    description = "Fetch a URL and extract its readable text content (max 10k chars)."
    requires_env: list[str] = []

    _MAX_TEXT_LENGTH = 10_000

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Fetch a web page.

        Parameters
        ----------
        url : str
            The URL to fetch.
        extract_text : bool
            If True (default), extract readable text; otherwise return raw HTML.
        """
        url: str = kwargs.get("url", "")
        extract_text: bool = kwargs.get("extract_text", True)

        if not url:
            return {"error": "No URL provided."}

        try:
            import httpx  # type: ignore[import-untyped]

            resp = httpx.get(url, timeout=15, follow_redirects=True)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("WebFetch failed for %s: %s", url, exc)
            return {"error": str(exc), "url": url, "status_code": 0}

        result: dict[str, Any] = {
            "url": url,
            "status_code": resp.status_code,
        }

        if not extract_text:
            result["text"] = resp.text[:self._MAX_TEXT_LENGTH]
            result["title"] = ""
            return result

        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove scripts and styles
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            text = soup.get_text(separator="\n", strip=True)
            # Collapse multiple blank lines
            lines = [line for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)[:self._MAX_TEXT_LENGTH]

            result["title"] = title
            result["text"] = clean_text
        except ImportError:
            # No BS4 — return raw text
            result["title"] = ""
            result["text"] = resp.text[:self._MAX_TEXT_LENGTH]

        return result
