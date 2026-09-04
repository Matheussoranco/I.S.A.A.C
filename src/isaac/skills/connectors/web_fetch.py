"""WebFetchConnector — fetch and extract readable text from a URL.

Uses httpx for HTTP requests and BeautifulSoup for content extraction.
Strips scripts, styles, and returns clean readable text (max 10k chars).
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Any, ClassVar
from urllib.parse import urljoin, urlparse

from isaac.skills.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class WebFetchConnector(BaseConnector):
    """Fetch a web page and extract its readable text content."""

    name = "web_fetch"
    description = "Fetch a URL and extract its readable text content (max 10k chars)."
    requires_env: ClassVar[list[str]] = []

    _MAX_TEXT_LENGTH = 10_000
    _MAX_RESPONSE_BYTES = 2 * 1024 * 1024
    _MAX_REDIRECTS = 5

    @classmethod
    def _validate_url(cls, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Only absolute http(s) URLs are allowed.")
        if parsed.username or parsed.password:
            raise ValueError("URLs containing credentials are not allowed.")

        try:
            addresses = {
                info[4][0]
                for info in socket.getaddrinfo(
                    parsed.hostname,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            }
        except OSError as exc:
            raise ValueError(f"Could not resolve host: {parsed.hostname}") from exc

        for address in addresses:
            ip = ipaddress.ip_address(address)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                raise ValueError("Requests to private or local network addresses are blocked.")

    @classmethod
    def _read_limited(cls, response: Any) -> bytes:
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > cls._MAX_RESPONSE_BYTES:
            raise ValueError("Response exceeds the 2 MiB safety limit.")
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > cls._MAX_RESPONSE_BYTES:
                raise ValueError("Response exceeds the 2 MiB safety limit.")
        return bytes(body)

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

            current_url = url
            with httpx.Client(timeout=15, follow_redirects=False) as client:
                for _ in range(self._MAX_REDIRECTS + 1):
                    self._validate_url(current_url)
                    with client.stream("GET", current_url) as response:
                        if 300 <= response.status_code < 400:
                            location = response.headers.get("location")
                            if not location:
                                raise ValueError(
                                    "Redirect response did not include a Location header."
                                )
                            current_url = urljoin(current_url, location)
                            continue
                        response.raise_for_status()
                        raw_body = self._read_limited(response)
                        encoding = response.encoding or "utf-8"
                        body = raw_body.decode(encoding, errors="replace")
                        status_code = response.status_code
                        break
                else:
                    raise ValueError("Too many redirects.")
        except Exception as exc:
            logger.error("WebFetch failed for %s: %s", url, exc)
            return {"error": str(exc), "url": url, "status_code": 0}

        result: dict[str, Any] = {
            "url": url,
            "status_code": status_code,
        }

        if not extract_text:
            result["text"] = body[: self._MAX_TEXT_LENGTH]
            result["title"] = ""
            return result

        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]

            soup = BeautifulSoup(body, "html.parser")

            # Remove scripts and styles
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            text = soup.get_text(separator="\n", strip=True)
            # Collapse multiple blank lines
            lines = [line for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)[: self._MAX_TEXT_LENGTH]

            result["title"] = title
            result["text"] = clean_text
        except ImportError:
            # No BS4 — return raw text
            result["title"] = ""
            result["text"] = body[: self._MAX_TEXT_LENGTH]

        return result
