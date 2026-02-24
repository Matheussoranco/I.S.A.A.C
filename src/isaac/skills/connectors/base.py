"""BaseConnector — abstract base class for all host-side connectors.

Every connector declares its name, description, and required env vars.
The registry uses ``is_available()`` to decide which connectors to expose.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """Abstract base for host-side skill connectors.

    Attributes
    ----------
    name:
        Short identifier (e.g. ``"web_search"``).
    description:
        Human-readable description shown to the LLM during tool selection.
    requires_env:
        List of environment variable names required.  Empty means no auth.
    """

    name: str = ""
    description: str = ""
    requires_env: list[str] = []

    def is_available(self) -> bool:
        """Check whether all required environment variables are set."""
        for var in self.requires_env:
            if not os.environ.get(var):
                return False
        return True

    @abstractmethod
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the connector action.

        Parameters
        ----------
        **kwargs:
            Connector-specific keyword arguments.

        Returns
        -------
        dict
            JSON-serialisable result dictionary.
        """

    def to_schema(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for the LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "available": self.is_available(),
        }
