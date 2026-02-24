"""Host-side connectors for external world access.

Connectors run on the host (NOT inside the Docker sandbox) and provide
controlled access to external services such as web search, file system,
GitHub, email, etc.
"""

from __future__ import annotations
