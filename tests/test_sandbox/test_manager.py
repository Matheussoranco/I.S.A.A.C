"""Tests for the SandboxManager (Docker SDK mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from isaac.sandbox.manager import SandboxManager
from isaac.sandbox.security import SecurityPolicy


class TestSandboxManager:
    def test_create_container(self) -> None:
        mock_client = MagicMock()
        mock_client.images.get.return_value = True
        mock_container = MagicMock()
        mock_container.short_id = "abc"
        mock_client.containers.create.return_value = mock_container

        with patch("docker.from_env", return_value=mock_client):
            mgr = SandboxManager("isaac-sandbox:latest", SecurityPolicy())
            container = mgr.create_container(["python", "/input/task.py"])

        assert container.short_id == "abc"
        mock_client.containers.create.assert_called_once()
        call_kwargs = mock_client.containers.create.call_args
        assert call_kwargs.kwargs.get("network_mode") == "none"
        assert call_kwargs.kwargs.get("read_only") is True

    def test_wait_returns_exit_code(self) -> None:
        mock_client = MagicMock()
        mock_client.images.get.return_value = True
        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}

        with patch("docker.from_env", return_value=mock_client):
            mgr = SandboxManager("test:latest", SecurityPolicy())
            code = mgr.wait(mock_container)

        assert code == 0

    def test_destroy(self) -> None:
        mock_client = MagicMock()
        mock_client.images.get.return_value = True
        mock_container = MagicMock()

        with patch("docker.from_env", return_value=mock_client):
            mgr = SandboxManager("test:latest", SecurityPolicy())
            mgr.destroy(mock_container)

        mock_container.remove.assert_called_once_with(force=True)
