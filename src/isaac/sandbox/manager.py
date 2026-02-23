"""Docker container lifecycle manager.

Handles image verification, container creation, and tear-down.  All
operations go through the Docker SDK — never through ``subprocess``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import docker
from docker.errors import DockerException, ImageNotFound

if TYPE_CHECKING:
    from docker.models.containers import Container

    from isaac.sandbox.security import SecurityPolicy

logger = logging.getLogger(__name__)


class SandboxManager:
    """Create, track, and destroy ephemeral Docker containers.

    Parameters
    ----------
    image:
        Docker image tag used for sandbox containers.
    policy:
        Security policy governing resource limits and isolation.
    """

    def __init__(self, image: str, policy: SecurityPolicy) -> None:
        self._image = image
        self._policy = policy
        self._client = docker.from_env()
        self._verify_image()

    # -- internal -----------------------------------------------------------

    def _verify_image(self) -> None:
        """Ensure the sandbox image exists locally."""
        try:
            self._client.images.get(self._image)
            logger.info("Sandbox image '%s' verified.", self._image)
        except ImageNotFound:
            logger.warning(
                "Sandbox image '%s' not found locally.  "
                "Build it with: docker build -t %s sandbox_image/",
                self._image,
                self._image,
            )

    # -- public API ---------------------------------------------------------

    def create_container(
        self,
        command: list[str],
        *,
        volumes: dict[str, dict[str, str]] | None = None,
        environment: dict[str, str] | None = None,
    ) -> Container:
        """Create (but do not start) an ephemeral container.

        Parameters
        ----------
        command:
            The command to execute inside the container (e.g.,
            ``["python", "/input/task.py"]``).
        volumes:
            Optional host→container bind-mounts.
        environment:
            Optional environment variables injected into the container.

        Returns
        -------
        Container
            A Docker container in *created* state.
        """
        kwargs = self._policy.to_container_kwargs()
        container: Container = self._client.containers.create(
            image=self._image,
            command=command,
            volumes=volumes or {},
            environment=environment or {},
            detach=True,
            auto_remove=False,  # we remove manually after log capture
            **kwargs,
        )
        logger.debug("Container %s created (image=%s).", container.short_id, self._image)
        return container

    def start(self, container: Container) -> None:
        """Start a previously created container."""
        container.start()
        logger.debug("Container %s started.", container.short_id)

    def wait(self, container: Container, timeout: int | None = None) -> int:
        """Block until the container exits.  Returns the exit code."""
        t = timeout or self._policy.timeout_seconds
        try:
            result = container.wait(timeout=t)
            exit_code: int = result.get("StatusCode", -1)
        except Exception:
            logger.warning("Container %s timed out after %ds — killing.", container.short_id, t)
            container.kill()
            exit_code = -1
        return exit_code

    def logs(self, container: Container) -> tuple[str, str]:
        """Return ``(stdout, stderr)`` from the container."""
        stdout = container.logs(stdout=True, stderr=False).decode(errors="replace")
        stderr = container.logs(stdout=False, stderr=True).decode(errors="replace")
        return stdout, stderr

    def destroy(self, container: Container) -> None:
        """Force-remove the container and reclaim resources."""
        try:
            container.remove(force=True)
            logger.debug("Container %s destroyed.", container.short_id)
        except DockerException as exc:
            logger.error("Failed to destroy container %s: %s", container.short_id, exc)

    def close(self) -> None:
        """Close the underlying Docker client."""
        self._client.close()
