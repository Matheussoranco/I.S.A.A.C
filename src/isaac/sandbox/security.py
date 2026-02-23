"""Sandbox security policy.

Defines the ``SecurityPolicy`` dataclass that controls every Docker container
flag.  All values are derived from :pymod:`isaac.config.settings` but can be
overridden per-execution when needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SecurityPolicy:
    """Immutable set of Docker security constraints.

    Default values enforce maximum isolation: no network, no capabilities,
    read-only root filesystem, non-root user, and strict resource limits.
    """

    # Network
    network_mode: str = "none"

    # Resource limits
    memory_limit: str = "256m"
    cpu_limit: float = 1.0
    pids_limit: int = 64

    # Privilege
    user: str = "65534:65534"  # nobody
    cap_drop: list[str] = field(default_factory=lambda: ["ALL"])
    security_opts: list[str] = field(
        default_factory=lambda: ["no-new-privileges"]
    )
    read_only_rootfs: bool = True

    # Tmpfs for writable scratch space
    tmpfs: dict[str, str] = field(
        default_factory=lambda: {"/tmp": "rw,noexec,nosuid,size=64m"}
    )

    # Execution timeout (seconds) — enforced at application level
    timeout_seconds: int = 30

    def to_container_kwargs(self) -> dict:
        """Convert to keyword arguments for ``docker.containers.run()``."""
        return {
            "network_mode": self.network_mode,
            "mem_limit": self.memory_limit,
            "nano_cpus": int(self.cpu_limit * 1e9),
            "pids_limit": self.pids_limit,
            "user": self.user,
            "cap_drop": self.cap_drop,
            "security_opt": self.security_opts,
            "read_only": self.read_only_rootfs,
            "tmpfs": self.tmpfs,
        }


def default_policy() -> SecurityPolicy:
    """Build a ``SecurityPolicy`` from the current application settings."""
    from isaac.config.settings import settings

    cfg = settings.sandbox
    return SecurityPolicy(
        network_mode=cfg.network,
        memory_limit=cfg.memory_limit,
        cpu_limit=cfg.cpu_limit,
        pids_limit=cfg.pids_limit,
        timeout_seconds=cfg.timeout_seconds,
        tmpfs={},  # code sandbox uses read-only rootfs; no tmpfs needed
    )


def ui_policy() -> SecurityPolicy:
    """Build a permissive ``SecurityPolicy`` for virtual-desktop containers.

    UI containers need write access to ``/tmp`` and ``/run`` for X11 sockets,
    and more memory/CPU for running a browser.  The rootfs is NOT read-only.
    """
    from isaac.config.settings import settings

    cfg = settings.ui_sandbox
    return SecurityPolicy(
        network_mode="none" if not cfg.allow_browser_network else "bridge",
        memory_limit=cfg.memory_limit,
        cpu_limit=cfg.cpu_limit,
        pids_limit=cfg.pids_limit,
        timeout_seconds=cfg.timeout_seconds,
        # UI containers must be writable (Xvfb writes to /tmp/.X11-unix)
        read_only_rootfs=False,
        tmpfs={}  # managed by the image itself
    )
