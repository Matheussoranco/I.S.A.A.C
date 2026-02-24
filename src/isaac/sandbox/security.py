"""Sandbox security policy.

Defines the ``SecurityPolicy`` dataclass that controls every Docker container
flag.  All values are derived from :pymod:`isaac.config.settings` but can be
overridden per-execution when needed.

Seccomp hardening
-----------------
A built-in seccomp profile is generated that only allows the system calls
required by the Python interpreter and standard libraries.  This blocks
dangerous syscalls like ``ptrace``, ``mount``, ``reboot``, ``kexec_load``,
``init_module``, and similar privilege-escalation vectors.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seccomp profile
# ---------------------------------------------------------------------------

def _default_seccomp_profile() -> dict[str, Any]:
    """Return a restrictive seccomp profile for code-execution containers.

    Default action is SCMP_ACT_ERRNO (deny), with an explicit allowlist
    of safe syscalls required for Python, pip, and basic I/O.
    """
    # Allowlisted syscalls — minimal set for CPython + NumPy
    allowed_syscalls = [
        # Process
        "exit", "exit_group", "getpid", "getppid", "gettid",
        "clone", "clone3", "fork", "vfork", "wait4", "waitid",
        "execve", "execveat",
        # Memory
        "mmap", "munmap", "mprotect", "mremap", "brk",
        "madvise", "mlock", "munlock",
        # File I/O
        "open", "openat", "close", "read", "write", "pread64", "pwrite64",
        "readv", "writev", "lseek", "fstat", "newfstatat", "stat",
        "fstatfs", "statfs", "statx",
        "access", "faccessat", "faccessat2",
        "dup", "dup2", "dup3", "fcntl",
        "ioctl", "flock",
        "mkdir", "mkdirat", "rmdir", "unlink", "unlinkat",
        "rename", "renameat", "renameat2",
        "readlink", "readlinkat",
        "getcwd", "chdir", "fchdir",
        "getdents", "getdents64",
        "ftruncate", "truncate",
        "fallocate", "copy_file_range",
        "sendfile",
        # Pipe / socket (for subprocess stdout/stderr)
        "pipe", "pipe2", "socketpair",
        "socket", "connect", "bind", "listen",
        "accept", "accept4",
        "getsockname", "getpeername",
        "setsockopt", "getsockopt",
        "sendto", "recvfrom", "sendmsg", "recvmsg",
        "shutdown",
        "select", "pselect6",
        # Signals
        "rt_sigaction", "rt_sigprocmask", "rt_sigreturn",
        "sigaltstack", "kill", "tgkill",
        # Time
        "clock_gettime", "clock_getres", "clock_nanosleep",
        "nanosleep", "gettimeofday",
        # Epoll / poll
        "epoll_create", "epoll_create1", "epoll_ctl", "epoll_wait",
        "epoll_pwait", "epoll_pwait2",
        "poll", "ppoll",
        "eventfd", "eventfd2",
        # Misc
        "getrandom", "arch_prctl", "prctl", "set_tid_address",
        "set_robust_list", "get_robust_list",
        "futex", "futex_waitv",
        "sched_yield", "sched_getaffinity",
        "getuid", "getgid", "geteuid", "getegid",
        "getgroups", "setgroups",
        "uname", "sysinfo",
        "umask", "chown", "fchown", "fchownat",
        "chmod", "fchmod", "fchmodat",
        "utimensat",
        "timerfd_create", "timerfd_settime", "timerfd_gettime",
        "memfd_create",
        "prlimit64", "getrlimit", "setrlimit",
        "rseq",
    ]

    return {
        "defaultAction": "SCMP_ACT_ERRNO",
        "archMap": [
            {"architecture": "SCMP_ARCH_X86_64", "subArchitectures": ["SCMP_ARCH_X86", "SCMP_ARCH_X32"]},
            {"architecture": "SCMP_ARCH_AARCH64", "subArchitectures": ["SCMP_ARCH_ARM"]},
        ],
        "syscalls": [
            {
                "names": allowed_syscalls,
                "action": "SCMP_ACT_ALLOW",
            }
        ],
    }


def write_seccomp_profile(target_dir: Path | None = None) -> Path:
    """Write the seccomp JSON profile to disk and return the path.

    The Docker ``--security-opt=seccomp=<path>`` flag needs a file path.
    """
    if target_dir is None:
        try:
            from isaac.config.settings import get_settings
            target_dir = get_settings().isaac_home / "security"
        except Exception:
            target_dir = Path.home() / ".isaac" / "security"

    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "seccomp.json"
    path.write_text(json.dumps(_default_seccomp_profile(), indent=2), encoding="utf-8")
    logger.info("Seccomp profile written to %s", path)
    return path


@dataclass(frozen=True)
class SecurityPolicy:
    """Immutable set of Docker security constraints.

    Default values enforce maximum isolation: no network, no capabilities,
    read-only root filesystem, non-root user, and strict resource limits.
    Includes optional seccomp profile for syscall filtering.
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

    # Seccomp
    seccomp_profile_path: str = ""
    """Path to a seccomp JSON profile.  Empty string means Docker default."""

    # Tmpfs for writable scratch space
    tmpfs: dict[str, str] = field(
        default_factory=lambda: {"/tmp": "rw,noexec,nosuid,size=64m"}
    )

    # Execution timeout (seconds) — enforced at application level
    timeout_seconds: int = 30

    def to_container_kwargs(self) -> dict:
        """Convert to keyword arguments for ``docker.containers.run()``."""
        sec_opts = list(self.security_opts)
        if self.seccomp_profile_path:
            import platform as _platform
            if _platform.system() == "Windows":
                # Seccomp is a Linux kernel feature; Docker Desktop on Windows
                # runs containers in a VM that has seccomp but cannot accept a
                # Windows host-side file path.  Pass the profile content inline.
                try:
                    import json as _json
                    from pathlib import Path as _Path
                    content = _Path(self.seccomp_profile_path).read_text(encoding="utf-8")
                    # Minify to a single line — Docker API expects inline JSON
                    sec_opts.append(f"seccomp={_json.dumps(_json.loads(content))}")
                except Exception:
                    logger.debug("Seccomp inline load failed — skipping on Windows.")
            else:
                sec_opts.append(f"seccomp={self.seccomp_profile_path}")

        return {
            "network_mode": self.network_mode,
            "mem_limit": self.memory_limit,
            "nano_cpus": int(self.cpu_limit * 1e9),
            "pids_limit": self.pids_limit,
            "user": self.user,
            "cap_drop": self.cap_drop,
            "security_opt": sec_opts,
            "read_only": self.read_only_rootfs,
            "tmpfs": self.tmpfs,
        }


def default_policy() -> SecurityPolicy:
    """Build a ``SecurityPolicy`` from the current application settings.

    Automatically generates and references the seccomp profile.
    """
    from isaac.config.settings import settings

    cfg = settings.sandbox

    # Write seccomp profile and reference it
    seccomp_path = ""
    try:
        path = write_seccomp_profile()
        seccomp_path = str(path)
    except Exception as exc:
        logger.debug("Seccomp profile generation skipped: %s", exc)

    return SecurityPolicy(
        network_mode=cfg.network,
        memory_limit=cfg.memory_limit,
        cpu_limit=cfg.cpu_limit,
        pids_limit=cfg.pids_limit,
        timeout_seconds=cfg.timeout_seconds,
        seccomp_profile_path=seccomp_path,
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
