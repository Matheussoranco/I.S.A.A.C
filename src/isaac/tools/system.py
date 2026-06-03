"""System Info Tool — read-only situational awareness about the host PC.

Gives a specialist the facts it needs to plan PC tasks: OS and version, CPU
core count and load, memory, and disk usage per drive.  Read-only and risk
level 1 — it never changes anything.  Uses ``psutil`` when available and
degrades gracefully to the standard library otherwise.
"""

from __future__ import annotations

import os
import platform
import shutil
from typing import Any

from isaac.tools.base import IsaacTool, ToolResult


class SystemInfoTool(IsaacTool):
    """Report read-only information about the host machine."""

    name = "system_info"
    description = (
        "Report read-only facts about the host PC: operating system and version, "
        "CPU cores and load, total/available memory, and disk usage per drive. "
        "Use to understand the environment before planning system or file tasks."
    )
    risk_level = 1
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> ToolResult:
        lines: list[str] = []
        meta: dict[str, Any] = {}

        # ── OS / platform ────────────────────────────────────────────────
        uname = platform.uname()
        meta["os"] = uname.system
        meta["os_release"] = uname.release
        meta["os_version"] = uname.version
        meta["machine"] = uname.machine
        meta["python"] = platform.python_version()
        meta["cpu_count"] = os.cpu_count() or 0
        lines.append(f"OS:        {uname.system} {uname.release} ({uname.machine})")
        lines.append(f"Hostname:  {uname.node}")
        lines.append(f"Python:    {platform.python_version()}")
        lines.append(f"CPU cores: {os.cpu_count() or 'unknown'}")

        # ── Memory / load (psutil optional) ──────────────────────────────
        try:
            import psutil  # type: ignore[import-untyped]

            vm = psutil.virtual_memory()
            meta["memory_total_gb"] = round(vm.total / 1e9, 2)
            meta["memory_available_gb"] = round(vm.available / 1e9, 2)
            meta["memory_percent"] = vm.percent
            lines.append(
                f"Memory:    {vm.available / 1e9:.1f} GB free / {vm.total / 1e9:.1f} GB "
                f"({vm.percent:.0f}% used)"
            )
            try:
                meta["cpu_percent"] = psutil.cpu_percent(interval=0.1)
                lines.append(f"CPU load:  {meta['cpu_percent']:.0f}%")
            except Exception:
                pass
        except Exception:
            lines.append("Memory:    (install 'psutil' for memory/CPU stats)")

        # ── Disk usage ───────────────────────────────────────────────────
        lines.append("Disks:")
        drives: list[str] = []
        if platform.system() == "Windows":
            import string

            drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
        else:
            drives = ["/"]
        disk_meta = []
        for d in drives:
            try:
                usage = shutil.disk_usage(d)
                free_gb, total_gb = usage.free / 1e9, usage.total / 1e9
                pct = (usage.used / usage.total * 100) if usage.total else 0
                lines.append(
                    f"  {d:<6} {free_gb:6.1f} GB free / {total_gb:6.1f} GB ({pct:.0f}% used)"
                )
                disk_meta.append(
                    {"drive": d, "free_gb": round(free_gb, 1), "total_gb": round(total_gb, 1)}
                )
            except Exception:
                continue
        meta["disks"] = disk_meta

        return ToolResult(success=True, output="\n".join(lines), metadata=meta)
