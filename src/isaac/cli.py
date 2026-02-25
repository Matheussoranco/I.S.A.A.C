"""I.S.A.A.C. CLI — Typer-based unified entry point.

Commands
--------
run         Start the interactive cognitive loop (default).
serve       Start the Telegram gateway + heartbeat scheduler.
audit       View / verify the audit log.
memory      Query the memory layers.
tools       List registered tools.
tokens      Manage capability tokens.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import typer  # type: ignore[import-untyped]
except ImportError:
    # Fallback: if Typer is not installed, provide a minimal CLI via argparse
    typer = None  # type: ignore[assignment]

if typer is not None:
    app = typer.Typer(
        name="isaac",
        help="I.S.A.A.C. — Intelligent System for Autonomous Action and Cognition",
        add_completion=False,
    )
else:
    app = None  # type: ignore[assignment]


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


if typer is not None:
    assert app is not None

    @app.command()
    def run(
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
        classic: bool = typer.Option(False, "--classic", help="Use the classic plain-text REPL."),
    ) -> None:
        """Start the interactive cognitive loop (REPL)."""
        _setup_logging(verbose)

        if classic:
            # Legacy plain-text REPL
            from isaac.core.graph import build_and_run
            from isaac.scheduler.heartbeat import start_scheduler, stop_scheduler
            from isaac.tools import register_all_tools

            register_all_tools()
            start_scheduler()
            try:
                code = build_and_run()
            finally:
                stop_scheduler()
            raise typer.Exit(code)

        # Rich terminal UI (default)
        from isaac.interfaces.repl import run_repl
        code = run_repl()
        raise typer.Exit(code)

    @app.command()
    def serve(
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ) -> None:
        """Start the Telegram gateway + heartbeat scheduler (daemon mode)."""
        _setup_logging(verbose)
        import asyncio
        from isaac.interfaces.telegram_gateway import start_bot
        from isaac.scheduler.heartbeat import start_scheduler, stop_scheduler
        from isaac.tools import register_all_tools

        register_all_tools()
        start_scheduler()

        typer.echo("Starting Telegram gateway... Press Ctrl+C to stop.")
        try:
            asyncio.run(start_bot())
        except KeyboardInterrupt:
            typer.echo("\nShutting down.")
        finally:
            stop_scheduler()

    @app.command()
    def audit(
        verify: bool = typer.Option(False, "--verify", help="Verify audit chain integrity."),
        last: int = typer.Option(10, "--last", "-n", help="Show last N entries."),
    ) -> None:
        """View or verify the audit log."""
        _setup_logging()
        from isaac.security.audit import get_audit_log

        log = get_audit_log()

        if verify:
            valid, count = log.verify_chain()
            status = "VALID" if valid else "BROKEN"
            typer.echo(f"Audit chain: {status} ({count} entries verified)")
            if not valid:
                raise typer.Exit(1)
        else:
            entries = log.recent(last)
            if not entries:
                typer.echo("No audit entries.")
                return
            for entry in entries:
                typer.echo(
                    f"[{entry.timestamp}] {entry.category}/{entry.action} "
                    f"actor={entry.actor} hash={entry.entry_hash[:12]}..."
                )
                if entry.details:
                    typer.echo(f"  details: {entry.details}")

    @app.command()
    def memory(
        query: str = typer.Argument("recent", help="Search query for memory recall."),
        k: int = typer.Option(5, "--k", help="Number of results per layer."),
    ) -> None:
        """Query the unified memory system."""
        _setup_logging()
        from isaac.memory.manager import get_memory_manager

        mm = get_memory_manager()
        result = mm.recall(query, k=k)
        typer.echo(result.combined_context or "No memories found.")

    @app.command()
    def tools() -> None:
        """List all registered tools."""
        _setup_logging()
        from isaac.tools import register_all_tools
        from isaac.tools.base import get_tool_registry

        register_all_tools()
        registry = get_tool_registry()
        all_tools = registry.list_all()

        if not all_tools:
            typer.echo("No tools registered.")
            return

        for tool in all_tools:
            approval = " [APPROVAL REQUIRED]" if tool.requires_approval else ""
            sandbox = " [SANDBOX]" if tool.sandbox_required else ""
            typer.echo(
                f"  {tool.name:20s} risk={tool.risk_level}  {tool.description[:60]}{approval}{sandbox}"
            )

    @app.command()
    def cron(
        action: str = typer.Argument("list", help="Action: list, add, remove, pause, resume, start, stop, status."),
        name: str = typer.Option("", "--name", help="Task name (for add)."),
        schedule: str = typer.Option("0 * * * *", "--schedule", "-s", help="Cron expression (for add)."),
        command: str = typer.Option("", "--command", "-c", help="Command string (for add)."),
        task_id: str = typer.Option("", "--id", help="Task ID (for remove/pause/resume)."),
    ) -> None:
        """Manage background cron tasks."""
        _setup_logging()
        from isaac.background.cron_engine import (
            add_task,
            is_cron_running,
            list_tasks,
            pause_task,
            remove_task,
            resume_task,
            start_cron_daemon,
            stop_cron_daemon,
        )

        if action == "list":
            tasks = list_tasks()
            if not tasks:
                typer.echo("No cron tasks.")
                return
            for t in tasks:
                status = "enabled" if t["enabled"] else "PAUSED"
                typer.echo(
                    f"  {t['id']}  [{status}]  {t['schedule']}  "
                    f"{t['name'] or t['command'][:40]}  "
                    f"last={t['last_run'] or 'never'}  result={t['last_status'] or '-'}"
                )
        elif action == "add":
            if not command:
                typer.echo("Provide --command (-c).")
                raise typer.Exit(1)
            task = add_task(name=name or command[:30], schedule=schedule, command=command)
            typer.echo(f"Created task: {task.id} ({task.name})")
        elif action == "remove":
            if not task_id:
                typer.echo("Provide --id.")
                raise typer.Exit(1)
            ok = remove_task(task_id)
            typer.echo("Removed." if ok else "Task not found.")
        elif action == "pause":
            if not task_id:
                typer.echo("Provide --id.")
                raise typer.Exit(1)
            ok = pause_task(task_id)
            typer.echo("Paused." if ok else "Task not found.")
        elif action == "resume":
            if not task_id:
                typer.echo("Provide --id.")
                raise typer.Exit(1)
            ok = resume_task(task_id)
            typer.echo("Resumed." if ok else "Task not found.")
        elif action == "start":
            start_cron_daemon()
            typer.echo("Cron daemon started.")
        elif action == "stop":
            stop_cron_daemon()
            typer.echo("Cron daemon stopped.")
        elif action == "status":
            running = is_cron_running()
            typer.echo(f"Cron daemon: {'RUNNING' if running else 'STOPPED'}")
            tasks = list_tasks()
            typer.echo(f"Tasks: {len(tasks)} total, {sum(1 for t in tasks if t['enabled'])} enabled")
        else:
            typer.echo(f"Unknown action: {action}")

    @app.command()
    def connectors() -> None:
        """List all registered connectors and their availability."""
        _setup_logging()
        from isaac.skills.connectors.registry import get_registry

        reg = get_registry()
        if not reg:
            typer.echo("No connectors found.")
            return

        for name, connector in sorted(reg.items()):
            avail = "✓" if connector.is_available() else "✗"
            env = ", ".join(connector.requires_env) if connector.requires_env else "none"
            typer.echo(f"  [{avail}] {name:15s}  env={env:30s}  {connector.description[:50]}")

    @app.command()
    def tokens(
        action: str = typer.Argument("list", help="Action: list, issue, revoke, cleanup."),
        tool_name: str = typer.Option("*", "--tool", help="Tool name for issue/revoke."),
        token_id: str = typer.Option("", "--id", help="Token ID for revoke."),
        ttl: int = typer.Option(24, "--ttl", help="TTL in hours for issue."),
    ) -> None:
        """Manage capability tokens."""
        _setup_logging()
        from isaac.security.capabilities import get_token_store

        store = get_token_store()

        if action == "list":
            active = store.list_active()
            if not active:
                typer.echo("No active tokens.")
                return
            for t in active:
                typer.echo(
                    f"  {t.token_id[:12]}...  tool={t.tool_name}  "
                    f"action={t.action}  uses={t.use_count}/{t.max_uses or '∞'}  "
                    f"expires={t.expires_at}"
                )
        elif action == "issue":
            token = store.issue(tool_name, ttl_hours=ttl, issued_by="cli")
            typer.echo(f"Issued token: {token.token_id}")
        elif action == "revoke":
            if not token_id:
                typer.echo("Provide --id to revoke.")
                raise typer.Exit(1)
            ok = store.revoke(token_id, revoked_by="cli")
            typer.echo("Revoked." if ok else "Token not found.")
        elif action == "cleanup":
            n = store.cleanup_expired()
            typer.echo(f"Cleaned up {n} expired tokens.")
        else:
            typer.echo(f"Unknown action: {action}")


def main() -> int:
    """Entry point — delegates to Typer if available, else basic argparse."""
    if app is not None:
        app()
        return 0
    else:
        # Fallback for environments without Typer — use Rich REPL
        _setup_logging()
        try:
            from isaac.interfaces.repl import run_repl
            return run_repl()
        except ImportError:
            from isaac.core.graph import build_and_run
            return build_and_run()
