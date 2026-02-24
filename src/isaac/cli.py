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

    @app.command()
    def run(
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    ) -> None:
        """Start the interactive cognitive loop (REPL)."""
        _setup_logging(verbose)
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
        # Fallback for environments without Typer
        _setup_logging()
        from isaac.core.graph import build_and_run
        return build_and_run()
