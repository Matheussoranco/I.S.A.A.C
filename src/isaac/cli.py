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
    def agent(
        task: str = typer.Argument(..., help="The task for the agent to accomplish."),
        max_iters: int = typer.Option(12, "--max-iters", "-n", help="Max tool-use rounds."),
        max_seconds: float = typer.Option(
            600.0, "--max-seconds", help="Wall-clock budget for the run (0 = unlimited)."
        ),
        auto_approve: bool = typer.Option(
            False, "--auto-approve", "-y", help="Run high-risk (4-5) tools without approval."
        ),
        tools_csv: str = typer.Option(
            "", "--tools", help="Comma-separated tool names to restrict the agent to."
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    ) -> None:
        """Run the autonomous tool-use agent on a single task (Claude-Code style).

        The agent is given the full built-in toolbox — web search, a persistent
        browser, a Python runner, and a sandboxed file workspace — and iterates
        (call tool → observe → decide) until it produces a final answer.

        Example::

            isaac agent "Find the current stable Python version and save it to version.txt"
        """
        _setup_logging(verbose)

        try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console()
            _echo = console.print
            _rich = True
        except Exception:
            console = None
            _rich = False

            def _echo(*a: object, **k: object) -> None:  # type: ignore[misc]
                typer.echo(" ".join(str(x) for x in a))

        def on_event(kind: str, data: dict) -> None:
            if kind == "iteration":
                _echo(f"\n[dim]— step {data['n']} —[/dim]" if _rich else f"\n— step {data['n']} —")
            elif kind == "thought" and data.get("text"):
                _echo(f"[cyan]{data['text']}[/cyan]" if _rich else data["text"])
            elif kind == "tool_call":
                _echo(
                    f"[yellow]→ {data['name']}[/yellow] {data.get('args', {})}"
                    if _rich
                    else f"→ {data['name']} {data.get('args', {})}"
                )
            elif kind == "tool_result":
                mark = "ok" if data.get("success") else "ERR"
                snippet = (data.get("output", "") or "").replace("\n", " ")[:200]
                _echo(
                    f"[green]  ✓[/green] {snippet}"
                    if data.get("success") and _rich
                    else f"  [{mark}] {snippet}"
                )
            elif kind == "error":
                msg = data.get("message", "")
                _echo(f"[red]! {msg}[/red]" if _rich else f"! {msg}")

        import sys

        from isaac.agents.agent_loop import build_default_agent
        from isaac.agents.trace import TraceStore
        from isaac.tools import register_all_tools

        register_all_tools()
        only = [t.strip() for t in tools_csv.split(",") if t.strip()] or None

        # Real human-in-the-loop approval for risk-4/5 tools when a terminal
        # is attached (instead of all-or-nothing blocking).
        approval_callback = None
        if not auto_approve and sys.stdin.isatty():

            def approval_callback(name: str, args: dict, risk: int) -> bool:
                return typer.confirm(f"Allow high-risk tool '{name}' (risk {risk}) with {args}?")

        try:
            trace_store: TraceStore | None = TraceStore()
        except Exception:
            trace_store = None

        loop = build_default_agent(
            max_iterations=max_iters,
            auto_approve=auto_approve,
            on_event=on_event,
            only=only,
            max_wall_seconds=max_seconds,
            approval_callback=approval_callback,
            trace_store=trace_store,
        )
        result = loop.run(task)

        if _rich and console is not None:
            console.print(
                Panel(
                    result.output or "(no output)",
                    title=f"Result  ·  {result.iterations} steps  ·  "
                    f"{len(result.tool_calls)} tool calls  ·  {result.stopped_reason}",
                    border_style="green" if result.success else "yellow",
                )
            )
        else:
            typer.echo("\n=== RESULT ===")
            typer.echo(result.output or "(no output)")
        raise typer.Exit(0 if result.success else 1)

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

    @app.command(name="mcp-serve")
    def mcp_serve(
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    ) -> None:
        """Run the MCP stdio server — expose I.S.A.A.C. as a Claude tool provider.

        Add to Claude Code via:
            claude mcp add isaac -- isaac mcp-serve
        """
        _setup_logging(verbose)
        from isaac.mcp.server import run_server

        run_server()

    @app.command()
    def learn(
        task_type: str = typer.Option("", "--type", "-t", help="Filter by task type."),
        failures: bool = typer.Option(False, "--failures", "-f", help="Show failure analysis."),
    ) -> None:
        """Show self-improvement statistics from the MetaLearner."""
        _setup_logging()
        import json

        from isaac.meta.learner import get_learner

        learner = get_learner()
        if failures:
            data = learner.analyse_failures()
        else:
            data = learner.get_stats(task_type=task_type or None)
        typer.echo(json.dumps(data, indent=2))

    @app.command()
    def transcribe(
        audio_file: str = typer.Argument(..., help="Path to audio file (.wav/.mp3/etc.)"),
        model: str = typer.Option("base", "--model", "-m", help="Whisper model size."),
        language: str | None = typer.Option(
            None, "--lang", "-l", help="Language code (e.g. 'en')."
        ),
    ) -> None:
        """Transcribe an audio file to text using local Whisper (faster-whisper)."""
        _setup_logging()
        from isaac.multimodal.audio import transcribe as do_transcribe

        text = do_transcribe(audio_file, model=model, language=language)  # type: ignore[arg-type]
        typer.echo(text)

    @app.command()
    def speak(
        text: str = typer.Argument(..., help="Text to synthesise."),
        output: str | None = typer.Option(
            None, "--out", "-o", help="Save to file instead of playing."
        ),
        engine: str = typer.Option(
            "auto", "--engine", "-e", help="TTS engine: pyttsx3/kokoro/openai/auto."
        ),
    ) -> None:
        """Convert text to speech using local TTS (pyttsx3/kokoro/openai)."""
        _setup_logging()
        from isaac.multimodal.audio import speak as do_speak

        result = do_speak(text, output_path=output, engine=engine)  # type: ignore[arg-type]
        if result:
            typer.echo(f"Saved to: {result}")

    @app.command()
    def extract(
        doc_path: str = typer.Argument(..., help="Document path (.pdf/.docx/.pptx/.png/etc.)"),
        pages_only: bool = typer.Option(
            False, "--pages", "-p", help="Show page-by-page (PDF only)."
        ),
    ) -> None:
        """Extract text from a document (PDF, DOCX, PPTX, image)."""
        _setup_logging()
        from isaac.multimodal.document import extract_pages, extract_text

        if pages_only:
            pages = extract_pages(doc_path)
            for i, page in enumerate(pages, 1):
                typer.echo(f"\n--- Page {i} ---\n{page}")
        else:
            typer.echo(extract_text(doc_path))

    @app.command()
    def prove(
        constraints_json: str = typer.Argument(..., help="JSON array of constraint strings."),
        variables_json: str = typer.Option("{}", "--vars", "-v", help="JSON object: {name: sort}."),
    ) -> None:
        """Check satisfiability of symbolic constraints using Z3."""
        _setup_logging()
        import json

        from isaac.reasoning.theorem_prover import TheoremProver

        constraints = json.loads(constraints_json)
        variables = json.loads(variables_json)
        prover = TheoremProver()
        result = prover.check_sat(constraints, variables or None)
        typer.echo(json.dumps(result, indent=2))

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
                f"  {tool.name:20s} risk={tool.risk_level}  "
                f"{tool.description[:60]}{approval}{sandbox}"
            )

    @app.command()
    def cron(
        action: str = typer.Argument(
            "list", help="Action: list, add, remove, pause, resume, start, stop, status."
        ),
        name: str = typer.Option("", "--name", help="Task name (for add)."),
        schedule: str = typer.Option(
            "0 * * * *", "--schedule", "-s", help="Cron expression (for add)."
        ),
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
            typer.echo(
                f"Tasks: {len(tasks)} total, {sum(1 for t in tasks if t['enabled'])} enabled"
            )
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
    def voice(
        hands_free: bool = typer.Option(
            False, "--hands-free", "-f", help="Continuous listening mode."
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ) -> None:
        """Start the conversational voice REPL (mic ↔ STT ↔ agent ↔ TTS ↔ speaker)."""
        _setup_logging(verbose)
        try:
            from isaac.tools import register_all_tools

            register_all_tools()
        except Exception:
            pass
        from isaac.interfaces.voice_repl import run_voice_repl

        code = run_voice_repl(hands_free=hands_free)
        raise typer.Exit(code)

    @app.command()
    def vision(
        image: str = typer.Argument(..., help="Path or URL of the image to analyse."),
        prompt: str = typer.Option(
            "Describe this image in detail.",
            "--prompt",
            "-p",
            help="Question to ask about the image.",
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ) -> None:
        """Ask the local vision-language model a question about an image."""
        _setup_logging(verbose)
        from isaac.multimodal.vision.vision_lm import get_vision_lm

        try:
            answer = get_vision_lm().ask(prompt, image)
        except Exception as exc:
            typer.echo(f"Vision call failed: {exc}", err=True)
            raise typer.Exit(1) from exc
        typer.echo(answer)

    @app.command()
    def improve(
        report: bool = typer.Option(
            False, "--report", "-r", help="Print the last critique report."
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ) -> None:
        """Run one self-improvement cycle (curation + critique + telemetry prune)."""
        _setup_logging(verbose)
        from isaac.improvement import run_improvement_cycle

        result = run_improvement_cycle()
        promoted = sum(1 for d in result.curation_decisions if d.get("action") == "promote")
        deprecated = sum(1 for d in result.curation_decisions if d.get("action") == "deprecate")
        typer.echo(
            f"Improvement cycle complete in {result.finished_at - result.started_at:.1f}s — "
            f"promoted={promoted}, deprecated={deprecated}, pruned_rows={result.pruned_rows}"
        )
        if result.critique_summary:
            typer.echo(f"\nCritique: {result.critique_summary}")
        if result.critique_action:
            typer.echo(f"Suggested action: {result.critique_action}")
        if result.errors:
            typer.echo(f"\nErrors during cycle: {result.errors}")
        if report and result.curation_decisions:
            typer.echo("\nCuration decisions:")
            for d in result.curation_decisions:
                typer.echo(
                    f"  - {d['action']:10s} {d['skill_name']}  "
                    f"(runs={d['runs']}, sr={d['success_rate']:.2f})"
                )

    @app.command()
    def trace(
        run_id: str = typer.Argument("", help="Run ID to inspect (blank = list recent runs)."),
        last: int = typer.Option(20, "--last", "-n", help="How many recent runs to list."),
    ) -> None:
        """Inspect persisted agent run traces (see also: isaac agent)."""
        _setup_logging()
        import json as _json
        from datetime import datetime as _dt

        from isaac.agents.trace import TraceStore

        store = TraceStore()
        if not run_id:
            runs = store.recent_runs(last)
            if not runs:
                typer.echo('No agent runs recorded yet. Run: isaac agent "<task>"')
                return
            for r in runs:
                date = _dt.fromtimestamp(r["started_at"]).strftime("%Y-%m-%d %H:%M")
                typer.echo(
                    f"  {r['run_id']}  {date}  [{r['stopped_reason'] or 'running'}] "
                    f"iters={r['iterations']}  {r['task'][:60]}"
                )
            return

        events = store.run_events(run_id)
        if not events:
            typer.echo(f"No trace found for run '{run_id}'.")
            raise typer.Exit(1)
        for e in events:
            ts = _dt.fromtimestamp(e["ts"]).strftime("%H:%M:%S")
            try:
                data = _json.loads(e["data_json"])
            except Exception:
                data = {}
            detail = ", ".join(f"{k}={str(v)[:80]}" for k, v in data.items())
            typer.echo(f"  {e['seq']:>3} {ts}  {e['kind']:<12s} {detail}")

    @app.command(name="eval")
    def eval_cmd(
        suite: str = typer.Argument(
            "", help="Path to a JSONL task suite (e.g. evals/golden_v1.jsonl)."
        ),
        report: bool = typer.Option(
            False, "--report", "-r", help="Show recent recorded runs instead of running."
        ),
        fmt: str = typer.Option(
            "jsonl", "--format", "-f", help="Suite format: 'jsonl' (native), 'gaia', or 'arc'."
        ),
        level: int = typer.Option(1, "--level", help="GAIA difficulty level (gaia format only)."),
        download: bool = typer.Option(
            False,
            "--download",
            help="Download the benchmark dataset first (GAIA needs HF auth; ARC is ungated).",
        ),
        limit: int = typer.Option(0, "--limit", "-n", help="Run only the first N tasks."),
        task_id: str = typer.Option("", "--task", help="Run only the task with this id."),
        auto_approve: bool = typer.Option(
            False, "--auto-approve", "-y", help="Allow high-risk tools during eval runs."
        ),
        db: str = typer.Option("", "--db", help="Results DB path (default ~/.isaac/eval.db)."),
        no_store: bool = typer.Option(False, "--no-store", help="Do not persist this run."),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ) -> None:
        """Run a task suite against the agent and score it (reproducible evals).

        Loads a JSONL suite, runs each task through the AgentLoop (or the
        specialist team for ``runner: team`` tasks), scores answers with
        programmatic checkers, and records the run to a SQLite results DB.

        Example::

            isaac eval evals/golden_v1.jsonl
            isaac eval --report
        """
        _setup_logging(verbose)
        from pathlib import Path as _Path

        from isaac.config.settings import get_settings
        from isaac.eval import EvalStore, load_suite, run_suite
        from isaac.eval.report import format_recent, format_summary
        from isaac.eval.runner import default_runner

        db_path = _Path(db) if db else get_settings().isaac_home / "eval.db"
        store = EvalStore(db_path)

        if report and not suite:
            typer.echo(format_recent(store))
            return

        suite_label: str
        if fmt == "gaia":
            from isaac.eval.gaia import download_gaia, load_gaia_tasks

            if download:
                typer.echo("Downloading GAIA validation split from Hugging Face ...")
                suite = str(download_gaia(suite or None))
                typer.echo(f"Downloaded to {suite}")
            if not suite:
                typer.echo(
                    "Provide the GAIA split directory (containing metadata.jsonl) "
                    "or add --download."
                )
                raise typer.Exit(2)
            tasks = load_gaia_tasks(suite, level=level)
            suite_label = f"gaia-2023-l{level}-validation"
        elif fmt == "arc":
            from isaac.eval.arc import download_arc, load_arc_tasks

            if download:
                typer.echo("Downloading ARC-AGI-1 (public, ungated) from GitHub ...")
                suite = str(download_arc(suite or None))
                typer.echo(f"Downloaded to {suite}")
            if not suite:
                typer.echo(
                    "Provide the ARC split directory (containing *.json tasks) or add --download."
                )
                raise typer.Exit(2)
            tasks = load_arc_tasks(suite)
            suite_label = f"arc-agi-1-{_Path(suite).name}"
        else:
            if not suite:
                typer.echo("Provide a suite path (e.g. evals/golden_v1.jsonl) or --report.")
                raise typer.Exit(2)
            tasks = load_suite(suite)
            suite_label = _Path(suite).stem
        if task_id:
            tasks = [t for t in tasks if t.id == task_id]
            if not tasks:
                typer.echo(f"No task with id '{task_id}' in {suite}.")
                raise typer.Exit(2)
        if limit > 0:
            tasks = tasks[:limit]

        run_kwargs: dict = {}
        if fmt == "arc":
            # Symbolic solver run — no agent tools, no LLM; record it as such.
            from isaac.eval.arc import arc_runner

            runner = arc_runner(suite)
            run_kwargs = {"model": "arc-synthesis (symbolic, no LLM)", "provider": "none"}
        else:
            from isaac.tools import register_all_tools

            register_all_tools()
            runner = default_runner(auto_approve=auto_approve)

        def on_event(kind: str, data: dict) -> None:
            if kind == "task_start":
                typer.echo(f"[{data['n']}/{data['total']}] {data['task_id']} ...")
            elif kind == "task_done":
                mark = "PASS" if data["passed"] else "FAIL"
                typer.echo(f"    {mark}  ({data['duration_ms']:.0f}ms)")

        summary = run_suite(
            tasks,
            runner,
            suite_name=suite_label,
            store=None if no_store else store,
            on_event=on_event,
            **run_kwargs,
        )
        typer.echo("\n" + format_summary(summary))
        raise typer.Exit(0 if summary.accuracy == 1.0 else 1)

    @app.command()
    def doctor() -> None:
        """Preflight check: Python, settings, Ollama, Docker, and optional extras.

        Exits non-zero only when a *core* requirement is broken; missing
        optional capabilities are reported as warnings with the fix.
        """
        _setup_logging()
        from isaac.doctor import has_failures, run_checks

        results = run_checks()
        marks = {"ok": "✓", "warn": "!", "fail": "✗"}
        for r in results:
            typer.echo(f"  [{marks.get(r.status, '?')}] {r.name:18s} {r.detail}")
        if has_failures(results):
            typer.echo("\nCore checks failed — fix the items marked ✗ above.")
            raise typer.Exit(1)
        typer.echo("\nAll core checks passed.")

    @app.command()
    def models() -> None:
        """List available providers and detect locally-installed models."""
        _setup_logging()
        from isaac.config.settings import settings
        from isaac.llm.providers import LOCAL_PROVIDERS, PROVIDERS
        from isaac.llm.providers.ollama import health_check
        from isaac.llm.providers.ollama import list_models as list_ollama_models

        typer.echo("Registered providers:")
        for name in sorted(PROVIDERS):
            tag = "local" if name in LOCAL_PROVIDERS else "cloud"
            typer.echo(f"  - {name:15s} [{tag}]")

        typer.echo("\nOllama:")
        if health_check(settings.ollama_base_url):
            tags = list_ollama_models(settings.ollama_base_url)
            typer.echo(f"  reachable at {settings.ollama_base_url}")
            for t in tags:
                typer.echo(f"    - {t}")
        else:
            typer.echo(f"  not reachable at {settings.ollama_base_url}")

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

    @app.command()
    def team(
        goal: str = typer.Argument(..., help="The high-level goal for the specialist team."),
        auto_approve: bool = typer.Option(
            False,
            "--auto-approve",
            "-y",
            help="Allow high-risk (4-5) tool actions without approval.",
        ),
        max_workers: int = typer.Option(
            4, "--workers", "-w", help="Max specialists to run in parallel."
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v"),
    ) -> None:
        """Decompose a goal and dispatch it to the specialist team (orchestrator).

        The manager LLM breaks the goal into subtasks, assigns each to the best
        specialist (coder, file_organizer, researcher, designer, operator, …),
        runs independent subtasks in parallel, and synthesises one final answer.

        Example::

            isaac team "Research the top 3 local vector DBs and write a comparison to compare.md"
        """
        _setup_logging(verbose)
        from isaac.specialists import Orchestrator
        from isaac.tools import register_all_tools

        register_all_tools()

        def on_event(kind: str, data: dict) -> None:
            if kind == "plan":
                typer.echo("Plan:")
                for s in data.get("plan", []):
                    dep = f"  <- {', '.join(s['depends_on'])}" if s.get("depends_on") else ""
                    typer.echo(f"  [{s['id']}] {s['specialist']}: {s['description']}{dep}")
            elif kind == "subtask_start":
                typer.echo(f"\n> {data['specialist']} ({data['id']}): {data['description']}")
            elif kind == "subtask_done":
                mark = "ok" if data.get("success") else "ERR"
                typer.echo(f"  [{mark}]")

        result = Orchestrator(
            max_workers=max_workers, auto_approve=auto_approve, on_event=on_event
        ).run(goal)
        typer.echo("\n=== RESULT ===")
        typer.echo(result.final_output or "(no output)")
        raise typer.Exit(0 if result.success else 1)

    @app.command()
    def specialists() -> None:
        """List the specialist team and the tools each one is allowed to use."""
        _setup_logging()
        from isaac.specialists import list_specialists

        for c in list_specialists():
            tools = "all" if c["tools"] is None else (", ".join(c["tools"]) or "none")
            typer.echo(f"  {c['name']:15s} {c['title']:24s} risk<={c['max_risk']}")
            typer.echo(f"      {c['domain']}")
            typer.echo(f"      tools: {tools}")

    @app.command()
    def persona(
        action: str = typer.Argument(
            "list", help="Action: list, show, new, activate, delete, examples."
        ),
        slug: str = typer.Option("", "--slug", "-s", help="Persona slug."),
    ) -> None:
        """Create, list, and activate user-built personas (custom agent identities)."""
        _setup_logging()
        import json as _json

        from isaac.identity import persona_builder as pb

        if action == "list":
            active = pb.active_persona()
            names = pb.list_personas()
            if not names:
                typer.echo("No personas yet. Try: isaac persona examples  (or)  isaac persona new")
                return
            for s in names:
                typer.echo(f"{'* ' if s == active else '  '}{s}")
        elif action == "examples":
            saved = pb.install_examples()
            typer.echo(f"Installed examples: {', '.join(saved) or '(none new)'}")
        elif action == "show":
            if not slug:
                typer.echo("Provide --slug.")
                raise typer.Exit(1)
            typer.echo(_json.dumps(pb.load_persona(slug), indent=2, ensure_ascii=False))
        elif action == "activate":
            if not slug:
                typer.echo("Provide --slug.")
                raise typer.Exit(1)
            p = pb.activate_persona(slug)
            typer.echo(f"Activated persona '{slug}' ({p.get('name')}).")
            typer.echo(f"Set ISAAC_SOUL_PATH={pb.active_soul_path()} in .env to persist it.")
        elif action == "delete":
            if not slug:
                typer.echo("Provide --slug.")
                raise typer.Exit(1)
            typer.echo("Deleted." if pb.delete_persona(slug) else "Not found.")
        elif action == "new":
            answers: dict[str, str] = {}
            for q in pb.interactive_questions():
                val = typer.prompt(q["prompt"], default=None if q["required"] else "")
                if val:
                    answers[q["key"]] = val
            s, p = pb.create_and_activate(answers, activate=True)
            typer.echo(f"Created and activated persona '{s}' ({p['name']}).")
            typer.echo(f"Set ISAAC_SOUL_PATH={pb.active_soul_path()} in .env to persist it.")
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
