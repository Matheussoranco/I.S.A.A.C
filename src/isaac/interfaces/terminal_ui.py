"""Rich Terminal UI — beautiful CLI rendering for I.S.A.A.C.

Provides a ``TerminalUI`` helper that encapsulates all Rich console
operations:  banners, panels, spinners, streaming token output,
code-block highlighting, and node-progress indicators.

Inspired by Cline and Claude Code terminal interfaces.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

ISAAC_THEME = Theme(
    {
        "isaac.name": "bold cyan",
        "isaac.dim": "dim white",
        "isaac.success": "bold green",
        "isaac.error": "bold red",
        "isaac.warning": "bold yellow",
        "isaac.info": "bold blue",
        "isaac.node": "bold magenta",
        "isaac.user": "bold white",
        "isaac.token": "cyan",
        "isaac.hint": "dim italic",
        "isaac.phase": "bold bright_cyan",
        "isaac.border": "bright_cyan",
    }
)

# Phase display metadata
PHASE_ICONS: dict[str, tuple[str, str]] = {
    "guard":              ("\U0001f6e1",  "Guard"),           # 🛡
    "perception":         ("\U0001f441",  "Perception"),      # 👁
    "direct_response":    ("\u26a1",      "Direct Response"), # ⚡
    "explorer":           ("\U0001f50d",  "Explorer"),        # 🔍
    "planner":            ("\U0001f4cb",  "Planner"),         # 📋
    "connector_execution":("\U0001f517",  "Connectors"),      # 🔗
    "synthesis":          ("\U0001f9ea",  "Synthesis"),       # 🧪
    "sandbox":            ("\U0001f4e6",  "Sandbox"),         # 📦
    "computer_use":       ("\U0001f5a5",  "Computer Use"),    # 🖥
    "reflection":         ("\U0001f914",  "Reflection"),      # 🤔
    "skill_abstraction":  ("\U0001f4be",  "Skill Save"),      # 💾
}


class TerminalUI:
    """Encapsulates all Rich-based rendering for the I.S.A.A.C. REPL."""

    def __init__(self) -> None:
        self.console = Console(theme=ISAAC_THEME, highlight=False)
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def print_banner(self) -> None:
        """Print the startup banner."""
        logo = Text()
        logo.append("  ___ ", style="bold cyan")
        logo.append(" ____  ", style="bold cyan")
        logo.append("  _    ", style="bold cyan")
        logo.append("  _    ", style="bold cyan")
        logo.append("  ___\n", style="bold cyan")
        logo.append(" |_ _|", style="bold cyan")
        logo.append("|  __| ", style="bold cyan")
        logo.append(" / \\  ", style="bold cyan")
        logo.append(" / \\  ", style="bold cyan")
        logo.append(" / __|\n", style="bold cyan")
        logo.append("  | | ", style="bold cyan")
        logo.append(" \\__ \\ ", style="bold cyan")
        logo.append("/ _ \\ ", style="bold cyan")
        logo.append("/ _ \\ ", style="bold cyan")
        logo.append("| (__ \n", style="bold cyan")
        logo.append(" |___|", style="bold cyan")
        logo.append("|____/", style="bold cyan")
        logo.append("/_/ \\_\\", style="bold cyan")
        logo.append("/_/ \\_\\", style="bold cyan")
        logo.append(" \\___|\n", style="bold cyan")

        subtitle = Text(
            "Intelligent System for Autonomous Action and Cognition",
            style="italic bright_white",
        )
        version_text = Text("v0.2.0", style="dim cyan")
        combined = Text.assemble(subtitle, "  ", version_text)

        banner_group = Group(logo, combined)

        self.console.print()
        self.console.print(
            Panel(
                banner_group,
                border_style="bright_cyan",
                padding=(0, 2),
            )
        )
        self.console.print()
        self.console.print(
            "  [isaac.hint]Type your task below.  "
            "Commands: /help, /clear, /status, /exit[/isaac.hint]"
        )
        self.console.print()

    # ------------------------------------------------------------------
    # Input prompt
    # ------------------------------------------------------------------

    def get_prompt_tokens(self) -> list[tuple[str, str]]:
        """Return styled prompt tokens for prompt_toolkit."""
        return [
            ("class:prompt_bracket", "\u276f "),  # ❯
        ]

    # ------------------------------------------------------------------
    # Node progress
    # ------------------------------------------------------------------

    def start_thinking(self) -> None:
        """Record the start timestamp for latency tracking."""
        self._start_time = time.monotonic()

    def print_phase(self, phase: str) -> None:
        """Print a phase-transition indicator."""
        icon, label = PHASE_ICONS.get(phase, ("\u2022", phase.replace("_", " ").title()))
        self.console.print(f"  [isaac.dim]{icon}[/isaac.dim]  [isaac.node]{label}[/isaac.node]")

    def print_thinking(self) -> None:
        """Print a lightweight 'thinking...' line."""
        self.console.print("  [isaac.dim]...[/isaac.dim]", end="")

    def clear_line(self) -> None:
        """Overwrite the current line (for clearing spinners)."""
        self.console.print("\r" + " " * 60 + "\r", end="")

    # ------------------------------------------------------------------
    # Response rendering
    # ------------------------------------------------------------------

    def print_assistant_response(self, text: str) -> None:
        """Render an assistant response inside a styled panel with Markdown."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        elapsed_str = f"{elapsed:.1f}s" if elapsed else ""

        content = Markdown(text)

        title = Text.assemble(
            (" I.S.A.A.C. ", Style(bold=True, color="bright_cyan")),
        )
        subtitle = Text(elapsed_str, style="dim") if elapsed_str else None

        self.console.print()
        self.console.print(
            Panel(
                content,
                title=title,
                subtitle=subtitle,
                subtitle_align="right",
                border_style="bright_cyan",
                padding=(1, 2),
            )
        )
        self.console.print()

    def stream_token(self, token: str) -> None:
        """Write a single streaming token (no newline, no panel — raw output)."""
        self.console.print(token, end="", style="isaac.token", highlight=False)

    def start_stream(self) -> None:
        """Begin a streaming response block."""
        self.console.print()
        title = Text(" I.S.A.A.C. ", style="bold bright_cyan")
        self.console.print(
            Rule(title=title, style="bright_cyan")
        )
        self.console.print()

    def end_stream(self, elapsed: float | None = None) -> None:
        """End a streaming response block."""
        if elapsed is None and self._start_time:
            elapsed = time.monotonic() - self._start_time
        suffix = Text(f" {elapsed:.1f}s ", style="dim") if elapsed else Text("")
        self.console.print()
        self.console.print(Rule(title=suffix, style="dim bright_cyan"))
        self.console.print()

    # ------------------------------------------------------------------
    # Execution summary
    # ------------------------------------------------------------------

    def print_execution_summary(
        self,
        exit_code: int,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        """Render sandbox execution results."""
        status_style = "isaac.success" if exit_code == 0 else "isaac.error"
        status_icon = "\u2714" if exit_code == 0 else "\u2718"  # ✔ / ✘

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="dim", width=12)
        table.add_column()

        table.add_row("exit_code", Text(f"{status_icon} {exit_code}", style=status_style))

        if stdout.strip():
            # Try to detect and highlight code
            display = stdout.strip()[:500]
            table.add_row("stdout", Text(display, style="white"))

        if stderr.strip():
            display = stderr.strip()[:300]
            table.add_row("stderr", Text(display, style="isaac.warning"))

        self.console.print(
            Panel(table, title="[dim]Execution[/dim]", border_style="dim", padding=(0, 1))
        )

    def print_ui_summary(
        self,
        actions_count: int,
        ui_cycle: int,
        last_action_success: bool,
        last_action_type: str,
        last_action_desc: str,
    ) -> None:
        """Render UI-mode execution results."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="dim", width=14)
        table.add_column()

        table.add_row("actions", str(actions_count))
        table.add_row("ui_cycle", str(ui_cycle))
        status = "[isaac.success]OK[/isaac.success]" if last_action_success else "[isaac.error]FAIL[/isaac.error]"
        table.add_row("last_action", f"{status}  {last_action_type} - {last_action_desc[:60]}")

        self.console.print(
            Panel(table, title="[dim]UI Actions[/dim]", border_style="dim", padding=(0, 1))
        )

    def print_mode_badge(self, mode: str, phase: str) -> None:
        """Print a small mode/phase badge after execution."""
        mode_colors = {
            "direct": "green",
            "code": "blue",
            "computer_use": "yellow",
            "hybrid": "magenta",
        }
        color = mode_colors.get(mode, "white")
        self.console.print(
            f"  [{color}]{mode}[/{color}] [dim]| {phase}[/dim]",
        )

    # ------------------------------------------------------------------
    # Errors & warnings
    # ------------------------------------------------------------------

    def print_error(self, message: str) -> None:
        """Render an error message."""
        self.console.print(
            Panel(
                Text(message, style="red"),
                title="[bold red]Error[/bold red]",
                border_style="red",
                padding=(0, 2),
            )
        )

    def print_warning(self, message: str) -> None:
        """Render a warning message."""
        self.console.print(f"  [isaac.warning]\u26a0 {message}[/isaac.warning]")

    def print_info(self, message: str) -> None:
        """Render an informational message."""
        self.console.print(f"  [isaac.info]{message}[/isaac.info]")

    # ------------------------------------------------------------------
    # Slash-command helpers
    # ------------------------------------------------------------------

    def print_help(self) -> None:
        """Print the slash-command help table."""
        table = Table(title="Commands", border_style="dim", show_header=True, header_style="bold")
        table.add_column("Command", style="cyan")
        table.add_column("Description")

        table.add_row("/help", "Show this help")
        table.add_row("/clear", "Clear the terminal")
        table.add_row("/status", "Show system status")
        table.add_row("/compact", "Toggle compact output mode")
        table.add_row("/exit, /quit", "Exit I.S.A.A.C.")

        self.console.print(table)

    def print_status(
        self,
        model: str = "unknown",
        tools_count: int = 0,
        memory_ok: bool = False,
        scheduler_ok: bool = False,
    ) -> None:
        """Print system status summary."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="dim", width=14)
        table.add_column()

        table.add_row("model", model)
        table.add_row("tools", str(tools_count))
        ok = "[isaac.success]OK[/isaac.success]"
        fail = "[isaac.error]DOWN[/isaac.error]"
        table.add_row("memory", ok if memory_ok else fail)
        table.add_row("scheduler", ok if scheduler_ok else fail)

        self.console.print(
            Panel(table, title="[bold]System Status[/bold]", border_style="bright_cyan")
        )

    def clear(self) -> None:
        """Clear the console."""
        self.console.clear()

    def print_goodbye(self) -> None:
        """Print the exit message."""
        self.console.print("\n  [isaac.dim]Shutting down. Goodbye.[/isaac.dim]\n")
