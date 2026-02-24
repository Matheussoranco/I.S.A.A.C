"""Telegram Gateway — bidirectional operator interface via python-telegram-bot v21+.

Commands
--------
/start    — welcome message
/status   — current agent status
/tasks    — show pending plan steps
/memory   — show recent memory context
/approve  — approve the pending approval
/reject   — reject the pending approval

Only whitelisted user IDs (``TELEGRAM_ALLOWED_USERS``) can interact.
All other users receive a "not authorised" reply.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level reference to the running Application — filled by ``start_bot``.
_application: Any | None = None

# Shared approval queue — approval node writes, Telegram handler resolves.
_pending_approvals: list[Any] = []


def set_pending_approvals(approvals: list[Any]) -> None:
    """Set the pending approvals list so Telegram handlers can resolve them."""
    global _pending_approvals
    _pending_approvals = approvals


def send_notification(text: str) -> None:
    """Send a message to all allowed Telegram users (best-effort, fire-and-forget)."""
    try:
        from isaac.config.settings import get_settings

        settings = get_settings()
        token = settings.telegram_bot_token
        allowed = settings.telegram_allowed_users

        if not token or not allowed:
            return

        import urllib.request
        import urllib.parse
        import json

        for user_id in allowed:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = json.dumps({"chat_id": user_id, "text": text}).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(req, timeout=5)
            except Exception as exc:
                logger.debug("Failed to send Telegram notification to %s: %s", user_id, exc)
    except Exception as exc:
        logger.debug("Telegram notification error: %s", exc)


async def start_bot() -> None:
    """Start the Telegram bot in the background.

    This coroutine is meant to be called from the scheduler or the
    main entry point.  It runs the ``python-telegram-bot`` polling
    loop until stopped.
    """
    global _application

    try:
        from telegram import Update
        from telegram.ext import (
            Application,
            CommandHandler,
            ContextTypes,
            filters,
        )
    except ImportError:
        logger.warning("python-telegram-bot not installed — Telegram gateway disabled.")
        return

    from isaac.config.settings import get_settings

    settings = get_settings()
    token = settings.telegram_bot_token
    allowed_users = set(settings.telegram_allowed_users)

    if not token:
        logger.info("TELEGRAM_BOT_TOKEN not set — Telegram gateway disabled.")
        return

    # ── Auth filter ────────────────────────────────────────

    class AllowedFilter(filters.BaseFilter):  # type: ignore[misc]
        """Allow only whitelisted user IDs."""

        def filter(self, message: Any) -> bool:
            if not message or not message.from_user:
                return False
            return str(message.from_user.id) in allowed_users

    allowed_filter = AllowedFilter()

    # ── Handlers ───────────────────────────────────────────

    async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_message is not None
        await update.effective_message.reply_text(
            "🤖 I.S.A.A.C. Telegram Gateway\n\n"
            "Commands: /status /tasks /memory /approve /reject"
        )

    async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_message is not None
        text = (
            "📊 Status\n"
            f"Pending approvals: {len(_pending_approvals)}\n"
            "Use /tasks to see current plan."
        )
        await update.effective_message.reply_text(text)

    async def cmd_tasks(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_message is not None
        try:
            from pathlib import Path
            from isaac.config.settings import get_settings

            tasks_file = get_settings().isaac_home / "TASKS.md"
            if tasks_file.exists():
                content = tasks_file.read_text(encoding="utf-8")[:3000]
                await update.effective_message.reply_text(f"📋 Tasks:\n{content}")
            else:
                await update.effective_message.reply_text("No TASKS.md found.")
        except Exception as exc:
            await update.effective_message.reply_text(f"Error reading tasks: {exc}")

    async def cmd_memory(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_message is not None
        try:
            from isaac.memory.manager import get_memory_manager

            mm = get_memory_manager()
            result = mm.recall("recent activity", k=3)
            text = result.combined_context[:3000] if result.combined_context else "No memories."
            await update.effective_message.reply_text(f"🧠 Memory:\n{text}")
        except Exception as exc:
            await update.effective_message.reply_text(f"Error: {exc}")

    async def cmd_approve(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_message is not None
        resolved = 0
        for a in _pending_approvals:
            if a.approved is None:
                a.approved = True
                a.resolved_by = f"telegram:{update.effective_user.id if update.effective_user else 'unknown'}"
                resolved += 1
        if resolved:
            await update.effective_message.reply_text(f"✅ Approved {resolved} pending action(s).")
        else:
            await update.effective_message.reply_text("No pending approvals.")

    async def cmd_reject(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_message is not None
        resolved = 0
        for a in _pending_approvals:
            if a.approved is None:
                a.approved = False
                a.resolved_by = f"telegram:{update.effective_user.id if update.effective_user else 'unknown'}"
                resolved += 1
        if resolved:
            await update.effective_message.reply_text(f"❌ Rejected {resolved} pending action(s).")
        else:
            await update.effective_message.reply_text("No pending approvals.")

    async def cmd_unauthorized(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_message is not None
        await update.effective_message.reply_text("⛔ Not authorised.")

    # ── Build application ─────────────────────────────────

    app = Application.builder().token(token).build()

    # Authorised handlers
    app.add_handler(CommandHandler("start", cmd_start, filters=allowed_filter))
    app.add_handler(CommandHandler("status", cmd_status, filters=allowed_filter))
    app.add_handler(CommandHandler("tasks", cmd_tasks, filters=allowed_filter))
    app.add_handler(CommandHandler("memory", cmd_memory, filters=allowed_filter))
    app.add_handler(CommandHandler("approve", cmd_approve, filters=allowed_filter))
    app.add_handler(CommandHandler("reject", cmd_reject, filters=allowed_filter))

    # Catch-all for non-authorised users
    app.add_handler(CommandHandler("start", cmd_unauthorized))

    _application = app

    logger.info("Telegram gateway starting polling...")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()  # type: ignore[union-attr]

    # Keep running until the application is stopped externally
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await app.updater.stop()  # type: ignore[union-attr]
        await app.stop()
        await app.shutdown()


async def stop_bot() -> None:
    """Gracefully stop the Telegram bot if running."""
    global _application
    if _application is not None:
        try:
            if _application.updater and _application.updater.running:
                await _application.updater.stop()
            await _application.stop()
            await _application.shutdown()
        except Exception as exc:
            logger.debug("Error stopping Telegram bot: %s", exc)
        finally:
            _application = None
