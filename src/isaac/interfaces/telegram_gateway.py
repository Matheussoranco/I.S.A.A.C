"""Telegram Gateway — bidirectional operator interface via python-telegram-bot v21+.

Commands
--------
/start    — welcome message
/status   — current agent status
/tasks    — show pending plan steps
/memory   — show recent memory context

Only whitelisted user IDs (``TELEGRAM_ALLOWED_USERS``) can interact.
All other users receive a "not authorised" reply.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from isaac.interfaces.gateway_base import AgentRunner, Gateway

logger = logging.getLogger(__name__)

# Module-level reference to the running Application — filled by ``start_bot``.
_application: Any | None = None


def send_message_to_all(text: str) -> int:
    """Send *text* to every allowed Telegram user; return the delivery count.

    Raises :class:`RuntimeError` when Telegram is not configured or every
    recipient failed, so background jobs can record a real delivery failure.
    """
    from isaac.config.settings import get_settings, parse_id_list

    settings = get_settings()
    token = settings.telegram_bot_token
    recipients = parse_id_list(settings.telegram_allowed_users)
    if not token or not recipients:
        raise RuntimeError(
            "Telegram is not configured. Set ISAAC_TELEGRAM_BOT_TOKEN and "
            "ISAAC_TELEGRAM_ALLOWED_USERS."
        )

    import json
    import urllib.request

    sent = 0
    errors: list[str] = []
    for user_id in recipients:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({"chat_id": user_id, "text": text[:4096]}).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                raise RuntimeError("Telegram did not confirm delivery")
            sent += 1
        except Exception as exc:
            errors.append(f"{user_id}: {exc}")
    if sent == 0:
        raise RuntimeError("Telegram delivery failed: " + "; ".join(errors))
    return sent


def send_notification(text: str) -> None:
    """Send a message to all allowed Telegram users (best-effort, fire-and-forget)."""
    try:
        send_message_to_all(text)
    except Exception as exc:
        logger.debug("Telegram notification error: %s", exc)


@Gateway.register("telegram")
class TelegramGateway(Gateway):
    """Telegram gateway built on python-telegram-bot v21+."""

    def __init__(self, agent_runner: AgentRunner | None = None) -> None:
        super().__init__(agent_runner)
        self._application: Any | None = None

    @property
    def name(self) -> str:
        return "telegram"

    async def send_message(self, channel_id: str, text: str) -> None:
        """Send a message to a Telegram chat, splitting long replies."""
        if self._application is None:
            raise RuntimeError("Telegram gateway is not running.")
        chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)] or [""]
        for chunk in chunks:
            await self._application.bot.send_message(chat_id=channel_id, text=chunk)

    async def start(self) -> None:
        """Start the Telegram bot polling loop until stopped."""
        global _application

        try:
            from telegram import Update
            from telegram.ext import (
                Application,
                CommandHandler,
                ContextTypes,
                MessageHandler,
                filters,
            )
        except ImportError:
            logger.warning("python-telegram-bot not installed — Telegram gateway disabled.")
            return

        from isaac.config.settings import get_settings, parse_id_list

        settings = get_settings()
        token = settings.telegram_bot_token
        allowed_users = set(parse_id_list(settings.telegram_allowed_users))

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
                "🤖 I.S.A.A.C. Telegram Gateway\n\nCommands: /status /tasks /memory"
            )

        async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            assert update.effective_message is not None
            text = "📊 Online. Use /tasks to see the current plan."
            await update.effective_message.reply_text(text)

        async def cmd_tasks(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            assert update.effective_message is not None
            try:
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

        async def cmd_unauthorized(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            assert update.effective_message is not None
            await update.effective_message.reply_text("⛔ Not authorised.")

        async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
            message = update.effective_message
            user = message.from_user if message else None
            if message is None or not message.text or user is None:
                return
            await self._handle_incoming(str(message.chat_id), str(user.id), message.text)

        # ── Build application ─────────────────────────────────

        app = Application.builder().token(token).build()

        # Authorised handlers
        app.add_handler(CommandHandler("start", cmd_start, filters=allowed_filter))
        app.add_handler(CommandHandler("status", cmd_status, filters=allowed_filter))
        app.add_handler(CommandHandler("tasks", cmd_tasks, filters=allowed_filter))
        app.add_handler(CommandHandler("memory", cmd_memory, filters=allowed_filter))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND & allowed_filter, handle_message)
        )

        # Catch-all for non-authorised users
        app.add_handler(CommandHandler("start", cmd_unauthorized))
        app.add_handler(MessageHandler(filters.COMMAND & ~allowed_filter, cmd_unauthorized))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND & ~allowed_filter, cmd_unauthorized)
        )

        self._application = app
        _application = app
        self._running = True

        logger.info("Telegram gateway starting polling...")
        await app.initialize()
        await app.start()
        await app.updater.start_polling()  # type: ignore[union-attr]

        # Keep running until the application is stopped externally
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await app.updater.stop()  # type: ignore[union-attr]
            await app.stop()
            await app.shutdown()
            self._running = False

    async def stop(self) -> None:
        """Gracefully stop the Telegram bot if running."""
        global _application
        self._running = False
        app = self._application or _application
        if app is not None:
            try:
                if app.updater and app.updater.running:
                    await app.updater.stop()
                await app.stop()
                await app.shutdown()
            except Exception as exc:
                logger.debug("Error stopping Telegram bot: %s", exc)
            finally:
                self._application = None
                _application = None


async def start_bot(agent_runner: AgentRunner | None = None) -> None:
    """Start the Telegram bot in the background (thin wrapper).

    This coroutine is meant to be called from the scheduler or the
    main entry point.  It runs the ``python-telegram-bot`` polling
    loop until stopped.
    """
    await TelegramGateway(agent_runner=agent_runner).start()


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
