"""Local web interface for I.S.A.A.C.

The UI is intentionally a thin client over the existing :class:`AgentLoop`.
Agent and browser events are sent over one WebSocket, while high-risk approval
requests travel back through the same channel.  No API keys are ever exposed to
the browser.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from isaac.agents.agent_loop import AgentLoop, build_default_agent
from isaac.agents.computer_agent import ComputerAgentRunner
from isaac.agents.trace import TraceStore
from isaac.config.settings import get_settings
from isaac.interfaces.conversation_store import ConversationStore
from isaac.llm.provider import build_llm_for_profile
from isaac.security.credentials import credential_available, set_credential

_ASSETS = Path(__file__).with_name("web_assets")
_APP_TITLE = "I.S.A.A.C."
_APP_VERSION = "1.6.2"
_APPROVAL_TIMEOUT_SECONDS = 120.0
_MAX_HISTORY_CHARS = 40_000

AgentBuilder = Callable[..., AgentLoop]
ComputerRunnerBuilder = Callable[..., ComputerAgentRunner]
ProfileLLMBuilder = Callable[..., Any]


@dataclass
class _ApprovalWaiter:
    ready: threading.Event = field(default_factory=threading.Event)
    approved: bool = False


@dataclass
class UISession:
    """Thread-safe bridge between one browser tab and one agent run."""

    event_loop: asyncio.AbstractEventLoop
    store: ConversationStore
    conversation_id: str
    provider: str
    model: str
    reasoning_effort: str = "medium"
    events: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    history: list[dict[str, str]] = field(default_factory=list)
    run_task: asyncio.Task[None] | None = None
    approvals: dict[str, _ApprovalWaiter] = field(default_factory=dict)
    approval_lock: threading.Lock = field(default_factory=threading.Lock)

    async def emit(self, kind: str, **data: Any) -> None:
        await self.events.put({"type": kind, "data": data})

    def emit_threadsafe(self, kind: str, data: dict[str, Any]) -> None:
        """Queue an event from the worker thread running the agent."""
        try:
            self.event_loop.call_soon_threadsafe(
                self.events.put_nowait,
                {"type": kind, "data": data},
            )
        except RuntimeError:
            # The tab closed while a provider call was finishing.
            return

    def request_approval(self, name: str, args: dict[str, Any], risk: int) -> bool:
        request_id = uuid.uuid4().hex
        waiter = _ApprovalWaiter()
        with self.approval_lock:
            self.approvals[request_id] = waiter
        self.emit_threadsafe(
            "approval_request",
            {
                "request_id": request_id,
                "name": name,
                "args": args,
                "risk": risk,
                "timeout_seconds": _APPROVAL_TIMEOUT_SECONDS,
            },
        )
        waiter.ready.wait(_APPROVAL_TIMEOUT_SECONDS)
        with self.approval_lock:
            self.approvals.pop(request_id, None)
        return waiter.approved if waiter.ready.is_set() else False

    def resolve_approval(self, request_id: str, decision: str) -> bool:
        with self.approval_lock:
            waiter = self.approvals.get(request_id)
            if waiter is None:
                return False
            waiter.approved = decision == "approve_once"
            waiter.ready.set()
            return True

    def release_approvals(self) -> None:
        with self.approval_lock:
            for waiter in self.approvals.values():
                waiter.approved = False
                waiter.ready.set()

    def conversation_context(self) -> str:
        lines: list[str] = []
        for item in self.history:
            role = "User" if item["role"] == "user" else "Assistant"
            lines.append(f"{role}: {item['content']}")
        context = "\n\n".join(lines)
        return context[-_MAX_HISTORY_CHARS:]


def create_app(
    *,
    agent_builder: AgentBuilder = build_default_agent,
    computer_runner_builder: ComputerRunnerBuilder = ComputerAgentRunner,
    conversation_store: ConversationStore | None = None,
    llm_builder: ProfileLLMBuilder = build_llm_for_profile,
) -> FastAPI:
    """Build the local-only FastAPI application.

    ``agent_builder`` is injectable so the transport can be tested without a
    live model or provider credentials.
    """

    return _create_app(
        agent_builder=agent_builder,
        computer_runner_builder=computer_runner_builder,
        conversation_store=conversation_store,
        llm_builder=llm_builder,
    )


def _create_app(
    *,
    agent_builder: AgentBuilder = build_default_agent,
    computer_runner_builder: ComputerRunnerBuilder = ComputerAgentRunner,
    conversation_store: ConversationStore | None = None,
    llm_builder: ProfileLLMBuilder = build_llm_for_profile,
) -> FastAPI:
    settings = get_settings()
    store = conversation_store or ConversationStore(
        settings.isaac_home / "ui" / "conversations.sqlite3"
    )
    app = FastAPI(title=_APP_TITLE, version=_APP_VERSION, docs_url=None, redoc_url=None)
    app.mount("/assets", StaticFiles(directory=_ASSETS), name="assets")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(_ASSETS / "index.html")

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        settings = get_settings()
        model = settings.llm.strong_model or settings.llm.model_name
        return {
            "name": settings.agent_name,
            "provider": settings.llm.llm_provider,
            "model": model,
            "version": _APP_VERSION,
            "local_first": settings.local_first,
            "profiles": _model_profiles(settings),
            "openai_configured": credential_available("openai"),
            "anthropic_configured": credential_available("anthropic"),
        }

    @app.websocket("/ws")
    async def agent_socket(websocket: WebSocket) -> None:
        if not _same_origin_websocket(websocket):
            await websocket.close(code=1008, reason="Origin not allowed")
            return
        await websocket.accept()
        settings = get_settings()
        conversation_id = store.ensure_latest()
        session = UISession(
            asyncio.get_running_loop(),
            store=store,
            conversation_id=conversation_id,
            provider=settings.llm.llm_provider,
            model=settings.llm.strong_model or settings.llm.model_name,
            history=store.load(conversation_id),
        )
        await session.emit(
            "connected",
            status="ready",
            provider=session.provider,
            model=session.model,
            reasoning_effort=session.reasoning_effort,
        )
        await _emit_conversations(session)
        await session.emit(
            "history_loaded",
            conversation_id=session.conversation_id,
            messages=session.history,
        )

        sender = asyncio.create_task(_send_events(websocket, session))
        try:
            while True:
                payload = await websocket.receive_json()
                action = str(payload.get("action", ""))
                if action == "run":
                    prompt = str(payload.get("message", "")).strip()
                    if not prompt:
                        await session.emit("error", message="Digite uma mensagem para o agente.")
                        continue
                    if session.run_task is not None and not session.run_task.done():
                        await session.emit(
                            "error", message="O agente já está executando uma tarefa."
                        )
                        continue
                    session.cancelled.clear()
                    session.run_task = asyncio.create_task(
                        _run_agent(
                            session,
                            prompt,
                            agent_builder=agent_builder,
                            max_iterations=_bounded_int(payload.get("max_iterations"), 12, 1, 50),
                            tools=_clean_tool_names(payload.get("tools")),
                            mode=_clean_mode(payload.get("mode")),
                            computer_runner_builder=computer_runner_builder,
                            llm_builder=llm_builder,
                        )
                    )
                elif action == "cancel":
                    session.cancelled.set()
                    session.release_approvals()
                    await session.emit("cancelling", message="Cancelamento solicitado…")
                elif action == "approval":
                    request_id = str(payload.get("request_id", ""))
                    decision = str(payload.get("decision", ""))
                    if not decision:
                        decision = "approve_once" if payload.get("approved") else "deny"
                    resolved = session.resolve_approval(request_id, decision)
                    if not resolved:
                        await session.emit(
                            "error", message="Essa solicitação de aprovação já expirou."
                        )
                elif action == "configure":
                    if session.run_task is not None and not session.run_task.done():
                        await session.emit(
                            "error", message="Pare a tarefa antes de trocar o modelo."
                        )
                        continue
                    try:
                        session.provider = _clean_provider(payload.get("provider"))
                        session.model = _clean_model(payload.get("model"))
                        session.reasoning_effort = _clean_reasoning(payload.get("reasoning_effort"))
                        secret = str(payload.get("api_key", "")).strip()
                        if secret:
                            if session.provider not in {"openai", "anthropic"}:
                                raise ValueError(
                                    "Credenciais só podem ser salvas para OpenAI ou Anthropic."
                                )
                            set_credential(session.provider, secret)
                    except (ValueError, RuntimeError) as exc:
                        await session.emit("error", message=str(exc))
                        continue
                    await session.emit(
                        "configured",
                        provider=session.provider,
                        model=session.model,
                        reasoning_effort=session.reasoning_effort,
                        credential_configured=(
                            credential_available(session.provider)
                            if session.provider in {"openai", "anthropic"}
                            else False
                        ),
                    )
                elif action in {"clear", "new_chat"}:
                    if session.run_task is None or session.run_task.done():
                        session.conversation_id = store.create()
                        session.history = []
                        await _emit_conversations(session)
                        await session.emit(
                            "history_loaded",
                            conversation_id=session.conversation_id,
                            messages=[],
                        )
                elif action == "select_chat":
                    if session.run_task is not None and not session.run_task.done():
                        await session.emit(
                            "error", message="Pare a tarefa antes de trocar de conversa."
                        )
                        continue
                    conversation_id = str(payload.get("conversation_id", ""))
                    try:
                        history = store.load(conversation_id)
                    except KeyError:
                        await session.emit("error", message="Conversa não encontrada.")
                        continue
                    session.conversation_id = conversation_id
                    session.history = history
                    await session.emit(
                        "history_loaded",
                        conversation_id=conversation_id,
                        messages=history,
                    )
        except WebSocketDisconnect:
            pass
        finally:
            session.cancelled.set()
            session.release_approvals()
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender

    return app


async def _send_events(websocket: WebSocket, session: UISession) -> None:
    while True:
        event = await session.events.get()
        await websocket.send_json(event)


async def _run_agent(
    session: UISession,
    prompt: str,
    *,
    agent_builder: AgentBuilder,
    max_iterations: int,
    tools: list[str] | None,
    mode: str,
    computer_runner_builder: ComputerRunnerBuilder,
    llm_builder: ProfileLLMBuilder,
) -> None:
    context = session.conversation_context()
    session.history.append({"role": "user", "content": prompt})
    session.store.add(session.conversation_id, "user", prompt)
    await session.emit("run_started", message=prompt)
    await _emit_conversations(session)

    try:
        try:
            trace_store: TraceStore | None = TraceStore()
        except Exception:
            trace_store = None

        if mode == "computer" and session.provider == "openai":
            runner = computer_runner_builder(
                model=session.model,
                reasoning_effort=session.reasoning_effort,
                max_cycles=max_iterations,
                on_event=session.emit_threadsafe,
                approval_callback=session.request_approval,
                should_stop=session.cancelled.is_set,
            )
            result = await asyncio.to_thread(runner.run, prompt, context)
        else:
            llm = llm_builder(
                session.provider,
                session.model,
                reasoning_effort=session.reasoning_effort,
            )
            agent = agent_builder(
                llm=llm,
                max_iterations=max_iterations,
                max_wall_seconds=600.0,
                only=tools,
                on_event=session.emit_threadsafe,
                browser_event_callback=session.emit_threadsafe,
                desktop_event_callback=session.emit_threadsafe,
                approval_callback=session.request_approval,
                should_stop=session.cancelled.is_set,
                trace_store=trace_store,
            )
            result = await asyncio.to_thread(agent.run, prompt, context)
        session.history.append({"role": "assistant", "content": result.output})
        session.store.add(session.conversation_id, "assistant", result.output)
        await _emit_conversations(session)
        await session.emit(
            "run_complete",
            output=result.output,
            success=result.success,
            stopped_reason=result.stopped_reason,
            iterations=result.iterations,
            tool_calls=len(result.tool_calls),
        )
    except Exception as exc:
        await session.emit("error", message=f"Falha ao executar o agente: {exc}")
        await session.emit(
            "run_complete",
            output="",
            success=False,
            stopped_reason="error",
            iterations=0,
            tool_calls=0,
        )


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, number))


def _clean_tool_names(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    names = [str(item).strip() for item in value if str(item).strip()]
    return names or None


async def _emit_conversations(session: UISession) -> None:
    await session.emit(
        "conversation_list",
        conversations=session.store.list(),
        active_id=session.conversation_id,
    )


def _clean_provider(value: Any) -> str:
    provider = str(value or "").lower().strip()
    allowed = {"ollama", "llamacpp", "openai_compat", "openai", "anthropic"}
    if provider not in allowed:
        raise ValueError("Provedor de modelo inválido.")
    return provider


def _clean_model(value: Any) -> str:
    model = str(value or "").strip()
    if not model or len(model) > 160 or any(char in model for char in "\r\n\0"):
        raise ValueError("Nome de modelo inválido.")
    return model


def _clean_reasoning(value: Any) -> str:
    effort = str(value or "medium").lower().strip()
    if effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Nível de raciocínio inválido.")
    return effort


def _clean_mode(value: Any) -> str:
    return "computer" if str(value or "").lower().strip() == "computer" else "agent"


def _same_origin_websocket(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    if not origin:
        return True
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and parsed.netloc == websocket.headers.get("host")


def _model_profiles(settings: Any) -> list[dict[str, Any]]:
    current_model = settings.llm.strong_model or settings.llm.model_name
    profiles = [
        {
            "provider": settings.llm.llm_provider,
            "model": current_model,
            "label": f"Atual · {current_model}",
        },
        {"provider": "openai", "model": "gpt-5.6-sol", "label": "GPT-5.6 Sol"},
        {"provider": "openai", "model": "gpt-5.6-terra", "label": "GPT-5.6 Terra"},
        {"provider": "openai", "model": "gpt-5.6-luna", "label": "GPT-5.6 Luna"},
    ]
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for profile in profiles:
        key = (str(profile["provider"]), str(profile["model"]))
        if key not in seen:
            unique.append(profile)
            seen.add(key)
    return unique


app = create_app()
