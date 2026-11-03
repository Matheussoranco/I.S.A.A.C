from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from isaac.agents.agent_loop import AgentRunResult, build_default_agent
from isaac.security.api_auth import validate_api_key

app = FastAPI(title="I.S.A.A.C. OpenAI-Compatible API")

# --- Models ---


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    stream: bool | None = False
    temperature: float | None = 0.7
    max_tokens: int | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


class ModelResponse(BaseModel):
    object: str = "list"
    data: list[dict]


class EmbeddingRequest(BaseModel):
    model: str
    input: Any  # str or List[str]


# --- Helpers ---


def _content_text(content: Any) -> str:
    """Flatten OpenAI text content from a string or content-part list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
        return "\n".join(parts)
    return ""


def openai_messages_to_task(messages: list[dict]) -> tuple[str, str]:
    """Convert OpenAI messages to ``(task, context)``.

    The last user message is the task; system, assistant, and earlier user
    messages are folded into the context.
    """
    user_messages: list[str] = []
    context_lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        text = _content_text(message.get("content")).strip()
        if not text:
            continue
        if role == "user":
            user_messages.append(text)
        elif role in {"system", "assistant"}:
            context_lines.append(f"{role.capitalize()}: {text}")

    if not user_messages:
        return "No task provided.", "\n".join(context_lines)

    task = user_messages[-1]
    context_lines.extend(f"User: {text}" for text in user_messages[:-1])
    return task, "\n".join(context_lines)


def _finish_reason(reason: str) -> str:
    """Map agent stop reasons onto OpenAI ``finish_reason`` values."""
    if reason in {"max_iterations", "budget_exhausted"}:
        return "length"
    return "stop"


def _usage(task: str, context: str, output: str) -> dict[str, int]:
    prompt_tokens = max(1, len(task + context) // 4)
    completion_tokens = max(0, len(output) // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


# --- Endpoints ---


@app.get("/v1/models", dependencies=[Depends(validate_api_key)])
async def list_models():
    return ModelResponse(data=[{"id": "isaac-strong", "object": "model", "owned_by": "isaac"}])


@app.post("/v1/embeddings", dependencies=[Depends(validate_api_key)])
async def create_embedding(request: EmbeddingRequest):
    raise HTTPException(status_code=501, detail="Embeddings not implemented in this proxy.")


@app.post("/v1/chat/completions", dependencies=[Depends(validate_api_key)])
async def create_chat_completion(request: ChatCompletionRequest):
    task, context = openai_messages_to_task(request.messages)
    agent = build_default_agent()

    if request.stream:
        return StreamingResponse(
            stream_isaac_response(agent, task, context, request.model),
            media_type="text/event-stream",
        )

    result: AgentRunResult = await agent.arun(task, context=context)

    return ChatCompletionResponse(
        id=f"chatcmpl-{result.run_id}",
        created=int(time.time()),
        model=request.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.output},
                "finish_reason": _finish_reason(result.stopped_reason),
            }
        ],
        usage=_usage(task, context, result.output),
    )


async def stream_isaac_response(
    agent: Any, task: str, context: str, model_name: str
) -> AsyncGenerator[str, None]:
    """Stream an agent reply as OpenAI-compatible SSE chunks."""
    import json

    result = await agent.arun(task, context=context)
    created = int(time.time())
    chunk_id = f"chatcmpl-{result.run_id}"

    def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant"})
    if result.output:
        yield chunk({"content": result.output})
    yield chunk({}, _finish_reason(result.stopped_reason))
    yield "data: [DONE]\n\n"
