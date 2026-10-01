from typing import AsyncGenerator, Any, List, Optional
import uuid
import time
import asyncio
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from isaac.agents.agent_loop import build_default_agent, AgentRunResult
from isaac.security.api_auth import validate_api_key

app = FastAPI(title="I.S.A.A.C. OpenAI-Compatible API")

# --- Models ---

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[dict]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class ChatCompletionResponseChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[dict]

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

class ModelResponse(BaseModel):
    object: str = "list"
    data: List[dict]

class EmbeddingRequest(BaseModel):
    model: str
    input: Any # str or List[str]

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict

# --- Helpers ---

def openai_messages_to_task(messages: List[dict]) -> tuple[str, str]:
    """
    Converts OpenAI message format to (task, context).
    The last user message is the task, everything prior is context.
    """
    user_messages = [m["content"] for m in messages if m["role"] == "user"]
    if not user_messages:
        return "No task provided.", ""
    
    task = user_messages[-1]
    context = "\n".join(user_messages[:-1])
    return task, context

# --- Endpoints ---

@app.get("/v1/models", dependencies=[Depends(validate_api_key)])
async def list_models():
    return ModelResponse(data=[{"id": "isaac-strong", "object": "model", "owned_by": "isaac"}])

@app.post("/v1/embeddings", dependencies=[Depends(validate_api_key)])
async def create_embedding(request: EmbeddingRequest):
    # Basic placeholder for embeddings since I.S.A.A.C. focuses on agentic loops
    # In a full impl, this would call the underlying LLM's embedding endpoint
    raise HTTPException(status_code=501, detail="Embeddings not implemented in this proxy.")

@app.post("/v1/chat/completions", dependencies=[Depends(validate_api_key)])
async def create_chat_completion(request: ChatCompletionRequest):
    task, context = openai_messages_to_task(request.messages)
    
    # Build a fresh agent for the request
    # In production, you might cache agents or use a pool
    agent = build_default_agent()
    
    if request.stream:
        return StreamingResponse(
            stream_isaac_response(agent, task, context, request.model),
            media_type="text/event-stream"
        )
    
    # Non-streaming run
    result: AgentRunResult = await agent.arun(task, context=context)
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{result.run_id}",
        created=int(time.time()),
        model=request.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": result.output},
            "finish_reason": "stop" if result.stopped_reason == "final" else result.stopped_reason
        }],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} # Approximation
    )

async def stream_isaac_response(agent, task, context, model_name) -> AsyncGenerator[str, None]:
    run_id = uuid.uuid4().hex[:12]
    created = int(time.time())
    
    # We use a queue to bridge the agent's callback to the SSE stream
    queue = asyncio.Queue()

    def on_token(token: str):
        # This is called by the AgentLoop's stream_callback
        # Since on_token might be called from a different thread/context, we use call_soon_threadsafe
        asyncio.get_event_loop().call_soon_threadsafe(queue.put_nowait, token)

    # Initialize agent with the streaming callback
    # Since we already built the agent in the endpoint, we need to ensure the callback is set
    # Note: AgentLoop.stream_callback is set at __init__, so we should ideally build it here.
    # To avoid duplication, let's assume we pass it to build_default_agent in the endpoint.
    
    # FIX: Redefine Agent construction in the endpoint to use this callback.
    # For now, this helper is used by a revised endpoint.
    pass

# Revision: We need the endpoint to handle the queue logic.
