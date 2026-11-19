"""Kanban web dashboard — FastAPI-based web UI for kanban boards."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from isaac.config.settings import get_settings
from isaac.kanban import (
    DispatchConfig,
    add_comment,
    add_task_dep,
    claim_task,
    create_task,
    get_active_claim,
    get_board,
    get_board_stats,
    get_columns,
    get_comments,
    get_dependents,
    get_task,
    get_task_deps,
    get_tasks,
    init_db,
    init_kanban_board,
    list_boards,
    move_task,
    release_task_claims,
    remove_task_dep,
    run_dispatcher_once,
    start_dispatcher,
    stop_dispatcher,
    update_task,
)


# Dependency for kanban database path
def get_kanban_db_path() -> Path:
    """Get the kanban database path from settings."""
    return get_settings().isaac_home / "kanban.db"


app = FastAPI(title="I.S.A.A.C. Kanban Dashboard")

# Setup templates
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))


# Ensure DB is initialized on startup
@app.on_event("startup")
async def startup() -> None:
    db_path = get_settings().isaac_home / "kanban.db"
    init_db(db_path)


# Pydantic models for API


class BoardCreate(BaseModel):
    name: str


class TaskCreate(BaseModel):
    title: str
    description: str = ""
    priority: int = 0
    assignee: str | None = None
    due: str | None = None
    tags: str = ""


class TaskUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    priority: int | None = None
    assignee: str | None = None
    due: str | None = None
    tags: str | None = None
    column: str | None = None


class ClaimCreate(BaseModel):
    worker_id: str
    profile: str = "default"
    stale_minutes: int = 60


class CommentCreate(BaseModel):
    author: str = "dashboard"
    body: str


# HTML Routes


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Main dashboard page."""
    return templates.TemplateResponse(request, "dashboard.html")


DBDep = Annotated[Path, Depends(get_kanban_db_path)]

# API Routes - Boards


@app.get("/api/boards")
async def api_list_boards(db_path: DBDep) -> list[dict]:
    """List all boards."""
    boards = list_boards(db_path)
    return [
        {"id": b.id, "name": b.name, "created_at": b.created_at, "updated_at": b.updated_at}
        for b in boards
    ]


@app.post("/api/boards")
async def api_create_board(board: BoardCreate, db_path: DBDep) -> dict:
    """Create a new board."""
    board_obj, columns = init_kanban_board(db_path, board.name)
    return {
        "board": {"id": board_obj.id, "name": board_obj.name, "created_at": board_obj.created_at},
        "columns": [
            {"id": c.id, "name": c.name, "position": c.position, "wip_limit": c.wip_limit}
            for c in columns
        ],
    }


@app.get("/api/boards/{board_id}")
async def api_get_board(board_id: str, db_path: DBDep) -> dict:
    """Get board details."""
    board = get_board(db_path, board_id)
    if not board:
        raise HTTPException(status_code=404, detail="Board not found")
    return {
        "id": board.id,
        "name": board.name,
        "created_at": board.created_at,
        "updated_at": board.updated_at,
    }


# API Routes - Columns


@app.get("/api/boards/{board_id}/columns")
async def api_get_columns(board_id: str) -> list[dict]:
    """Get columns for a board."""
    db_path = get_settings().isaac_home / "kanban.db"
    board = get_board(db_path, board_id)
    if not board:
        raise HTTPException(status_code=404, detail="Board not found")
    columns = get_columns(db_path, board.id)
    return [
        {"id": c.id, "name": c.name, "position": c.position, "wip_limit": c.wip_limit}
        for c in columns
    ]


@app.post("/api/boards/{board_id}/columns")
async def api_create_column(
    board_id: str,
    name: str = Query(...),
    position: int | None = None,
    wip_limit: int | None = None,
) -> dict:
    """Create a new column."""
    from isaac.kanban import create_column

    db_path = get_settings().isaac_home / "kanban.db"
    board = get_board(db_path, board_id)
    if not board:
        raise HTTPException(status_code=404, detail="Board not found")
    col = create_column(db_path, board.id, name, position, wip_limit)
    return {"id": col.id, "name": col.name, "position": col.position, "wip_limit": col.wip_limit}


# API Routes - Tasks


@app.get("/api/boards/{board_id}/tasks")
async def api_list_tasks(board_id: str, column_id: str | None = None) -> list[dict]:
    """List tasks for a board, optionally filtered by column."""
    db_path = get_settings().isaac_home / "kanban.db"
    board = get_board(db_path, board_id)
    if not board:
        raise HTTPException(status_code=404, detail="Board not found")
    tasks = get_tasks(db_path, board.id, column_id)
    cols = {c.id: c.name for c in get_columns(db_path, board.id)}
    return [
        {
            "id": t.id,
            "title": t.title,
            "description": t.description,
            "column_id": t.column_id,
            "column_name": cols.get(t.column_id, t.column_id),
            "priority": t.priority,
            "assignee": t.assignee,
            "blocked": t.is_blocked,
            "tags": t.tag_list,
            "created_at": t.created_at,
            "updated_at": t.updated_at,
            "due_at": t.due_at,
        }
        for t in tasks
    ]


@app.post("/api/boards/{board_id}/tasks")
async def api_create_task(board_id: str, task: TaskCreate) -> dict:
    """Create a new task."""
    db_path = get_settings().isaac_home / "kanban.db"
    board = get_board(db_path, board_id)
    if not board:
        raise HTTPException(status_code=404, detail="Board not found")
    t = create_task(
        db_path,
        board.id,
        task.title,
        task.description,
        task.priority,
        task.assignee,
        task.due,
        task.tags,
    )
    return {"id": t.id, "title": t.title, "column_id": t.column_id}


@app.get("/api/boards/{board_id}/tasks/{task_id}")
async def api_get_task(board_id: str, task_id: str) -> dict:
    """Get task details with dependencies, claims, comments."""
    db_path = get_settings().isaac_home / "kanban.db"
    task = get_task(db_path, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    deps = get_task_deps(db_path, task_id)
    dependents = get_dependents(db_path, task_id)
    claim = get_active_claim(db_path, task_id)
    comments = get_comments(db_path, task_id)
    cols = {c.id: c.name for c in get_columns(db_path, board_id)}

    return {
        "task": {
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "column": cols.get(task.column_id, task.column_id),
            "column_id": task.column_id,
            "priority": task.priority,
            "assignee": task.assignee,
            "blocked": task.is_blocked,
            "tags": task.tag_list,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "due_at": task.due_at,
            "depends_on": deps,
            "dependents": dependents,
        },
        "claim": {
            "worker_id": claim.worker_id,
            "profile": claim.profile,
            "claimed_at": claim.claimed_at,
            "stale": claim.is_claim_stale(claim) if hasattr(claim, "is_claim_stale") else False,
        }
        if claim
        else None,
        "comments": [
            {"author": c.author, "body": c.body, "created_at": c.created_at} for c in comments
        ],
    }


@app.patch("/api/boards/{board_id}/tasks/{task_id}")
async def api_update_task(board_id: str, task_id: str, task: TaskUpdate) -> dict:
    """Update a task."""
    db_path = get_settings().isaac_home / "kanban.db"
    t = get_task(db_path, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.column:
        cols = {c.name.lower(): c.id for c in get_columns(db_path, board_id)}
        col_id = cols.get(task.column.lower())
        if not col_id:
            raise HTTPException(status_code=404, detail=f"Column not found: {task.column}")
        move_task(db_path, task_id, col_id)

    new_priority = task.priority
    updated = update_task(
        db_path,
        task_id,
        title=task.title,
        description=task.description,
        priority=new_priority,
        assignee=task.assignee,
        due_at=task.due,
        tags=task.tags,
    )
    return {"id": updated.id, "title": updated.title, "column_id": updated.column_id}


@app.post("/api/boards/{board_id}/tasks/{task_id}/claim")
async def api_claim_task(board_id: str, task_id: str, claim: ClaimCreate) -> dict:
    """Claim a task."""
    db_path = get_settings().isaac_home / "kanban.db"
    try:
        cl = claim_task(db_path, task_id, claim.worker_id, claim.profile, claim.stale_minutes)
        return {
            "id": cl.id,
            "task_id": cl.task_id,
            "worker_id": cl.worker_id,
            "profile": cl.profile,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/boards/{board_id}/tasks/{task_id}/release")
async def api_release_task(board_id: str, task_id: str) -> dict:
    """Release claims on a task."""
    db_path = get_settings().isaac_home / "kanban.db"
    count = release_task_claims(db_path, task_id)
    return {"released": count}


@app.post("/api/boards/{board_id}/tasks/{task_id}/comments")
async def api_add_comment(board_id: str, task_id: str, comment: CommentCreate) -> dict:
    """Add a comment to a task."""
    db_path = get_settings().isaac_home / "kanban.db"
    c = add_comment(db_path, task_id, comment.author, comment.body)
    return {
        "id": c.id,
        "task_id": c.task_id,
        "author": c.author,
        "body": c.body,
        "created_at": c.created_at,
    }


# API Routes - Dependencies


@app.post("/api/boards/{board_id}/tasks/{task_id}/deps")
async def api_add_dep(board_id: str, task_id: str, depends_on: str = Query(...)) -> dict:
    """Add a dependency."""
    db_path = get_settings().isaac_home / "kanban.db"
    add_task_dep(db_path, task_id, depends_on)
    return {"task_id": task_id, "depends_on": depends_on}


@app.delete("/api/boards/{board_id}/tasks/{task_id}/deps")
async def api_remove_dep(board_id: str, task_id: str, depends_on: str = Query(...)) -> dict:
    """Remove a dependency."""
    db_path = get_settings().isaac_home / "kanban.db"
    remove_task_dep(db_path, task_id, depends_on)
    return {"task_id": task_id, "depends_on": depends_on}


# API Routes - Stats


@app.get("/api/boards/{board_id}/stats")
async def api_get_stats(board_id: str) -> dict:
    """Get board statistics."""
    db_path = get_settings().isaac_home / "kanban.db"
    stats = get_board_stats(db_path, board_id)
    return {
        "board_id": stats.board_id,
        "board_name": stats.board_name,
        "total_tasks": stats.total_tasks,
        "by_column": stats.by_column,
        "by_priority": stats.by_priority,
        "blocked_count": stats.blocked_count,
        "claimed_count": stats.claimed_count,
        "stale_claims": stats.stale_claims,
    }


# API Routes - Dispatcher


@app.post("/api/dispatcher/run")
async def api_run_dispatcher(board_name: str = Query("default")) -> dict:
    """Run a single dispatcher cycle."""
    config = DispatchConfig(board_name=board_name)
    assigned = run_dispatcher_once(config)
    return {"assigned": assigned}


@app.post("/api/dispatcher/start")
async def api_start_dispatcher(board_name: str = Query("default")) -> dict:
    """Start the global dispatcher."""
    config = DispatchConfig(board_name=board_name)
    start_dispatcher(config)
    return {"started": True, "board": board_name}


@app.post("/api/dispatcher/stop")
async def api_stop_dispatcher() -> dict:
    """Stop the global dispatcher."""
    stop_dispatcher()
    return {"stopped": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
