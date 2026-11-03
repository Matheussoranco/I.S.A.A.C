from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException

from isaac.config.profiles import get_active_profile, list_profiles, validate_profile_name
from isaac.config.settings import clear_settings_cache
from isaac.interfaces.dashboard import (
    api_keys,
    config_editor,
    gateway_status,
    memory_browser,
    session_search,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/desktop", tags=["desktop"])


# --- Cron Endpoints ---
@router.get("/cron/list")
async def cron_list():
    from isaac.background.cron_engine import list_tasks

    jobs = [
        {
            "job_id": t["id"],
            "name": t["name"] or t["command"][:40],
            "schedule": t["schedule"],
            "enabled": t["enabled"],
            "approved": t["approved"],
            "last_run": t["last_run"],
            "last_status": t["last_status"],
        }
        for t in list_tasks()
    ]
    return {"ok": True, "jobs": jobs}


@router.post("/cron/toggle")
async def cron_toggle(payload: dict[str, Any]):
    job_id = payload.get("job_id")
    if not job_id:
        raise HTTPException(400, "Missing job_id")
    from isaac.background.cron_engine import pause_task, resume_task

    enabled = bool(payload.get("enabled"))
    changed = resume_task(job_id) if enabled else pause_task(job_id)
    if not changed:
        raise HTTPException(404, f"No cron job: {job_id}")
    return {"ok": True, "enabled": enabled}


@router.get("/cron/logs/{job_id}")
async def cron_logs(job_id: str):
    from isaac.background.cron_engine import task_logs

    return {"ok": True, "logs": task_logs(job_id)}


# --- Profile Endpoints ---
@router.get("/profile/list")
async def profile_list():
    active = get_active_profile()
    profiles = list_profiles()
    return {"profiles": [{"id": p, "name": p, "active": p == active} for p in profiles]}


@router.post("/profile/switch")
async def profile_switch(payload: dict[str, Any]):
    profile_id = payload.get("profile_id")
    if not profile_id:
        raise HTTPException(400, "Missing profile_id")
    try:
        name = validate_profile_name(str(profile_id))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    os.environ["ISAAC_PROFILE"] = name
    clear_settings_cache()
    return {"ok": True, "active_profile": name, "restart_required": True}


# --- Skills Endpoints ---
@router.get("/skills/list")
async def skills_list():
    from isaac.memory.procedural import ProceduralMemory

    pm = ProceduralMemory()
    skills = []
    for name in pm.list_all():
        record = pm.get_record(name)
        skills.append(
            {
                "skill_id": name,
                "name": name,
                "enabled": not (record.deprecated if record else False),
                "success_rate": pm.get_success_rate(name),
                "uses": record.total_invocations if record else 0,
            }
        )
    return {"ok": True, "skills": skills}


@router.post("/skills/toggle")
async def skills_toggle(payload: dict[str, Any]):
    skill_id = payload.get("skill_id")
    if not skill_id:
        raise HTTPException(400, "Missing skill_id")
    from isaac.memory.procedural import ProceduralMemory

    pm = ProceduralMemory()
    if pm.get_record(skill_id) is None:
        raise HTTPException(404, f"No such skill: {skill_id}")
    enabled = bool(payload.get("enabled"))
    if enabled:
        pm.restore(skill_id)
    else:
        pm.deprecate(skill_id)
    return {"ok": True, "enabled": enabled}


# --- Command Palette Endpoints ---
@router.post("/command/run")
async def command_run(payload: dict[str, Any]):
    cmd = payload.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        raise HTTPException(400, "Missing command")
    try:
        from isaac.cli import run_command

        result = await asyncio.to_thread(run_command, cmd)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@router.get("/palette/search")
async def palette_search(q: str):
    # Mock search: in reality, this would search command history and available CLI commands
    results = [
        {"label": f"Run: {q}", "cmd": q, "type": "cmd"},
        {"label": f"History: {q} (Yesterday)", "cmd": q, "type": "history"},
    ]
    return {"results": results}


# --- Dashboard Endpoints ---


# Config Editor
@router.get("/dashboard/config")
async def dash_config_get(profile: str | None = None):
    return config_editor.get_config_content(profile)


@router.post("/dashboard/config")
async def dash_config_set(payload: dict, profile: str | None = None):
    return config_editor.update_config(payload, profile)


# API Keys
@router.get("/dashboard/keys")
async def dash_keys_list():
    return api_keys.list_keys()


@router.post("/dashboard/keys")
async def dash_keys_set(payload: dict):
    provider = payload.get("provider")
    value = payload.get("value", "")
    if not provider:
        raise HTTPException(400, "Missing provider")
    return api_keys.set_key(provider, value)


# Memory Browser
@router.get("/dashboard/memory/search")
async def dash_mem_search(query: str):
    return memory_browser.search_memory(query)


@router.get("/dashboard/memory/graph")
async def dash_mem_graph():
    return memory_browser.inspect_semantic_graph()


# Gateway Status
@router.get("/dashboard/gateways")
async def dash_gateways_status():
    return gateway_status.get_gateway_status()


# Session Search
@router.get("/dashboard/sessions/search")
async def dash_session_search(query: str, limit: int = 20):
    return session_search.search_sessions(query, limit=limit)
