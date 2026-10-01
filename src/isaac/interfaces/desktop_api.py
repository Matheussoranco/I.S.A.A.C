from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException

from isaac.config.profiles import get_active_profile, list_profiles
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
    try:
        from isaac.core.cron import list_cron_jobs  # Hypothesized path

        jobs = list_cron_jobs()
        return {"ok": True, "jobs": jobs}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/cron/toggle")
async def cron_toggle(payload: dict[str, Any]):
    job_id = payload.get("job_id")
    enabled = payload.get("enabled")
    if not job_id:
        raise HTTPException(400, "Missing job_id")
    try:
        from isaac.core.cron import toggle_cron_job

        toggle_cron_job(job_id, enabled)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@router.get("/cron/logs/{job_id}")
async def cron_logs(job_id: str):
    try:
        from isaac.core.cron import get_cron_logs

        logs = get_cron_logs(job_id)
        return {"ok": True, "logs": logs}
    except Exception as e:
        raise HTTPException(500, str(e)) from e


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
    # In a real scenario, switching os.environ is not persistent across processes.
    # But for the NativeAppServer, we might need to restart or update internal state.
    os.environ["ISAAC_PROFILE"] = profile_id
    return {"ok": True, "active_profile": profile_id}


# --- Skills Endpoints ---
@router.get("/skills/list")
async def skills_list():
    try:
        from isaac.skills.manager import list_skills  # Hypothesized path

        skills = list_skills()
        return {"ok": True, "skills": skills}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/skills/toggle")
async def skills_toggle(payload: dict[str, Any]):
    skill_id = payload.get("skill_id")
    enabled = payload.get("enabled")
    if not skill_id:
        raise HTTPException(400, "Missing skill_id")
    try:
        from isaac.skills.manager import toggle_skill

        toggle_skill(skill_id, enabled)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e)) from e


# --- Command Palette Endpoints ---
@router.post("/command/run")
async def command_run(payload: dict[str, Any]):
    cmd = payload.get("command")
    if not cmd:
        raise HTTPException(400, "Missing command")
    try:
        # This would typically trigger a CLI command via a subprocess or internal dispatcher
        from isaac.cli import run_command

        result = run_command(cmd)
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
