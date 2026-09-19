from __future__ import annotations

import socket
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from isaac.doctor import _check_ollama, _check_selected_provider, has_failures, run_checks
from isaac.llm.provider import _capped_llm, build_llm_for_profile, get_llm, provider_api_key
from isaac.tools.browser import BrowserTool


@pytest.fixture
def runtime(monkeypatch):
    page = SimpleNamespace(goto=AsyncMock(), title=AsyncMock(return_value="Example"))
    context = SimpleNamespace(
        pages=[page], close=AsyncMock(), route=AsyncMock(), route_web_socket=AsyncMock()
    )
    engines = {
        name: SimpleNamespace(launch_persistent_context=AsyncMock(return_value=context))
        for name in ("chromium", "firefox", "webkit")
    }
    pw = SimpleNamespace(**engines, stop=AsyncMock())
    api = ModuleType("playwright.async_api")
    api.async_playwright = MagicMock(return_value=SimpleNamespace(start=AsyncMock(return_value=pw)))
    monkeypatch.setitem(sys.modules, "playwright", ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.async_api", api)
    return pw, context, page


@pytest.mark.parametrize(
    ("engine", "channel", "expected_engine", "expected_channel"),
    [
        ("chromium", None, "chromium", None),
        ("firefox", None, "firefox", None),
        ("webkit", None, "webkit", None),
        ("chrome", None, "chromium", "chrome"),
        ("edge", None, "chromium", "msedge"),
        ("chromium", "edge", "chromium", "msedge"),
        ("chromium", "chrome", "chromium", "chrome"),
        ("chromium", "chromium", "chromium", "chromium"),
    ],
)
async def test_selected_browser_uses_disposable_isolated_profile(
    runtime, engine, channel, expected_engine, expected_channel
):
    pw, context, page = runtime
    tool = BrowserTool(engine=engine, channel=channel)
    assert await tool._ensure_page() is page
    assert await tool._ensure_page() is page
    launch = getattr(pw, expected_engine).launch_persistent_context
    launch.assert_awaited_once()
    profile = Path(launch.call_args.args[0])
    assert profile.is_dir()
    assert profile.name.startswith("isaac-browser-")
    options = launch.call_args.kwargs
    assert options.get("channel") == expected_channel
    assert options["service_workers"] == "block"
    assert options["accept_downloads"] is False
    assert "user_agent" not in options
    assert "args" not in options
    if expected_engine == "chromium":
        assert options["chromium_sandbox"] is True
    else:
        assert "chromium_sandbox" not in options
    context.route.assert_awaited_once_with("**/*", tool._route_request)
    context.route_web_socket.assert_awaited_once_with("**/*", tool._block_websocket)
    await tool.aclose()
    assert not profile.exists()
    context.close.assert_awaited_once()
    pw.stop.assert_awaited_once()
    await tool.aclose()


@pytest.mark.parametrize(
    ("engine", "channel"),
    [
        ("safari", None),
        ("firefox", "chrome"),
        ("webkit", "msedge"),
        ("chromium", "personal"),
        ("chrome", "edge"),
    ],
)
def test_invalid_browser_selection_is_rejected(engine, channel):
    with pytest.raises(ValueError):
        BrowserTool(engine=engine, channel=channel)


def test_personal_profile_cannot_be_supplied():
    with pytest.raises(TypeError):
        BrowserTool(user_data_dir="personal-profile")


async def test_profiles_are_not_shared_between_instances(runtime):
    first, second = BrowserTool(), BrowserTool()
    try:
        await first._ensure_page()
        await second._ensure_page()
        assert first._profile.name != second._profile.name
    finally:
        await first.aclose()
        await second.aclose()


async def test_launch_failure_cleans_up_and_names_selected_engine(runtime):
    pw, _, _ = runtime
    pw.firefox.launch_persistent_context.side_effect = RuntimeError("executable missing")
    tool = BrowserTool(engine="firefox")
    result = await tool.execute(action="current")
    profile = Path(pw.firefox.launch_persistent_context.call_args.args[0])
    assert not result.success
    assert "playwright install firefox" in result.error
    assert not profile.exists()
    assert tool._pw is None
    pw.stop.assert_awaited_once()


async def test_setup_failure_closes_context_and_profile(runtime):
    pw, context, _ = runtime
    context.route.side_effect = RuntimeError("route failed")
    tool = BrowserTool()
    with pytest.raises(RuntimeError, match="route failed"):
        await tool._ensure_page()
    profile = Path(pw.chromium.launch_persistent_context.call_args.args[0])
    assert not profile.exists()
    context.close.assert_awaited_once()
    pw.stop.assert_awaited_once()


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "data:text/html,hello",
        "file:///C:/secret",
        "about:config",
        "chrome://settings",
        "edge://settings",
        "ftp://example.com",
        "mailto:x@example.com",
        "https://example.com\\@127.0.0.1",
        "https://exam\nple.com",
        "https://user:pass@example.com",
    ],
)
async def test_dangerous_urls_rejected_before_browser_launch(monkeypatch, url):
    launch = AsyncMock()
    tool = BrowserTool()
    monkeypatch.setattr(tool, "_ensure_page", launch)
    result = await tool.execute(action="navigate", url=url)
    assert not result.success
    launch.assert_not_awaited()


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1"])
async def test_private_dns_targets_are_blocked(monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(0, 0, 0, "", (address, 443))])
    with pytest.raises(ValueError, match="private or local"):
        await BrowserTool()._navigation_url("https://public-looking.example")


async def test_bare_hostname_is_normalized(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(0, 0, 0, "", ("93.184.216.34", 443))]
    )
    assert await BrowserTool()._navigation_url("example.com/path") == "https://example.com/path"


async def test_request_guard_blocks_private_subresources_and_redirect_targets(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(0, 0, 0, "", ("127.0.0.1", 80))])
    route = SimpleNamespace(
        request=SimpleNamespace(url="http://internal.example"), abort=AsyncMock(), fetch=AsyncMock()
    )
    await BrowserTool()._route_request(route)
    route.abort.assert_awaited_once_with("blockedbyclient")
    route.fetch.assert_not_awaited()


async def test_request_guard_does_not_follow_unvalidated_redirects(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(0, 0, 0, "", ("93.184.216.34", 443))]
    )
    response = SimpleNamespace(dispose=AsyncMock())
    route = SimpleNamespace(
        request=SimpleNamespace(url="https://example.com"),
        abort=AsyncMock(),
        fetch=AsyncMock(return_value=response),
        fulfill=AsyncMock(),
    )
    await BrowserTool()._route_request(route)
    assert route.fetch.call_args.kwargs["max_redirects"] == 0
    route.fulfill.assert_awaited_once_with(response=response)
    response.dispose.assert_awaited_once()
    route.abort.assert_not_awaited()


async def test_websockets_are_closed_without_connecting():
    route = SimpleNamespace(close=AsyncMock(), connect_to_server=MagicMock())
    await BrowserTool()._block_websocket(route)
    route.close.assert_awaited_once()
    route.connect_to_server.assert_not_called()


@pytest.fixture
def provider_settings(monkeypatch):
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            llm_provider="openai",
            model_name="glm-5:cloud",
            base_url="",
            temperature=0.2,
            fast_model="",
            strong_model="",
            fast_temperature=-1,
            strong_temperature=-1,
        ),
        openai_api_key="",
        anthropic_api_key="",
        openai_compat_base_url="",
        ollama_base_url="http://localhost:11434",
        ollama_light_model="model",
        ollama_heavy_model="model",
    )
    monkeypatch.setattr("isaac.config.settings.get_settings", lambda: settings)
    monkeypatch.setattr("isaac.config.settings.settings", settings)
    monkeypatch.setattr("isaac.security.credentials.get_credential", lambda provider: "")
    monkeypatch.setattr(
        "httpx.get",
        MagicMock(
            return_value=SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"data": [{"id": settings.llm.model_name}]},
            )
        ),
    )
    get_llm.cache_clear()
    yield settings
    get_llm.cache_clear()


@pytest.mark.parametrize("key", ["", " ", "...", "sk-your-key-here", "not-needed"])
def test_doctor_fails_selected_openai_without_usable_key(provider_settings, key):
    provider_settings.openai_api_key = key
    result = _check_selected_provider()
    assert result.status == "fail"
    assert "OPENAI_API_KEY" in result.detail


def test_doctor_includes_selected_provider_failure(provider_settings):
    results = run_checks()
    assert has_failures(results)
    assert next(result for result in results if result.name == "selected-provider").status == "fail"


def test_doctor_rejects_ollama_cloud_tag_on_default_openai(provider_settings):
    provider_settings.openai_api_key = "valid-looking-key"
    result = _check_selected_provider()
    assert result.status == "fail"
    assert ":cloud" in result.detail


@pytest.mark.parametrize(
    "builder",
    [
        lambda: get_llm(),
        lambda: _capped_llm(200),
        lambda: build_llm_for_profile("openai", "gpt-4o"),
    ],
)
def test_all_cloud_builders_reject_missing_credentials(provider_settings, builder):
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        builder()


@pytest.mark.parametrize("endpoint", ["https://api.openai.com/v1", "https://remote.example/v1"])
def test_remote_endpoint_does_not_bypass_key_requirement(provider_settings, endpoint):
    provider_settings.llm.base_url = endpoint
    assert _check_selected_provider().status == "fail"


def test_loopback_endpoint_has_explicit_keyless_client_value(provider_settings):
    provider_settings.llm.base_url = "http://localhost:11434/v1"
    assert provider_api_key("openai", provider_settings) == "not-needed"
    assert _check_selected_provider().status == "ok"


def test_keyring_credentials_work_for_doctor_and_runtime(provider_settings, monkeypatch):
    provider_settings.llm.model_name = "gpt-4o"
    monkeypatch.setattr("isaac.security.credentials.get_credential", lambda provider: "keyring-key")
    assert provider_api_key("openai", provider_settings) == "keyring-key"
    result = _check_selected_provider()
    assert result.status == "ok"
    assert "keyring-key" not in result.detail
    assert "not probed" in result.detail


def test_anthropic_requires_its_own_key(provider_settings):
    provider_settings.llm.llm_provider = "anthropic"
    provider_settings.openai_api_key = "valid-looking-key"
    result = _check_selected_provider()
    assert result.status == "fail"
    assert "ANTHROPIC_API_KEY" in result.detail


def test_compat_requires_endpoint(provider_settings):
    provider_settings.llm.llm_provider = "openai_compat"
    assert _check_selected_provider().status == "fail"
    provider_settings.openai_compat_base_url = "http://localhost:8080/v1"
    assert _check_selected_provider().status == "ok"


def test_selected_local_endpoint_missing_model_fails(provider_settings, monkeypatch):
    provider_settings.llm.base_url = "http://localhost:11434/v1"
    monkeypatch.setattr(
        "httpx.get",
        MagicMock(
            return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": []})
        ),
    )
    result = _check_selected_provider()
    assert result.status == "fail"
    assert "glm-5:cloud" in result.detail


def test_selected_local_endpoint_unreachable_fails(provider_settings, monkeypatch):
    provider_settings.llm.base_url = "http://localhost:11434/v1"
    monkeypatch.setattr("httpx.get", MagicMock(side_effect=ConnectionError("unreachable")))
    assert _check_selected_provider().status == "fail"


async def test_request_fetch_failure_aborts_instead_of_leaving_navigation_hanging(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(0, 0, 0, "", ("93.184.216.34", 443))]
    )
    route = SimpleNamespace(
        request=SimpleNamespace(url="https://example.com"),
        abort=AsyncMock(),
        fetch=AsyncMock(side_effect=TimeoutError()),
    )
    await BrowserTool()._route_request(route)
    route.abort.assert_awaited_once_with("failed")


def test_selected_ollama_with_empty_model_list_fails(provider_settings, monkeypatch):
    provider_settings.llm.llm_provider = "ollama"
    monkeypatch.setattr("isaac.llm.providers.ollama.list_models", lambda url: [])
    assert _check_ollama().status == "fail"


def test_selected_ollama_probe_exception_fails(provider_settings, monkeypatch):
    provider_settings.llm.llm_provider = "ollama"
    monkeypatch.setattr(
        "isaac.llm.providers.ollama.health_check",
        MagicMock(side_effect=RuntimeError("unreachable")),
    )
    assert _check_ollama().status == "fail"
