"""Real configuration, runtime scopes and image payloads across shared vision consumers."""

import asyncio
import base64
import json
import threading
from contextvars import copy_context
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from agent.auxiliary_client import reset_runtime_main, set_runtime_main
from agent.image_routing import build_native_content_parts, decide_image_input_mode
from hermes_cli.config import atomic_config_write, load_config
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def vision_runtime(tmp_path):
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            payload = json.dumps({
                "id": "chat-test", "object": "chat.completion", "created": 1,
                "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "A red square."}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = set_hermes_home_override(tmp_path)
    url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        atomic_config_write(tmp_path / "config.yaml", {
            "agent": {"vision_capability_first": True},
            "model": {"provider": "persisted", "default": "text-main"},
            "auxiliary": {"vision": {"provider": "describer", "model": "aux-vision"}},
            "vision": {"max_calls_per_image": 0},
            "providers": {
                "persisted": {"models": {"vision-main": {"supports_vision": False}}},
                "active": {"base_url": url, "api_key": "fake-active-test-only",
                           "models": {"vision-main": {"supports_vision": True},
                                      "text-main": {"supports_vision": False}}},
                "xiaomi": {"models": {"vision-main": {"supports_vision": True}}},
                "describer": {"base_url": url, "api_key": "fake-loopback-test-only"},
            },
        })
        image = tmp_path / "red.png"
        Image.new("RGB", (16, 16), "red").save(image)
        yield SimpleNamespace(image=image, url=url, requests=requests)
    finally:
        reset_hermes_home_override(home)
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _runtime(env, model="vision-main", provider="custom"):
    return set_runtime_main(provider, model, requested_provider="active" if provider == "custom" else provider,
                            base_url=env.url, api_key="fake-main-test-only")


def _assert_image(content):
    images = [part for part in content if part.get("type") == "image_url"]
    assert len(images) == 1
    raw = base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1])
    with Image.open(BytesIO(raw)) as image:
        assert image.size == (16, 16)
        assert image.convert("RGB").getpixel((0, 0))[0] > 240


@pytest.fixture(params=["static", "callable"])
def route_agent(vision_runtime, request):
    from openai import OpenAI
    from agent.agent_runtime_helpers import anthropic_prompt_cache_policy
    from agent.primary_runtime_snapshot import _build_primary_runtime_snapshot

    env = vision_runtime
    clients = []
    primary_key = (lambda: "fake-main-test-only") if request.param == "callable" else "fake-main-test-only"

    def create_client(kwargs, **_):
        client = OpenAI(**kwargs)
        clients.append(client)
        return client

    # Compression and LM Studio loading are unrelated to image routing. Keep route resolution,
    # client installation, config, snapshots and runtime publication real.
    compressor = SimpleNamespace(context_length=32768, threshold_tokens=16384,
                                 update_model=lambda **_: None)
    agent = SimpleNamespace(
        model="vision-main", provider="custom", requested_provider="active", base_url=env.url,
        api_key=primary_key, api_mode="chat_completions", session_id="route-vision",
        auth_mode="api_key", _credential_pool=None, _cache_disabled=False,
        _use_prompt_caching=False, _use_native_cache_layout=False, context_compressor=compressor,
        _client_kwargs={"base_url": env.url, "api_key": primary_key},
        _ensure_lmstudio_runtime_loaded=lambda: None, _create_openai_client=create_client,
    )
    agent._anthropic_prompt_cache_policy = lambda **kw: anthropic_prompt_cache_policy(agent, **kw)
    agent.client = create_client(agent._client_kwargs)
    agent._primary_runtime = _build_primary_runtime_snapshot(agent, agent.api_mode)
    # Skip metadata probes during fallback; snapshot restoration still exercises update_model.
    agent.context_compressor = None
    try:
        yield agent, compressor
    finally:
        for client in clients:
            client.close()
        if agent.client not in clients:
            agent.client.close()


def _assert_vision_exchange(env, agent, native):
    from agent.tool_dispatch_helpers import make_tool_result_message
    from tools.vision_tools import _handle_vision_analyze, _should_use_native_vision_fast_path

    assert _should_use_native_vision_fast_path() is native
    before = len(env.requests)
    result = asyncio.run(_handle_vision_analyze({"image_url": str(env.image), "question": "Color?"}))
    if native:
        assert isinstance(result, dict) and result["_multimodal"] is True
        _assert_image(result["content"])
        assert len(env.requests) == before
        content = result["content"]
    else:
        assert json.loads(result)["analysis"] == "A red square."
        assert len(env.requests) == before + 1
        assert env.requests[-1]["model"] == "aux-vision"
        assert "temperature" not in env.requests[-1]
        _assert_image(env.requests[-1]["messages"][0]["content"])
        content = result
    message = make_tool_result_message("vision_analyze", content, "call-vision")
    agent.client.chat.completions.create(model=agent.model, messages=[
        {"role": "user", "content": "Color?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-vision",
         "type": "function", "function": {"name": "vision_analyze", "arguments": "{}"}}]},
        message,
    ])
    sent = env.requests[-1]
    assert sent["model"] == agent.model
    assert sent["messages"][-1]["tool_call_id"] == "call-vision"
    if native:
        _assert_image(sent["messages"][-1]["content"])
    else:
        assert "A red square." in sent["messages"][-1]["content"]
        assert "data:image" not in json.dumps(sent)


@pytest.mark.parametrize("restore", ["snapshot", "primary"])
def test_actual_route_binding_refreshes_vision_runtime_and_restores_it(vision_runtime, route_agent, restore):
    from agent.auxiliary_client import _normalize_main_runtime, scoped_runtime_main
    from agent.route_binding import bind_route_entry, reinstall_runtime_snapshot
    from agent.agent_runtime_helpers import restore_primary_runtime
    from agent.turn_context import _publish_runtime_main

    env = vision_runtime
    agent, compressor = route_agent
    with scoped_runtime_main({"provider": "outer", "model": "outer-main"}):
        # Route changes happen in the owning turn context; tool workers inherit the new binding.
        def turn():
            _publish_runtime_main(agent)
            _assert_vision_exchange(env, agent, native=True)
            before_swap = copy_context()
            bound = bind_route_entry(agent, {
                "base_url": env.url, "api_key": "fake-fallback-test-only", "api_mode": "chat_completions",
            }, "active", "text-main")
            assert bound == ("vision-main", "custom")
            copy_context().run(_assert_vision_exchange, env, agent, False)
            assert _normalize_main_runtime(None)["provider"] == "active"
            assert before_swap.run(_normalize_main_runtime, None)["model"] == "vision-main"
            agent.context_compressor = compressor
            if restore == "snapshot":
                reinstall_runtime_snapshot(agent, agent._primary_runtime)
            else:
                assert restore_primary_runtime(agent) is True
            copy_context().run(_assert_vision_exchange, env, agent, True)
            runtime = _normalize_main_runtime(None)
            assert (runtime["provider"], runtime["requested_provider"], runtime["model"]) == (
                "custom", "active", "vision-main")
            assert runtime["api_key"] is agent._primary_runtime["api_key"]
            assert runtime["cache_scope"] == "route-vision"

        copy_context().run(turn)
        assert _normalize_main_runtime(None)["model"] == "outer-main"


def test_primary_restore_publishes_reselected_pool_credential(vision_runtime, route_agent):
    from agent.auxiliary_client import _normalize_main_runtime, scoped_runtime_main
    from agent.client_lifecycle import ClientLifecycleMixin
    from agent.credential_pool import CredentialPool, PooledCredential
    from agent.agent_runtime_helpers import restore_primary_runtime

    env = vision_runtime
    agent, compressor = route_agent
    agent.context_compressor = compressor
    agent._fallback_activated = True
    entry = PooledCredential.from_dict("active", {
        "id": "selected", "access_token": "fake-reselected-test-only", "base_url": env.url,
    })
    agent._credential_pool = CredentialPool("active", [entry])
    agent._reapply_route_client_config = lambda **_: None

    def replace_client(**_):
        agent.client.close()
        agent.client = agent._create_openai_client(agent._client_kwargs)

    agent._replace_primary_openai_client = replace_client
    agent._swap_credential = lambda selected: ClientLifecycleMixin._swap_credential(agent, selected)
    with scoped_runtime_main({"provider": "active", "model": "text-main"}):
        assert restore_primary_runtime(agent) is True
        runtime = _normalize_main_runtime(None)
        assert runtime["api_key"] == "fake-reselected-test-only"
        assert runtime["base_url"] == env.url
        _assert_vision_exchange(env, agent, native=True)


def test_transport_veto_keeps_capable_model_on_auxiliary_path(vision_runtime):
    from tools.vision_tools import _handle_vision_analyze

    env = vision_runtime
    token = _runtime(env, provider="xiaomi")
    try:
        assert decide_image_input_mode("xiaomi", "vision-main", load_config()) == "native"
        result = asyncio.run(_handle_vision_analyze({"image_url": str(env.image), "question": "Color?"}))
        assert json.loads(result)["analysis"] == "A red square."
        assert [request["model"] for request in env.requests] == ["aux-vision"]
    finally:
        reset_runtime_main(token)


@pytest.mark.parametrize("consumer", ["browser", "computer", "mcp"])
@pytest.mark.parametrize("model,native", [("vision-main", True), ("text-main", False)])
def test_shared_consumers_return_pixels_only_for_active_capable_model(vision_runtime, consumer, model, native):
    env = vision_runtime
    token = _runtime(env, model)
    try:
        if consumer == "browser":
            from tools.browser_use_cli import _native_screenshot_result
            result = _native_screenshot_result({"stdout": "captured"}, str(env.image))
        elif consumer == "computer":
            from tools.computer_use.backend import CaptureResult
            from tools.computer_use.tool import _capture_response
            result = _capture_response(CaptureResult(
                mode="vision", width=16, height=16, elements=[], app="test", window_title="red",
                png_b64=base64.b64encode(env.image.read_bytes()).decode(),
                png_bytes_len=env.image.stat().st_size))
        else:
            from tools.mcp_tool_content import _cache_mcp_image_block, _mcp_result_with_native_images
            tag = _cache_mcp_image_block(SimpleNamespace(
                mimeType="image/png", data=base64.b64encode(env.image.read_bytes()).decode()))
            assert tag.startswith("MEDIA:")
            result = _mcp_result_with_native_images(tag, [tag.removeprefix("MEDIA:")])
        if native:
            assert isinstance(result, dict) and result["_multimodal"] is True
            _assert_image(result["content"])
            assert env.requests == []
        elif consumer == "computer":
            payload = json.loads(result)
            assert payload["vision_analysis"] == "A red square."
            assert "data:image" not in result
            assert [request["model"] for request in env.requests] == ["aux-vision"]
        elif consumer == "browser":
            assert result is None
        else:
            assert isinstance(result, str) and result.startswith("MEDIA:")
    finally:
        reset_runtime_main(token)


@pytest.mark.parametrize("model,want", [("vision-main", "native"), ("text-main", "text")])
def test_gateway_inbound_routes_pixels_or_real_auxiliary_description(vision_runtime, model, want):
    from gateway.run import GatewayRunner

    env = vision_runtime
    token = _runtime(env, model)
    try:
        runner = GatewayRunner.__new__(GatewayRunner)
        assert runner._decide_image_input_mode() == want
        if want == "native":
            parts, skipped = build_native_content_parts("Color?", [str(env.image)])
            assert skipped == []
            _assert_image(parts)
            assert env.requests == []
        else:
            text = asyncio.run(runner._enrich_message_with_vision("Color?", [str(env.image)]))
            assert "A red square." in text
            assert "Color?" in text
            assert "data:image" not in text
            assert [request["model"] for request in env.requests] == ["aux-vision"]
    finally:
        reset_runtime_main(token)
