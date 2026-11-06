"""Opt-in precedence: pinned auxiliary vision remains the default route."""

import pytest

from agent.image_routing import decide_image_input_mode


@pytest.mark.parametrize(
    "flag,capability,want",
    [
        (True, True, "native"),
        (True, False, "text"),
        (True, None, "text"),
        (False, True, "text"),
        (None, True, "text"),
        ("false", True, "text"),
        ("true", True, "text"),
        (1, True, "text"),
        (0, True, "text"),
        ([], True, "text"),
    ],
)
def test_auto_pinned_aux_requires_literal_true(flag, capability, want, monkeypatch):
    cfg = {
        "agent": {} if flag is None else {"vision_capability_first": flag},
        "auxiliary": {"vision": {"provider": "describer", "model": "aux-vision"}},
        "providers": {"active": {"models": {"main": {"supports_vision": capability}}}},
    }
    # Only unknown capability reaches external probes; the policy and metadata reader stay real.
    monkeypatch.setattr("agent.image_routing._VISION_PROBES", ())
    assert decide_image_input_mode("active", "main", cfg) == want


@pytest.mark.parametrize("flag", [True, False, "false"])
@pytest.mark.parametrize("mode,capability", [("native", False), ("text", True)])
def test_explicit_mode_keeps_precedence(flag, mode, capability):
    cfg = {
        "agent": {"image_input_mode": mode, "vision_capability_first": flag},
        "auxiliary": {"vision": {"provider": "describer", "model": "aux-vision"}},
        "providers": {"active": {"models": {"main": {"supports_vision": capability}}}},
    }
    assert decide_image_input_mode("active", "main", cfg) == mode


@pytest.mark.parametrize("flag", [True, False, None])
def test_unpinned_aux_keeps_native_capable_route(flag):
    cfg = {
        "agent": {} if flag is None else {"vision_capability_first": flag},
        "providers": {"active": {"models": {"main": {"supports_vision": True}}}},
    }
    assert decide_image_input_mode("active", "main", cfg) == "native"


def test_requested_custom_identity_beats_persisted_default():
    cfg = {
        "agent": {"vision_capability_first": True},
        "model": {"provider": "persisted"},
        "auxiliary": {"vision": {"provider": "describer", "model": "aux-vision"}},
        "custom_providers": [
            {"name": "persisted", "models": {"main": {"supports_vision": False}}},
            {"name": "active", "models": {"main": {"supports_vision": True}}},
        ],
    }
    assert decide_image_input_mode("custom", "main", cfg, requested_provider="active") == "native"


@pytest.mark.parametrize("loader_name", ["load_config", "load_user_config_effective"])
def test_real_config_profiles_a_b_a_keep_policy_and_pinned_describer_isolated(tmp_path, loader_name):
    from agent.auxiliary_client import resolve_vision_provider_client
    from hermes_cli import config, config_effective
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    homes = [tmp_path / "a", tmp_path / "b"]
    for home, enabled, model in zip(homes, [True, False], ["aux-a", "aux-b"]):
        home.mkdir()
        token = set_hermes_home_override(home)
        try:
            config.atomic_config_write(home / "config.yaml", {
                "agent": {"vision_capability_first": True} if enabled else {},
                "model": {"provider": "active", "default": "main"},
                "auxiliary": {"vision": {"provider": "describer", "model": model}},
                "providers": {
                    "active": {"models": {"main": {"supports_vision": True}}},
                    "describer": {"base_url": "http://127.0.0.1:9/v1",
                                  "api_key": "fake-" + home.name + "-test-only"},
                },
            })
        finally:
            reset_hermes_home_override(token)

    loader = getattr(config if loader_name == "load_config" else config_effective, loader_name)
    for home, want, aux_model in [(homes[0], "native", "aux-a"), (homes[1], "text", "aux-b"),
                                  (homes[0], "native", "aux-a")]:
        token = set_hermes_home_override(home)
        try:
            cfg = loader()
            assert decide_image_input_mode("custom", "main", cfg, requested_provider="active") == want
            provider, client, model = resolve_vision_provider_client()
            assert (provider, model) == ("describer", aux_model)
            assert client is not None
            assert str(client.base_url) == "http://127.0.0.1:9/v1/"
            assert client.api_key == "fake-" + home.name + "-test-only"
        finally:
            reset_hermes_home_override(token)
