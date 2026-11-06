"""Structural diagnostics for the strict capability-first boolean."""

import pytest

from hermes_cli.config import atomic_config_write, load_config, validate_config_structure
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.mark.parametrize("value", ["false", "true", 1, 0, None, [], {}])
def test_invalid_capability_first_reports_error_without_refusing_write(value, tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        atomic_config_write(tmp_path / "config.yaml", {"agent": {"vision_capability_first": value}})
        cfg = load_config()
        issues = [issue for issue in validate_config_structure(cfg)
                  if "agent.vision_capability_first" in issue.message]
        assert len(issues) == 1
        assert issues[0].severity == "error"
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("agent", [{}, {"vision_capability_first": True}, {"vision_capability_first": False}])
def test_boolean_and_absent_values_have_no_diagnostic(agent):
    assert not [issue for issue in validate_config_structure({"agent": agent})
                if "agent.vision_capability_first" in issue.message]


def test_loader_default_preserves_aux_first(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        atomic_config_write(tmp_path / "config.yaml", {})
        assert load_config()["agent"]["vision_capability_first"] is False
    finally:
        reset_hermes_home_override(token)
