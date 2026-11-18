"""``discover_models: false`` must silence every ``/models`` read, not just the picker.

Overmind fork decision (2026-11-18): Bifrost is a curated gateway whose ``/models``
entries publish bare ids with no context/pricing metadata, yet the TTL endpoint-metadata
probe re-fetched it from context and pricing paths every few minutes per process, and —
because RFC-1918 counts as local — prefixed each cycle with an LM Studio/Ollama
type-detection waterfall the gateway could only 404. The opt-out now gates
``fetch_endpoint_model_metadata`` itself.
"""

import contextlib

import pytest

from agent import model_metadata as mm
from hermes_cli import config as hconfig


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mm._endpoint_blackhole_cache.clear()
    yield tmp_path
    mm._endpoint_blackhole_cache.clear()


def _never_stream(*args, **kwargs):
    @contextlib.contextmanager
    def _ctx():
        raise AssertionError("probe must not run when discover_models is false")
        yield  # pragma: no cover
    return _ctx()


def _patch_config(monkeypatch, entries):
    monkeypatch.setattr(hconfig, "get_compatible_custom_providers", lambda config=None: entries)


@pytest.mark.parametrize("value", [False, "false", "no", "0"])
def test_optout_suppresses_probe(isolated_home, monkeypatch, value):
    _patch_config(monkeypatch, [{"base_url": "http://gw.test:8083/v1", "discover_models": value}])
    monkeypatch.setattr(mm.model_metadata_http, "stream", _never_stream)
    assert mm.fetch_endpoint_model_metadata("http://gw.test:8083/v1", api_key="k") == {}


def test_curated_entry_neighbour_does_not_leak(isolated_home, monkeypatch):
    """An opt-out entry must not silence a different base URL that left discovery on."""
    _patch_config(monkeypatch, [
        {"base_url": "http://gw.test:8083/v1", "discover_models": False},
        {"base_url": "http://other.test:9000/v1", "discover_models": True},
    ])
    calls = []

    @contextlib.contextmanager
    def _fail_after_first(url, *a, **k):
        calls.append(url)
        raise OSError("stop after first candidate")
        yield  # pragma: no cover

    monkeypatch.setattr(mm.model_metadata_http, "stream", _fail_after_first)
    assert mm.fetch_endpoint_model_metadata("http://gw.test:8083/v1", api_key="k") == {}
    assert calls == []
    assert mm.fetch_endpoint_model_metadata("http://other.test:9000/v1", api_key="k") == {}
    assert calls and "models" in calls[0]


def test_unknown_base_url_still_probes(isolated_home, monkeypatch):
    _patch_config(monkeypatch, [{"base_url": "http://gw.test:8083/v1", "discover_models": False}])
    calls = []

    @contextlib.contextmanager
    def _fail_after_first(url, *a, **k):
        calls.append(url)
        raise OSError("stop after first candidate")
        yield  # pragma: no cover

    monkeypatch.setattr(mm.model_metadata_http, "stream", _fail_after_first)
    assert mm.fetch_endpoint_model_metadata("http://elsewhere.test:1234/v1", api_key="k") == {}
    assert calls and "models" in calls[0]
