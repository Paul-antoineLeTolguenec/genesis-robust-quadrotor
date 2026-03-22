"""Placeholder test — validates CI test harness setup. Remove after Phase 2 is complete."""
import pytest


@pytest.mark.unit
def test_harness_ok():
    """CI smoke test: pytest runs, markers are registered."""
    assert True


@pytest.mark.unit
def test_torch_importable():
    """torch must be importable (listed in dev extras)."""
    import torch  # noqa: F401

    assert torch.__version__, "torch.__version__ is empty"


@pytest.mark.unit
def test_genesis_robust_rl_importable():
    """Package must be importable after `uv sync --extra dev`."""
    try:
        import genesis_robust_rl  # noqa: F401
    except ImportError as e:
        pytest.fail(f"genesis_robust_rl not importable — run `uv sync --extra dev` before testing: {e}")
