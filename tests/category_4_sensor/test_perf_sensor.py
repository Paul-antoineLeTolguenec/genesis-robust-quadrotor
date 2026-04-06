"""Performance tests for category 4 — sensor perturbations.

P1 — apply() overhead (CPU, n_envs=1 and 512).
"""

import time

import pytest
import torch

from tests.category_4_sensor.conftest import OBS_DIM

PERF_WARMUP = 50
PERF_STEPS = 500
MAX_US_PER_STEP = 500  # microseconds


@pytest.mark.perf
@pytest.mark.parametrize("n_envs_perf", [1, 512])
def test_apply_overhead_cpu(perturbation, n_envs_perf, request):
    """apply() on CPU must be < 500 µs/step."""
    # Recreate perturbation with the perf n_envs
    # Get the factory function from the fixture
    factory = request.node.callspec.params.get("perturbation")
    if factory is None:
        pytest.skip("cannot determine factory")
    p = factory(n_envs_perf)
    p.tick(is_reset=True)
    obs = torch.randn(n_envs_perf, OBS_DIM)

    # Warmup
    for _ in range(PERF_WARMUP):
        p.tick(is_reset=False)
        p.apply(obs.clone())

    # Measure
    start = time.perf_counter()
    for _ in range(PERF_STEPS):
        p.tick(is_reset=False)
        p.apply(obs.clone())
    elapsed_us = (time.perf_counter() - start) * 1e6 / PERF_STEPS
    assert elapsed_us < MAX_US_PER_STEP, (
        f"apply() too slow: {elapsed_us:.1f} µs/step (limit {MAX_US_PER_STEP} µs)"
    )
