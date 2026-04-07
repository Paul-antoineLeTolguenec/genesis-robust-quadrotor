"""Tests for CurriculumScheduler."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from genesis_robust_rl.adversarial.curriculum import CurriculumScheduler


def _make_mock_env() -> MagicMock:
    env = MagicMock()
    env.set_curriculum_scale = MagicMock()
    return env


class TestCurriculumScheduler:
    """Schedule computation and callback integration tests."""

    def test_linear_start(self):
        env = _make_mock_env()
        sched = CurriculumScheduler(env, schedule="linear", start_scale=0.0, end_scale=1.0)
        sched(step=0, total_steps=100)
        env.set_curriculum_scale.assert_called_once_with(0.0)

    def test_linear_end(self):
        env = _make_mock_env()
        sched = CurriculumScheduler(env, schedule="linear", start_scale=0.0, end_scale=1.0)
        sched(step=100, total_steps=100)
        env.set_curriculum_scale.assert_called_once_with(1.0)

    def test_linear_midpoint(self):
        env = _make_mock_env()
        sched = CurriculumScheduler(env, schedule="linear", start_scale=0.0, end_scale=1.0)
        sched(step=50, total_steps=100)
        env.set_curriculum_scale.assert_called_once_with(0.5)

    def test_cosine_start(self):
        env = _make_mock_env()
        sched = CurriculumScheduler(env, schedule="cosine", start_scale=0.0, end_scale=1.0)
        sched(step=0, total_steps=100)
        env.set_curriculum_scale.assert_called_once_with(0.0)

    def test_cosine_end(self):
        env = _make_mock_env()
        sched = CurriculumScheduler(env, schedule="cosine", start_scale=0.0, end_scale=1.0)
        sched(step=100, total_steps=100)
        env.set_curriculum_scale.assert_called_once_with(1.0)

    def test_step_before_midpoint(self):
        env = _make_mock_env()
        sched = CurriculumScheduler(env, schedule="step", start_scale=0.0, end_scale=1.0)
        sched(step=10, total_steps=100)
        env.set_curriculum_scale.assert_called_once_with(0.0)

    def test_step_after_midpoint(self):
        env = _make_mock_env()
        sched = CurriculumScheduler(env, schedule="step", start_scale=0.0, end_scale=1.0)
        sched(step=60, total_steps=100)
        env.set_curriculum_scale.assert_called_once_with(1.0)

    def test_custom_callable(self):
        env = _make_mock_env()
        sched = CurriculumScheduler(env, schedule=lambda s, t: 0.42)
        sched(step=0, total_steps=100)
        env.set_curriculum_scale.assert_called_once_with(0.42)

    def test_unknown_schedule_raises(self):
        env = _make_mock_env()
        with pytest.raises(ValueError, match="Unknown schedule"):
            CurriculumScheduler(env, schedule="exponential")

    def test_kwargs_forwarded(self):
        """Extra kwargs don't crash the callback."""
        env = _make_mock_env()
        sched = CurriculumScheduler(env, schedule="linear")
        sched(step=0, total_steps=100, drone_metrics={}, adv_metrics={})
        env.set_curriculum_scale.assert_called_once()
