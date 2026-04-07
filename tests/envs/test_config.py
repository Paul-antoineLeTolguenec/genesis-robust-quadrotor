"""Tests for PerturbationConfig, SensorConfig, DesyncConfig."""

import pytest

from genesis_robust_rl.envs.config import DesyncConfig, PerturbationConfig, SensorConfig
from genesis_robust_rl.sensor_models import AccelerometerModel, BarometerModel, GyroscopeModel

# ---------- PerturbationConfig ----------


class TestPerturbationConfig:
    """Test PerturbationConfig validation and helpers."""

    def test_empty_config_ok(self):
        cfg = PerturbationConfig()
        assert cfg.all_perturbations() == []

    def test_duplicate_id_raises(self):
        """Duplicate IDs across lists must raise ValueError."""
        from unittest.mock import MagicMock

        p1 = MagicMock()
        p1.id = "dup"
        p2 = MagicMock()
        p2.id = "dup"
        with pytest.raises(ValueError, match="Duplicate perturbation ID"):
            PerturbationConfig(physics=[p1], motor=[p2])

    def test_unique_ids_ok(self):
        from unittest.mock import MagicMock

        p1 = MagicMock()
        p1.id = "a"
        p2 = MagicMock()
        p2.id = "b"
        cfg = PerturbationConfig(physics=[p1], motor=[p2])
        assert len(cfg.all_perturbations()) == 2

    def test_all_perturbations_order(self):
        """Order must be physics -> motor -> sensors -> actions."""
        from unittest.mock import MagicMock

        items = []
        for name in ("phys", "motor", "sensor", "action"):
            m = MagicMock()
            m.id = name
            items.append(m)
        cfg = PerturbationConfig(
            physics=[items[0]],
            motor=[items[1]],
            sensors=[items[2]],
            actions=[items[3]],
        )
        ids = [p.id for p in cfg.all_perturbations()]
        assert ids == ["phys", "motor", "sensor", "action"]


# ---------- SensorConfig ----------


class TestSensorConfig:
    """Test SensorConfig helpers."""

    def test_empty_sensor_config(self):
        cfg = SensorConfig()
        assert cfg.active_sensors() == []
        assert cfg.total_obs_dim() == 0

    def test_single_sensor(self):
        cfg = SensorConfig(gyroscope=GyroscopeModel(n_envs=4))
        active = cfg.active_sensors()
        assert len(active) == 1
        assert active[0][0] == "gyroscope"
        assert cfg.total_obs_dim() == 3

    def test_multiple_sensors(self):
        cfg = SensorConfig(
            gyroscope=GyroscopeModel(n_envs=4),
            accelerometer=AccelerometerModel(n_envs=4),
            barometer=BarometerModel(n_envs=4),
        )
        assert cfg.total_obs_dim() == 3 + 3 + 1
        names = [name for name, _ in cfg.active_sensors()]
        assert names == ["gyroscope", "accelerometer", "barometer"]

    def test_declaration_order(self):
        """Active sensors must follow declaration order, not insertion order."""
        cfg = SensorConfig(
            barometer=BarometerModel(n_envs=4),
            gyroscope=GyroscopeModel(n_envs=4),
        )
        names = [name for name, _ in cfg.active_sensors()]
        assert names == ["gyroscope", "barometer"]


# ---------- DesyncConfig ----------


class TestDesyncConfig:
    """Test DesyncConfig validation."""

    def test_valid_config(self):
        cfg = DesyncConfig(
            obs_delay_range=(0, 3),
            action_delay_range=(1, 5),
            correlation=0.8,
        )
        assert cfg.obs_delay_range == (0, 3)

    def test_invalid_obs_range(self):
        with pytest.raises(ValueError, match="obs_delay_range"):
            DesyncConfig(obs_delay_range=(3, 1), action_delay_range=(0, 1))

    def test_negative_obs_range(self):
        with pytest.raises(ValueError, match="obs_delay_range"):
            DesyncConfig(obs_delay_range=(-1, 3), action_delay_range=(0, 1))

    def test_invalid_action_range(self):
        with pytest.raises(ValueError, match="action_delay_range"):
            DesyncConfig(obs_delay_range=(0, 1), action_delay_range=(5, 2))

    def test_correlation_out_of_range(self):
        with pytest.raises(ValueError, match="correlation"):
            DesyncConfig(obs_delay_range=(0, 1), action_delay_range=(0, 1), correlation=1.5)

    def test_correlation_negative(self):
        with pytest.raises(ValueError, match="correlation"):
            DesyncConfig(obs_delay_range=(0, 1), action_delay_range=(0, 1), correlation=-0.1)
