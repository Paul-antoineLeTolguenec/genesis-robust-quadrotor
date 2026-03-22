"""Category 1 — Robot physical dynamics perturbations.

Implemented perturbations:
  1.1  MassShift                 — per-env body mass shift via set_links_mass_shift()
  1.2  COMShift                  — per-env center-of-mass shift via set_links_COM_shift()
  1.3  InertiaTensor             — per-env inertia scaling via Huygens-Steiner approximation
  1.4  MotorArmature             — per-env rotor inertia scaling via set_dofs_armature()
  1.5  FrictionRatio             — per-env geometry friction ratio via set_geoms_friction_ratio()
  1.6  PositionGainKp            — per-env DOF position gain via set_dofs_kp()
  1.7  VelocityGainKv            — per-env DOF velocity gain via set_dofs_kv()
  1.8  JointStiffness            — per-env DOF joint stiffness via set_dofs_stiffness()
  1.9  JointDamping              — per-env DOF joint damping via set_dofs_damping()
  1.10 AeroDragCoeff             — per-env aerodynamic drag coefficient via apply_links_external_force()
  1.11 GroundEffect              — per-env Cheeseman-Bennett thrust augmentation via apply_links_external_force()
  1.12 ChassisGeometryAsymmetry  — per-env arm-length deviations propagated as CoM + mass shifts
  1.13 PropellerBladeDamage      — per-env blade efficiency loss via net thrust correction force
  1.14 StructuralFlexibility     — per-env spring-damper residual torque via apply_links_external_torque()
  1.15 BatteryVoltageSag         — stateful per-env SoC depletion → KF scaling via thrust correction force
"""
from __future__ import annotations

import math
from typing import Any, Callable

import torch
from torch import Tensor

from genesis_robust_rl.perturbations.base import (
    EnvState,
    ExternalWrenchPerturbation,
    GenesisSetterPerturbation,
    PhysicsPerturbation,
    register,
)


@register("mass_shift")
class MassShift(GenesisSetterPerturbation):
    """1.1 — Per-env body mass shift applied via set_links_mass_shift().

    Adds a scalar Δm [kg] to the drone body mass each episode (default) or step.
    The setter takes effect at the next scene.step() call (post_physics¹ timing).

    Args:
        setter_fn: bound method drone.set_links_mass_shift — provided by the env.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform; {"mean", "std"} for gaussian.
        bounds: hard clamp [min_kg, max_kg] applied after curriculum scaling.
        nominal: unperturbed mass shift (0.0 kg).
        lipschitz_k: max |Δm| per step in adversarial mode (None = per_episode only).
    """

    def __init__(
        self,
        setter_fn: Callable[[Tensor, Tensor], None],
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-0.5, 1.0),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": -0.05, "high": 0.1}
        super().__init__(
            setter_fn=setter_fn,
            id="mass_shift",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )


@register("com_shift")
class COMShift(GenesisSetterPerturbation):
    """1.2 — Per-env center-of-mass shift applied via set_links_COM_shift().

    Adds a 3D vector Δr [m] to the drone body CoM each episode (default) or step.
    The setter takes effect at the next scene.step() call (post_physics¹ timing).

    Args:
        setter_fn: bound method drone.set_links_COM_shift — provided by the env.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform; {"mean", "std"} for gaussian.
        bounds: hard clamp [min_m, max_m] applied per axis after curriculum scaling.
        nominal: unperturbed CoM shift (default [0.0, 0.0, 0.0] m).
        lipschitz_k: max ‖Δr‖ per step in adversarial mode (None = per_episode only).
    """

    def __init__(
        self,
        setter_fn: Callable[[Tensor, Tensor], None],
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-0.05, 0.05),
        nominal: list[float] | None = None,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": -0.02, "high": 0.02}
        if nominal is None:
            nominal = [0.0, 0.0, 0.0]
        super().__init__(
            setter_fn=setter_fn,
            id="com_shift",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )


@register("inertia_tensor")
class InertiaTensor(PhysicsPerturbation):
    """1.3 — Per-env inertia tensor scaling via Huygens-Steiner approximation.

    Genesis has no direct inertia tensor setter. This class approximates a per-axis
    inertia scaling by jointly adjusting mass shift and CoM shift using the
    parallel-axis (Huygens-Steiner) theorem:

        I_perturbed ≈ I_nom * scale   (per axis)

    Implementation: for each axis j, the inertia is I_j = m * r_j^2 where r_j is
    the effective radius of gyration. Scaling I_j by s_j is achieved by scaling
    the CoM offset along that axis by sqrt(s_j) (keeping mass constant) — or
    equivalently by scaling mass while adjusting CoM offset. We use a combined
    approach: scale = sqrt(s) applied to an effective CoM arm, then use
    set_links_mass_shift and set_links_COM_shift to inject both effects.

    Concretely, for a scale vector s ∈ ℝ³:
      - Δm   = m_nom * (mean(s) - 1)          # isotropic mass contribution
      - Δr_j = r_nom_j * (sqrt(s_j) - 1)      # per-axis CoM arm adjustment

    Since Genesis setters take relative values, we expose the scaling factor
    directly as the perturbation value (dimension=(3,), bounds=[0.5, 1.5]).

    Args:
        mass_setter_fn: bound method drone.set_links_mass_shift — from env.
        com_setter_fn: bound method drone.set_links_COM_shift — from env.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        m_nom: nominal body mass [kg] used for Huygens-Steiner mass delta.
        r_nom: nominal radii of gyration per axis [m, m, m].
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform; {"mean", "std"} for gaussian.
        bounds: hard clamp [min_scale, max_scale] applied per axis.
        nominal: unperturbed scale ([1.0, 1.0, 1.0]).
    """

    def __init__(
        self,
        mass_setter_fn: Callable[[Tensor, Tensor], None],
        com_setter_fn: Callable[[Tensor, Tensor], None],
        n_envs: int,
        dt: float = 0.01,
        m_nom: float = 0.5,
        r_nom: list[float] | None = None,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.5, 1.5),
        nominal: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.8, "high": 1.2}
        if nominal is None:
            nominal = [1.0, 1.0, 1.0]
        if r_nom is None:
            r_nom = [0.05, 0.05, 0.03]  # typical quadrotor radii of gyration [m]

        self.m_nom = m_nom
        self.r_nom = torch.tensor(r_nom, dtype=torch.float32)  # [3]
        self._mass_setter_fn = mass_setter_fn
        self._com_setter_fn = com_setter_fn

        super().__init__(
            id="inertia_tensor",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=None,
            **kwargs,
        )

    def apply(self, scene: Any, drone: Any, env_state: EnvState) -> None:
        """Apply Huygens-Steiner approximation: call both mass and CoM setters.

        For scale vector s ∈ [n_envs, 3]:
          Δm   = m_nom * (mean(s, axis=1) - 1)      shape [n_envs]
          Δr_j = r_nom_j * (sqrt(s_j) - 1)          shape [n_envs, 3]
        """
        assert self._current_value is not None, "call tick() before apply()"
        scale = self._current_value  # [n_envs, 3]
        envs_idx = torch.arange(self.n_envs)

        # Mass delta: isotropic contribution from mean scale
        delta_m = self.m_nom * (scale.mean(dim=1, keepdim=False) - 1.0)  # [n_envs]

        # CoM delta: per-axis arm adjustment via sqrt of scale
        # r_nom broadcast: [1, 3] * [n_envs, 3]
        delta_r = self.r_nom.unsqueeze(0) * (scale.sqrt() - 1.0)  # [n_envs, 3]

        self._mass_setter_fn(delta_m, envs_idx)
        self._com_setter_fn(delta_r, envs_idx)


@register("motor_armature")
class MotorArmature(GenesisSetterPerturbation):
    """1.4 — Per-env rotor inertia (armature) scaling via set_dofs_armature().

    Scales the nominal armature value by a multiplicative factor per env and per episode.
    The setter triggers an expensive physics recompute — must only be called at episode
    reset, never at every step.

    **Application hook:** `reset` — the env calls apply() only on episode reset.
    **Risk:** medium — set_dofs_armature() triggers a solver recompute.

    The perturbation value is a scalar multiplier in [0.5, 1.5] representing the ratio
    `a_perturbed / a_nom`. The env is responsible for multiplying by the nominal value
    before passing to the Genesis setter if needed; the perturbation exposes the ratio
    directly so the env can scale appropriately.

    Args:
        setter_fn: bound method drone.set_dofs_armature — provided by the env.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform; {"mean", "std"} for gaussian.
        bounds: hard clamp [min_ratio, max_ratio] — default (0.5, 1.5).
        nominal: unperturbed armature ratio (1.0 = nominal).
    """

    def __init__(
        self,
        setter_fn: Callable[[Tensor, Tensor], None],
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.5, 1.5),
        nominal: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.7, "high": 1.3}
        super().__init__(
            setter_fn=setter_fn,
            id="motor_armature",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(),
            lipschitz_k=None,
            **kwargs,
        )


@register("friction_ratio")
class FrictionRatio(GenesisSetterPerturbation):
    """1.5 — Per-env geometry friction ratio via set_geoms_friction_ratio().

    Scales the contact friction coefficient by a multiplicative ratio per env.
    Applied per episode (default) or per step via the Genesis native setter.
    The setter operates per-env without triggering recompilation.

    **Application hook:** `reset` or `pre_physics` — per-step supported (dynamic mode).
    **Risk:** low.

    The perturbation value is a scalar ratio in [0.1, 3.0] representing
    `mu_perturbed / mu_nom`. A value of 1.0 leaves friction unchanged.

    Args:
        setter_fn: bound method drone.set_geoms_friction_ratio — provided by the env.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform; {"mean", "std"} for gaussian.
        bounds: hard clamp [min_ratio, max_ratio] — default (0.1, 3.0).
        nominal: unperturbed friction ratio (1.0 = nominal).
        lipschitz_k: max |Δratio| per step in adversarial dynamic mode (0.05 recommended).
    """

    def __init__(
        self,
        setter_fn: Callable[[Tensor, Tensor], None],
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.1, 3.0),
        nominal: float = 1.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.5, "high": 1.5}
        super().__init__(
            setter_fn=setter_fn,
            id="friction_ratio",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )


@register("position_gain_kp")
class PositionGainKp(GenesisSetterPerturbation):
    """1.6 — Per-env DOF position gain (Kp) via set_dofs_kp().

    Multiplies the nominal Kp gain by a per-env scalar ratio sampled from the
    configured distribution. Applied per episode (default) or per step.

    **Application hook:** `reset` or `pre_physics` — per-step supported (dynamic mode).
    **Risk:** low.

    The perturbation value is a scalar ratio in [0.5, 2.0] representing
    `Kp_perturbed / Kp_nom`. A value of 1.0 leaves gain unchanged.

    Args:
        setter_fn: bound method drone.set_dofs_kp — provided by the env.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform; {"mean", "std"} for gaussian.
        bounds: hard clamp [min_ratio, max_ratio] — default (0.5, 2.0).
        nominal: unperturbed Kp ratio (1.0 = nominal).
        lipschitz_k: max |Δratio| per step in adversarial dynamic mode.
    """

    def __init__(
        self,
        setter_fn: Callable[[Tensor, Tensor], None],
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.5, 2.0),
        nominal: float = 1.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.5, "high": 2.0}
        super().__init__(
            setter_fn=setter_fn,
            id="position_gain_kp",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )


@register("velocity_gain_kv")
class VelocityGainKv(GenesisSetterPerturbation):
    """1.7 — Per-env DOF velocity gain (Kv) via set_dofs_kv().

    Multiplies the nominal Kv gain by a per-env scalar ratio sampled from the
    configured distribution. Applied per episode (default) or per step.

    **Application hook:** `reset` or `pre_physics` — per-step supported (dynamic mode).
    **Risk:** low.

    The perturbation value is a scalar ratio in [0.5, 2.0] representing
    `Kv_perturbed / Kv_nom`. A value of 1.0 leaves gain unchanged.

    Args:
        setter_fn: bound method drone.set_dofs_kv — provided by the env.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform; {"mean", "std"} for gaussian.
        bounds: hard clamp [min_ratio, max_ratio] — default (0.5, 2.0).
        nominal: unperturbed Kv ratio (1.0 = nominal).
        lipschitz_k: max |Δratio| per step in adversarial dynamic mode.
    """

    def __init__(
        self,
        setter_fn: Callable[[Tensor, Tensor], None],
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.5, 2.0),
        nominal: float = 1.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.5, "high": 2.0}
        super().__init__(
            setter_fn=setter_fn,
            id="velocity_gain_kv",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )


@register("joint_stiffness")
class JointStiffness(GenesisSetterPerturbation):
    """1.8 — Per-env DOF joint stiffness via set_dofs_stiffness().

    Multiplies the nominal joint stiffness by a per-env scalar ratio sampled from
    the configured distribution. Applied per episode (default) or per step.

    **Application hook:** `reset` or `pre_physics` — per-step supported (dynamic mode).
    **Risk:** low.

    The perturbation value is a scalar ratio in [0.5, 2.0] representing
    `stiffness_perturbed / stiffness_nom`. A value of 1.0 leaves stiffness unchanged.

    Args:
        setter_fn: bound method drone.set_dofs_stiffness — provided by the env.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform; {"mean", "std"} for gaussian.
        bounds: hard clamp [min_ratio, max_ratio] — default (0.5, 2.0).
        nominal: unperturbed stiffness ratio (1.0 = nominal).
        lipschitz_k: max |Δratio| per step in adversarial dynamic mode.
    """

    def __init__(
        self,
        setter_fn: Callable[[Tensor, Tensor], None],
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.5, 2.0),
        nominal: float = 1.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.5, "high": 2.0}
        super().__init__(
            setter_fn=setter_fn,
            id="joint_stiffness",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )


@register("joint_damping")
class JointDamping(GenesisSetterPerturbation):
    """1.9 — Per-env DOF joint damping via set_dofs_damping().

    Multiplies the nominal joint damping by a per-env scalar ratio sampled from
    the configured distribution. Applied per episode (default) or per step.

    **Application hook:** `reset` or `pre_physics` — per-step supported (dynamic mode).
    **Risk:** low.

    The perturbation value is a scalar ratio in [0.5, 2.0] representing
    `damping_perturbed / damping_nom`. A value of 1.0 leaves damping unchanged.

    Args:
        setter_fn: bound method drone.set_dofs_damping — provided by the env.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform; {"mean", "std"} for gaussian.
        bounds: hard clamp [min_ratio, max_ratio] — default (0.5, 2.0).
        nominal: unperturbed damping ratio (1.0 = nominal).
        lipschitz_k: max |Δratio| per step in adversarial dynamic mode.
    """

    def __init__(
        self,
        setter_fn: Callable[[Tensor, Tensor], None],
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.5, 2.0),
        nominal: float = 1.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.5, "high": 2.0}
        super().__init__(
            setter_fn=setter_fn,
            id="joint_damping",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )


@register("aero_drag_coeff")
class AeroDragCoeff(ExternalWrenchPerturbation):
    """1.10 — Per-env aerodynamic drag coefficient via apply_links_external_force().

    Injects a velocity-squared drag force F = -Cd_perturbed * v^2 (per axis) as an
    external wrench on the drone body link. The perturbation value is a multiplicative
    scaling vector applied to the nominal drag coefficient Cd_nom along each body axis.

    The drag force for environment i at each step is:

        F_i = -Cd_i * sign(v_i) * v_i^2       (element-wise, [n_envs, 3])

    where Cd_i = _current_value[i] * Cd_nom is the perturbed coefficient per axis, and
    v_i = env_state.vel[i] is the body linear velocity.

    **Application hook:** pre_physics — injected before scene.step() as external force.
    **Frame:** world (default; caller may override to "local" for body-frame drag).
    **Risk:** low.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        Cd_nom: nominal drag coefficient per axis [N*s2/m2] (default 0.1 for each axis).
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform; {"mean", "std"} for gaussian.
        bounds: hard clamp [min_multiplier, max_multiplier] on the Cd scaling factor.
        nominal: unperturbed Cd multiplier per axis ([1.0, 1.0, 1.0]).
        lipschitz_k: max |delta_multiplier| per step in adversarial dynamic mode (None = per_episode).
        frame: "world" or "local" — reference frame for the force application.
        link_idx: index of the drone body link (default 0).
        duration_mode: "continuous" or "pulse".
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        Cd_nom: list[float] | None = None,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.5, 2.0),
        nominal: list[float] | None = None,
        lipschitz_k: float | None = None,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if Cd_nom is None:
            Cd_nom = [0.1, 0.1, 0.1]
        if distribution_params is None:
            distribution_params = {"low": 0.5, "high": 2.0}
        if nominal is None:
            nominal = [1.0, 1.0, 1.0]

        # Store Cd_nom as tensor for vectorized force computation
        self._Cd_nom = torch.tensor(Cd_nom, dtype=torch.float32)  # [3]

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            id="aero_drag_coeff",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(3,),
            lipschitz_k=lipschitz_k,
            **kwargs,
        )

    def _compute_wrench(self, env_state: EnvState) -> torch.Tensor:
        """Compute aerodynamic drag force F = -Cd_perturbed * sign(v) * v^2.

        Returns:
            Tensor[n_envs, 3] — drag force vector in the configured reference frame.
        """
        assert self._current_value is not None, "call tick() before apply()"
        # _current_value: [n_envs, 3] — per-env Cd multiplier per axis
        # _Cd_nom: [3] — broadcast over n_envs
        Cd = self._current_value * self._Cd_nom.to(self._current_value.device)  # [n_envs, 3]
        vel = env_state.vel  # [n_envs, 3]
        # F = -Cd * sign(v) * v^2  (preserves drag direction)
        force = -Cd * vel.sign() * vel.pow(2)  # [n_envs, 3]
        return force


@register("ground_effect")
class GroundEffect(ExternalWrenchPerturbation):
    """1.11 — Per-env ground-effect thrust augmentation via apply_links_external_force().

    Models the Cheeseman-Bennett ground effect: rotor thrust increases near the ground
    due to reduced induced velocity. The correction factor is:

        k_ge = 1 / (1 - (R / (4h))^2)   for h > R/2
        k_ge = clamped to max_k_ge       for h <= R/2 (near-singularity protection)

    where h = env_state.pos[:,2] (altitude AGL, m) and R = rotor_radius (m).

    The additional thrust injected as an upward external force per environment is:

        ΔF_i = (k_ge_i - 1) * T_nominal   [N, upward world-Z]

    At high altitude (h >> R), k_ge → 1 → ΔF → 0 (no ground effect).

    This perturbation uses distribution="constant" because the force is fully determined
    by the physics model; there is no stochastic sampling.

    **Application hook:** pre_physics — injected as external force before scene.step().
    **Frame:** world (upward +Z force).
    **Risk:** medium.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        rotor_radius: effective rotor radius R [m] (default 0.1 m).
        nominal_thrust: nominal total thrust at hover [N] (default 9.81 * 0.5 N).
        max_k_ge: maximum ground-effect multiplier (clamp for h <= R/2, default 2.0).
        frame: reference frame for force application (default "world").
        link_idx: drone body link index (default 0).
        duration_mode: "continuous" or "pulse".
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        rotor_radius: float = 0.1,
        nominal_thrust: float = 9.81 * 0.5,
        max_k_ge: float = 2.0,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        self._rotor_radius = rotor_radius
        self._nominal_thrust = nominal_thrust
        self._max_k_ge = max_k_ge

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            id="ground_effect",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "dynamic"),
            frequency=kwargs.pop("frequency", "per_step"),
            scope="per_env",
            distribution="constant",
            distribution_params={"value": 0.0},
            bounds=(0.0, (max_k_ge - 1.0) * nominal_thrust),
            nominal=0.0,
            dimension=(3,),
            lipschitz_k=None,
            **kwargs,
        )

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute upward ground-effect force via Cheeseman-Bennett model.

        For each environment:
          h = altitude AGL (env_state.pos[:, 2]), clamped to h >= R/2
          k_ge = 1 / (1 - (R / (4h))^2), clamped to [1, max_k_ge]
          ΔF = (k_ge - 1) * nominal_thrust  (upward +Z in world frame)

        Returns:
            Tensor[n_envs, 3] — external force; only Z-component is non-zero.
        """
        R = self._rotor_radius
        h = env_state.pos[:, 2]  # [n_envs] altitude AGL

        # Clamp altitude to avoid singularity at h = R/4
        h_safe = h.clamp(min=R / 2.0)

        # Cheeseman-Bennett multiplier
        ratio = R / (4.0 * h_safe)          # [n_envs]
        k_ge = 1.0 / (1.0 - ratio.pow(2))   # [n_envs]

        # Clamp to physical range [1, max_k_ge] — no negative effect, cap singularity
        k_ge = k_ge.clamp(1.0, self._max_k_ge)

        # Zero-out effect where altitude is above the influence zone (h > 4R)
        above_zone = h > 4.0 * R
        k_ge = torch.where(above_zone, torch.ones_like(k_ge), k_ge)

        # Additional thrust force (upward +Z, world frame)
        delta_f = (k_ge - 1.0) * self._nominal_thrust  # [n_envs]

        # Build force tensor [n_envs, 3] — only Z component
        force = torch.zeros(self.n_envs, 3, dtype=torch.float32, device=h.device)
        force[:, 2] = delta_f

        return force


# ---------------------------------------------------------------------------
# ARM_ANGLES for a standard quadrotor (NED / X-frame convention)
# Motor positions at 45°, 135°, 225°, 315° from the forward axis.
# ---------------------------------------------------------------------------
_ARM_ANGLES = torch.tensor(
    [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4],
    dtype=torch.float32,
)  # [4]


@register("chassis_geometry_asymmetry")
class ChassisGeometryAsymmetry(PhysicsPerturbation):
    """1.12 — Per-env arm-length deviations propagated as CoM + mass shifts.

    Models structural asymmetry of the quadrotor chassis by perturbing each arm
    length independently. A deviation vector δ ∈ ℝ⁴ (one scalar per arm, in metres)
    is sampled per episode. The geometric asymmetry is then propagated into equivalent
    CoM and mass shifts using the planar mass-moment identity:

        Δx_com = Σ_j (δ_j × cos(θ_j)) / n_arms
        Δy_com = Σ_j (δ_j × cos(θ_j + π/2)) / n_arms
        Δz_com = 0

    where θ_j = [π/4, 3π/4, 5π/4, 7π/4] are the standard quadrotor arm angles.

    The mass shift is zero (arm deviations affect geometry, not mass):
        Δm = 0

    Both setters are called at each episode reset via apply().

    Args:
        mass_setter_fn: bound method drone.set_links_mass_shift — provided by env.
        com_setter_fn:  bound method drone.set_links_COM_shift  — provided by env.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform; {"mean", "std"} for gaussian.
        bounds: hard clamp [min_dev, max_dev] in metres applied per arm after curriculum scaling.
        nominal: unperturbed arm deviations ([0.0, 0.0, 0.0, 0.0] m).
        arm_length: nominal arm length [m] — used to convert relative bounds to absolute.
    """

    # Standard quadrotor arm angles (class-level constant, allocated once)
    _ARM_ANGLES: torch.Tensor = _ARM_ANGLES  # [4]

    def __init__(
        self,
        mass_setter_fn: Callable[[Tensor, Tensor], None],
        com_setter_fn: Callable[[Tensor, Tensor], None],
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-0.05, 0.05),
        nominal: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": -0.05, "high": 0.05}
        if nominal is None:
            nominal = [0.0, 0.0, 0.0, 0.0]

        self._mass_setter_fn = mass_setter_fn
        self._com_setter_fn = com_setter_fn

        super().__init__(
            id="chassis_geometry_asymmetry",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(4,),
            lipschitz_k=None,
            **kwargs,
        )

    def apply(self, scene: Any, drone: Any, env_state: EnvState) -> None:
        """Propagate arm-length deviations into CoM shift and zero mass shift.

        For arm deviations δ ∈ [n_envs, 4]:
          Δx_com_i = Σ_j δ_{i,j} × cos(θ_j)       / 4
          Δy_com_i = Σ_j δ_{i,j} × cos(θ_j + π/2) / 4
          Δz_com_i = 0

        Calls mass_setter_fn(Δm=0, envs_idx) and com_setter_fn(Δr, envs_idx).
        """
        assert self._current_value is not None, "call tick() before apply()"
        deviations = self._current_value  # [n_envs, 4]
        envs_idx = torch.arange(self.n_envs, device=deviations.device)

        angles = self._ARM_ANGLES.to(deviations.device)  # [4]
        cos_x = torch.cos(angles)                         # [4]
        cos_y = torch.cos(angles + math.pi / 2)           # [4] = -sin(angles)

        # Planar CoM shift: [n_envs, 4] @ [4] → [n_envs] / n_arms
        delta_x = (deviations * cos_x).sum(dim=1) / 4.0  # [n_envs]
        delta_y = (deviations * cos_y).sum(dim=1) / 4.0  # [n_envs]

        delta_com = torch.stack(
            [delta_x, delta_y, torch.zeros_like(delta_x)], dim=1
        )  # [n_envs, 3]

        # Arm deviations do not change mass (geometry only)
        delta_m = torch.zeros(self.n_envs, dtype=torch.float32, device=deviations.device)

        self._mass_setter_fn(delta_m, envs_idx)
        self._com_setter_fn(delta_com, envs_idx)


@register("propeller_blade_damage")
class PropellerBladeDamage(ExternalWrenchPerturbation):
    """1.13 — Per-env propeller blade damage via per-propeller KF scaling.

    Models structural blade damage as a per-propeller efficiency ratio η_j ∈ [0.5, 1.0].
    The damage effect is propagated as a net corrective vertical force applied to the
    drone body link, representing the reduction in effective thrust relative to nominal:

        ΔF_z = Σ_j (η_j − 1) × KF × ω_j²     (N, world +Z upward)

    At nominal efficiency [1,1,1,1]: ΔF_z = 0 (no damage, no perturbation).
    With partial damage η_j < 1: ΔF_z < 0 (net thrust reduction).

    **Application hook:** reset — efficiency sampled per episode, applied each step.
    **Frame:** world (default). **Risk:** low.

    Note: apply() does NOT overwrite _current_value so that privileged obs returns
    the efficiency vector (more informative than the derived force).

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        KF: nominal thrust coefficient [N/(RPM²)] — default 3.16e-10.
        distribution: "uniform" or "beta".
        distribution_params: {"low", "high"} for uniform; {"alpha", "beta"} for beta.
        bounds: hard clamp [min_eff, max_eff] per propeller — default (0.5, 1.0).
        nominal: unperturbed efficiency ([1.0, 1.0, 1.0, 1.0]).
        frame: reference frame for force ("world" or "local").
        link_idx: body link index (default 0).
        duration_mode: "continuous" or "pulse".
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        KF: float = 3.16e-10,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.5, 1.0),
        nominal: list[float] | None = None,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.5, "high": 1.0}
        if nominal is None:
            nominal = [1.0, 1.0, 1.0, 1.0]

        self._KF = KF

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            id="propeller_blade_damage",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(4,),
            lipschitz_k=None,
            **kwargs,
        )

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute net vertical thrust correction from blade efficiency loss.

        Returns:
            Tensor[n_envs, 3] — corrective force; only the Z-component is non-zero.
        """
        assert self._current_value is not None, "call tick() before apply()"
        efficiency = self._current_value  # [n_envs, 4]
        rpm = env_state.rpm  # [n_envs, 4]

        # delta_thrust_j = (η_j - 1) * KF * ω_j²  — negative for damage (η < 1)
        delta_per_prop = (efficiency - 1.0) * self._KF * rpm.pow(2)  # [n_envs, 4]
        delta_f_z = delta_per_prop.sum(dim=1)  # [n_envs]

        force = torch.zeros(self.n_envs, 3, dtype=torch.float32, device=rpm.device)
        force[:, 2] = delta_f_z
        return force

    def apply(self, scene: Any, drone: Any, env_state: EnvState) -> None:
        """Apply thrust correction — preserves _current_value (efficiency ratios).

        Overrides base to avoid overwriting _current_value with the force vector,
        keeping the efficiency vector available for privileged observation.
        """
        force = self._compute_wrench(env_state)
        local = self.frame == "local"
        envs_idx = torch.arange(self.n_envs, device=force.device)
        scene.rigid_solver.apply_links_external_force(force, self.link_idx, envs_idx, local=local)


@register("structural_flexibility")
class StructuralFlexibility(ExternalWrenchPerturbation):
    """1.14 — Per-env structural flexibility modeled as a spring-damper residual torque.

    Samples stiffness k [N·m/rad] and damping b [N·m·s/rad] per env per episode.
    The flexibility effect is applied as an isotropic residual torque about all three
    body axes at each physics step:

        τ = −(k × θ_approx + b × ω_body)

    where θ_approx = ω_body × dt is a first-order approximation of the angular
    deformation angle, and ω_body = env_state.ang_vel is the body angular velocity.

    At nominal [k, b] = [0, 0] (curriculum_scale=0): τ = 0 (rigid body).
    At full perturbation: τ introduces damping and spring-like resistance.

    **Application hook:** reset (params sampled per episode, torque applied each step).
    **Frame:** local (body frame). **Risk:** medium.
    **Dimension:** (2,) — stores [k, b] as _current_value for privileged obs.

    Note: apply() calls apply_links_external_torque (not force) and does NOT
    overwrite _current_value so that privileged obs returns [k, b].

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution: "uniform" (default; only supported distribution).
        distribution_params: {"k_low", "k_high", "b_low", "b_high"}.
        bounds: hard clamp [0, 500] applied element-wise to [k, b].
        nominal: [0.0, 0.0] — zero stiffness/damping = rigid body.
        frame: reference frame for torque ("local" or "world") — default "local".
        link_idx: body link index (default 0).
        duration_mode: "continuous" or "pulse".
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 500.0),
        nominal: list[float] | None = None,
        frame: str = "local",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"k_low": 50.0, "k_high": 500.0, "b_low": 0.1, "b_high": 5.0}
        if nominal is None:
            nominal = [0.0, 0.0]

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            wrench_type="torque",
            id="structural_flexibility",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "fixed"),
            frequency=kwargs.pop("frequency", "per_episode"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(2,),
            lipschitz_k=None,
            **kwargs,
        )

    def _draw(self) -> Tensor:
        """Draw [k, b] independently from their respective uniform ranges."""
        p = self.distribution_params
        n = self.n_envs
        k = torch.empty(n).uniform_(p["k_low"], p["k_high"])   # [n_envs]
        b = torch.empty(n).uniform_(p["b_low"], p["b_high"])   # [n_envs]
        return torch.stack([k, b], dim=1)  # [n_envs, 2]

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute spring-damper residual torque.

        τ = −(k × θ_approx + b × ω_body),  θ_approx = ω_body × dt
        Returns Tensor[n_envs, 3].
        """
        assert self._current_value is not None, "call tick() before apply()"
        k = self._current_value[:, 0].unsqueeze(1)  # [n_envs, 1]
        b = self._current_value[:, 1].unsqueeze(1)  # [n_envs, 1]
        ang_vel = env_state.ang_vel                  # [n_envs, 3]
        theta = ang_vel * self.dt                    # first-order angular deformation
        return -(k * theta + b * ang_vel)            # [n_envs, 3]

    def apply(self, scene: Any, drone: Any, env_state: EnvState) -> None:
        """Apply spring-damper torque via apply_links_external_torque.

        Overrides base to: (1) call torque instead of force, (2) preserve
        _current_value ([k, b]) for privileged observation.
        """
        torque = self._compute_wrench(env_state)
        local = self.frame == "local"
        envs_idx = torch.arange(self.n_envs, device=torque.device)
        scene.rigid_solver.apply_links_external_torque(torque, self.link_idx, envs_idx, local=local)


@register("battery_voltage_sag")
class BatteryVoltageSag(ExternalWrenchPerturbation):
    """1.15 — Stateful per-env battery voltage sag via SoC depletion.

    Models battery discharge as a monotonically decreasing state of charge (SoC).
    At each step, SoC decreases by a per-env discharge rate:

        SoC_{t+1} = max(SoC_t − r × dt,  0)

    The voltage ratio (≈ normalized terminal voltage) is:

        v_ratio = 0.7 + 0.3 × SoC   ∈ [0.7, 1.0]

    The voltage sag reduces effective motor KF: KF_eff = KF_nom × v_ratio².
    The net corrective thrust force is:

        ΔF_z = Σ_j (v_ratio² − 1) × KF_nom × ω_j²     (N, world +Z upward)

    At full charge (SoC=1): v_ratio=1.0, ΔF_z=0 (no sag).
    At empty (SoC=0): v_ratio=0.7, ΔF_z < 0 (30% voltage drop → ~49% KF reduction).

    **Stateful:** reset() samples initial SoC and discharge_rate; step() advances SoC.
    **Application hook:** pre_physics — applied each step.
    **Frame:** world. **Risk:** low.

    Note: apply() does NOT overwrite _current_value so that privileged obs returns
    the voltage_ratio (scalar per env), not the derived force.

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        KF: nominal thrust coefficient [N/(RPM²)] — default 3.16e-10.
        distribution: "uniform" (initial SoC).
        distribution_params: {"low": min_SoC, "high": max_SoC,
                               "discharge_rate_low": min_r, "discharge_rate_high": max_r}.
        bounds: voltage_ratio bounds — default (0.7, 1.0).
        nominal: 1.0 (full charge, no sag).
        frame: "world" (default).
        link_idx: body link index (default 0).
        duration_mode: "continuous" (default).
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        KF: float = 3.16e-10,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.7, 1.0),
        nominal: float = 1.0,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {
                "low": 0.5,
                "high": 1.0,
                "discharge_rate_low": 0.001,
                "discharge_rate_high": 0.01,
            }

        self._KF = KF
        # Initialize state tensors at nominal (full charge, no discharge)
        self._soc: Tensor = torch.ones(n_envs)
        self._discharge_rate: Tensor = torch.zeros(n_envs)

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            id="battery_voltage_sag",
            n_envs=n_envs,
            dt=dt,
            value_mode=kwargs.pop("value_mode", "dynamic"),
            frequency=kwargs.pop("frequency", "per_step"),
            scope="per_env",
            distribution=distribution,
            distribution_params=distribution_params,
            bounds=bounds,
            nominal=nominal,
            dimension=(),
            lipschitz_k=None,
            **kwargs,
        )
        self.is_stateful = True
        # Set initial _current_value to nominal voltage_ratio
        self._current_value = torch.ones(n_envs)

    def _voltage_ratio(self) -> Tensor:
        """Map SoC ∈ [0, 1] to voltage_ratio ∈ [0.7, 1.0]."""
        return (0.7 + 0.3 * self._soc).clamp(0.7, 1.0)

    def reset(self, env_ids: Tensor) -> None:
        """Sample initial SoC and discharge rate for selected envs.

        Applies curriculum_scale to scale the SoC deviation from full charge (1.0)
        and the discharge_rate deviation from zero (0.0).
        """
        p = self.distribution_params
        n = len(env_ids)

        # SoC: curriculum_scale compresses toward nominal (1.0 = full charge)
        soc_raw = torch.empty(n).uniform_(p["low"], p["high"])
        self._soc[env_ids] = (
            1.0 + (soc_raw - 1.0) * self.curriculum_scale
        ).clamp(0.0, 1.0)

        # Discharge rate: curriculum_scale compresses toward 0 (no discharge)
        dr_raw = torch.empty(n).uniform_(
            p["discharge_rate_low"], p["discharge_rate_high"]
        )
        self._discharge_rate[env_ids] = (dr_raw * self.curriculum_scale).clamp(
            0.0, p["discharge_rate_high"]
        )

        self._current_value = self._voltage_ratio()

    def sample(self) -> Tensor:
        """Return current voltage ratio derived from SoC state (no distribution draw).

        Overrides base sample() — BatteryVoltageSag dynamics are fully governed by
        reset()/step(); sample() just exposes the current state as _current_value.
        """
        self._current_value = self._voltage_ratio()
        return self._current_value

    def step(self) -> None:
        """Advance battery discharge by one timestep."""
        self._soc = (self._soc - self._discharge_rate * self.dt).clamp(0.0, 1.0)
        self._current_value = self._voltage_ratio()

    def _compute_wrench(self, env_state: EnvState) -> Tensor:
        """Compute net corrective thrust force from voltage sag.

        ΔF_z = Σ_j (v_ratio² − 1) × KF × ω_j²

        Returns:
            Tensor[n_envs, 3] — corrective force; only Z-component is non-zero.
        """
        assert self._current_value is not None, "call tick() before apply()"
        vr = self._current_value  # [n_envs] voltage_ratio
        rpm = env_state.rpm       # [n_envs, 4]

        # (v_ratio^2 - 1) < 0 for vr < 1 — negative thrust correction (sag reduces thrust)
        delta_per_prop = (vr.pow(2).unsqueeze(1) - 1.0) * self._KF * rpm.pow(2)  # [n_envs, 4]
        delta_f_z = delta_per_prop.sum(dim=1)  # [n_envs]

        force = torch.zeros(self.n_envs, 3, dtype=torch.float32, device=rpm.device)
        force[:, 2] = delta_f_z
        return force

    def apply(self, scene: Any, drone: Any, env_state: EnvState) -> None:
        """Apply voltage-sag thrust correction — preserves _current_value (voltage_ratio).

        Overrides base to avoid overwriting _current_value with the force vector,
        keeping the voltage_ratio available for privileged observation.
        """
        force = self._compute_wrench(env_state)
        local = self.frame == "local"
        envs_idx = torch.arange(self.n_envs, device=force.device)
        scene.rigid_solver.apply_links_external_force(force, self.link_idx, envs_idx, local=local)
