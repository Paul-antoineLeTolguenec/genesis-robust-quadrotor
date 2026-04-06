"""Category 7 — Payload & configuration perturbations.

Implemented perturbations:
  7.1  PayloadMass             — per-env payload mass shift via set_links_mass_shift()
  7.2  PayloadCOMOffset        — per-env payload CoM offset via set_links_COM_shift()
  7.3  AsymmetricPropGuardDrag — per-env asymmetric drag per arm via external force
"""

from __future__ import annotations

from typing import Any, Callable

from torch import Tensor

from genesis_robust_rl.perturbations.base import (
    EnvState,
    ExternalWrenchPerturbation,
    GenesisSetterPerturbation,
    register,
)


@register("payload_mass")
class PayloadMass(GenesisSetterPerturbation):
    """7.1 — Per-env payload mass uncertainty via set_links_mass_shift().

    Adds a scalar Δm [kg] representing an unknown payload mass attached to the drone.
    Unlike 1.1 MassShift (which can be negative for lighter-than-nominal bodies), payload
    mass is always non-negative: bounds default to [0, 0.5] kg.

    Args:
        setter_fn: bound method drone.set_links_mass_shift — provided by the env.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform; {"mean", "std"} for gaussian.
        bounds: hard clamp [min_kg, max_kg] — default [0.0, 0.5].
        nominal: unperturbed payload mass (0.0 kg = no payload).
        lipschitz_k: max |Δm| per step in adversarial mode (None = per_episode only).
    """

    def __init__(
        self,
        setter_fn: Callable[[Tensor, Tensor], None],
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.0, 0.5),
        nominal: float = 0.0,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.0, "high": 0.5}
        super().__init__(
            setter_fn=setter_fn,
            id="payload_mass",
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


@register("payload_com_offset")
class PayloadCOMOffset(GenesisSetterPerturbation):
    """7.2 — Per-env payload center-of-mass offset via set_links_COM_shift().

    Adds a 3D vector Δr [m] representing the CoM displacement caused by an asymmetric
    payload. Each axis is independently sampled within [-0.1, +0.1] m.

    Args:
        setter_fn: bound method drone.set_links_COM_shift — provided by the env.
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform; {"mean", "std"} for gaussian.
        bounds: hard clamp [min_m, max_m] per axis — default [-0.1, 0.1].
        nominal: unperturbed CoM offset ([0, 0, 0] m).
        lipschitz_k: max |Δr| per step in adversarial mode (None = per_episode only).
    """

    def __init__(
        self,
        setter_fn: Callable[[Tensor, Tensor], None],
        n_envs: int,
        dt: float = 0.01,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (-0.1, 0.1),
        nominal: list[float] | None = None,
        lipschitz_k: float | None = None,
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": -0.1, "high": 0.1}
        if nominal is None:
            nominal = [0.0, 0.0, 0.0]
        super().__init__(
            setter_fn=setter_fn,
            id="payload_com_offset",
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


@register("asymmetric_prop_guard_drag")
class AsymmetricPropGuardDrag(ExternalWrenchPerturbation):
    """7.3 — Per-env asymmetric propeller guard drag via external force.

    Models manufacturing asymmetry in propeller guards by applying per-arm drag scaling
    ratios. Each arm j has a drag ratio r_j ∈ [0.8, 1.2] that scales a nominal drag
    coefficient Cd_nom. The resulting force per arm is:

        F_j = -r_j × Cd_nom × sign(v) × v²    (in the velocity direction)

    where v = env_state.vel is the body linear velocity. The 4 per-arm forces are summed
    into a single resultant force [n_envs, 3] applied as an external wrench.

    The arm positions (CF2X quadrotor X-configuration) define the moment arm for torque
    generation, but the primary effect here is the net drag force asymmetry.

    _current_value stores the 4 drag ratios [n_envs, 4] for privileged observation
    (preserve_current_value=True), while _compute_wrench returns [n_envs, 3].

    Args:
        n_envs: number of parallel environments.
        dt: simulation timestep (s).
        Cd_nom: nominal drag coefficient per arm [N·s²/m²] (default 0.025).
        distribution: "uniform" or "gaussian".
        distribution_params: {"low", "high"} for uniform.
        bounds: hard clamp [min_ratio, max_ratio] per arm — default [0.8, 1.2].
        nominal: unperturbed drag ratio per arm ([1.0, 1.0, 1.0, 1.0]).
        frame: reference frame for the force ("world" or "local").
        link_idx: body link index (default 0).
        duration_mode: "continuous" or "pulse".
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
        Cd_nom: float = 0.025,
        distribution: str = "uniform",
        distribution_params: dict | None = None,
        bounds: tuple[float, float] = (0.8, 1.2),
        nominal: list[float] | None = None,
        frame: str = "world",
        link_idx: int = 0,
        duration_mode: str = "continuous",
        **kwargs: Any,
    ) -> None:
        if distribution_params is None:
            distribution_params = {"low": 0.8, "high": 1.2}
        if nominal is None:
            nominal = [1.0, 1.0, 1.0, 1.0]

        self._Cd_nom = Cd_nom

        super().__init__(
            frame=frame,
            link_idx=link_idx,
            duration_mode=duration_mode,
            preserve_current_value=True,
            id="asymmetric_prop_guard_drag",
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
        """Compute net asymmetric drag force from per-arm drag ratios.

        Each arm's drag = r_j × Cd_nom × v² (opposing velocity). The 4 arm drags
        are summed into a single resultant 3D force.

        Returns:
            Tensor[n_envs, 3] — net drag force vector.
        """
        assert self._current_value is not None, "call tick() before apply()"
        ratios = self._current_value  # [n_envs, 4]
        vel = env_state.vel  # [n_envs, 3]

        # Mean drag ratio across arms → effective drag scaling [n_envs, 1]
        # Asymmetry effect: if all ratios = 1.0, net drag = 4 × Cd_nom × drag
        # We compute: F = -sum(r_j) × Cd_nom × sign(v) × v²
        drag_sum = ratios.sum(dim=1, keepdim=True)  # [n_envs, 1]
        force = -drag_sum * self._Cd_nom * vel.sign() * vel.pow(2)  # [n_envs, 3]
        return force
