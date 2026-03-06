# genesis-robust-rl

A robust reinforcement learning wrapper for quadrotor drones using the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator.

## Goals

- Structured perturbation catalog (wind, motor failure, sensor noise, delays...)
- Rectangular and Lipschitz-continuous perturbation dynamics
- Adversarial API: adversary sets perturbations at each timestep
- Distributionally robust training support
- Gymnasium-compatible environment
- Sim-to-real transfer focus

## Status

See [ROADMAP.md](ROADMAP.md) for current progress.

## Setup

```bash
git clone https://github.com/Paul-antoineLeTolguenec/genesis-robust-rl.git
cd genesis-robust-rl
uv sync --extra dev
```

## Run tests

```bash
uv run pytest
```
