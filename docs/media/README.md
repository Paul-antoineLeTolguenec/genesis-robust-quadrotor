# Demo Media

Scripts to regenerate the hover demo GIFs embedded in the project README.

## Requirements

- Genesis installed (`uv sync --extra genesis`)
- Media extras for MP4 → GIF conversion (`uv sync --extra media`)

## Regenerate the GIFs

```bash
# Record MP4s (one Genesis init per process)
uv run python docs/media/record_demo.py --scenario baseline
uv run python docs/media/record_demo.py --scenario perturbed

# Convert MP4 → GIF
uv run --extra media python docs/media/mp4_to_gif.py docs/media/baseline.mp4
uv run --extra media python docs/media/mp4_to_gif.py docs/media/perturbed.mp4
```

Outputs `docs/media/baseline.gif` and `docs/media/perturbed.gif`.

## Scenarios

| Scenario | Perturbations | Expected visual |
|----------|---------------|-----------------|
| `baseline` | none | Stable hover at z ≈ 0.5 m |
| `perturbed` | `WindGust` (mag 0.004–0.009 N, prob 8%/step) | Drifted trajectory, up to ~0.3 m horizontal |

Both run for 240 steps at dt=0.01 s (8 s sim time) with 30 fps video output,
calibrated hover RPM = 14 476 (CF2X, 27 g, KF=3.16e-10).

The `perturbed` scenario passes `link_idx=drone.base_link_idx` to `WindGust`
so the force targets the drone base link, not the first-added plane entity.

## Notes

- Genesis camera uses offscreen rendering (`GUI=False`) — no display required.
- Validated on macOS CPU backend (Mac M4 Pro); should also work on Linux + CUDA.
- `gs.init()` can only be called once per Python process, so each scenario runs
  in its own invocation.
- MP4s are gitignored (regeneratable); only the GIFs are committed.
