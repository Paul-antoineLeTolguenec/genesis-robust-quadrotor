"""Shared plotting framework for docs/impl/plot_category_*.py.

Enforces a rigorous, reproducible workflow:
  1. Raw measurements are written to ``docs/impl/data/<name>.csv`` *before* any
     figure is rendered.
  2. A sibling ``docs/impl/data/<name>.meta.json`` captures hardware metadata,
     full configuration, and summary statistics.
  3. Figures are rendered with a unified Plotly template (titles, axes, hardware
     footer) so every PNG embedded in the docs has consistent units, stats and
     provenance.

Import and use:

    from _plot_framework import (
        HardwareMeta, stats_summary, log_and_plot, make_figure, save_figure,
    )

All generated files (CSV, JSON, PNG, HTML) live under ``docs/impl/``.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import plotly.graph_objects as go
import plotly.io as pio

DOCS_IMPL = Path(__file__).resolve().parent
ASSETS = DOCS_IMPL / "assets"
DATA = DOCS_IMPL / "data"
ASSETS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Plotly unified template
# ---------------------------------------------------------------------------

PLOT_WIDTH = 820
PLOT_HEIGHT = 480
FONT_FAMILY = "Inter, Helvetica, Arial, sans-serif"
PNG_SCALE = 2  # HiDPI: 820x480 @ scale=2 → 1640x960 px

CATEGORY_COLORS = {
    1: "#1f77b4",  # physics
    2: "#ff7f0e",  # motor
    3: "#2ca02c",  # temporal
    4: "#d62728",  # sensor
    5: "#9467bd",  # wind
    6: "#8c564b",  # action
    7: "#e377c2",  # payload
    8: "#17becf",  # external
}


def apply_layout(
    fig: go.Figure,
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    subtitle: str | None = None,
    hardware_footer: str | None = None,
    xaxis_type: str = "linear",
    yaxis_type: str = "linear",
) -> go.Figure:
    """Apply the shared Plotly layout.

    ``title`` must state the exact metric *with units*. ``subtitle`` should
    include n_envs / n_rounds / n_samples when applicable. ``hardware_footer``
    is a short one-liner appended at the bottom.
    """
    full_title = title if subtitle is None else f"{title}<br><sub>{subtitle}</sub>"
    fig.update_layout(
        title=dict(text=full_title, x=0.5, xanchor="center", font=dict(size=15)),
        xaxis=dict(title=xaxis_title, type=xaxis_type, showgrid=True, gridcolor="#eee"),
        yaxis=dict(title=yaxis_title, type=yaxis_type, showgrid=True, gridcolor="#eee"),
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=12),
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        margin=dict(l=70, r=30, t=80, b=90 if hardware_footer else 60),
        legend=dict(font=dict(size=11), bgcolor="rgba(255,255,255,0.7)"),
    )
    if hardware_footer:
        fig.add_annotation(
            text=hardware_footer,
            xref="paper",
            yref="paper",
            x=0.0,
            y=-0.22,
            showarrow=False,
            font=dict(size=10, color="#666"),
            xanchor="left",
            align="left",
        )
    return fig


def make_figure() -> go.Figure:
    return go.Figure()


# ---------------------------------------------------------------------------
# Hardware metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareMeta:
    cpu: str
    arch: str
    cpu_count: int
    torch_num_threads: int
    memory_gb: float
    os: str
    os_release: str
    hostname: str
    python_version: str
    torch_version: str
    genesis_version: str
    backend: str
    git_sha: str
    git_dirty: bool
    date_utc: str

    def footer_line(self) -> str:
        """One-line hardware summary for plot footers."""
        return (
            f"{self.cpu} ({self.cpu_count} cores, torch threads={self.torch_num_threads}) — "
            f"{self.os} {self.os_release} — Genesis {self.genesis_version} "
            f"backend={self.backend} — torch {self.torch_version} — "
            f"{self.date_utc} — commit {self.git_sha[:8]}" + ("+dirty" if self.git_dirty else "")
        )


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _detect_cpu() -> str:
    system = platform.system()
    if system == "Darwin":
        model = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if model:
            return model
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _detect_versions() -> tuple[str, str, int]:
    torch_threads = 0
    try:
        import torch

        torch_v = torch.__version__
        torch_threads = torch.get_num_threads()
    except ImportError:
        torch_v = "n/a"
    try:
        import genesis as gs

        genesis_v = getattr(gs, "__version__", "unknown")
    except ImportError:
        genesis_v = "n/a"
    return torch_v, genesis_v, torch_threads


def _detect_memory_gb() -> float:
    """Total physical memory in GB, or 0.0 if unavailable."""
    system = platform.system()
    try:
        if system == "Darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            if out.isdigit():
                return round(int(out) / (1024**3), 1)
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 1)
    except (OSError, ValueError):
        pass
    return 0.0


def collect_hardware_meta(backend: str = "cpu") -> HardwareMeta:
    torch_v, genesis_v, torch_threads = _detect_versions()
    git_sha = _run(["git", "rev-parse", "HEAD"]) or "unknown"
    git_status = _run(["git", "status", "--porcelain"])
    return HardwareMeta(
        cpu=_detect_cpu(),
        arch=platform.machine(),
        cpu_count=os.cpu_count() or 0,
        torch_num_threads=torch_threads,
        memory_gb=_detect_memory_gb(),
        os=platform.system(),
        os_release=platform.release(),
        hostname=socket.gethostname(),
        python_version=sys.version.split()[0],
        torch_version=torch_v,
        genesis_version=genesis_v,
        backend=backend,
        git_sha=git_sha,
        git_dirty=bool(git_status),
        date_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stats:
    n: int
    median: float
    mean: float
    stdev: float
    q1: float
    q3: float
    iqr: float
    min: float
    max: float


def stats_summary(values: Sequence[float]) -> Stats:
    """Compute median/mean/σ/quartiles — all basic stats in one call."""
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        raise ValueError("stats_summary: empty values")
    if n == 1:
        x = xs[0]
        return Stats(n=1, median=x, mean=x, stdev=0.0, q1=x, q3=x, iqr=0.0, min=x, max=x)
    median = statistics.median(xs)
    mean = statistics.fmean(xs)
    stdev = statistics.stdev(xs)
    # Quartiles via Tukey "inclusive" (same as numpy default for n>=2)
    quartiles = statistics.quantiles(xs, n=4, method="inclusive")
    q1, _, q3 = quartiles
    return Stats(
        n=n,
        median=median,
        mean=mean,
        stdev=stdev,
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        min=xs[0],
        max=xs[-1],
    )


# ---------------------------------------------------------------------------
# Logging: CSV + meta JSON before PNG
# ---------------------------------------------------------------------------


@dataclass
class PlotArtifacts:
    name: str
    csv_path: Path
    meta_path: Path
    png_path: Path
    html_path: Path | None


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> int:
    rows = list(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def _write_meta(
    path: Path,
    *,
    name: str,
    category: int | None,
    metric: str,
    unit: str,
    baseline_description: str | None,
    config: Mapping[str, Any],
    hardware: HardwareMeta,
    stats_by_series: Mapping[str, Stats],
    csv_rows: int,
) -> None:
    payload = {
        "name": name,
        "category": category,
        "metric": metric,
        "unit": unit,
        "baseline": baseline_description,
        "config": dict(config),
        "hardware": asdict(hardware),
        "stats": {k: asdict(v) for k, v in stats_by_series.items()},
        "stats_method": "inclusive (Tukey quartiles, numpy-compatible for n>=2)",
        "csv_rows": csv_rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def log_and_plot(
    *,
    name: str,
    category: int | None,
    metric: str,
    unit: str,
    baseline_description: str | None,
    config: Mapping[str, Any],
    hardware: HardwareMeta,
    csv_rows: Sequence[Mapping[str, Any]],
    csv_columns: Sequence[str],
    stats_by_series: Mapping[str, Stats],
    figure_fn: Callable[[HardwareMeta], go.Figure],
    save_html: bool = False,
) -> PlotArtifacts:
    """Canonical pipeline: CSV first, then meta JSON, then figure.

    This ordering is deliberate — if ``figure_fn`` crashes, the raw data is
    already on disk and the run is still auditable.
    """
    csv_path = DATA / f"{name}.csv"
    meta_path = DATA / f"{name}.meta.json"
    png_path = ASSETS / f"{name}.png"
    html_path = ASSETS / f"{name}.html" if save_html else None

    n_rows = _write_csv(csv_path, csv_rows, csv_columns)
    _write_meta(
        meta_path,
        name=name,
        category=category,
        metric=metric,
        unit=unit,
        baseline_description=baseline_description,
        config=config,
        hardware=hardware,
        stats_by_series=stats_by_series,
        csv_rows=n_rows,
    )

    fig = figure_fn(hardware)
    pio.write_image(fig, png_path, scale=PNG_SCALE)
    if html_path is not None:
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)

    return PlotArtifacts(
        name=name,
        csv_path=csv_path,
        meta_path=meta_path,
        png_path=png_path,
        html_path=html_path,
    )


# ---------------------------------------------------------------------------
# Convenience: serialize Stats / HardwareMeta
# ---------------------------------------------------------------------------


def save_figure(fig: go.Figure, name: str, *, save_html: bool = False) -> Path:
    """Render a standalone figure (no CSV logging). Use for plots whose data is
    already logged elsewhere or is purely derived from saved CSVs."""
    png_path = ASSETS / f"{name}.png"
    pio.write_image(fig, png_path, scale=PNG_SCALE)
    if save_html:
        fig.write_html(ASSETS / f"{name}.html", include_plotlyjs="cdn", full_html=True)
    return png_path
