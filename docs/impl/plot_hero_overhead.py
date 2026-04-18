"""Hero plot: overhead (%) across all 69 perturbations, 8 categories.

Consumes the CSVs produced by ``plot_perf_cat<1..8>.py`` (no re-measurement) and
renders a single horizontal bar chart, grouped by category and sorted by median
overhead at ``n_envs=16``. Also writes a summary CSV+meta for audit.

Run (after the 8 ``plot_perf_cat<N>.py`` scripts have been run):
    uv run python docs/impl/plot_hero_overhead.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _plot_framework import (  # noqa: E402
    CATEGORY_COLORS,
    DATA,
    apply_layout,
    collect_hardware_meta,
    log_and_plot,
    make_figure,
    stats_summary,
)

CATEGORY_LABELS = {
    1: "Physics",
    2: "Motor",
    3: "Temporal",
    4: "Sensor",
    5: "Wind",
    6: "Action",
    7: "Payload",
    8: "External",
}

N_ENVS_REPRESENTATIVE = 16


def _parse_slug(slug: str) -> tuple[int, str]:
    """Split ``cat3_packet_loss`` → ``(3, "packet_loss")``."""
    prefix, rest = slug.split("_", 1)
    assert prefix.startswith("cat"), slug
    return int(prefix[3:]), rest


def _load_overhead(csv_path: Path, n_envs: int) -> list[float]:
    """Return per-round overhead values at the given n_envs."""
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return [float(row["overhead_pct"]) for row in reader if int(row["n_envs"]) == n_envs]


def main() -> None:
    perf_csvs = sorted(DATA.glob("*_perf.csv"))
    if not perf_csvs:
        raise SystemExit("no *_perf.csv in docs/impl/data/ — run the 8 plot_perf_cat<N>.py first")

    entries: list[dict[str, Any]] = []
    for csv_path in perf_csvs:
        slug = csv_path.stem.removesuffix("_perf")
        category, name = _parse_slug(slug)
        overheads = _load_overhead(csv_path, N_ENVS_REPRESENTATIVE)
        if not overheads:
            continue
        stats = stats_summary(overheads)
        entries.append(
            {
                "slug": slug,
                "category": category,
                "name": name.replace("_", " "),
                "median": stats.median,
                "q1": stats.q1,
                "q3": stats.q3,
                "min": stats.min,
                "max": stats.max,
            }
        )

    entries.sort(key=lambda e: (e["category"], e["median"]))
    print(f"loaded {len(entries)} perturbations")

    hardware = collect_hardware_meta(backend="cpu")

    rows: list[dict[str, Any]] = [
        {
            "slug": e["slug"],
            "category": e["category"],
            "name": e["name"],
            "overhead_median_pct": e["median"],
            "overhead_q1_pct": e["q1"],
            "overhead_q3_pct": e["q3"],
            "overhead_min_pct": e["min"],
            "overhead_max_pct": e["max"],
        }
        for e in entries
    ]

    all_medians = [e["median"] for e in entries]
    stats_by_series = {
        "all": stats_summary(all_medians),
        **{
            f"cat{c}": stats_summary([e["median"] for e in entries if e["category"] == c])
            for c in sorted(set(e["category"] for e in entries))
        },
    }

    def figure_fn(hw):
        fig = make_figure()
        labels = [f"[cat{e['category']}] {e['name']}" for e in entries]
        medians = [e["median"] for e in entries]
        errs_plus = [e["q3"] - e["median"] for e in entries]
        errs_minus = [e["median"] - e["q1"] for e in entries]

        by_cat: dict[int, list[int]] = {}
        for i, e in enumerate(entries):
            by_cat.setdefault(e["category"], []).append(i)
        for cat in sorted(by_cat):
            idxs = by_cat[cat]
            fig.add_trace(
                go.Bar(
                    y=[labels[i] for i in idxs],
                    x=[medians[i] for i in idxs],
                    orientation="h",
                    marker=dict(color=CATEGORY_COLORS[cat]),
                    error_x=dict(
                        type="data",
                        array=[errs_plus[i] for i in idxs],
                        arrayminus=[errs_minus[i] for i in idxs],
                        color="#555",
                        thickness=1,
                        width=2,
                    ),
                    name=f"Cat {cat} — {CATEGORY_LABELS[cat]}",
                    hovertemplate="%{y}<br>overhead=%{x:.1f}%<extra></extra>",
                )
            )

        fig.add_vline(
            x=5.0,
            line_dash="dash",
            line_color="#d62728",
            annotation_text="5% budget",
            annotation_position="top",
            annotation_font=dict(size=10, color="#d62728"),
        )
        apply_layout(
            fig,
            title=(
                f"Overhead (%) — all {len(entries)} perturbations @ n_envs={N_ENVS_REPRESENTATIVE}"
            ),
            subtitle=(
                "median + IQR across 5 rounds × 100 steps — "
                "baseline = scene.step() only — Genesis CF2X"
            ),
            xaxis_title="overhead (%) — log-scale",
            yaxis_title="",
            xaxis_type="log",
            hardware_footer=hw.footer_line(),
        )
        fig.update_layout(
            height=max(600, 14 * len(entries)),
            barmode="group",
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.01,
                font=dict(size=10),
            ),
            margin=dict(l=240, r=160, t=80, b=110),
        )
        return fig

    log_and_plot(
        name="hero_overhead",
        category=None,
        metric="overhead",
        unit="%",
        baseline_description=(
            f"scene.step() only (no tick/apply) — all values at n_envs={N_ENVS_REPRESENTATIVE}"
        ),
        config={
            "source": "docs/impl/data/*_perf.csv",
            "n_envs_representative": N_ENVS_REPRESENTATIVE,
            "n_perturbations": len(entries),
            "genesis_backend": "cpu",
            "drone_urdf": "urdf/drones/cf2x.urdf",
        },
        hardware=hardware,
        csv_rows=rows,
        csv_columns=[
            "slug",
            "category",
            "name",
            "overhead_median_pct",
            "overhead_q1_pct",
            "overhead_q3_pct",
            "overhead_min_pct",
            "overhead_max_pct",
        ],
        stats_by_series=stats_by_series,
        figure_fn=figure_fn,
    )
    print("wrote hero_overhead.{csv,meta.json,png}")

    with (DATA / "hero_overhead.meta.json").open() as f:
        meta = json.load(f)
    print(f"median of all: {meta['stats']['all']['median']:+.2f} %")
    under_5 = sum(1 for e in entries if e["median"] <= 5.0)
    print(f"{under_5}/{len(entries)} perturbations ≤ 5% overhead (median)")


if __name__ == "__main__":
    main()
