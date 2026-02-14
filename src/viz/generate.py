"""Generate poster-ready figures and tables."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from src.config import FIGURES_DIR, PROCESSED_DIR, TABLES_DIR
from src.utils.io import ensure_dir


def _plot_event_study() -> Path:
    df = pd.read_csv(PROCESSED_DIR / "causal_effects_eventstudy.csv")
    out = FIGURES_DIR / "event_study_primary_from_table.png"

    primary = df[df["outcome"] == "rate_impaired_per100k"].sort_values("event_time")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(-1, color="gray", linestyle="--", linewidth=1)
    ax.errorbar(
        primary["event_time"],
        primary["coef"],
        yerr=[primary["coef"] - primary["ci_low"], primary["ci_high"] - primary["coef"]],
        fmt="o",
        capsize=4,
    )
    ax.set_title("Event Study: Alcohol-Impaired Fatality Rate")
    ax.set_xlabel("Event Time")
    ax.set_ylabel("Coefficient")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_model_comparison() -> Path:
    df = pd.read_csv(PROCESSED_DIR / "model_eval_summary.csv")
    out = FIGURES_DIR / "model_comparison_rmse.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = df.copy()
    plot_df["label"] = plot_df["task"] + " | " + plot_df["model"]
    plot_df = plot_df.sort_values("test_rmse")
    ax.barh(plot_df["label"], plot_df["test_rmse"])
    ax.set_title("Model Comparison (Test RMSE)")
    ax.set_xlabel("RMSE")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_state_map() -> Path:
    panel = pd.read_parquet(PROCESSED_DIR / "panel_state_year.parquet")
    base = panel[panel["year"] == panel["year"].min()][["state_abbrev", "rate_impaired_per100k"]].rename(
        columns={"rate_impaired_per100k": "rate_base"}
    )
    latest = panel[panel["year"] == panel["year"].max()][["state_abbrev", "rate_impaired_per100k"]].rename(
        columns={"rate_impaired_per100k": "rate_latest"}
    )
    delta = base.merge(latest, on="state_abbrev", how="inner")
    delta["delta_rate_impaired"] = delta["rate_latest"] - delta["rate_base"]

    fig = px.choropleth(
        delta,
        locations="state_abbrev",
        locationmode="USA-states",
        scope="usa",
        color="delta_rate_impaired",
        color_continuous_scale="RdBu_r",
        title="Change in Alcohol-Impaired Fatality Rate (Latest vs Earliest Year)",
    )

    out = FIGURES_DIR / "state_impact_map.html"
    fig.write_html(str(out))
    return out


def run() -> Dict[str, Path]:
    ensure_dir(FIGURES_DIR)
    ensure_dir(TABLES_DIR)

    p1 = _plot_event_study()
    p2 = _plot_model_comparison()
    p3 = _plot_state_map()

    # Also export a key summary table.
    eval_df = pd.read_csv(PROCESSED_DIR / "model_eval_summary.csv")
    eval_df.to_csv(TABLES_DIR / "model_eval_summary_copy.csv", index=False)

    return {
        "event_study_plot": p1,
        "model_comparison_plot": p2,
        "state_map": p3,
    }


if __name__ == "__main__":
    print(run())
