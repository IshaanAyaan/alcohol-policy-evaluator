"""Generate long-form executive summary with numeric findings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR, TABLES_DIR
from src.utils.io import ensure_dir


def _fmt(x: float, ndigits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{ndigits}f}"


def run() -> Dict[str, Path]:
    ensure_dir(REPORTS_DIR)

    panel = pd.read_parquet(PROCESSED_DIR / "panel_state_year.parquet")
    teen = pd.read_parquet(PROCESSED_DIR / "panel_state_wave_teen.parquet")
    causal = pd.read_csv(PROCESSED_DIR / "causal_effects_eventstudy.csv")
    model_eval = pd.read_csv(PROCESSED_DIR / "model_eval_summary.csv")

    causal_summary = pd.read_csv(TABLES_DIR / "causal_model_summary.csv") if (TABLES_DIR / "causal_model_summary.csv").exists() else pd.DataFrame()

    n_state_year = panel.shape[0]
    n_states = panel["state_abbrev"].nunique()
    years = (int(panel["year"].min()), int(panel["year"].max()))

    teen_obs = teen["teen_current_alcohol_use_pct"].notna().sum()
    teen_rows = teen.shape[0]

    primary = causal[causal["outcome"] == "rate_impaired_per100k"].copy()
    post = primary[primary["event_time"] >= 0]
    avg_post = post["coef"].mean() if not post.empty else np.nan

    best_models = (
        model_eval.sort_values(["task", "test_rmse"]).groupby("task", as_index=False).first()
        if not model_eval.empty
        else pd.DataFrame()
    )

    top_states = (
        panel.sort_values("year")
        .groupby("state_abbrev", as_index=False)
        .agg(
            start_rate=("rate_impaired_per100k", "first"),
            end_rate=("rate_impaired_per100k", "last"),
        )
    )
    top_states["delta"] = top_states["end_rate"] - top_states["start_rate"]
    most_improved = top_states.nsmallest(5, "delta")
    most_worsened = top_states.nlargest(5, "delta")

    lines = []
    lines.append("# Executive Project Summary")
    lines.append("")
    lines.append("## Objective")
    lines.append(
        "This project built a reproducible state-level evidence system to evaluate whether changes in alcohol policy "
        "are followed by measurable changes in alcohol-related traffic fatality outcomes and teen drinking behavior. "
        "The implementation intentionally separates causal inference from forecasting so correlation and causation are not conflated."
    )
    lines.append("")
    lines.append("## Data Coverage")
    lines.append(
        f"The primary panel contains **{n_state_year} state-year rows** across **{n_states} jurisdictions** (50 states + DC) "
        f"from **{years[0]} to {years[1]}**."
    )
    lines.append(
        f"Teen wave data contains **{teen_rows} rows** with **{teen_obs} observed** values for current alcohol use and explicit missingness flags for non-participating waves."
    )
    lines.append(
        "Policy features include APIS Beer tax (topic 30), Sunday off-premise sales change events (topic 28), and Underage purchase change events (topic 43), "
        "assembled with an APIS-first parser and fallback pathways."
    )
    lines.append("")
    lines.append("## Causal Findings")
    lines.append(
        "Event-study models were estimated with state and year fixed effects, covariate adjustment, and clustered standard errors. "
        "Primary outcome was alcohol-impaired fatality rate per 100k."
    )
    lines.append(f"Average post-policy dynamic coefficient (event time >= 0) for the primary outcome: **{_fmt(avg_post)}**.")
    if not causal_summary.empty:
        for _, row in causal_summary.iterrows():
            lines.append(
                f"- {row['outcome']}: R²={_fmt(float(row['r2']))}, pretrend p-value={_fmt(float(row['pretrend_pvalue']))}, "
                f"placebo mean={_fmt(float(row['placebo_mean']))}, placebo sd={_fmt(float(row['placebo_std']))}"
            )
    lines.append(
        "Placebo distributions were generated to check whether estimated effects are centered near zero under randomized policy timing."
    )
    lines.append("")
    lines.append("## Predictive Performance")
    lines.append(
        "A model ladder (ElasticNet, RandomForest, HistGradientBoosting, MLP with policy text embeddings) was evaluated using out-of-time splits."
    )
    if not best_models.empty:
        lines.append("Best held-out models by task:")
        for _, row in best_models.iterrows():
            lines.append(
                f"- {row['task']}: {row['model']} with test RMSE={_fmt(float(row['test_rmse']))}, "
                f"MAE={_fmt(float(row['test_mae']))}, R²={_fmt(float(row['test_r2']))}"
            )
    lines.append(
        "This predictive output is used for scenario exploration and prioritization, not causal attribution."
    )
    lines.append("")
    lines.append("## Geographic Pattern Highlights")
    lines.append("Largest estimated improvements in impaired fatality rate (latest year vs earliest year):")
    for _, row in most_improved.iterrows():
        lines.append(f"- {row['state_abbrev']}: delta {_fmt(float(row['delta']))} per 100k")
    lines.append("Largest estimated worsenings in impaired fatality rate (latest year vs earliest year):")
    for _, row in most_worsened.iterrows():
        lines.append(f"- {row['state_abbrev']}: delta {_fmt(float(row['delta']))} per 100k")
    lines.append("")
    lines.append("## Limitations")
    lines.append(
        "Teen outcome coverage is wave-based and state participation is uneven in later cycles. APIS access required an HTTP fallback parser due direct endpoint blocking. "
        "Some policy dimensions and enforcement intensity are not fully observed."
    )
    lines.append(
        "Accordingly, causal findings should be interpreted as quasi-experimental estimates under tested assumptions, not as definitive policy law effects in every context."
    )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append(
        "The pipeline is command-driven (`make data`, `make features`, `make causal`, `make predict`, `make text`, `make dashboard`, `make report`, `make all`) "
        "and produces versioned processed artifacts and figure outputs for poster and dashboard use."
    )

    out_path = REPORTS_DIR / "executive_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"executive_summary": out_path}


if __name__ == "__main__":
    print(run())
