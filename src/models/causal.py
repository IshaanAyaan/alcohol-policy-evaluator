"""Causal/event-study analysis for policy impact."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.config import FIGURES_DIR, PROCESSED_DIR, R_SCRIPT_PATH, TABLES_DIR
from src.utils.io import ensure_dir


def _prepare_data() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "panel_state_year_features.parquet")
    df = df.sort_values(["state_abbrev", "year"]).reset_index(drop=True)

    treat = (
        df.loc[df["beer_tax_increase_event"] == 1, ["state_abbrev", "year"]]
        .groupby("state_abbrev", as_index=False)["year"]
        .min()
        .rename(columns={"year": "treatment_year"})
    )
    df = df.merge(treat, on="state_abbrev", how="left")
    df["event_time"] = df["year"] - df["treatment_year"]
    return df


def _event_dummies(df: pd.DataFrame, min_k: int = -5, max_k: int = 5, omit_k: int = -1) -> List[str]:
    cols = []
    for k in range(min_k, max_k + 1):
        if k == omit_k:
            continue
        col = f"evt_{'m' + str(abs(k)) if k < 0 else 'p' + str(k)}"
        df[col] = ((df["event_time"] == k) & df["treatment_year"].notna()).astype(int)
        cols.append(col)
    return cols


def _fit_event_study(df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, Dict[str, float], sm.regression.linear_model.RegressionResultsWrapper]:
    work = df.copy()
    evt_cols = _event_dummies(work)

    formula = (
        f"{outcome} ~ {' + '.join(evt_cols)} + unemployment_rate + pcpi_nominal + vmt_per_capita + "
        "C(state_abbrev) + C(year)"
    )
    model = smf.ols(formula=formula, data=work).fit(
        cov_type="cluster", cov_kwds={"groups": work["state_abbrev"]}
    )

    rows = []
    for col in evt_cols:
        ci_low, ci_high = model.conf_int().loc[col]
        rows.append(
            {
                "outcome": outcome,
                "term": col,
                "event_time": _decode_event_col(col),
                "coef": model.params[col],
                "se": model.bse[col],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": model.pvalues[col],
            }
        )

    out = pd.DataFrame(rows).sort_values("event_time")

    # Joint pre-trend test for leads (-5,-4,-3,-2).
    lead_terms = [c for c in evt_cols if _decode_event_col(c) in {-5, -4, -3, -2}]
    pretrend_p = np.nan
    if lead_terms:
        hypo = " = 0, ".join(lead_terms) + " = 0"
        test = model.f_test(hypo)
        pretrend_p = float(test.pvalue)

    summary = {
        "nobs": float(model.nobs),
        "r2": float(model.rsquared),
        "pretrend_pvalue": pretrend_p,
    }
    return out, summary, model


def _decode_event_col(col: str) -> int:
    token = col.replace("evt_", "")
    if token.startswith("m"):
        return -int(token[1:])
    if token.startswith("p"):
        return int(token[1:])
    raise ValueError(col)


def _placebo_distribution(df: pd.DataFrame, outcome: str, n_draws: int = 100) -> pd.DataFrame:
    work = df.copy()
    treated_states = work.loc[work["treatment_year"].notna(), "state_abbrev"].drop_duplicates().tolist()
    if not treated_states:
        return pd.DataFrame(columns=["draw", "coef"])

    rng = np.random.default_rng(42)
    unique_years = sorted(work["year"].unique())

    rows = []
    for draw in range(n_draws):
        shuffled_years = rng.choice(unique_years, size=len(treated_states), replace=True)
        placebo_map = dict(zip(treated_states, shuffled_years, strict=False))
        work["placebo_treat_year"] = work["state_abbrev"].map(placebo_map)
        work["placebo_post"] = (
            work["placebo_treat_year"].notna() & (work["year"] >= work["placebo_treat_year"])
        ).astype(int)

        formula = (
            f"{outcome} ~ placebo_post + unemployment_rate + pcpi_nominal + vmt_per_capita + "
            "C(state_abbrev) + C(year)"
        )
        model = smf.ols(formula=formula, data=work).fit(
            cov_type="cluster", cov_kwds={"groups": work["state_abbrev"]}
        )
        rows.append({"draw": draw, "coef": float(model.params.get("placebo_post", np.nan))})

    return pd.DataFrame(rows)


def _save_event_plot(event_df: pd.DataFrame, outcome: str) -> Path:
    ensure_dir(FIGURES_DIR)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(-1, color="gray", linestyle="--", linewidth=1)
    ax.errorbar(
        event_df["event_time"],
        event_df["coef"],
        yerr=[event_df["coef"] - event_df["ci_low"], event_df["ci_high"] - event_df["coef"]],
        fmt="o",
        capsize=4,
    )
    ax.set_title(f"Event Study: {outcome}")
    ax.set_xlabel("Event time (years, -1 omitted)")
    ax.set_ylabel("Coefficient")
    out = FIGURES_DIR / f"event_study_{outcome}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _save_placebo_plot(placebo_df: pd.DataFrame, outcome: str) -> Path:
    ensure_dir(FIGURES_DIR)
    fig, ax = plt.subplots(figsize=(8, 5))
    if not placebo_df.empty:
        ax.hist(placebo_df["coef"].dropna(), bins=20, alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"Placebo Distribution ({outcome})")
    ax.set_xlabel("Estimated placebo coefficient")
    out = FIGURES_DIR / f"placebo_distribution_{outcome}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _run_r_did(input_csv: Path, output_csv: Path, meta_json: Path) -> None:
    if not output_csv.exists():
        pd.DataFrame(columns=["event_time", "att", "se", "crit_val"]).to_csv(output_csv, index=False)
    if not R_SCRIPT_PATH.exists():
        meta_json.write_text(json.dumps({"status": "R script missing"}, indent=2), encoding="utf-8")
        return

    cmd = [
        "Rscript",
        str(R_SCRIPT_PATH),
        str(input_csv),
        str(output_csv),
        str(meta_json),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        meta_json.write_text(json.dumps({"status": "failed", "error": str(exc)}, indent=2), encoding="utf-8")


def run() -> Dict[str, Path]:
    ensure_dir(PROCESSED_DIR)
    ensure_dir(TABLES_DIR)

    df = _prepare_data()
    outcomes = ["rate_impaired_per100k", "rate_alcohol_involved_per100k"]

    all_events = []
    model_summaries = []
    figure_paths = []

    for outcome in outcomes:
        subset = df.dropna(subset=[outcome, "unemployment_rate", "pcpi_nominal", "vmt_per_capita"]).copy()
        event_df, summary, _ = _fit_event_study(subset, outcome)
        placebo_df = _placebo_distribution(subset, outcome)

        event_df["pretrend_pvalue"] = summary["pretrend_pvalue"]
        all_events.append(event_df)

        model_summaries.append(
            {
                "outcome": outcome,
                "nobs": summary["nobs"],
                "r2": summary["r2"],
                "pretrend_pvalue": summary["pretrend_pvalue"],
                "placebo_mean": float(placebo_df["coef"].mean()) if not placebo_df.empty else np.nan,
                "placebo_std": float(placebo_df["coef"].std()) if not placebo_df.empty else np.nan,
            }
        )

        figure_paths.append(_save_event_plot(event_df, outcome))
        figure_paths.append(_save_placebo_plot(placebo_df, outcome))

    events_out = pd.concat(all_events, ignore_index=True)
    events_path = PROCESSED_DIR / "causal_effects_eventstudy.csv"
    events_out.to_csv(events_path, index=False)

    summary_path = TABLES_DIR / "causal_model_summary.csv"
    pd.DataFrame(model_summaries).to_csv(summary_path, index=False)

    # Run optional R Callaway-Sant'Anna implementation.
    did_input = PROCESSED_DIR / "panel_state_year_features.csv"
    df.to_csv(did_input, index=False)
    did_out = TABLES_DIR / "causal_did_r_output.csv"
    did_meta = TABLES_DIR / "causal_did_r_meta.json"
    _run_r_did(did_input, did_out, did_meta)

    return {
        "causal_effects_eventstudy": events_path,
        "causal_summary": summary_path,
        "causal_r_output": did_out,
        "causal_r_meta": did_meta,
        "figures": Path(str(figure_paths[0].parent)),
    }


if __name__ == "__main__":
    print(run())
