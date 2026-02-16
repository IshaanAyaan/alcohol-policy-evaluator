"""Streamlit dashboard for policy impact exploration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from src.config import PROCESSED_DIR
    from src.utils.states import ABBREV_TO_NAME
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import PROCESSED_DIR
    from src.utils.states import ABBREV_TO_NAME


@st.cache_data
def load_data():
    panel = pd.read_parquet(PROCESSED_DIR / "panel_state_year.parquet")
    teen = pd.read_parquet(PROCESSED_DIR / "panel_state_wave_teen.parquet")
    causal = pd.read_csv(PROCESSED_DIR / "causal_effects_eventstudy.csv")
    model_eval = pd.read_csv(PROCESSED_DIR / "model_eval_summary.csv")
    return panel, teen, causal, model_eval


def main() -> None:
    st.set_page_config(page_title="Alcohol Policy Impact Tracker", layout="wide")
    st.title("Alcohol Policy Impact Tracker")
    st.caption("Causal evidence and predictive trends for state alcohol policy shifts.")

    panel, teen, causal, model_eval = load_data()

    states = sorted(panel["state_abbrev"].unique())
    state = st.sidebar.selectbox("State", states, index=states.index("CA") if "CA" in states else 0)

    state_panel = panel[panel["state_abbrev"] == state].sort_values("year")
    state_teen = teen[teen["state_abbrev"] == state].sort_values("year")

    st.subheader(f"{ABBREV_TO_NAME.get(state, state)} ({state})")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            state_panel,
            x="year",
            y=["rate_impaired_per100k", "rate_alcohol_involved_per100k"],
            markers=True,
            title="Fatality Outcomes Over Time",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.line(
            state_teen,
            x="year",
            y=["teen_current_alcohol_use_pct", "teen_binge_pct"],
            markers=True,
            title="Teen Behavior Outcomes (Wave-Based)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Scenario Explorer")
    tax_delta = st.slider("Hypothetical beer-tax change (USD per gallon)", min_value=-1.0, max_value=1.0, value=0.1, step=0.05)

    post_effect = causal[(causal["outcome"] == "rate_impaired_per100k") & (causal["event_time"] >= 0)]["coef"].mean()
    post_effect = float(post_effect) if not np.isnan(post_effect) else 0.0
    implied_delta = post_effect * np.sign(tax_delta)

    latest_rate = state_panel["rate_impaired_per100k"].iloc[-1]
    scenario_rate = max(0.0, latest_rate + implied_delta)

    st.metric("Latest impaired fatality rate", f"{latest_rate:.2f} per 100k")
    st.metric("Scenario-estimated next-year rate", f"{scenario_rate:.2f} per 100k", delta=f"{scenario_rate-latest_rate:.2f}")

    st.subheader("Model Comparison")
    st.dataframe(model_eval.sort_values(["task", "test_rmse"]))

    st.subheader("Methods and Interpretation")
    st.write(
        "Causal and predictive outputs are separated. Event-study estimates summarize average post-policy dynamics, "
        "while model metrics reflect out-of-sample forecasting performance. Scenario values are directional summaries, "
        "not deterministic forecasts."
    )


if __name__ == "__main__":
    main()
