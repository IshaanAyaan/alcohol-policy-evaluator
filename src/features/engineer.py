"""Feature engineering for causal and predictive modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import PROCESSED_DIR
from src.utils.io import ensure_dir


def run() -> Dict[str, Path]:
    ensure_dir(PROCESSED_DIR)

    panel = pd.read_parquet(PROCESSED_DIR / "panel_state_year.parquet")
    panel = panel.sort_values(["state_abbrev", "year"]).reset_index(drop=True)

    panel["beer_tax_delta"] = panel.groupby("state_abbrev")["beer_tax_usd_per_gallon"].diff()
    panel["beer_tax_increase_event"] = (panel["beer_tax_delta"] > 0).astype(int)

    for col in [
        "rate_impaired_per100k",
        "rate_alcohol_involved_per100k",
        "beer_tax_usd_per_gallon",
        "unemployment_rate",
        "pcpi_nominal",
        "vmt_per_capita",
    ]:
        panel[f"{col}_lag1"] = panel.groupby("state_abbrev")[col].shift(1)
        panel[f"{col}_lag2"] = panel.groupby("state_abbrev")[col].shift(2)

    out_state_year = PROCESSED_DIR / "panel_state_year_features.parquet"
    panel.to_parquet(out_state_year, index=False)

    teen = pd.read_parquet(PROCESSED_DIR / "panel_state_wave_teen.parquet")
    teen = teen.merge(
        panel[
            [
                "state_abbrev",
                "year",
                "beer_tax_usd_per_gallon",
                "policy_change_count_sunday_sales",
                "policy_change_count_underage_purchase",
                "unemployment_rate",
                "pcpi_nominal",
                "vmt_per_capita",
                "rate_impaired_per100k",
            ]
        ],
        on=["state_abbrev", "year"],
        how="left",
    )
    teen = teen.sort_values(["state_abbrev", "year"]).reset_index(drop=True)
    teen["teen_current_alcohol_use_pct_lag1"] = teen.groupby("state_abbrev")[
        "teen_current_alcohol_use_pct"
    ].shift(1)

    out_teen = PROCESSED_DIR / "panel_state_wave_teen_features.parquet"
    teen.to_parquet(out_teen, index=False)

    return {
        "panel_state_year_features": out_state_year,
        "panel_state_wave_teen_features": out_teen,
    }


if __name__ == "__main__":
    print(run())
