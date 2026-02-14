"""Build processed state-year and teen-wave panels."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import ANALYSIS_YEARS, INTERMEDIATE_DIR, PROCESSED_DIR, TABLES_DIR
from src.utils.io import ensure_dir
from src.utils.states import ABBREV_TO_FIPS, ABBREV_TO_NAME, STATES


def _state_year_grid() -> pd.DataFrame:
    grid = pd.MultiIndex.from_product(
        [[s.abbrev for s in STATES], ANALYSIS_YEARS], names=["state_abbrev", "year"]
    ).to_frame(index=False)
    grid["state_fips"] = grid["state_abbrev"].map(ABBREV_TO_FIPS)
    grid["state_name"] = grid["state_abbrev"].map(ABBREV_TO_NAME)
    return grid


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def run() -> Dict[str, Path]:
    ensure_dir(PROCESSED_DIR)
    ensure_dir(TABLES_DIR)

    grid = _state_year_grid()

    apis_beer = _load_csv(INTERMEDIATE_DIR / "apis_beer_tax_state_year.csv")
    apis_changes = _load_csv(INTERMEDIATE_DIR / "apis_policy_changes_state_year.csv")
    fars = _load_csv(INTERMEDIATE_DIR / "fars_state_year.csv")
    cov = _load_csv(INTERMEDIATE_DIR / "covariates_fred_state_year.csv")
    fhwa = _load_csv(INTERMEDIATE_DIR / "fhwa_vmt_state_year.csv")

    apis_beer["year"] = pd.to_numeric(apis_beer["year"], errors="coerce").astype("Int64")
    apis_changes["year"] = pd.to_numeric(apis_changes["year"], errors="coerce").astype("Int64")
    cov["year"] = pd.to_numeric(cov["year"], errors="coerce").astype("Int64")
    fhwa["year"] = pd.to_numeric(fhwa["year"], errors="coerce").astype("Int64")

    fars["year"] = pd.to_numeric(fars["year"], errors="coerce").astype("Int64")
    fars["state_fips"] = fars["state_fips"].astype(str).str.zfill(2)
    fars["state_abbrev"] = fars["state_fips"].map({v: k for k, v in ABBREV_TO_FIPS.items()})

    panel = grid.merge(
        apis_beer[["state_abbrev", "year", "beer_tax_usd_per_gallon"]],
        on=["state_abbrev", "year"],
        how="left",
    )
    panel = panel.merge(
        apis_changes[
            [
                "state_abbrev",
                "year",
                "policy_change_count_sunday_sales",
                "policy_change_count_underage_purchase",
            ]
        ],
        on=["state_abbrev", "year"],
        how="left",
    )
    panel = panel.merge(
        fars[
            [
                "state_abbrev",
                "year",
                "fars_fatalities_total",
                "fars_fatalities_alcohol_involved",
                "fars_fatalities_impaired",
            ]
        ],
        on=["state_abbrev", "year"],
        how="left",
    )
    panel = panel.merge(
        cov[["state_abbrev", "year", "unemployment_rate", "pcpi_nominal", "population_thousands"]],
        on=["state_abbrev", "year"],
        how="left",
    )
    panel = panel.merge(
        fhwa[["state_abbrev", "year", "vmt_total_million"]],
        on=["state_abbrev", "year"],
        how="left",
    )

    for col in [
        "policy_change_count_sunday_sales",
        "policy_change_count_underage_purchase",
        "fars_fatalities_total",
        "fars_fatalities_alcohol_involved",
        "fars_fatalities_impaired",
    ]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0).astype(int)

    panel["population_thousands"] = pd.to_numeric(panel["population_thousands"], errors="coerce")
    panel["rate_alcohol_involved_per100k"] = (
        panel["fars_fatalities_alcohol_involved"] / (panel["population_thousands"] * 1000.0)
    ) * 100000.0
    panel["rate_impaired_per100k"] = (
        panel["fars_fatalities_impaired"] / (panel["population_thousands"] * 1000.0)
    ) * 100000.0
    panel["vmt_per_capita"] = (panel["vmt_total_million"] * 1_000_000.0) / (
        panel["population_thousands"] * 1000.0
    )

    panel = panel[
        [
            "state_fips",
            "state_abbrev",
            "state_name",
            "year",
            "beer_tax_usd_per_gallon",
            "policy_change_count_sunday_sales",
            "policy_change_count_underage_purchase",
            "fars_fatalities_total",
            "fars_fatalities_alcohol_involved",
            "fars_fatalities_impaired",
            "rate_alcohol_involved_per100k",
            "rate_impaired_per100k",
            "unemployment_rate",
            "pcpi_nominal",
            "population_thousands",
            "vmt_total_million",
            "vmt_per_capita",
        ]
    ].copy()

    panel = panel.sort_values(["state_abbrev", "year"]).reset_index(drop=True)

    panel_parquet = PROCESSED_DIR / "panel_state_year.parquet"
    panel_csv = PROCESSED_DIR / "panel_state_year.csv"
    panel.to_parquet(panel_parquet, index=False)
    panel.to_csv(panel_csv, index=False)

    teen = _load_csv(INTERMEDIATE_DIR / "teen_state_wave.csv")
    teen["year"] = pd.to_numeric(teen["year"], errors="coerce").astype("Int64")
    teen = teen[[
        "state_abbrev",
        "year",
        "teen_current_alcohol_use_pct",
        "teen_binge_pct",
        "teen_source",
        "coverage_flag",
    ]].copy()
    teen = teen.sort_values(["state_abbrev", "year"]).reset_index(drop=True)

    teen_parquet = PROCESSED_DIR / "panel_state_wave_teen.parquet"
    teen_csv = PROCESSED_DIR / "panel_state_wave_teen.csv"
    teen.to_parquet(teen_parquet, index=False)
    teen.to_csv(teen_csv, index=False)

    # Missingness report for QA.
    miss = panel.isna().mean().rename("missing_rate").reset_index().rename(columns={"index": "column"})
    miss.to_csv(TABLES_DIR / "panel_state_year_missingness.csv", index=False)

    return {
        "panel_state_year_parquet": panel_parquet,
        "panel_state_year_csv": panel_csv,
        "panel_state_wave_teen_parquet": teen_parquet,
        "panel_state_wave_teen_csv": teen_csv,
    }


if __name__ == "__main__":
    print(run())
