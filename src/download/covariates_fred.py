"""Download state-level covariates from FRED CSV endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import ANALYSIS_YEARS, FRED_BASE, INTERMEDIATE_DIR, RAW_DIR
from src.utils.io import ensure_dir, http_get
from src.utils.states import STATES


def _series_id(abbrev: str, metric: str) -> str:
    if metric == "unemployment_rate":
        return f"{abbrev}UR"
    if metric == "pcpi_nominal":
        return f"{abbrev}PCPI"
    if metric == "population_thousands":
        return f"{abbrev}POP"
    raise ValueError(f"Unknown metric: {metric}")


def _download_series(series_id: str) -> Path:
    out = RAW_DIR / "fred" / f"{series_id}.csv"
    ensure_dir(out.parent)
    if out.exists() and out.stat().st_size > 0:
        return out
    response = http_get(FRED_BASE, params={"id": series_id}, timeout=60)
    out.write_text(response.text, encoding="utf-8")
    return out


def _read_series(series_id: str) -> pd.DataFrame:
    path = _download_series(series_id)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    value_col = [c for c in df.columns if c != "observation_date"][0]
    out = df.rename(columns={"observation_date": "date", value_col: "value"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def _annualize_unemployment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["date", "value"]).copy()
    out["year"] = out["date"].dt.year
    out = out.groupby("year", as_index=False)["value"].mean()
    out = out.rename(columns={"value": "unemployment_rate"})
    return out


def _annual_series(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    out = df.dropna(subset=["date", "value"]).copy()
    out["year"] = out["date"].dt.year
    out = out.groupby("year", as_index=False)["value"].mean()
    out = out.rename(columns={"value": colname})
    return out


def run() -> Dict[str, Path]:
    ensure_dir(INTERMEDIATE_DIR)

    rows: List[pd.DataFrame] = []
    for state in STATES:
        ur = _annualize_unemployment(_read_series(_series_id(state.abbrev, "unemployment_rate")))
        pcpi = _annual_series(_read_series(_series_id(state.abbrev, "pcpi_nominal")), "pcpi_nominal")
        pop = _annual_series(
            _read_series(_series_id(state.abbrev, "population_thousands")), "population_thousands"
        )

        merged = ur.merge(pcpi, on="year", how="outer").merge(pop, on="year", how="outer")
        merged["state_abbrev"] = state.abbrev
        merged["state_name"] = state.name
        merged["state_fips"] = state.fips
        merged = merged[merged["year"].between(min(ANALYSIS_YEARS), max(ANALYSIS_YEARS))]
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True)
    out_path = INTERMEDIATE_DIR / "covariates_fred_state_year.csv"
    out.to_csv(out_path, index=False)
    return {"covariates_fred": out_path}


if __name__ == "__main__":
    print(run())
