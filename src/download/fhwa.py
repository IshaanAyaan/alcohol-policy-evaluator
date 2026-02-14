"""Download FHWA VMT table VM-2 and aggregate to state-year."""

from __future__ import annotations

import io
from typing import Dict, List

import pandas as pd
import requests

from src.config import ANALYSIS_YEARS, FHWA_VM2_TEMPLATE, FRED_BASE, INTERMEDIATE_DIR, RAW_DIR
from src.utils.io import ensure_dir
from src.utils.states import NORMALIZED_NAME_TO_ABBREV, STATES


def _flatten_columns(columns) -> List[str]:
    if not isinstance(columns, pd.MultiIndex):
        return [str(c).strip() for c in columns]
    out = []
    for tup in columns:
        parts = [str(x).strip() for x in tup if str(x).strip() and "Unnamed" not in str(x)]
        out.append("_".join(parts) if parts else "col")
    return out


def _download_year_html(year: int) -> str:
    out_path = RAW_DIR / "fhwa" / f"vm2_{year}.html"
    ensure_dir(out_path.parent)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path.read_text(encoding="utf-8", errors="ignore")

    url = FHWA_VM2_TEMPLATE.format(year=year)
    resp = requests.get(url, timeout=60, verify=False)
    if resp.status_code == 404:
        raise FileNotFoundError(f"FHWA VM-2 missing for year {year}")
    resp.raise_for_status()
    out_path.write_text(resp.text, encoding="utf-8")
    return resp.text


def _parse_year(year: int) -> pd.DataFrame:
    html = _download_year_html(year)
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError(f"No table found for FHWA year {year}")

    df = tables[0].copy()
    df.columns = _flatten_columns(df.columns)

    state_col_candidates = [c for c in df.columns if "STATE" in c.upper()]
    if not state_col_candidates:
        state_col_candidates = [df.columns[0]]
    state_col = state_col_candidates[0]

    total_candidates = [c for c in df.columns if "TOTAL" in c.upper()]
    # Prefer the most rightward TOTAL-like column as final total.
    total_col = total_candidates[-1] if total_candidates else df.columns[-1]

    out = df[[state_col, total_col]].copy()
    out = out.rename(columns={state_col: "state_name_raw", total_col: "vmt_total_million"})

    out["state_name_raw"] = out["state_name_raw"].astype(str).str.strip()
    out = out[~out["state_name_raw"].str.upper().isin({"UNITED STATES", "TOTAL", "NAN"})]
    out = out[~out["state_name_raw"].str.contains("Puerto Rico", case=False, na=False)]

    out["state_abbrev"] = out["state_name_raw"].map(NORMALIZED_NAME_TO_ABBREV)
    out = out.dropna(subset=["state_abbrev"]) 

    out["vmt_total_million"] = (
        out["vmt_total_million"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.strip()
        .replace({"": None, "nan": None})
        .pipe(pd.to_numeric, errors="coerce")
    )
    out["year"] = year
    return out[["state_abbrev", "year", "vmt_total_million"]]


def _national_vmt_ratios() -> pd.DataFrame:
    # Monthly national traffic volume, used to scale 2007 state values backward for 2003-2006.
    resp = requests.get(FRED_BASE, params={"id": "TRFVOLUSM227NFWA"}, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    val_col = [c for c in df.columns if c != "observation_date"][0]
    out = df.rename(columns={"observation_date": "date", val_col: "value"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["year"] = out["date"].dt.year
    out = out.groupby("year", as_index=False)["value"].mean()
    base = float(out.loc[out["year"] == 2007, "value"].iloc[0])
    out["ratio_to_2007"] = out["value"] / base
    return out[["year", "ratio_to_2007"]]


def run() -> Dict[str, str]:
    ensure_dir(INTERMEDIATE_DIR)
    frames = []
    missing_years = []
    for year in ANALYSIS_YEARS:
        try:
            frames.append(_parse_year(year))
        except FileNotFoundError:
            missing_years.append(year)

    out = pd.concat(frames, ignore_index=True)
    valid = {s.abbrev for s in STATES}
    out = out[out["state_abbrev"].isin(valid)]

    # Fill missing years (2003-2006) using 2007 state shares scaled by national trend ratios.
    if missing_years:
        ratios = _national_vmt_ratios()
        base_2007 = out[out["year"] == 2007][["state_abbrev", "vmt_total_million"]].copy()
        if not base_2007.empty:
            imputed_rows = []
            for year in sorted(missing_years):
                ratio_row = ratios.loc[ratios["year"] == year, "ratio_to_2007"]
                if ratio_row.empty:
                    continue
                scale = float(ratio_row.iloc[0])
                temp = base_2007.copy()
                temp["year"] = year
                temp["vmt_total_million"] = temp["vmt_total_million"] * scale
                imputed_rows.append(temp)
            if imputed_rows:
                out = pd.concat([out, pd.concat(imputed_rows, ignore_index=True)], ignore_index=True)

    out = out.drop_duplicates(subset=["state_abbrev", "year"], keep="last").sort_values(
        ["state_abbrev", "year"]
    )
    out_path = INTERMEDIATE_DIR / "fhwa_vmt_state_year.csv"
    out.to_csv(out_path, index=False)
    return {"fhwa_vmt": str(out_path)}


if __name__ == "__main__":
    print(run())
