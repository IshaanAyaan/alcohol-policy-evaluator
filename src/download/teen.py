"""Fetch teen alcohol behavior outcomes from YRBS sources."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import INTERMEDIATE_DIR, YRBS_EXPLORER_BASE, YRBS_SOCRATA_DATASET
from src.utils.io import ensure_dir, http_get
from src.utils.states import ABBREV_TO_NAME, STATES


def _query_socrata(short_question: str, value_colname: str) -> pd.DataFrame:
    params = {
        "$select": "year,locationabbr,locationdesc,greater_risk_data_value",
        "$where": (
            "topic='Alcohol and Other Drug Use' and "
            f"shortquestiontext='{short_question}' and "
            "sex='Total' and race='Total' and grade='Total' and stratificationtype='State'"
        ),
        "$limit": 50000,
    }
    url = f"https://data.cdc.gov/resource/{YRBS_SOCRATA_DATASET}.json"
    response = http_get(url, params=params, timeout=60)
    records = response.json()
    if not records:
        return pd.DataFrame(columns=["state_abbrev", "year", value_colname])

    df = pd.DataFrame(records)
    df = df.rename(
        columns={
            "locationabbr": "state_abbrev",
            "locationdesc": "state_name",
            "greater_risk_data_value": value_colname,
        }
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df[value_colname] = pd.to_numeric(df[value_colname], errors="coerce")
    df = df[df["state_abbrev"].isin([s.abbrev for s in STATES])]
    return df[["state_abbrev", "year", value_colname]].dropna(subset=["year"])


def _explorer_location_codes() -> Dict[str, str]:
    response = http_get(f"{YRBS_EXPLORER_BASE}/YrbsExplorerLocations", timeout=30)
    locs = pd.DataFrame(response.json())
    if locs.empty:
        return {}

    state_locs = locs.loc[locs["LocationType"] == "State"].copy()
    out: Dict[str, str] = {}

    for state in STATES:
        subset = state_locs.loc[state_locs["LocationDescription"] == state.name]
        if subset.empty:
            continue
        # Prefer canonical 2-letter codes when present.
        subset = subset.sort_values(by="LocationCode", key=lambda s: s.astype(str).str.len())
        out[state.abbrev] = subset.iloc[0]["LocationCode"]

    return out


def _extract_total_value(rows: List[dict]) -> Optional[float]:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if df.empty:
        return None

    filt = df.loc[(df.get("StratType") == "Total") & (df.get("SubStrat") == "Total")]
    if filt.empty:
        return None
    val = pd.to_numeric(filt.iloc[0].get("MainValue"), errors="coerce")
    if pd.isna(val):
        return None
    return float(val)


def _fetch_explorer_point(state_abbrev: str, location_id: str, year: int, question_id: str) -> Optional[float]:
    try:
        response = http_get(
            f"{YRBS_EXPLORER_BASE}/TableData",
            params={"QuestionId": question_id, "Yr": year, "LocationId": location_id},
            timeout=6,
            retries=1,
        )
        rows = response.json()
        return _extract_total_value(rows)
    except Exception:  # noqa: BLE001
        return None


def _fetch_explorer_extension() -> pd.DataFrame:
    years = [2019, 2021, 2023]
    codes = _explorer_location_codes()
    rows = []

    with ThreadPoolExecutor(max_workers=32) as pool:
        h42_futures = {}
        for state in STATES:
            location_id = codes.get(state.abbrev)
            if not location_id:
                for year in years:
                    rows.append(
                        {
                            "state_abbrev": state.abbrev,
                            "year": year,
                            "teen_current_alcohol_use_pct": None,
                            "teen_binge_pct": None,
                            "teen_source": "YRBS_Explorer",
                            "coverage_flag": "missing_no_location_code",
                        }
                    )
                continue

            for year in years:
                h42_futures[(state.abbrev, year, location_id)] = pool.submit(
                    _fetch_explorer_point, state.abbrev, location_id, year, "H42"
                )

        # Pull H42 first; only request H43 when H42 exists to cut total calls.
        for (state_abbrev, year, location_id), fut_h42 in h42_futures.items():
            h42 = fut_h42.result()
            h43 = None
            if h42 is not None:
                h43 = _fetch_explorer_point(state_abbrev, location_id, year, "H43")
            rows.append(
                {
                    "state_abbrev": state_abbrev,
                    "year": year,
                    "teen_current_alcohol_use_pct": h42,
                    "teen_binge_pct": h43,
                    "teen_source": "YRBS_Explorer",
                    "coverage_flag": "observed" if (h42 is not None or h43 is not None) else "missing_no_state_data",
                }
            )

    return pd.DataFrame(rows)


def run() -> Dict[str, str]:
    ensure_dir(INTERMEDIATE_DIR)

    socrata_current = _query_socrata("Current alcohol use", "teen_current_alcohol_use_pct")
    socrata_binge = _query_socrata("Current binge drinking", "teen_binge_pct")

    socrata = socrata_current.merge(
        socrata_binge,
        on=["state_abbrev", "year"],
        how="left",
    )
    socrata["teen_source"] = "YRBS_Socrata"
    socrata["coverage_flag"] = "observed"

    explorer = _fetch_explorer_extension()

    # Combine and let explorer replace/add for extension years when data is present.
    frames = [f for f in [socrata, explorer] if not f.empty]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = out.sort_values(["state_abbrev", "year", "teen_source"]).drop_duplicates(
        subset=["state_abbrev", "year", "teen_source"], keep="last"
    )

    # Consolidate to one row per state-year by preferring observed explorer rows in 2019+.
    pref = out.copy()
    pref["_priority"] = pref.apply(
        lambda r: 3
        if (r["teen_source"] == "YRBS_Explorer" and r["coverage_flag"] == "observed")
        else (2 if r["teen_source"] == "YRBS_Socrata" else 1),
        axis=1,
    )
    pref = pref.sort_values(["state_abbrev", "year", "_priority"], ascending=[True, True, False])
    pref = pref.drop_duplicates(subset=["state_abbrev", "year"], keep="first").drop(columns=["_priority"])

    out_path = INTERMEDIATE_DIR / "teen_state_wave.csv"
    pref.to_csv(out_path, index=False)
    return {"teen_state_wave": str(out_path)}


if __name__ == "__main__":
    print(run())
