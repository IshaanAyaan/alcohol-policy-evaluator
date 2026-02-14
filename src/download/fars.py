"""Download and aggregate FARS crash outcomes to state-year."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import ANALYSIS_YEARS, FARS_BASE, INTERMEDIATE_DIR, RAW_DIR
from src.utils.io import ensure_dir, http_get
from src.utils.states import STATES


def _zip_url(year: int) -> str:
    return f"{FARS_BASE}/{year}/National/FARS{year}NationalCSV.zip"


def _zip_path(year: int) -> Path:
    return RAW_DIR / "fars" / f"FARS{year}NationalCSV.zip"


def _download_zip(year: int) -> Path:
    out = _zip_path(year)
    ensure_dir(out.parent)
    if out.exists() and out.stat().st_size > 1024:
        return out
    resp = http_get(_zip_url(year), timeout=180)
    out.write_bytes(resp.content)
    return out


def _read_csv_from_zip(zip_path: Path, filename_suffix: str, usecols: List[str]) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        matches = [n for n in names if n.lower().endswith(filename_suffix.lower())]
        if not matches:
            raise FileNotFoundError(f"{filename_suffix} not found in {zip_path}")
        with zf.open(matches[0]) as handle:
            payload = handle.read()
        # Older FARS CSVs may contain non-UTF-8 bytes; attempt UTF-8 first, then latin-1.
        try:
            df = pd.read_csv(io.BytesIO(payload), usecols=usecols, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(payload), usecols=usecols, low_memory=False, encoding="latin-1")
    return df


def _aggregate_year(year: int) -> pd.DataFrame:
    zpath = _download_zip(year)

    accident = _read_csv_from_zip(zpath, "accident.csv", ["STATE", "ST_CASE", "FATALS"])
    vehicle = _read_csv_from_zip(zpath, "vehicle.csv", ["STATE", "ST_CASE", "VEH_NO", "DR_DRINK"])
    person = _read_csv_from_zip(zpath, "person.csv", ["STATE", "ST_CASE", "VEH_NO", "PER_TYP", "ALC_RES"])

    for df in (accident, vehicle, person):
        df["STATE"] = pd.to_numeric(df["STATE"], errors="coerce").astype("Int64")
        df["ST_CASE"] = pd.to_numeric(df["ST_CASE"], errors="coerce").astype("Int64")

    accident["FATALS"] = pd.to_numeric(accident["FATALS"], errors="coerce").fillna(0)
    vehicle["DR_DRINK"] = pd.to_numeric(vehicle["DR_DRINK"], errors="coerce")
    person["PER_TYP"] = pd.to_numeric(person["PER_TYP"], errors="coerce")
    person["ALC_RES"] = pd.to_numeric(person["ALC_RES"], errors="coerce")

    alcohol_cases = (
        vehicle.loc[vehicle["DR_DRINK"] == 1, ["STATE", "ST_CASE"]]
        .dropna()
        .drop_duplicates()
    )

    impaired_drivers = person[
        (person["PER_TYP"] == 1)
        & (person["ALC_RES"] >= 80)
        & (person["ALC_RES"] <= 940)
    ][["STATE", "ST_CASE"]].dropna()
    impaired_cases = impaired_drivers.drop_duplicates()

    totals = (
        accident.groupby("STATE", as_index=False)["FATALS"]
        .sum()
        .rename(columns={"FATALS": "fars_fatalities_total"})
    )

    alcohol_fatals = (
        accident.merge(alcohol_cases, on=["STATE", "ST_CASE"], how="inner")
        .groupby("STATE", as_index=False)["FATALS"]
        .sum()
        .rename(columns={"FATALS": "fars_fatalities_alcohol_involved"})
    )

    impaired_fatals = (
        accident.merge(impaired_cases, on=["STATE", "ST_CASE"], how="inner")
        .groupby("STATE", as_index=False)["FATALS"]
        .sum()
        .rename(columns={"FATALS": "fars_fatalities_impaired"})
    )

    merged = totals.merge(alcohol_fatals, on="STATE", how="left").merge(
        impaired_fatals, on="STATE", how="left"
    )
    merged["fars_fatalities_alcohol_involved"] = (
        merged["fars_fatalities_alcohol_involved"].fillna(0).astype(int)
    )
    merged["fars_fatalities_impaired"] = merged["fars_fatalities_impaired"].fillna(0).astype(int)
    merged["fars_fatalities_total"] = merged["fars_fatalities_total"].fillna(0).astype(int)
    merged["state_fips"] = merged["STATE"].astype(int).astype(str).str.zfill(2)
    merged["year"] = year

    out = merged[[
        "state_fips",
        "year",
        "fars_fatalities_total",
        "fars_fatalities_alcohol_involved",
        "fars_fatalities_impaired",
    ]].copy()
    return out


def run() -> Dict[str, Path]:
    ensure_dir(INTERMEDIATE_DIR)
    rows = []
    for year in ANALYSIS_YEARS:
        rows.append(_aggregate_year(year))

    out = pd.concat(rows, ignore_index=True)

    # Restrict to 50 states + DC only.
    valid_fips = {s.fips for s in STATES}
    out = out[out["state_fips"].isin(valid_fips)].copy()

    out_path = INTERMEDIATE_DIR / "fars_state_year.csv"
    out.to_csv(out_path, index=False)
    return {"fars_state_year": out_path}


if __name__ == "__main__":
    paths = run()
    for k, v in paths.items():
        print(f"{k}: {v}")
