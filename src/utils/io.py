"""I/O helper functions."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_manifest_row(manifest_path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(manifest_path.parent)
    df = pd.DataFrame([row])
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        out = pd.concat([existing, df], ignore_index=True)
    else:
        out = df
    out.to_csv(manifest_path, index=False)


def http_get(
    url: str,
    *,
    timeout: int = 90,
    retries: int = 3,
    backoff_seconds: float = 1.0,
    params: Optional[Dict[str, Any]] = None,
    verify: bool = True,
) -> requests.Response:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout, params=params, verify=verify)
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt == retries:
                break
            time.sleep(backoff_seconds * attempt)
    if last_err:
        raise last_err
    raise RuntimeError("http_get failed with unknown error")


def coerce_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"-": None, "": None, "nan": None, "None": None, "\xa0": None})
        .pipe(pd.to_numeric, errors="coerce")
    )
