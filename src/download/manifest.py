"""Generate raw data manifest with hashes for reproducibility."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import RAW_DIR
from src.utils.io import sha256_file


def run() -> Dict[str, Path]:
    rows = []
    for path in sorted(RAW_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(RAW_DIR)
        rows.append(
            {
                "relative_path": str(rel),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    out = pd.DataFrame(rows)
    out_path = RAW_DIR / "data_manifest.csv"
    out.to_csv(out_path, index=False)
    return {"data_manifest": out_path}


if __name__ == "__main__":
    print(run())
