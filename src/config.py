"""Project-wide configuration."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = ROOT / "logs"

START_YEAR = 2003
END_YEAR = 2023
ANALYSIS_YEARS = list(range(START_YEAR, END_YEAR + 1))

APIS_TOPICS = {
    "beer_tax": {"slug": "beer", "id": 30},
    "sunday_sales": {"slug": "bans-on-off-premises-sunday-sales", "id": 28},
    "underage_purchase": {"slug": "underage-purchase-of-alcohol", "id": 43},
}

R_JINA_PREFIX = "https://r.jina.ai/http://"
APIS_BASE = "alcoholpolicy.niaaa.nih.gov"
FARS_BASE = "https://static.nhtsa.gov/nhtsa/downloads/FARS"
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FHWA_VM2_TEMPLATE = "https://www.fhwa.dot.gov/policyinformation/statistics/{year}/vm2.cfm"
YRBS_SOCRATA_DATASET = "svam-8dhg"
YRBS_EXPLORER_BASE = "https://yrbs-explorer.services.cdc.gov/api"

R_SCRIPT_PATH = ROOT / "scripts" / "causal_did.R"
