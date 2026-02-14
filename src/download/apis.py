"""Download and parse APIS policy topic data via r.jina.ai fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.config import (
    ANALYSIS_YEARS,
    APIS_BASE,
    APIS_TOPICS,
    END_YEAR,
    INTERMEDIATE_DIR,
    RAW_DIR,
    R_JINA_PREFIX,
    START_YEAR,
)
from src.utils.io import ensure_dir, http_get
from src.utils.states import ABBREV_TO_NAME, NAME_TO_ABBREV, STATES


DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
DOLLAR_RE = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")


@dataclass
class ApisTopicPaths:
    slug: str
    topic_id: int

    @property
    def base_path(self) -> str:
        return f"apis-policy-topics/{self.slug}/{self.topic_id}"

    @property
    def urls(self) -> Dict[str, str]:
        return {
            "specific_date": f"{self.base_path}",
            "changes_over_time": f"{self.base_path}/changes-over-time",
            "timeline_of_changes": f"{self.base_path}/timeline-of-changes",
        }


def _normalize_state_name(raw: str) -> Optional[str]:
    raw = raw.strip()
    if raw in NAME_TO_ABBREV:
        return raw
    # Soft normalization for frequent variants.
    raw = raw.replace("  ", " ").replace("\u00a0", " ")
    if raw == "District Of Columbia":
        return "District of Columbia"
    return raw if raw in NAME_TO_ABBREV else None


def _rjina_fetch(path: str) -> str:
    path = path.lstrip("/")
    url = f"{R_JINA_PREFIX}{APIS_BASE}/{path}"
    return http_get(url, timeout=120).text


def _save_raw(topic_key: str, page_key: str, text: str) -> Path:
    out_dir = RAW_DIR / "apis"
    ensure_dir(out_dir)
    out_path = out_dir / f"{topic_key}_{page_key}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _extract_tokens(markdown_text: str) -> List[str]:
    tokens = []
    for line in markdown_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Drop markdown table separators and image lines.
        if set(line) <= {"|", "-", " ", ":"}:
            continue
        if line.startswith("!["):
            continue
        tokens.append(line)
    return tokens


def _find_state_row_blocks(tokens: Iterable[str]) -> List[Tuple[str, str, List[str]]]:
    tokens_list = list(tokens)
    state_by_name = {s.name: s.abbrev for s in STATES}
    blocks: List[Tuple[str, str, List[str]]] = []
    i = 0
    n = len(tokens_list)

    while i < n - 1:
        token = tokens_list[i]
        if token in state_by_name and tokens_list[i + 1] == state_by_name[token]:
            state_name = token
            state_abbrev = tokens_list[i + 1]
            j = i + 2
            while j < n:
                nxt = tokens_list[j]
                if nxt in state_by_name and j + 1 < n and tokens_list[j + 1] == state_by_name[nxt]:
                    break
                j += 1
            block = tokens_list[i:j]
            blocks.append((state_name, state_abbrev, block))
            i = j
        else:
            i += 1

    return blocks


def _parse_date_safe(text: str) -> Optional[datetime]:
    try:
        return datetime.strptime(text, "%m/%d/%Y")
    except ValueError:
        return None


def _parse_beer_current_values(specific_date_text: str) -> pd.DataFrame:
    tokens = _extract_tokens(specific_date_text)
    blocks = _find_state_row_blocks(tokens)

    rows = []
    for state_name, state_abbrev, block in blocks:
        as_of_dates = [tok for tok in block if DATE_RE.fullmatch(tok)]
        if not as_of_dates:
            continue
        dollars = []
        for tok in block:
            match = DOLLAR_RE.search(tok)
            if match:
                dollars.append(float(match.group(1)))
        beer_tax = dollars[0] if dollars else None
        rows.append(
            {
                "state_name": state_name,
                "state_abbrev": state_abbrev,
                "as_of": as_of_dates[0],
                "beer_tax_usd_per_gallon": beer_tax,
            }
        )

    return pd.DataFrame(rows)


def _parse_beer_changes_intervals(changes_over_time_text: str) -> pd.DataFrame:
    tokens = _extract_tokens(changes_over_time_text)
    blocks = _find_state_row_blocks(tokens)
    rows = []

    for state_name, state_abbrev, block in blocks:
        dates = [tok for tok in block if DATE_RE.fullmatch(tok)]
        if len(dates) < 2:
            continue
        start_date = dates[0]
        end_date = dates[1]

        dollars = []
        for tok in block:
            match = DOLLAR_RE.search(tok)
            if match:
                dollars.append(float(match.group(1)))
        beer_tax = dollars[0] if dollars else None

        rows.append(
            {
                "state_name": state_name,
                "state_abbrev": state_abbrev,
                "start_date": start_date,
                "end_date": end_date,
                "beer_tax_usd_per_gallon": beer_tax,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["start_dt"] = pd.to_datetime(out["start_date"], format="%m/%d/%Y", errors="coerce")
    out["end_dt"] = pd.to_datetime(out["end_date"], format="%m/%d/%Y", errors="coerce")
    return out


def _build_beer_tax_state_year(current_df: pd.DataFrame, intervals_df: pd.DataFrame) -> pd.DataFrame:
    all_states = pd.DataFrame(
        {
            "state_abbrev": [s.abbrev for s in STATES],
            "state_name": [s.name for s in STATES],
        }
    )
    current_map = current_df.set_index("state_abbrev")["beer_tax_usd_per_gallon"].to_dict()

    rows = []
    for _, srow in all_states.iterrows():
        state = srow["state_abbrev"]
        state_intervals = intervals_df.loc[intervals_df["state_abbrev"] == state].copy()
        state_intervals = state_intervals.sort_values("start_dt") if not state_intervals.empty else state_intervals

        for year in ANALYSIS_YEARS:
            date_point = datetime(year, 1, 1)
            matched_tax = None
            if not state_intervals.empty:
                mask = (state_intervals["start_dt"] <= date_point) & (date_point <= state_intervals["end_dt"])
                if mask.any():
                    matched_tax = state_intervals.loc[mask, "beer_tax_usd_per_gallon"].iloc[0]
            if matched_tax is None:
                matched_tax = current_map.get(state)
            rows.append(
                {
                    "state_abbrev": state,
                    "state_name": srow["state_name"],
                    "year": year,
                    "beer_tax_usd_per_gallon": matched_tax,
                }
            )

    out = pd.DataFrame(rows)
    out["beer_tax_usd_per_gallon"] = pd.to_numeric(out["beer_tax_usd_per_gallon"], errors="coerce")
    return out


def _parse_timeline_change_events(timeline_text: str, topic_key: str) -> pd.DataFrame:
    parts = timeline_text.split("Change for ")[1:]
    rows = []

    for part in parts:
        head = part.split("* **Jurisdiction:**", 1)[0].strip()
        if " in " not in head:
            continue
        state_raw = head.split(" in ", 1)[0].strip()
        state_name = _normalize_state_name(state_raw)
        if not state_name:
            continue
        state_abbrev = NAME_TO_ABBREV[state_name]

        date_match = re.search(r"Effective Date of Change:\*\*(\d{1,2}/\d{1,2}/\d{4})", part)
        if not date_match:
            continue
        effective_date = date_match.group(1)
        dt = _parse_date_safe(effective_date)
        if dt is None:
            continue

        change_match = re.search(r"\*\*Change:\*\*([^\n\r|]+)", part)
        change_text = change_match.group(1).strip() if change_match else "Policy change"

        rows.append(
            {
                "topic_key": topic_key,
                "state_name": state_name,
                "state_abbrev": state_abbrev,
                "effective_date": effective_date,
                "year": dt.year,
                "change_text": change_text,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[(out["year"] >= START_YEAR) & (out["year"] <= END_YEAR)].copy()
    return out


def _events_to_state_year_counts(events_df: pd.DataFrame) -> pd.DataFrame:
    all_grid = pd.MultiIndex.from_product(
        [[s.abbrev for s in STATES], ANALYSIS_YEARS], names=["state_abbrev", "year"]
    ).to_frame(index=False)

    def _topic_counts(topic_key: str, out_col: str) -> pd.DataFrame:
        subset = events_df.loc[events_df["topic_key"] == topic_key]
        if subset.empty:
            return all_grid.assign(**{out_col: 0})
        counts = (
            subset.groupby(["state_abbrev", "year"], as_index=False)
            .size()
            .rename(columns={"size": out_col})
        )
        merged = all_grid.merge(counts, on=["state_abbrev", "year"], how="left")
        merged[out_col] = merged[out_col].fillna(0).astype(int)
        return merged

    sunday = _topic_counts("sunday_sales", "policy_change_count_sunday_sales")
    underage = _topic_counts("underage_purchase", "policy_change_count_underage_purchase")

    out = sunday.merge(underage, on=["state_abbrev", "year"], how="left")
    out["policy_change_count_underage_purchase"] = (
        out["policy_change_count_underage_purchase"].fillna(0).astype(int)
    )
    return out


def _load_manual_fallback() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    manual_dir = RAW_DIR / "apis" / "manual"
    beer_path = manual_dir / "beer_tax_manual.csv"
    changes_path = manual_dir / "policy_changes_manual.csv"
    text_path = manual_dir / "policy_text_events_manual.csv"

    beer_df = pd.read_csv(beer_path) if beer_path.exists() else None
    changes_df = pd.read_csv(changes_path) if changes_path.exists() else None
    text_df = pd.read_csv(text_path) if text_path.exists() else None
    return beer_df, changes_df, text_df


def run() -> Dict[str, Path]:
    ensure_dir(INTERMEDIATE_DIR)
    results: Dict[str, Path] = {}

    try:
        topic_pages: Dict[str, Dict[str, str]] = {}
        for topic_key, topic_meta in APIS_TOPICS.items():
            paths = ApisTopicPaths(slug=topic_meta["slug"], topic_id=topic_meta["id"])
            topic_pages[topic_key] = {}
            for page_key, path in paths.urls.items():
                text = _rjina_fetch(path)
                _save_raw(topic_key, page_key, text)
                topic_pages[topic_key][page_key] = text

        beer_current = _parse_beer_current_values(topic_pages["beer_tax"]["specific_date"])
        beer_intervals = _parse_beer_changes_intervals(topic_pages["beer_tax"]["changes_over_time"])
        beer_tax_state_year = _build_beer_tax_state_year(beer_current, beer_intervals)

        events = []
        for topic_key in ("sunday_sales", "underage_purchase", "beer_tax"):
            tdf = _parse_timeline_change_events(topic_pages[topic_key]["timeline_of_changes"], topic_key)
            if not tdf.empty:
                events.append(tdf)
        events_df = pd.concat(events, ignore_index=True) if events else pd.DataFrame()

        changes_state_year = _events_to_state_year_counts(events_df)

        beer_out = INTERMEDIATE_DIR / "apis_beer_tax_state_year.csv"
        changes_out = INTERMEDIATE_DIR / "apis_policy_changes_state_year.csv"
        text_out = INTERMEDIATE_DIR / "apis_policy_text_events.csv"

        beer_tax_state_year.to_csv(beer_out, index=False)
        changes_state_year.to_csv(changes_out, index=False)
        events_df.to_csv(text_out, index=False)

        results.update({"beer_tax": beer_out, "policy_changes": changes_out, "policy_text": text_out})

    except Exception as exc:  # noqa: BLE001
        manual_beer, manual_changes, manual_text = _load_manual_fallback()
        if manual_beer is None or manual_changes is None:
            raise RuntimeError(
                "APIS parser failed and manual fallback files are missing in data/raw/apis/manual/."
            ) from exc

        beer_out = INTERMEDIATE_DIR / "apis_beer_tax_state_year.csv"
        changes_out = INTERMEDIATE_DIR / "apis_policy_changes_state_year.csv"
        text_out = INTERMEDIATE_DIR / "apis_policy_text_events.csv"

        manual_beer.to_csv(beer_out, index=False)
        manual_changes.to_csv(changes_out, index=False)
        if manual_text is not None:
            manual_text.to_csv(text_out, index=False)
        else:
            pd.DataFrame(
                columns=["topic_key", "state_name", "state_abbrev", "effective_date", "year", "change_text"]
            ).to_csv(text_out, index=False)

        results.update({"beer_tax": beer_out, "policy_changes": changes_out, "policy_text": text_out})

    return results


if __name__ == "__main__":
    out = run()
    for key, value in out.items():
        print(f"{key}: {value}")
