"""Predictive model ladder for crash and teen outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR, TABLES_DIR
from src.utils.io import ensure_dir


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    keys_val: pd.DataFrame
    keys_test: pd.DataFrame


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _build_split(df: pd.DataFrame, feature_cols: List[str], target_col: str, key_cols: List[str]) -> SplitData:
    data = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col]).copy()

    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    train = data[(data["year"] >= 2003) & (data["year"] <= 2016)].copy()
    val = data[(data["year"] >= 2017) & (data["year"] <= 2019)].copy()
    test = data[(data["year"] >= 2020) & (data["year"] <= 2023)].copy()

    # If no teen rows in some windows, back off to deterministic split by quantile year.
    if val.empty or test.empty:
        years = sorted(data["year"].unique())
        if len(years) >= 3:
            cut1 = years[int(len(years) * 0.6)]
            cut2 = years[int(len(years) * 0.8)]
            train = data[data["year"] <= cut1].copy()
            val = data[(data["year"] > cut1) & (data["year"] <= cut2)].copy()
            test = data[data["year"] > cut2].copy()

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_val = val[feature_cols]
    y_val = val[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        keys_val=val[key_cols].copy(),
        keys_test=test[key_cols].copy(),
    )


def _fit_models(split: SplitData, task: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    models = {
        "ElasticNet": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000)),
            ]
        ),
        "RandomForestRegressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        random_state=42,
                        min_samples_leaf=2,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoostingRegressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(random_state=42, max_depth=6)),
            ]
        ),
        "MLPRegressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        random_state=42,
                        max_iter=400,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }

    eval_rows = []
    pred_rows = []

    for model_name, model in models.items():
        if split.X_train.empty or split.X_val.empty or split.X_test.empty:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            model.fit(split.X_train, split.y_train)
            val_pred = model.predict(split.X_val)
            test_pred = model.predict(split.X_test)

        val_metrics = _evaluate(split.y_val.to_numpy(), val_pred)
        test_metrics = _evaluate(split.y_test.to_numpy(), test_pred)

        eval_rows.append(
            {
                "task": task,
                "model": model_name,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "n_train": int(split.X_train.shape[0]),
                "n_val": int(split.X_val.shape[0]),
                "n_test": int(split.X_test.shape[0]),
            }
        )

        pred_df = split.keys_test.copy()
        pred_df["task"] = task
        pred_df["model"] = model_name
        pred_df["y_true"] = split.y_test.to_numpy()
        pred_df["y_pred"] = test_pred
        pred_rows.append(pred_df)

    eval_df = pd.DataFrame(eval_rows)
    pred_out = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return eval_df, pred_out


def run() -> Dict[str, Path]:
    ensure_dir(PROCESSED_DIR)
    ensure_dir(TABLES_DIR)

    state = pd.read_parquet(PROCESSED_DIR / "panel_state_year_features.parquet")
    text = pd.read_parquet(PROCESSED_DIR / "policy_text_state_year.parquet")
    teen = pd.read_parquet(PROCESSED_DIR / "panel_state_wave_teen_features.parquet")

    emb_cols = [c for c in text.columns if c.startswith("emb_")]
    state = state.merge(text[["state_abbrev", "year", *emb_cols]], on=["state_abbrev", "year"], how="left")
    teen = teen.merge(text[["state_abbrev", "year", *emb_cols]], on=["state_abbrev", "year"], how="left")

    crash_features = [
        "beer_tax_usd_per_gallon",
        "policy_change_count_sunday_sales",
        "policy_change_count_underage_purchase",
        "unemployment_rate",
        "pcpi_nominal",
        "vmt_per_capita",
        "rate_impaired_per100k_lag1",
        "rate_impaired_per100k_lag2",
        "rate_alcohol_involved_per100k_lag1",
        "beer_tax_usd_per_gallon_lag1",
        "unemployment_rate_lag1",
    ] + emb_cols

    teen_features = [
        "beer_tax_usd_per_gallon",
        "policy_change_count_sunday_sales",
        "policy_change_count_underage_purchase",
        "unemployment_rate",
        "pcpi_nominal",
        "vmt_per_capita",
        "rate_impaired_per100k",
        "teen_current_alcohol_use_pct_lag1",
    ] + emb_cols

    crash_split = _build_split(
        state,
        feature_cols=[c for c in crash_features if c in state.columns],
        target_col="rate_impaired_per100k",
        key_cols=["state_abbrev", "year"],
    )
    crash_eval, crash_preds = _fit_models(crash_split, task="crash_rate_next_year")

    teen_split = _build_split(
        teen,
        feature_cols=[c for c in teen_features if c in teen.columns],
        target_col="teen_current_alcohol_use_pct",
        key_cols=["state_abbrev", "year"],
    )
    teen_eval, teen_preds = _fit_models(teen_split, task="teen_current_use_next_wave")

    eval_df = pd.concat([crash_eval, teen_eval], ignore_index=True)
    preds_df = pd.concat([crash_preds, teen_preds], ignore_index=True) if not crash_preds.empty or not teen_preds.empty else pd.DataFrame()

    eval_out = PROCESSED_DIR / "model_eval_summary.csv"
    preds_out = TABLES_DIR / "model_predictions_test.csv"

    eval_df.to_csv(eval_out, index=False)
    preds_df.to_csv(preds_out, index=False)

    return {"model_eval_summary": eval_out, "predictions": preds_out}


if __name__ == "__main__":
    print(run())
