"""Build policy text panel and embeddings for state-year rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import INTERMEDIATE_DIR, PROCESSED_DIR
from src.utils.io import ensure_dir


def _build_text_panel() -> pd.DataFrame:
    base = pd.read_parquet(PROCESSED_DIR / "panel_state_year.parquet")[["state_abbrev", "year"]].copy()
    events_path = INTERMEDIATE_DIR / "apis_policy_text_events.csv"

    if events_path.exists():
        events = pd.read_csv(events_path)
    else:
        events = pd.DataFrame(columns=["state_abbrev", "year", "topic_key", "change_text"])

    if not events.empty:
        events["year"] = pd.to_numeric(events["year"], errors="coerce")
        events["event_text"] = (
            events["topic_key"].fillna("topic")
            + ": "
            + events["change_text"].fillna("Policy change")
        )
        grouped = (
            events.groupby(["state_abbrev", "year"], as_index=False)
            .agg(policy_text=("event_text", lambda s: " ; ".join(x for x in s if isinstance(x, str))))
        )
    else:
        grouped = pd.DataFrame(columns=["state_abbrev", "year", "policy_text"])

    out = base.merge(grouped, on=["state_abbrev", "year"], how="left")
    out["policy_text"] = out["policy_text"].fillna("No APIS change recorded for tracked topics.")
    return out


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)


def _encode_transformer(texts: List[str]) -> Tuple[np.ndarray, str]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    vectors = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
            outputs = model(**encoded)
            pooled = _mean_pooling(outputs, encoded["attention_mask"])
            vectors.append(pooled.cpu().numpy())

    arr = np.vstack(vectors)
    return arr, model_id


def _encode_fallback(texts: List[str]) -> Tuple[np.ndarray, str]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=2048, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    n_comp = min(64, max(2, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    arr = svd.fit_transform(X)
    return arr, "tfidf_svd_fallback"


def run() -> Dict[str, Path]:
    ensure_dir(PROCESSED_DIR)

    panel = _build_text_panel()
    texts = panel["policy_text"].astype(str).tolist()

    encoder_name = "unknown"
    try:
        emb, encoder_name = _encode_transformer(texts)
    except Exception:
        emb, encoder_name = _encode_fallback(texts)

    emb_cols = [f"emb_{i:03d}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, columns=emb_cols)

    out = pd.concat([panel.reset_index(drop=True), emb_df], axis=1)
    out_path = PROCESSED_DIR / "policy_text_state_year.parquet"
    out.to_parquet(out_path, index=False)

    meta = {
        "encoder": encoder_name,
        "n_rows": int(out.shape[0]),
        "n_dims": int(emb.shape[1]),
    }
    meta_path = PROCESSED_DIR / "policy_text_state_year_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"policy_text_state_year": out_path, "policy_text_meta": meta_path}


if __name__ == "__main__":
    print(run())
