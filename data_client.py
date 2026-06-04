import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests


DEFAULT_POCKETBASE_URL = "http://YOUR_POCKETBASE_IP:8090"
DEFAULT_COLLECTION = "smart_money_stats"


def load_config(config_path: str = "config.json") -> Dict[str, object]:
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_pocketbase_url(config_path: str = "config.json") -> str:
    config = load_config(config_path)
    return str(config.get("POCKETBASE_URL", DEFAULT_POCKETBASE_URL))


def fetch_smart_money_data(
    pocketbase_url: Optional[str] = None,
    config_path: str = "config.json",
    collection_name: str = DEFAULT_COLLECTION,
    per_page: int = 500,
    timeout: int = 10,
) -> pd.DataFrame:
    url_root = pocketbase_url or load_pocketbase_url(config_path)
    url = f"{url_root.rstrip('/')}/api/collections/{collection_name}/records"

    items: List[Dict[str, object]] = []
    page = 1
    while True:
        params = {
            "perPage": per_page,
            "page": page,
            "sort": "-timestamp",
            "filter": "current_price > 0",
        }
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("items", [])
        if not batch:
            break

        items.extend(batch)
        if len(batch) < per_page:
            break
        page += 1

    if not items:
        return pd.DataFrame()

    items.reverse()
    df = pd.DataFrame(items).copy()
    if "timestamp" not in df.columns:
        return pd.DataFrame()

    source_columns = set(df.columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)

    core_defaults = {
        "current_price": np.nan,
        "long_traders": 0.0,
        "short_traders": 0.0,
        "total_traders": 0.0,
        "long_pos_usdt": 0.0,
        "short_pos_usdt": 0.0,
        "total_pos_usdt": 0.0,
        "long_unrealized_pnl": 0.0,
        "short_unrealized_pnl": 0.0,
    }
    optional_numeric_columns = [
        "funding_rate",
        "ls_ratio",
        "long_avg_entry",
        "short_avg_entry",
    ]
    numeric_hints = ("price", "traders", "pos", "pnl", "ratio", "entry", "rate")

    for col in df.columns:
        if col == "timestamp":
            continue
        if col in core_defaults or col in optional_numeric_columns or any(
            hint in col for hint in numeric_hints
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, default in core_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if not pd.isna(default):
            df[col] = df[col].fillna(default)

    for col in optional_numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    short_unrealized = pd.to_numeric(
        df.get("short_unrealized_pnl", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).fillna(0.0)
    df["short_unrealized_pnl_raw"] = short_unrealized
    df["short_unrealized_pnl_plot"] = -short_unrealized.abs()

    df = df[df["current_price"].fillna(0) > 0].copy()

    if {"long_traders", "short_traders"}.issubset(source_columns):
        df = df[
            (df["long_traders"].fillna(0) > 0) | (df["short_traders"].fillna(0) > 0)
        ].copy()

    return df.reset_index(drop=True)
