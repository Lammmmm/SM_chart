from typing import Dict

import numpy as np
import pandas as pd


def has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def get_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def safe_divide(a, b):
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        if isinstance(a, pd.Series):
            left = pd.to_numeric(a, errors="coerce")
            index = left.index
        else:
            index = b.index
            left = pd.Series(a, index=index, dtype="float64")

        if isinstance(b, pd.Series):
            right = pd.to_numeric(b, errors="coerce")
        else:
            right = pd.Series(b, index=index, dtype="float64")

        result = left.div(right.replace(0, np.nan))
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if b is None or pd.isna(b) or b == 0:
        return 0.0

    result = a / b
    if pd.isna(result) or not np.isfinite(result):
        return 0.0
    return result


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 2)
    min_periods = min(window, max(5, window // 3))
    clean = pd.to_numeric(series, errors="coerce")
    rolling_mean = clean.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = clean.rolling(window=window, min_periods=min_periods).std(ddof=0)
    return safe_divide(clean - rolling_mean, rolling_std.replace(0, np.nan))


def normalize_ls_ratio(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if not valid.empty and (valid > 10).mean() > 0.5:
        s = s / 100
    s = s.where(s >= 0, np.nan)
    return s.replace([np.inf, -np.inf], np.nan)


def add_derived_metrics(df: pd.DataFrame, params: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    smooth_window = int(params["smooth_window"])
    z_window = int(params["z_window"])

    out["long_pos_delta"] = get_series(out, "long_pos_usdt", 0.0).diff().fillna(0.0)
    out["short_pos_delta_raw"] = get_series(out, "short_pos_usdt", 0.0).diff().fillna(0.0)
    out["short_pos_delta_plot"] = -out["short_pos_delta_raw"]
    out["net_pos_flow"] = out["long_pos_delta"] - out["short_pos_delta_raw"]

    out["long_traders_delta"] = get_series(out, "long_traders", 0.0).diff().fillna(0.0)
    out["short_traders_delta"] = get_series(out, "short_traders", 0.0).diff().fillna(0.0)
    out["trader_crowding_delta"] = out["long_traders_delta"] - out["short_traders_delta"]

    out["long_pain"] = safe_divide(
        get_series(out, "long_unrealized_pnl", 0.0),
        get_series(out, "long_pos_usdt", 0.0),
    )
    out["short_pain"] = safe_divide(
        out.get("short_unrealized_pnl_raw", pd.Series(0.0, index=out.index)),
        get_series(out, "short_pos_usdt", 0.0),
    )

    trader_total = get_series(out, "long_traders", 0.0).fillna(0.0) + get_series(
        out, "short_traders", 0.0
    ).fillna(0.0)
    trader_based_long = safe_divide(get_series(out, "long_traders", 0.0), trader_total) * 100

    if has_col(out, "ls_ratio"):
        normalized_ls_ratio = normalize_ls_ratio(get_series(out, "ls_ratio", np.nan))
        ratio_total = normalized_ls_ratio + 1.0
        ratio_based_long = safe_divide(
            normalized_ls_ratio,
            ratio_total.where(ratio_total > 0, np.nan),
        ) * 100
        valid_ratio = normalized_ls_ratio.notna() & ratio_total.gt(0)
        long_percent = pd.Series(
            np.where(
                valid_ratio,
                ratio_based_long,
                np.where(trader_total > 0, trader_based_long, 50.0),
            ),
            index=out.index,
            dtype="float64",
        )
    else:
        long_percent = pd.Series(
            np.where(trader_total > 0, trader_based_long, 50.0),
            index=out.index,
            dtype="float64",
        )

    out["long_percent"] = long_percent.clip(lower=0, upper=100)
    out["short_percent"] = (100 - out["long_percent"]).clip(lower=0, upper=100)

    out["price_ret_5"] = get_series(out, "current_price", np.nan).pct_change(5).fillna(0.0)
    out["price_ret_15"] = get_series(out, "current_price", np.nan).pct_change(15).fillna(0.0)
    out["price_ma_20"] = get_series(out, "current_price", np.nan).rolling(20, min_periods=1).mean()
    out["price_ma_60"] = get_series(out, "current_price", np.nan).rolling(60, min_periods=1).mean()
    out["price_above_ma20"] = out["current_price"] > out["price_ma_20"]
    out["price_below_ma20"] = out["current_price"] < out["price_ma_20"]

    smooth_targets = [
        "long_pos_delta",
        "short_pos_delta_raw",
        "net_pos_flow",
        "long_pain",
        "short_pain",
        "long_percent",
        "short_percent",
    ]
    for col in smooth_targets:
        out[f"{col}_smooth"] = out[col].rolling(smooth_window, min_periods=1).mean()

    out["short_pos_delta_plot_smooth"] = -out["short_pos_delta_raw_smooth"]

    z_targets = [
        ("net_pos_flow", "net_pos_flow_z"),
        ("long_pos_delta", "long_pos_delta_z"),
        ("short_pos_delta_raw", "short_pos_delta_z"),
        ("long_pain", "long_pain_z"),
        ("short_pain", "short_pain_z"),
    ]
    for source_col, target_col in z_targets:
        out[target_col] = rolling_zscore(out[source_col], z_window)

    return out
