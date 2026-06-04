import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from backtest_page import render_backtest_page
import data_client as shared_data_client
import features as shared_features
import signals as shared_signals
from ui_shell import apply_common_style, render_console_header, render_workspace_switch


config_path = "config.json"
POCKETBASE_URL = "http://YOUR_POCKETBASE_IP:8090"
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        POCKETBASE_URL = config.get("POCKETBASE_URL", POCKETBASE_URL)


BULL_COLOR = "#00C087"
BEAR_COLOR = "#F6465D"
NEUTRAL_COLOR = "#8B949E"
WARNING_COLOR = "#F0B90B"
SQUEEZE_COLOR = "#A970FF"
PRICE_COLOR = "#E6EDF3"
MA20_COLOR = "#F0B90B"
MA60_COLOR = "#4F7CFF"
BACKGROUND = "#0B0E11"
PANEL_BG = "#11161D"
GRID_COLOR = "rgba(255,255,255,0.08)"
ZERO_LINE_COLOR = "rgba(255,255,255,0.18)"

TIME_RANGE_OPTIONS = {
    "最近 6 小时": pd.Timedelta(hours=6),
    "最近 12 小时": pd.Timedelta(hours=12),
    "最近 24 小时": pd.Timedelta(hours=24),
    "最近 3 天": pd.Timedelta(days=3),
    "最近 7 天": pd.Timedelta(days=7),
    "全部": None,
}

SIGNAL_META = {
    "bottom_watch": {"label": "潜在反弹", "reason_col": "bottom_reason", "priority": 3},
    "top_watch": {"label": "潜在回落", "reason_col": "top_reason", "priority": 4},
    "short_squeeze_watch": {
        "label": "空头轧空",
        "reason_col": "short_squeeze_reason",
        "priority": 1,
    },
    "long_liquidation_watch": {
        "label": "多头踩踏",
        "reason_col": "long_liquidation_reason",
        "priority": 2,
    },
}


st.set_page_config(
    page_title="Smart Money 统一控制台",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_common_style()


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


def suppress_repeated_events(event_series: pd.Series, cooldown_bars: int) -> pd.Series:
    cooldown_bars = max(int(cooldown_bars), 0)
    event_series = event_series.fillna(False).astype(bool)
    result = pd.Series(False, index=event_series.index, dtype="bool")
    last_trigger_idx = None

    for i, triggered in enumerate(event_series.to_numpy()):
        if not triggered:
            continue
        if last_trigger_idx is None or i - last_trigger_idx > cooldown_bars:
            result.iloc[i] = True
            last_trigger_idx = i

    return result


def consecutive_true_count(series: pd.Series) -> pd.Series:
    s = series.fillna(False).astype(bool)
    groups = (s != s.shift(fill_value=False)).cumsum()
    counts = s.groupby(groups).cumcount() + 1
    return counts.where(s, 0).astype(int)


def maybe_get_event_col(df: pd.DataFrame, signal_col: str) -> str:
    event_col = f"{signal_col}_event"
    return event_col if event_col in df.columns else signal_col


def render_status_card(label: str, value: str, color: str, subtext: str = "") -> None:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="status-value" style="color:{color};">{value}</div>
            <div class="status-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_signed_percent(value: float) -> str:
    return f"{value * 100:+.2f}%"


def format_percent_value(value: float) -> str:
    return f"{value:.1f}%"


def format_price(value: float) -> str:
    return f"{value:,.2f}"


def safe_float(value, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(numeric) else float(numeric)


def color_from_market_state(market_state: str) -> str:
    if "强偏多" in market_state or market_state == "偏多":
        return BULL_COLOR
    if "强偏空" in market_state or market_state == "偏空":
        return BEAR_COLOR
    return WARNING_COLOR if "确认" in market_state else NEUTRAL_COLOR


def pain_color_for_side(side: str, value: float) -> str:
    if side == "long":
        if value > 0:
            return BULL_COLOR
        if value < 0:
            return BEAR_COLOR
        return NEUTRAL_COLOR

    if value < 0:
        return BULL_COLOR
    if value > 0:
        return BEAR_COLOR
    return NEUTRAL_COLOR


def add_reference_line(
    fig: go.Figure,
    y: float,
    yref: str,
    color: str,
    dash: str = "dot",
    width: int = 1,
) -> None:
    fig.add_shape(
        type="line",
        xref="paper",
        x0=0,
        x1=1,
        yref=yref,
        y0=y,
        y1=y,
        line=dict(color=color, dash=dash, width=width),
        layer="below",
    )


def add_time_range_filter(df: pd.DataFrame, range_label: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    delta = TIME_RANGE_OPTIONS.get(range_label)
    if delta is None:
        return df.copy()

    cutoff = df["timestamp"].max() - delta
    filtered = df[df["timestamp"] >= cutoff].copy()
    return filtered.reset_index(drop=True)


def build_sidebar_options() -> Dict[str, object]:
    st.sidebar.header("雷达参数")
    mode = st.sidebar.radio("界面模式", ["作战模式", "高级模式"], index=0, horizontal=True)
    range_label = st.sidebar.selectbox(
        "数据范围",
        list(TIME_RANGE_OPTIONS.keys()),
        index=2,
    )
    smooth_window = st.sidebar.slider("平滑窗口 smooth_window", 1, 30, 5)
    z_window = st.sidebar.slider("z-score 窗口 z_window", 20, 240, 60, step=5)

    st.sidebar.subheader("信号阈值")
    price_move_threshold = st.sidebar.slider(
        "price_move_threshold",
        min_value=0.001,
        max_value=0.030,
        value=0.005,
        step=0.001,
        format="%.3f",
    )
    pain_threshold = st.sidebar.slider(
        "pain_threshold",
        min_value=0.005,
        max_value=0.150,
        value=0.030,
        step=0.005,
        format="%.3f",
    )
    crowding_high = st.sidebar.slider("crowding_high", 55, 95, 70)
    crowding_low = st.sidebar.slider("crowding_low", 5, 45, 30)
    flow_z_threshold = st.sidebar.slider(
        "flow_z_threshold",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        format="%.1f",
    )
    event_cooldown_bars = st.sidebar.slider("event_cooldown_bars", 1, 30, 5)

    st.sidebar.subheader("显示选项")
    show_signal_markers = st.sidebar.checkbox("显示信号标记", value=True)
    show_ma = st.sidebar.checkbox("显示均线", value=True)
    show_flow_panel = st.sidebar.checkbox("显示净仓位脉冲", value=False)
    show_pain_panel = st.sidebar.checkbox("显示盈亏压力", value=False)
    show_crowding_panel = st.sidebar.checkbox("显示多空拥挤度", value=False)
    show_entries = st.sidebar.checkbox("显示多空成本线", value=False)
    show_debug_indicators = st.sidebar.checkbox("显示全部调试指标", value=False)
    show_event_table = st.sidebar.checkbox("显示事件表", value=True)

    if crowding_low >= crowding_high:
        st.sidebar.warning("crowding_low 应小于 crowding_high，程序将按更保守的区间处理。")
        crowding_low = max(5, crowding_high - 5)

    if mode == "高级模式":
        show_flow_panel = True
        show_pain_panel = True
        show_crowding_panel = True

    st.sidebar.caption("这套雷达用于辅助观察潜在转折，不应视为百分百交易点。")

    return {
        "mode": mode,
        "range_label": range_label,
        "smooth_window": smooth_window,
        "z_window": z_window,
        "price_move_threshold": price_move_threshold,
        "pain_threshold": pain_threshold,
        "crowding_high": crowding_high,
        "crowding_low": crowding_low,
        "flow_z_threshold": flow_z_threshold,
        "event_cooldown_bars": event_cooldown_bars,
        "show_signal_markers": show_signal_markers,
        "show_ma": show_ma,
        "show_flow_panel": show_flow_panel,
        "show_pain_panel": show_pain_panel,
        "show_crowding_panel": show_crowding_panel,
        "show_entries": show_entries,
        "show_debug_indicators": show_debug_indicators,
        "show_event_table": show_event_table,
    }


@st.cache_data(ttl=60)
def fetch_and_process_data() -> pd.DataFrame:
    url = f"{POCKETBASE_URL.rstrip('/')}/api/collections/smart_money_stats/records"
    try:
        items: List[Dict[str, object]] = []
        page = 1
        while True:
            params = {
                "perPage": 500,
                "page": page,
                "sort": "-timestamp",
                "filter": "current_price > 0",
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("items", [])
            if not batch:
                break

            items.extend(batch)
            if len(batch) < 500:
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
            if pd.isna(default):
                continue
            df[col] = df[col].fillna(default)

        for col in optional_numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["short_unrealized_pnl_raw"] = get_series(df, "short_unrealized_pnl", 0.0).fillna(0.0)
        # Plot-only mirror: keep the raw short PnL untouched for signal semantics.
        df["short_unrealized_pnl_plot"] = -df["short_unrealized_pnl_raw"].abs()

        df = df[df["current_price"].fillna(0) > 0].copy()

        if {"long_traders", "short_traders"}.issubset(source_columns):
            df = df[
                (df["long_traders"].fillna(0) > 0) | (df["short_traders"].fillna(0) > 0)
            ].copy()

        return df.reset_index(drop=True)

    except Exception as exc:
        st.error(f"数据拉取或清洗失败: {exc}")
        return pd.DataFrame()


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

    out["long_pain"] = safe_divide(get_series(out, "long_unrealized_pnl", 0.0), get_series(out, "long_pos_usdt", 0.0))
    # Raw short PnL is preserved; signal logic reads this series as the short-side
    # profit/loss pressure indicator instead of using the mirrored plot field.
    out["short_pain"] = safe_divide(out["short_unrealized_pnl_raw"], get_series(out, "short_pos_usdt", 0.0))

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


def build_reason_series(
    index: pd.Index,
    checks: List[tuple],
    default_text: str,
) -> pd.Series:
    arrays = [
        (pd.Series(cond, index=index).fillna(False).to_numpy(dtype=bool), text)
        for cond, text in checks
    ]
    reasons: List[str] = []
    for pos in range(len(index)):
        parts = [text for arr, text in arrays if arr[pos]]
        reasons.append(" + ".join(parts[:4]) if parts else default_text)
    return pd.Series(reasons, index=index, dtype="object")


def add_signal_columns(df: pd.DataFrame, params: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    price_move_threshold = float(params["price_move_threshold"])
    pain_threshold = float(params["pain_threshold"])
    crowding_high = float(params["crowding_high"])
    crowding_low = float(params["crowding_low"])
    flow_z_threshold = float(params["flow_z_threshold"])
    cooldown_bars = int(params["event_cooldown_bars"])

    price_weakening_strict = (out["price_ret_5"] < 0) | (out["current_price"] < out["price_ma_20"])
    price_near_ma20 = out["current_price"] <= out["price_ma_20"] * (1 + price_move_threshold)

    bottom_checks = {
        "price_flush": out["price_ret_15"] < -price_move_threshold,
        "longs_hurt": out["long_pain_smooth"] < -pain_threshold,
        "shorts_profiting": out["short_pain_smooth"] > pain_threshold,
        "shorts_covering": out["short_pos_delta_raw_smooth"] < 0,
        "bullish_flow": out["net_pos_flow_smooth"] > 0,
        "short_crowded": (out["long_percent_smooth"] < crowding_low)
        | (out["short_percent_smooth"] > crowding_high),
        "flow_burst": out["net_pos_flow_z"] > flow_z_threshold,
        "cover_burst": out["short_pos_delta_z"] < -flow_z_threshold,
    }
    out["bottom_score"] = sum(cond.astype(int) for cond in bottom_checks.values())
    out["bottom_watch"] = out["bottom_score"] >= 4
    out["bottom_reason"] = build_reason_series(
        out.index,
        [
            (bottom_checks["short_crowded"], "空头拥挤"),
            (bottom_checks["shorts_covering"], "空头减仓"),
            (bottom_checks["bullish_flow"], "多头回流"),
            (bottom_checks["price_flush"], "价格急跌"),
            (bottom_checks["longs_hurt"], "多头承压"),
            (bottom_checks["shorts_profiting"], "空头浮盈"),
        ],
        "价格承压后的反弹观察",
    )

    top_checks = {
        "price_stretch": out["price_ret_15"] > price_move_threshold,
        "longs_profiting": out["long_pain_smooth"] > pain_threshold,
        "long_crowded": out["long_percent_smooth"] > crowding_high,
        "longs_chasing": out["long_pos_delta_smooth"] > 0,
        "shorts_reloading": out["short_pos_delta_raw_smooth"] > 0,
        "price_weakening": price_weakening_strict,
        "price_near_ma20": price_near_ma20,
        "long_chase_z": out["long_pos_delta_z"] > flow_z_threshold,
        "short_reload_z": out["short_pos_delta_z"] > flow_z_threshold,
    }
    out["top_score"] = (
        top_checks["price_stretch"].astype(int)
        + top_checks["longs_profiting"].astype(int)
        + top_checks["long_crowded"].astype(int)
        + top_checks["longs_chasing"].astype(int)
        + top_checks["shorts_reloading"].astype(int)
        + top_checks["price_weakening"].astype(int) * 2
        + top_checks["price_near_ma20"].astype(int)
        + top_checks["long_chase_z"].astype(int)
        + top_checks["short_reload_z"].astype(int)
    )
    out["top_watch"] = out["top_score"] >= 4
    out["top_reason"] = build_reason_series(
        out.index,
        [
            (top_checks["long_crowded"], "多头拥挤"),
            (top_checks["longs_chasing"], "多头追高"),
            (top_checks["shorts_reloading"], "空头加仓"),
            (top_checks["price_weakening"], "价格走弱"),
            (top_checks["price_near_ma20"], "价格贴近MA20"),
            (top_checks["price_stretch"], "价格拉升"),
            (top_checks["longs_profiting"], "多头浮盈高位"),
        ],
        "上涨后的回落观察",
    )

    short_entry_available = has_col(out, "short_avg_entry")
    short_pain_turn = (out["short_pain_smooth"].diff().fillna(0.0) < -(pain_threshold / 2)) | (
        out["short_pain_smooth"] < 0
    )
    if short_entry_available:
        squeeze_checks = {
            "short_crowded": out["short_percent_smooth"] > crowding_high,
            "break_short_entry": out["current_price"] > get_series(out, "short_avg_entry", np.nan),
            "shorts_covering": out["short_pos_delta_raw_smooth"] < 0,
            "short_pain_turn": short_pain_turn,
        }
        out["short_squeeze_score"] = sum(cond.astype(int) for cond in squeeze_checks.values())
        out["short_squeeze_watch"] = out["short_squeeze_score"] >= 3
        out["short_squeeze_reason"] = build_reason_series(
            out.index,
            [
                (squeeze_checks["short_crowded"], "空头拥挤"),
                (squeeze_checks["break_short_entry"], "价格突破空头成本线"),
                (squeeze_checks["shorts_covering"], "空头减仓"),
                (squeeze_checks["short_pain_turn"], "空头盈利回吐"),
            ],
            "空头轧空观察",
        )
    else:
        squeeze_checks = {
            "short_crowded": out["short_percent_smooth"] > crowding_high,
            "price_rebound": out["price_ret_5"] > price_move_threshold,
            "shorts_covering": out["short_pos_delta_raw_smooth"] < 0,
            "bullish_flow": out["net_pos_flow_smooth"] > 0,
        }
        out["short_squeeze_score"] = sum(cond.astype(int) for cond in squeeze_checks.values())
        out["short_squeeze_watch"] = out["short_squeeze_score"] >= 3
        out["short_squeeze_reason"] = build_reason_series(
            out.index,
            [
                (squeeze_checks["short_crowded"], "空头拥挤"),
                (squeeze_checks["price_rebound"], "价格快速反弹"),
                (squeeze_checks["shorts_covering"], "空头减仓"),
                (squeeze_checks["bullish_flow"], "多头回流"),
            ],
            "空头轧空观察",
        )

    long_entry_available = has_col(out, "long_avg_entry")
    if long_entry_available:
        liquidation_checks = {
            "long_crowded": out["long_percent_smooth"] > crowding_high,
            "break_long_entry": out["current_price"] < get_series(out, "long_avg_entry", np.nan),
            "longs_cutting": out["long_pos_delta_smooth"] < 0,
            "long_pain": out["long_pain_smooth"] < -pain_threshold,
        }
        out["long_liquidation_score"] = sum(cond.astype(int) for cond in liquidation_checks.values())
        out["long_liquidation_watch"] = out["long_liquidation_score"] >= 3
        out["long_liquidation_reason"] = build_reason_series(
            out.index,
            [
                (liquidation_checks["long_crowded"], "多头拥挤"),
                (liquidation_checks["break_long_entry"], "价格跌破多头成本线"),
                (liquidation_checks["longs_cutting"], "多头减仓"),
                (liquidation_checks["long_pain"], "多头止损压力"),
            ],
            "多头踩踏观察",
        )
    else:
        liquidation_checks = {
            "long_crowded": out["long_percent_smooth"] > crowding_high,
            "price_drop": out["price_ret_5"] < -price_move_threshold,
            "longs_cutting": out["long_pos_delta_smooth"] < 0,
            "bearish_flow": out["net_pos_flow_smooth"] < 0,
        }
        out["long_liquidation_score"] = sum(cond.astype(int) for cond in liquidation_checks.values())
        out["long_liquidation_watch"] = out["long_liquidation_score"] >= 3
        out["long_liquidation_reason"] = build_reason_series(
            out.index,
            [
                (liquidation_checks["long_crowded"], "多头拥挤"),
                (liquidation_checks["price_drop"], "价格快速走弱"),
                (liquidation_checks["longs_cutting"], "多头减仓"),
                (liquidation_checks["bearish_flow"], "空头回流"),
            ],
            "多头踩踏观察",
        )

    bullish_score = (
        (out["net_pos_flow_smooth"] > 0).astype(int) * 20
        + (out["short_pos_delta_raw_smooth"] < 0).astype(int) * 15
        + out["bottom_watch"].astype(int) * 25
        + out["short_squeeze_watch"].astype(int) * 25
        + out["price_above_ma20"].astype(int) * 15
    )
    bearish_score = (
        (out["net_pos_flow_smooth"] < 0).astype(int) * 20
        + (out["long_pos_delta_smooth"] < 0).astype(int) * 15
        + out["top_watch"].astype(int) * 25
        + out["long_liquidation_watch"].astype(int) * 25
        + out["price_below_ma20"].astype(int) * 15
    )
    out["smart_score"] = (bullish_score - bearish_score).clip(-100, 100)

    out["market_state"] = np.select(
        [
            out["smart_score"] >= 60,
            out["smart_score"] >= 25,
            out["smart_score"] <= -60,
            out["smart_score"] <= -25,
        ],
        [
            "强偏多 / 反弹确认",
            "偏多",
            "强偏空 / 下跌确认",
            "偏空",
        ],
        default="中性",
    )

    signal_cols = [
        "bottom_watch",
        "top_watch",
        "short_squeeze_watch",
        "long_liquidation_watch",
    ]

    for col in signal_cols:
        watch_series = out[col].fillna(False).astype(bool)
        out[f"{col}_duration"] = consecutive_true_count(watch_series)
        event_series = watch_series & (~watch_series.shift(1, fill_value=False))
        out[f"{col}_event"] = suppress_repeated_events(event_series, cooldown_bars)

    return out


def build_signal_table(df: pd.DataFrame) -> pd.DataFrame:
    events: List[Dict[str, object]] = []
    for signal_col, meta in SIGNAL_META.items():
        if signal_col not in df.columns:
            continue

        event_col = maybe_get_event_col(df, signal_col)
        signal_rows = df[df[event_col].fillna(False)].copy()
        if signal_rows.empty:
            continue

        reason_col = meta["reason_col"]
        for _, row in signal_rows.iterrows():
            events.append(
                {
                    "timestamp": row["timestamp"],
                    "signal_type": meta["label"],
                    "trade_meaning": get_trade_meaning(meta["label"]),
                    "current_price": row.get("current_price", np.nan),
                    "smart_score": row.get("smart_score", np.nan),
                    "long_percent_smooth": row.get("long_percent_smooth", np.nan),
                    "short_percent_smooth": row.get("short_percent_smooth", np.nan),
                    "long_pain_smooth": row.get("long_pain_smooth", np.nan),
                    "short_pain_smooth": row.get("short_pain_smooth", np.nan),
                    "reason": row.get(reason_col, ""),
                    "priority": meta["priority"],
                }
            )

    if not events:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "signal_type",
                "trade_meaning",
                "current_price",
                "smart_score",
                "long_percent_smooth",
                "short_percent_smooth",
                "long_pain_smooth",
                "short_pain_smooth",
                "reason",
            ]
        )

    signal_df = pd.DataFrame(events).copy()
    signal_df = signal_df.sort_values(
        ["timestamp", "priority"],
        ascending=[False, True],
    ).drop_duplicates(subset=["timestamp"], keep="first")
    signal_df = signal_df.sort_values("timestamp", ascending=False).head(20).copy()
    signal_df["timestamp"] = pd.to_datetime(signal_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    signal_df["current_price"] = pd.to_numeric(signal_df["current_price"], errors="coerce").round(2)
    signal_df["smart_score"] = pd.to_numeric(signal_df["smart_score"], errors="coerce").round(0).astype("Int64")
    signal_df["long_percent_smooth"] = pd.to_numeric(
        signal_df["long_percent_smooth"], errors="coerce"
    ).round(1)
    signal_df["short_percent_smooth"] = pd.to_numeric(
        signal_df["short_percent_smooth"], errors="coerce"
    ).round(1)
    signal_df["long_pain_smooth"] = (
        pd.to_numeric(signal_df["long_pain_smooth"], errors="coerce") * 100
    ).round(2)
    signal_df["short_pain_smooth"] = (
        pd.to_numeric(signal_df["short_pain_smooth"], errors="coerce") * 100
    ).round(2)
    signal_df = signal_df.drop(columns=["priority"], errors="ignore")

    return signal_df.rename(
        columns={
            "timestamp": "时间",
            "signal_type": "信号类型",
            "trade_meaning": "交易含义",
            "current_price": "BTC价格",
            "smart_score": "Smart Score",
            "long_percent_smooth": "多头占比(%)",
            "short_percent_smooth": "空头占比(%)",
            "long_pain_smooth": "多头盈亏压力(%)",
            "short_pain_smooth": "空头盈亏压力(%)",
            "reason": "原因",
        }
    )


def get_current_watch_status(latest: pd.Series) -> tuple[str, str, str]:
    watch_priority = [
        ("short_squeeze_watch", "空头轧空观察中", SQUEEZE_COLOR),
        ("long_liquidation_watch", "多头踩踏观察中", WARNING_COLOR),
        ("bottom_watch", "潜在反弹观察中", BULL_COLOR),
        ("top_watch", "潜在回落观察中", BEAR_COLOR),
    ]

    for signal_col, label, color in watch_priority:
        if bool(latest.get(signal_col, False)):
            duration_value = pd.to_numeric(latest.get(f"{signal_col}_duration", 0), errors="coerce")
            duration = int(0 if pd.isna(duration_value) else duration_value)
            return label, f"已持续 {duration} 根数据", color

    return "无明显预警", "当前最新一根数据未处于 watch 状态", NEUTRAL_COLOR


def describe_pressure(side: str, value: float) -> str:
    value = safe_float(value, 0.0)
    percent_text = f"{abs(value) * 100:.2f}%"

    if side == "long":
        return f"多头浮盈 {percent_text}" if value >= 0 else f"多头亏损 {percent_text}"
    return f"空头浮盈 {percent_text}" if value >= 0 else f"空头亏损 {percent_text}"


def get_trade_meaning(signal_type: str) -> str:
    meaning_map = {
        "潜在反弹": "空头拥挤且获利较厚，不宜追空，等待价格确认",
        "空头轧空": "空头开始回补，反弹可能加速",
        "潜在回落": "多头拥挤且追高，谨防回落",
        "多头踩踏": "多头亏损扩大并减仓，趋势可能继续下行",
    }
    return meaning_map.get(signal_type, "结合价格确认后再决策")


def get_latest_event_snapshot(df: pd.DataFrame) -> Dict[str, object]:
    event_rows: List[Dict[str, object]] = []
    for signal_col, meta in SIGNAL_META.items():
        if signal_col not in df.columns:
            continue

        event_col = maybe_get_event_col(df, signal_col)
        signal_rows = df[df[event_col].fillna(False)].copy()
        if signal_rows.empty:
            continue

        last_index = signal_rows.index[-1]
        row = signal_rows.iloc[-1]
        event_rows.append(
            {
                "signal_type": meta["label"],
                "reason": row.get(meta["reason_col"], ""),
                "trade_meaning": get_trade_meaning(meta["label"]),
                "timestamp": row.get("timestamp"),
                "priority": meta["priority"],
                "bars_ago": int(max(df.index.max() - last_index, 0)),
            }
        )

    if not event_rows:
        return {
            "signal_type": "暂无信号",
            "reason": "",
            "trade_meaning": "当前没有新的转折事件，优先观察价格与 Smart Score 是否共振。",
            "timestamp": None,
            "priority": 99,
            "bars_ago": None,
        }

    event_df = pd.DataFrame(event_rows).sort_values(
        ["timestamp", "priority"],
        ascending=[False, True],
    )
    return event_df.iloc[0].to_dict()


def build_tactical_summary(df: pd.DataFrame) -> Dict[str, object]:
    latest = df.iloc[-1]
    latest_event = get_latest_event_snapshot(df)
    current_price = safe_float(latest.get("current_price", 0), 0.0)
    long_percent = safe_float(latest.get("long_percent_smooth", 50), 50.0)
    short_percent = safe_float(latest.get("short_percent_smooth", 50), 50.0)
    long_pain = safe_float(latest.get("long_pain_smooth", 0), 0.0)
    short_pain = safe_float(latest.get("short_pain_smooth", 0), 0.0)
    smart_score = safe_float(latest.get("smart_score", 0), 0.0)
    price_above_ma20 = bool(latest.get("price_above_ma20", False))
    price_below_ma20 = bool(latest.get("price_below_ma20", False))
    net_flow = safe_float(latest.get("net_pos_flow_smooth", 0), 0.0)

    reason_candidates: List[str] = []
    if short_percent >= 60:
        reason_candidates.append(f"空头占比 {short_percent:.1f}%")
    elif long_percent >= 60:
        reason_candidates.append(f"多头占比 {long_percent:.1f}%")
    else:
        reason_candidates.append(f"多空占比较均衡 {long_percent:.1f}% / {short_percent:.1f}%")

    reason_candidates.append(describe_pressure("short", short_pain))
    reason_candidates.append(describe_pressure("long", long_pain))
    if price_above_ma20:
        reason_candidates.append("价格位于 MA20 上方")
    elif price_below_ma20:
        reason_candidates.append("价格位于 MA20 下方")

    summary_title = "中性震荡，无明显优势"
    summary_color = NEUTRAL_COLOR
    action_text = "继续等待，先看方向确认"
    action_color = NEUTRAL_COLOR
    confirmation_status = "未确认"
    confirmation_rules = [
        "等待价格明确站上或跌破 MA20",
        "等待 Smart Score 离开中性区间",
        "等待净仓位流向出现持续方向",
    ]
    invalidation_rules = [
        "信号反复切换时避免追单",
        "价格重新回到震荡区间",
        "净仓位流向缺乏延续性",
    ]

    if bool(latest.get("short_squeeze_watch", False)):
        summary_title = (
            "空头轧空观察中，反弹正在确认"
            if price_above_ma20 or smart_score >= 25
            else "空头轧空观察中，等待价格确认"
        )
        summary_color = SQUEEZE_COLOR
        action_text = "不追空，等回踩或确认后处理"
        action_color = SQUEEZE_COLOR
        confirmation_status = "反弹确认中" if price_above_ma20 or smart_score >= 25 else "等待确认"
        confirmation_rules = [
            "价格站回 MA20 并保持其上",
            "Smart Score 修复到 -25 以上",
            "净仓位流向持续转正",
        ]
        invalidation_rules = [
            "价格重新跌回 MA20 下方",
            "Smart Score 再次走弱并接近 -60",
            "净仓位流向重新转负",
        ]
    elif bool(latest.get("long_liquidation_watch", False)):
        summary_title = "多头踩踏观察中，趋势偏空"
        summary_color = WARNING_COLOR
        action_text = "不抄底，等止跌确认"
        action_color = WARNING_COLOR
        confirmation_status = "下跌确认中" if price_below_ma20 and smart_score <= -25 else "等待确认"
        confirmation_rules = [
            "价格持续位于 MA20 下方",
            "Smart Score 维持在 -25 以下",
            "净仓位流向持续转负",
        ]
        invalidation_rules = [
            "价格重新站回 MA20",
            "Smart Score 修复到 -25 以上",
            "净仓位流向重新转正",
        ]
    elif bool(latest.get("bottom_watch", False)):
        trend_text = "偏空趋势中" if smart_score <= -25 or price_below_ma20 else "震荡偏弱中"
        summary_title = f"{trend_text}，出现反弹预警，不宜追空，等待确认"
        summary_color = BULL_COLOR
        action_text = "先等价格确认，不急着抄底"
        action_color = BULL_COLOR
        confirmation_status = "等待反弹确认"
        confirmation_rules = [
            "价格站回 MA20",
            "Smart Score 修复到 -25 以上",
            "净仓位流向持续转正",
        ]
        invalidation_rules = [
            "价格继续跌破前低",
            "Smart Score 跌破 -60",
            "净仓位流向重新转负",
        ]
    elif bool(latest.get("top_watch", False)):
        trend_text = "偏多趋势中" if smart_score >= 25 or price_above_ma20 else "震荡偏强中"
        summary_title = f"{trend_text}，出现回落预警，不宜追多，等待确认"
        summary_color = BEAR_COLOR
        action_text = "不追多，等回落确认"
        action_color = BEAR_COLOR
        confirmation_status = "等待回落确认"
        confirmation_rules = [
            "价格跌回 MA20 下方",
            "Smart Score 回落到 25 以下",
            "净仓位流向持续转弱或转负",
        ]
        invalidation_rules = [
            "价格重新站稳 MA20 上方",
            "Smart Score 重回 60 上方",
            "净仓位流向重新转正",
        ]
    elif smart_score >= 60 and price_above_ma20:
        summary_title = "强偏多，反弹已确认，等待回踩机会，不宜追高"
        summary_color = BULL_COLOR
        action_text = "偏多看待，等回踩而非追高"
        action_color = BULL_COLOR
        confirmation_status = "已确认"
        confirmation_rules = [
            "价格继续站稳 MA20 上方",
            "Smart Score 维持在 25 以上",
            "净仓位流向保持偏多",
        ]
        invalidation_rules = [
            "价格重新跌破 MA20",
            "Smart Score 跌回 25 以下",
            "净仓位流向重新转负",
        ]
    elif smart_score <= -60 and price_below_ma20:
        summary_title = "强偏空，下跌确认中，谨慎追空，等待反抽后确认"
        summary_color = BEAR_COLOR
        action_text = "偏空看待，等反抽而非追空"
        action_color = BEAR_COLOR
        confirmation_status = "已确认"
        confirmation_rules = [
            "价格继续位于 MA20 下方",
            "Smart Score 维持在 -25 以下",
            "净仓位流向持续偏空",
        ]
        invalidation_rules = [
            "价格重新站回 MA20",
            "Smart Score 收复到 -25 以上",
            "净仓位流向重新转正",
        ]
    elif smart_score >= 25:
        summary_title = "偏多运行中，暂无明显做空优势，等待确认后再跟随"
        summary_color = BULL_COLOR
        action_text = "不逆势做空，等更好确认"
        action_color = BULL_COLOR
        confirmation_status = "偏多未完全确认"
    elif smart_score <= -25:
        summary_title = "偏空运行中，反弹未确认前不宜抄底"
        summary_color = BEAR_COLOR
        action_text = "不急着抄底，先看止跌证据"
        action_color = BEAR_COLOR
        confirmation_status = "偏空未完全确认"

    if net_flow > 0:
        reason_candidates.append("净仓位流向偏多")
    elif net_flow < 0:
        reason_candidates.append("净仓位流向偏空")

    if latest_event["signal_type"] != "暂无信号" and latest_event["bars_ago"] is not None and latest_event["bars_ago"] <= 3:
        reason_candidates.append(f"最近事件：{latest_event['signal_type']}")

    return {
        "summary_title": summary_title,
        "summary_reason": reason_candidates[:3],
        "confirmation_rules": confirmation_rules[:3],
        "invalidation_rules": invalidation_rules[:3],
        "summary_color": summary_color,
        "action_text": action_text,
        "action_color": action_color,
        "confirmation_status": confirmation_status,
        "latest_event_text": latest_event["signal_type"],
        "latest_event_trade_meaning": latest_event["trade_meaning"],
    }


def render_tactical_summary(summary: Dict[str, object]) -> None:
    reason_html = "".join(f"<li>{item}</li>" for item in summary["summary_reason"])
    confirmation_html = "".join(f"<li>{item}</li>" for item in summary["confirmation_rules"])
    invalidation_html = "".join(f"<li>{item}</li>" for item in summary["invalidation_rules"])

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(180deg, rgba(17,22,29,1) 0%, rgba(11,14,17,1) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-left: 5px solid {summary["summary_color"]};
            border-radius: 16px;
            padding: 18px 20px 14px 20px;
            margin: 6px 0 14px 0;
        ">
            <div style="font-size:0.88rem;color:#8b949e;margin-bottom:8px;">作战结论</div>
            <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
                <span style="padding:4px 10px;border-radius:999px;background:rgba(255,255,255,0.05);color:#c9d1d9;font-size:0.82rem;">
                    当前动作：<span style="color:{summary["action_color"]};font-weight:700;">{summary["action_text"]}</span>
                </span>
                <span style="padding:4px 10px;border-radius:999px;background:rgba(255,255,255,0.05);color:#c9d1d9;font-size:0.82rem;">
                    确认状态：{summary["confirmation_status"]}
                </span>
                <span style="padding:4px 10px;border-radius:999px;background:rgba(255,255,255,0.05);color:#c9d1d9;font-size:0.82rem;">
                    最近事件：{summary["latest_event_text"]}
                </span>
            </div>
            <div style="font-size:1.35rem;font-weight:700;color:{summary["summary_color"]};margin-bottom:12px;">
                {summary["summary_title"]}
            </div>
            <div style="color:#9aa4af;font-size:0.9rem;margin-bottom:12px;">
                事件含义：{summary["latest_event_trade_meaning"]}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;">
                <div>
                    <div style="color:#c9d1d9;font-weight:600;margin-bottom:6px;">当前原因</div>
                    <ul style="margin:0;padding-left:18px;color:#9aa4af;">{reason_html}</ul>
                </div>
                <div>
                    <div style="color:#c9d1d9;font-weight:600;margin-bottom:6px;">确认条件</div>
                    <ul style="margin:0;padding-left:18px;color:#9aa4af;">{confirmation_html}</ul>
                </div>
                <div>
                    <div style="color:#c9d1d9;font-weight:600;margin-bottom:6px;">失效条件</div>
                    <ul style="margin:0;padding-left:18px;color:#9aa4af;">{invalidation_html}</ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_debug_panel(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    debug_items = {
        "bottom_score": latest.get("bottom_score", 0),
        "top_score": latest.get("top_score", 0),
        "short_squeeze_score": latest.get("short_squeeze_score", 0),
        "long_liquidation_score": latest.get("long_liquidation_score", 0),
        "net_pos_flow_z": latest.get("net_pos_flow_z", 0),
        "long_pos_delta_z": latest.get("long_pos_delta_z", 0),
        "short_pos_delta_z": latest.get("short_pos_delta_z", 0),
        "long_pain_z": latest.get("long_pain_z", 0),
        "short_pain_z": latest.get("short_pain_z", 0),
    }
    debug_df = pd.DataFrame(
        {
            "指标": list(debug_items.keys()),
            "当前值": [pd.to_numeric(v, errors="coerce") for v in debug_items.values()],
        }
    )
    debug_df["当前值"] = pd.to_numeric(debug_df["当前值"], errors="coerce").round(3)

    with st.expander("调试指标"):
        st.dataframe(debug_df, use_container_width=True, hide_index=True)


def format_signal_table_for_mode(signal_table: pd.DataFrame, mode: str) -> pd.DataFrame:
    if signal_table.empty:
        return signal_table

    if mode == "作战模式":
        compact_columns = [
            "时间",
            "信号类型",
            "交易含义",
            "BTC价格",
            "Smart Score",
        ]
        available_columns = [col for col in compact_columns if col in signal_table.columns]
        return signal_table.loc[:, available_columns].head(8).copy()

    return signal_table.copy()


def build_status_cards(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    current_price = safe_float(latest.get("current_price", 0), 0.0)
    signal_table = build_signal_table(df)
    latest_signal = "暂无信号"
    latest_signal_hint = "当前范围内暂无触发"
    if not signal_table.empty:
        latest_signal = str(signal_table.iloc[0]["信号类型"])
        latest_signal_hint = str(signal_table.iloc[0]["时间"])

    watch_label, watch_hint, watch_color = get_current_watch_status(latest)
    long_percent = safe_float(latest.get("long_percent_smooth", 50), 50.0)
    short_percent = safe_float(latest.get("short_percent_smooth", 50), 50.0)
    smart_score = safe_float(latest.get("smart_score", 0), 0.0)
    long_pain = safe_float(latest.get("long_pain_smooth", 0), 0.0)
    short_pain = safe_float(latest.get("short_pain_smooth", 0), 0.0)

    if bool(latest.get("short_squeeze_watch", False)):
        risk_value = "空头轧空风险"
        risk_color = SQUEEZE_COLOR
        risk_hint = "反弹可能加速，避免追空"
    elif bool(latest.get("long_liquidation_watch", False)):
        risk_value = "多头踩踏风险"
        risk_color = WARNING_COLOR
        risk_hint = "下跌可能延续，谨慎抄底"
    elif bool(latest.get("bottom_watch", False)):
        risk_value = "反弹预警"
        risk_color = BULL_COLOR
        risk_hint = "等待价格确认后再跟随"
    elif bool(latest.get("top_watch", False)):
        risk_value = "回落预警"
        risk_color = BEAR_COLOR
        risk_hint = "谨防高位回落"
    else:
        risk_value = "暂无明显风险"
        risk_color = NEUTRAL_COLOR
        risk_hint = "当前没有新的强预警"

    if short_percent >= 55:
        crowding_value = f"空头拥挤 {short_percent:.1f}%"
        crowding_color = BEAR_COLOR
        crowding_hint = "空头占优，注意轧空风险"
    elif long_percent >= 55:
        crowding_value = f"多头拥挤 {long_percent:.1f}%"
        crowding_color = BULL_COLOR
        crowding_hint = "多头占优，注意踩踏风险"
    else:
        crowding_value = "多空均衡"
        crowding_color = NEUTRAL_COLOR
        crowding_hint = f"多头 {long_percent:.1f}% / 空头 {short_percent:.1f}%"

    if bool(latest.get("price_above_ma20", False)) and smart_score >= 25:
        confirm_value = "反弹确认中"
        confirm_color = BULL_COLOR
        confirm_hint = "价格与 Smart Score 同步转强"
    elif bool(latest.get("price_below_ma20", False)) and smart_score <= -25:
        confirm_value = "下跌确认中"
        confirm_color = BEAR_COLOR
        confirm_hint = "价格与 Smart Score 同步转弱"
    else:
        confirm_value = "等待确认"
        confirm_color = WARNING_COLOR
        confirm_hint = "价格或 Smart Score 尚未共振"

    cards = [
        ("市场状态", str(latest["market_state"]), color_from_market_state(str(latest["market_state"])), "当前综合偏向"),
        ("反弹/下跌风险", risk_value, risk_color, risk_hint),
        ("拥挤方向", crowding_value, crowding_color, crowding_hint),
        ("确认状态", confirm_value, confirm_color, confirm_hint),
    ]

    for column, (label, value, color, subtext) in zip(st.columns(4), cards):
        with column:
            render_status_card(label, value, color, subtext)

    with st.expander("展开查看详细状态"):
        detail_cards = [
            ("BTC 当前价格", format_price(current_price), PRICE_COLOR, "USDT"),
            ("Smart Score", f"{int(round(smart_score))}", color_from_market_state(str(latest["market_state"])), "范围 -100 ~ +100"),
            ("多头占比", format_percent_value(long_percent), BULL_COLOR, "平滑后拥挤度"),
            ("空头占比", format_percent_value(short_percent), BEAR_COLOR, "平滑后拥挤度"),
            ("多头盈亏压力", format_signed_percent(long_pain), pain_color_for_side("long", long_pain), "压力比例"),
            ("空头盈亏压力", format_signed_percent(short_pain), pain_color_for_side("short", short_pain), "压力比例"),
            ("最近信号", latest_signal, WARNING_COLOR if latest_signal == "暂无信号" else SQUEEZE_COLOR, latest_signal_hint),
            ("当前预警状态", watch_label, watch_color, watch_hint),
        ]

        detail_columns = st.columns(4)
        for idx, (label, value, color, subtext) in enumerate(detail_cards):
            with detail_columns[idx % 4]:
                render_status_card(label, value, color, subtext)


def build_main_figure(df: pd.DataFrame, options: Dict[str, object]) -> go.Figure:
    fig = go.Figure()

    mode = str(options.get("mode", "作战模式"))
    is_advanced = mode == "高级模式"
    show_flow = bool(options.get("show_flow_panel", False))
    show_pain = bool(options.get("show_pain_panel", False))
    show_crowding = bool(options.get("show_crowding_panel", False))
    show_entries = bool(options.get("show_entries", False)) and is_advanced
    show_signal_markers = bool(options.get("show_signal_markers", True))

    timestamps = df["timestamp"]
    current_price = df["current_price"]
    smart_score_series = pd.to_numeric(df["smart_score"], errors="coerce").fillna(0.0)
    price_and_score = np.column_stack([current_price.to_numpy(), smart_score_series.to_numpy()])

    if is_advanced:
        domains = {
            "price": [0.63, 1.00],
            "flow": [0.45, 0.60],
            "pain": [0.28, 0.42],
            "crowding": [0.13, 0.25],
            "score": [0.00, 0.10],
        }
        figure_height = 1120
    else:
        domains = {
            "price": [0.28, 1.00],
            "score": [0.00, 0.20],
        }
        figure_height = 720

    price_hover = (
        "时间 %{x|%Y-%m-%d %H:%M:%S}"
        "<br>BTC %{y:,.2f} USDT"
        + (
            "<extra></extra>"
            if is_advanced
            else "<br>Smart Score %{customdata[1]:.0f}<extra></extra>"
        )
    )

    ma_hover = (
        "时间 %{x|%Y-%m-%d %H:%M:%S}"
        "<br>BTC %{customdata[0]:,.2f} USDT"
        + (
            "<br>MA %{y:,.2f}<extra></extra>"
            if is_advanced
            else "<br>Smart Score %{customdata[1]:.0f}<extra></extra>"
        )
    )

    score_hover = (
        "时间 %{x|%Y-%m-%d %H:%M:%S}"
        "<br>BTC %{customdata[0]:,.2f} USDT"
        "<br>Smart Score %{y:.0f}"
        "<extra></extra>"
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=current_price,
            name="BTC 价格",
            mode="lines",
            line=dict(color=PRICE_COLOR, width=2.4),
            customdata=price_and_score if not is_advanced else current_price.to_numpy(),
            hovertemplate=price_hover,
            xaxis="x",
            yaxis="y",
        )
    )

    if options["show_ma"]:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["price_ma_20"],
                name="MA20",
                mode="lines",
                line=dict(color=MA20_COLOR, width=1.5),
                customdata=price_and_score,
                hovertemplate=ma_hover,
                xaxis="x",
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["price_ma_60"],
                name="MA60",
                mode="lines",
                line=dict(color=MA60_COLOR, width=1.3, dash="dot"),
                customdata=price_and_score,
                hovertemplate=ma_hover,
                xaxis="x",
                yaxis="y",
            )
        )

    if show_entries and has_col(df, "long_avg_entry"):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=get_series(df, "long_avg_entry", np.nan),
                name="多头成本线",
                mode="lines",
                line=dict(color=BULL_COLOR, width=1.0, dash="dash"),
                customdata=price_and_score,
                hovertemplate=(
                    "时间 %{x|%Y-%m-%d %H:%M:%S}"
                    "<br>BTC %{customdata[0]:,.2f} USDT"
                    "<br>多头成本线 %{y:,.2f}"
                    "<extra></extra>"
                ),
                xaxis="x",
                yaxis="y",
            )
        )

    if show_entries and has_col(df, "short_avg_entry"):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=get_series(df, "short_avg_entry", np.nan),
                name="空头成本线",
                mode="lines",
                line=dict(color=BEAR_COLOR, width=1.0, dash="dash"),
                customdata=price_and_score,
                hovertemplate=(
                    "时间 %{x|%Y-%m-%d %H:%M:%S}"
                    "<br>BTC %{customdata[0]:,.2f} USDT"
                    "<br>空头成本线 %{y:,.2f}"
                    "<extra></extra>"
                ),
                xaxis="x",
                yaxis="y",
            )
        )

    if show_signal_markers:
        marker_specs = [
            ("bottom_watch", "bottom_reason", "潜在反弹", "▲", BULL_COLOR, "triangle-up"),
            ("top_watch", "top_reason", "潜在回落", "▼", BEAR_COLOR, "triangle-down"),
            ("short_squeeze_watch", "short_squeeze_reason", "空头轧空", "⚡", SQUEEZE_COLOR, "diamond"),
            ("long_liquidation_watch", "long_liquidation_reason", "多头踩踏", "💥", WARNING_COLOR, "x"),
        ]
        for signal_col, reason_col, label, text_symbol, color, marker_symbol in marker_specs:
            event_col = maybe_get_event_col(df, signal_col)
            signal_rows = df[df[event_col].fillna(False)].copy()
            if signal_rows.empty:
                continue

            reason_values = signal_rows.get(
                reason_col,
                pd.Series("", index=signal_rows.index, dtype="object"),
            ).fillna("")
            smart_scores = pd.to_numeric(
                signal_rows.get(
                    "smart_score",
                    pd.Series(0.0, index=signal_rows.index, dtype="float64"),
                ),
                errors="coerce",
            ).fillna(0.0)

            fig.add_trace(
                go.Scatter(
                    x=signal_rows["timestamp"],
                    y=signal_rows["current_price"],
                    name=label,
                    mode="markers+text",
                    text=[text_symbol] * len(signal_rows),
                    textposition="top center",
                    textfont=dict(color=color, size=12),
                    marker=dict(
                        color=color,
                        size=10,
                        symbol=marker_symbol,
                        line=dict(color=BACKGROUND, width=1),
                    ),
                    customdata=np.column_stack(
                        [
                            reason_values.to_numpy(dtype=object),
                            smart_scores.to_numpy(),
                        ]
                    ),
                    hovertemplate=(
                        "时间 %{x|%Y-%m-%d %H:%M:%S}"
                        "<br>BTC %{y:,.2f} USDT"
                        f"<br>信号类型 {label}"
                        "<br>Smart Score %{customdata[1]:.0f}"
                        "<br>原因 %{customdata[0]}"
                        "<extra></extra>"
                    ),
                    xaxis="x",
                    yaxis="y",
                    showlegend=False,
                )
            )

    if is_advanced and show_flow:
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=df["net_pos_flow_smooth"],
                name="净仓位流向",
                marker=dict(
                    color=np.where(df["net_pos_flow_smooth"] >= 0, BULL_COLOR, BEAR_COLOR),
                    opacity=0.42,
                ),
                customdata=current_price.to_numpy(),
                hovertemplate=(
                    "时间 %{x|%Y-%m-%d %H:%M:%S}"
                    "<br>BTC %{customdata:,.2f} USDT"
                    "<br>净仓位流向 %{y:,.0f}"
                    "<extra></extra>"
                ),
                xaxis="x",
                yaxis="y2",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["long_pos_delta_smooth"],
                name="多头仓位变化",
                mode="lines",
                line=dict(color=BULL_COLOR, width=1.5),
                customdata=current_price.to_numpy(),
                hovertemplate=(
                    "时间 %{x|%Y-%m-%d %H:%M:%S}"
                    "<br>BTC %{customdata:,.2f} USDT"
                    "<br>多头仓位变化 %{y:,.0f}"
                    "<extra></extra>"
                ),
                xaxis="x",
                yaxis="y2",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["short_pos_delta_plot_smooth"],
                name="空头仓位变化(镜像)",
                mode="lines",
                line=dict(color=SQUEEZE_COLOR, width=1.5),
                customdata=np.stack([current_price.to_numpy(), df["short_pos_delta_raw_smooth"]], axis=-1),
                hovertemplate=(
                    "时间 %{x|%Y-%m-%d %H:%M:%S}"
                    "<br>BTC %{customdata[0]:,.2f} USDT"
                    "<br>空头仓位变化(原值) %{customdata[1]:,.0f}"
                    "<extra></extra>"
                ),
                xaxis="x",
                yaxis="y2",
            )
        )

    if is_advanced and show_pain:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["long_pain_smooth"],
                name="多头盈亏压力",
                mode="lines",
                line=dict(color=BULL_COLOR, width=1.5),
                customdata=current_price.to_numpy(),
                hovertemplate=(
                    "时间 %{x|%Y-%m-%d %H:%M:%S}"
                    "<br>BTC %{customdata:,.2f} USDT"
                    "<br>多头盈亏压力 %{y:.2%}"
                    "<extra></extra>"
                ),
                xaxis="x",
                yaxis="y3",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["short_pain_smooth"],
                name="空头盈亏压力",
                mode="lines",
                line=dict(color=BEAR_COLOR, width=1.5),
                customdata=current_price.to_numpy(),
                hovertemplate=(
                    "时间 %{x|%Y-%m-%d %H:%M:%S}"
                    "<br>BTC %{customdata:,.2f} USDT"
                    "<br>空头盈亏压力 %{y:.2%}"
                    "<extra></extra>"
                ),
                xaxis="x",
                yaxis="y3",
            )
        )

    if is_advanced and show_crowding:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["long_percent_smooth"],
                name="多头占比",
                mode="lines",
                line=dict(color=BULL_COLOR, width=1.5),
                customdata=current_price.to_numpy(),
                hovertemplate=(
                    "时间 %{x|%Y-%m-%d %H:%M:%S}"
                    "<br>BTC %{customdata:,.2f} USDT"
                    "<br>多头占比 %{y:.1f}%"
                    "<extra></extra>"
                ),
                xaxis="x",
                yaxis="y4",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df["short_percent_smooth"],
                name="空头占比",
                mode="lines",
                line=dict(color=BEAR_COLOR, width=1.5),
                customdata=current_price.to_numpy(),
                hovertemplate=(
                    "时间 %{x|%Y-%m-%d %H:%M:%S}"
                    "<br>BTC %{customdata:,.2f} USDT"
                    "<br>空头占比 %{y:.1f}%"
                    "<extra></extra>"
                ),
                xaxis="x",
                yaxis="y4",
            )
        )

    positive_score = smart_score_series.where(smart_score_series >= 0)
    negative_score = smart_score_series.where(smart_score_series <= 0)
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=positive_score,
            name="Smart Score(偏多)",
            mode="lines",
            line=dict(color=BULL_COLOR, width=2.0),
            fill="tozeroy",
            fillcolor="rgba(0, 192, 135, 0.18)",
            customdata=price_and_score,
            hovertemplate=score_hover,
            xaxis="x",
            yaxis="y5",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=negative_score,
            name="Smart Score(偏空)",
            mode="lines",
            line=dict(color=BEAR_COLOR, width=2.0),
            fill="tozeroy",
            fillcolor="rgba(246, 70, 93, 0.18)",
            customdata=price_and_score,
            hovertemplate=score_hover,
            xaxis="x",
            yaxis="y5",
        )
    )

    if is_advanced and show_flow:
        add_reference_line(fig, 0, "y2", ZERO_LINE_COLOR, dash="solid")
    if is_advanced and show_pain:
        add_reference_line(fig, 0, "y3", ZERO_LINE_COLOR, dash="solid")
        add_reference_line(fig, float(options["pain_threshold"]), "y3", WARNING_COLOR)
        add_reference_line(fig, -float(options["pain_threshold"]), "y3", WARNING_COLOR)
    if is_advanced and show_crowding:
        add_reference_line(fig, float(options["crowding_high"]), "y4", WARNING_COLOR)
        add_reference_line(fig, float(options["crowding_low"]), "y4", WARNING_COLOR)
        fig.add_shape(
            type="rect",
            xref="paper",
            x0=0,
            x1=1,
            yref="y4",
            y0=float(options["crowding_high"]),
            y1=100,
            fillcolor="rgba(240,185,11,0.10)",
            line=dict(width=0),
            layer="below",
        )
        fig.add_shape(
            type="rect",
            xref="paper",
            x0=0,
            x1=1,
            yref="y4",
            y0=0,
            y1=float(options["crowding_low"]),
            fillcolor="rgba(169,112,255,0.10)",
            line=dict(width=0),
            layer="below",
        )

    add_reference_line(fig, 60, "y5", BULL_COLOR)
    add_reference_line(fig, 25, "y5", BULL_COLOR, dash="dash")
    add_reference_line(fig, 0, "y5", ZERO_LINE_COLOR, dash="solid")
    add_reference_line(fig, -25, "y5", BEAR_COLOR, dash="dash")
    add_reference_line(fig, -60, "y5", BEAR_COLOR)

    fig.update_layout(
        height=figure_height,
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=dict(color="#C9D1D9"),
        margin=dict(l=14, r=24, t=24, b=40),
        hovermode="x unified",
        hoversubplots="axis",
        barmode="relative",
        showlegend=is_advanced,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="rgba(15,20,27,0.96)",
            bordercolor="#2B3139",
            font_size=12,
        ),
        xaxis=dict(
            domain=[0, 1],
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#AAB2BF",
            spikethickness=1,
            rangeslider_visible=False,
            tickformat="%m-%d %H:%M",
            hoverformat="%Y-%m-%d %H:%M:%S",
            color="#9AA4AF",
        ),
        yaxis=dict(
            domain=domains["price"],
            anchor="x",
            title="价格战术图",
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
        ),
        yaxis5=dict(
            domain=domains["score"],
            anchor="x",
            title="Smart Score",
            range=[-100, 100],
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
        ),
    )

    if is_advanced:
        fig.update_layout(
            yaxis2=dict(
                domain=domains["flow"],
                anchor="x",
                title="净仓位脉冲",
                showgrid=True,
                gridcolor=GRID_COLOR,
                zeroline=False,
            ),
            yaxis3=dict(
                domain=domains["pain"],
                anchor="x",
                title="盈亏压力",
                tickformat=".0%",
                showgrid=True,
                gridcolor=GRID_COLOR,
                zeroline=False,
            ),
            yaxis4=dict(
                domain=domains["crowding"],
                anchor="x",
                title="多空拥挤度",
                range=[0, 100],
                ticksuffix="%",
                showgrid=True,
                gridcolor=GRID_COLOR,
                zeroline=False,
            ),
        )

    return fig


# Shared module bindings: keep the radar UI intact while letting app.py reuse the
# extracted data/feature/signal logic used by the backtest platform.
SIGNAL_META = shared_signals.SIGNAL_META
has_col = shared_features.has_col
get_series = shared_features.get_series
safe_divide = shared_features.safe_divide
rolling_zscore = shared_features.rolling_zscore
normalize_ls_ratio = shared_features.normalize_ls_ratio
suppress_repeated_events = shared_signals.suppress_repeated_events
consecutive_true_count = shared_signals.consecutive_true_count
maybe_get_event_col = shared_signals.maybe_get_event_col
build_reason_series = shared_signals.build_reason_series
get_trade_meaning = shared_signals.get_trade_meaning
add_derived_metrics = shared_features.add_derived_metrics
add_signal_columns = shared_signals.add_signal_columns


@st.cache_data(ttl=60)
def load_shared_radar_data() -> pd.DataFrame:
    return shared_data_client.fetch_smart_money_data()


def fetch_and_process_data() -> pd.DataFrame:
    try:
        return load_shared_radar_data()
    except Exception as exc:
        st.error(f"数据拉取或清洗失败: {exc}")
        return pd.DataFrame()


def render_radar_page() -> None:
    render_console_header(
        "统一控制台 / 实时雷达",
        "📡 Smart Money 转折信号雷达",
        "信号仅用于观察与预警，不构成确定性买卖点，也不承诺预测准确率。",
    )
    st.caption("当前在统一控制台内查看实时雷达，侧栏顶部可随时切到策略回测。")

    options = build_sidebar_options()

    with st.spinner("正在扫描 Smart Money 数据..."):
        raw_df = fetch_and_process_data()

    if raw_df.empty:
        st.warning("暂无可用数据，请检查 PocketBase 地址、网络连接或数据采集状态。")
        st.stop()

    metrics_df = add_derived_metrics(raw_df, options)
    signal_df = add_signal_columns(metrics_df, options)
    display_df = add_time_range_filter(signal_df, str(options["range_label"]))

    if display_df.empty:
        st.warning("当前筛选范围内没有数据，请扩大时间范围后重试。")
        st.stop()

    tactical_summary = build_tactical_summary(display_df)
    render_tactical_summary(tactical_summary)

    build_status_cards(display_df)
    st.caption("最近信号 = 最近一次 event｜当前预警状态 = 当前 watch｜Smart Score = 当前综合偏向。三者含义不同，不应混用。")

    main_figure = build_main_figure(display_df, options)
    st.plotly_chart(main_figure, use_container_width=True, config={"displaylogo": False})

    if options.get("show_debug_indicators", False):
        render_debug_panel(display_df)

    if options["show_event_table"]:
        st.subheader("最近信号事件")
        table_df = build_signal_table(display_df)
        if table_df.empty:
            st.info("当前范围内还没有触发信号。")
        else:
            display_table = format_signal_table_for_mode(table_df, str(options.get("mode", "作战模式")))
            st.dataframe(display_table, use_container_width=True)


selected_page = render_workspace_switch(default_page="实时雷达")

if selected_page == "策略回测":
    render_backtest_page(embedded_in_console=True)
else:
    render_radar_page()
