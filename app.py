import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


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
    "bottom_watch": {"label": "潜在反弹", "reason_col": "bottom_reason"},
    "top_watch": {"label": "潜在回落", "reason_col": "top_reason"},
    "short_squeeze_watch": {
        "label": "空头轧空",
        "reason_col": "short_squeeze_reason",
    },
    "long_liquidation_watch": {
        "label": "多头踩踏",
        "reason_col": "long_liquidation_reason",
    },
}


st.set_page_config(
    page_title="Smart Money 转折信号雷达",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }
    .status-card {
        background: linear-gradient(180deg, #151b23 0%, #0f141b 100%);
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 92px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .status-label {
        color: #8b949e;
        font-size: 0.82rem;
        margin-bottom: 0.35rem;
    }
    .status-value {
        font-size: 1.16rem;
        font-weight: 700;
        line-height: 1.35;
    }
    .status-sub {
        color: #6b7280;
        font-size: 0.76rem;
        margin-top: 0.3rem;
    }
    [data-testid="stSidebar"] {
        background: #0f141b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📡 Smart Money 转折信号雷达")
st.caption("信号仅用于观察与预警，不构成确定性买卖点，也不承诺预测准确率。")


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

    st.sidebar.subheader("显示选项")
    show_signal_markers = st.sidebar.checkbox("显示信号标记", value=True)
    show_ma = st.sidebar.checkbox("显示均线", value=True)
    show_entries = st.sidebar.checkbox("显示平均开仓价", value=True)
    show_event_table = st.sidebar.checkbox("显示事件表", value=True)

    if crowding_low >= crowding_high:
        st.sidebar.warning("crowding_low 应小于 crowding_high，程序将按更保守的区间处理。")
        crowding_low = max(5, crowding_high - 5)

    st.sidebar.caption("这套雷达用于辅助观察潜在转折，不应视为百分百交易点。")

    return {
        "range_label": range_label,
        "smooth_window": smooth_window,
        "z_window": z_window,
        "price_move_threshold": price_move_threshold,
        "pain_threshold": pain_threshold,
        "crowding_high": crowding_high,
        "crowding_low": crowding_low,
        "flow_z_threshold": flow_z_threshold,
        "show_signal_markers": show_signal_markers,
        "show_ma": show_ma,
        "show_entries": show_entries,
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
        ratio_total = get_series(out, "ls_ratio", 0.0) + 1.0
        ratio_based_long = safe_divide(get_series(out, "ls_ratio", 0.0), ratio_total) * 100
        use_ratio = ratio_total > 0
        long_percent = pd.Series(
            np.where(
                use_ratio,
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

    price_near_or_below_ma20 = out["current_price"] <= out["price_ma_20"] * (1 + price_move_threshold)

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
        "price_weakening": (out["price_ret_5"] < 0) | price_near_or_below_ma20,
        "long_chase_z": out["long_pos_delta_z"] > flow_z_threshold,
        "short_reload_z": out["short_pos_delta_z"] > flow_z_threshold,
    }
    out["top_score"] = sum(cond.astype(int) for cond in top_checks.values())
    out["top_watch"] = out["top_score"] >= 4
    out["top_reason"] = build_reason_series(
        out.index,
        [
            (top_checks["long_crowded"], "多头拥挤"),
            (top_checks["longs_chasing"], "多头追高"),
            (top_checks["shorts_reloading"], "空头加仓"),
            (top_checks["price_weakening"], "价格走弱"),
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
    return out


def build_signal_table(df: pd.DataFrame) -> pd.DataFrame:
    events: List[Dict[str, object]] = []
    for signal_col, meta in SIGNAL_META.items():
        if signal_col not in df.columns:
            continue

        signal_rows = df[df[signal_col].fillna(False)].copy()
        if signal_rows.empty:
            continue

        reason_col = meta["reason_col"]
        for _, row in signal_rows.iterrows():
            events.append(
                {
                    "timestamp": row["timestamp"],
                    "signal_type": meta["label"],
                    "current_price": row.get("current_price", np.nan),
                    "smart_score": row.get("smart_score", np.nan),
                    "long_percent_smooth": row.get("long_percent_smooth", np.nan),
                    "short_percent_smooth": row.get("short_percent_smooth", np.nan),
                    "long_pain_smooth": row.get("long_pain_smooth", np.nan),
                    "short_pain_smooth": row.get("short_pain_smooth", np.nan),
                    "reason": row.get(reason_col, ""),
                }
            )

    if not events:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "signal_type",
                "current_price",
                "smart_score",
                "long_percent_smooth",
                "short_percent_smooth",
                "long_pain_smooth",
                "short_pain_smooth",
                "reason",
            ]
        )

    signal_df = pd.DataFrame(events).sort_values("timestamp", ascending=False).head(20).copy()
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

    return signal_df.rename(
        columns={
            "timestamp": "时间",
            "signal_type": "信号类型",
            "current_price": "BTC价格",
            "smart_score": "Smart Score",
            "long_percent_smooth": "多头占比(%)",
            "short_percent_smooth": "空头占比(%)",
            "long_pain_smooth": "多头盈亏压力(%)",
            "short_pain_smooth": "空头盈亏压力(%)",
            "reason": "原因",
        }
    )


def build_status_cards(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    signal_table = build_signal_table(df)
    latest_signal = "暂无信号"
    latest_signal_hint = "当前范围内暂无触发"
    if not signal_table.empty:
        latest_signal = str(signal_table.iloc[0]["信号类型"])
        latest_signal_hint = str(signal_table.iloc[0]["时间"])

    cards = [
        ("当前状态", str(latest["market_state"]), color_from_market_state(str(latest["market_state"])), "Smart Money 状态"),
        ("Smart Score", f"{int(round(latest['smart_score']))}", color_from_market_state(str(latest["market_state"])), "范围 -100 ~ +100"),
        ("BTC 当前价格", format_price(float(latest["current_price"])), PRICE_COLOR, "USDT"),
        ("多头占比", format_percent_value(float(latest["long_percent_smooth"])), BULL_COLOR, "平滑后拥挤度"),
        ("空头占比", format_percent_value(float(latest["short_percent_smooth"])), BEAR_COLOR, "平滑后拥挤度"),
        ("多头痛苦", format_signed_percent(float(latest["long_pain_smooth"])), pain_color_for_side("long", float(latest["long_pain_smooth"])), "盈亏压力"),
        ("空头痛苦", format_signed_percent(float(latest["short_pain_smooth"])), pain_color_for_side("short", float(latest["short_pain_smooth"])), "盈亏压力"),
        ("最近信号", latest_signal, WARNING_COLOR if latest_signal == "暂无信号" else SQUEEZE_COLOR, latest_signal_hint),
    ]

    first_row = st.columns(4)
    second_row = st.columns(4)
    all_columns = first_row + second_row

    for idx, (label, value, color, subtext) in enumerate(cards):
        with all_columns[idx]:
            render_status_card(label, value, color, subtext)


def build_main_figure(df: pd.DataFrame, options: Dict[str, object]) -> go.Figure:
    fig = go.Figure()

    domains = {
        "price": [0.63, 1.00],
        "flow": [0.45, 0.60],
        "pain": [0.28, 0.42],
        "crowding": [0.13, 0.25],
        "score": [0.00, 0.10],
    }

    timestamps = df["timestamp"]
    current_price = df["current_price"]
    custom_price = current_price.to_numpy()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=current_price,
            name="BTC 价格",
            mode="lines",
            line=dict(color=PRICE_COLOR, width=2.3),
            customdata=custom_price,
            hovertemplate=(
                "时间 %{x|%Y-%m-%d %H:%M:%S}"
                "<br>BTC %{y:,.2f} USDT"
                "<extra></extra>"
            ),
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
                line=dict(color=MA20_COLOR, width=1.4),
                customdata=custom_price,
                hovertemplate=(
                    "时间 %{x|%Y-%m-%d %H:%M:%S}"
                    "<br>BTC %{customdata:,.2f} USDT"
                    "<br>MA20 %{y:,.2f}"
                    "<extra></extra>"
                ),
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
                line=dict(color=MA60_COLOR, width=1.2, dash="dot"),
                customdata=custom_price,
                hovertemplate=(
                    "时间 %{x|%Y-%m-%d %H:%M:%S}"
                    "<br>BTC %{customdata:,.2f} USDT"
                    "<br>MA60 %{y:,.2f}"
                    "<extra></extra>"
                ),
                xaxis="x",
                yaxis="y",
            )
        )

    if options["show_entries"] and has_col(df, "long_avg_entry"):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=get_series(df, "long_avg_entry", np.nan),
                name="Long Avg Entry",
                mode="lines",
                line=dict(color=BULL_COLOR, width=1.0, dash="dash"),
                customdata=custom_price,
                hovertemplate=(
                    "时间 %{x|%Y-%m-%d %H:%M:%S}"
                    "<br>BTC %{customdata:,.2f} USDT"
                    "<br>多头成本线 %{y:,.2f}"
                    "<extra></extra>"
                ),
                xaxis="x",
                yaxis="y",
            )
        )

    if options["show_entries"] and has_col(df, "short_avg_entry"):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=get_series(df, "short_avg_entry", np.nan),
                name="Short Avg Entry",
                mode="lines",
                line=dict(color=BEAR_COLOR, width=1.0, dash="dash"),
                customdata=custom_price,
                hovertemplate=(
                    "时间 %{x|%Y-%m-%d %H:%M:%S}"
                    "<br>BTC %{customdata:,.2f} USDT"
                    "<br>空头成本线 %{y:,.2f}"
                    "<extra></extra>"
                ),
                xaxis="x",
                yaxis="y",
            )
        )

    if options["show_signal_markers"]:
        marker_specs = [
            ("bottom_watch", "bottom_reason", "潜在反弹", "▲", BULL_COLOR, "triangle-up"),
            ("top_watch", "top_reason", "潜在回落", "▼", BEAR_COLOR, "triangle-down"),
            ("short_squeeze_watch", "short_squeeze_reason", "空头轧空", "⚡", SQUEEZE_COLOR, "diamond"),
            ("long_liquidation_watch", "long_liquidation_reason", "多头踩踏", "💥", WARNING_COLOR, "x"),
        ]
        for signal_col, reason_col, label, text_symbol, color, marker_symbol in marker_specs:
            signal_rows = df[df[signal_col].fillna(False)].copy()
            if signal_rows.empty:
                continue

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
                    customdata=signal_rows[reason_col],
                    hovertemplate=(
                        "时间 %{x|%Y-%m-%d %H:%M:%S}"
                        "<br>BTC %{y:,.2f} USDT"
                        f"<br>信号 {label}"
                        "<br>原因 %{customdata}"
                        "<extra></extra>"
                    ),
                    xaxis="x",
                    yaxis="y",
                )
            )

    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=df["net_pos_flow_smooth"],
            name="净仓位流向",
            marker=dict(
                color=np.where(df["net_pos_flow_smooth"] >= 0, BULL_COLOR, BEAR_COLOR),
                opacity=0.42,
            ),
            customdata=custom_price,
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
            customdata=custom_price,
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
            customdata=np.stack([custom_price, df["short_pos_delta_raw_smooth"]], axis=-1),
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

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=df["long_pain_smooth"],
            name="多头盈亏压力",
            mode="lines",
            line=dict(color=BULL_COLOR, width=1.5),
            customdata=custom_price,
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
            customdata=custom_price,
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

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=df["long_percent_smooth"],
            name="多头占比",
            mode="lines",
            line=dict(color=BULL_COLOR, width=1.5),
            customdata=custom_price,
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
            customdata=custom_price,
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

    positive_score = df["smart_score"].where(df["smart_score"] >= 0)
    negative_score = df["smart_score"].where(df["smart_score"] <= 0)
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=positive_score,
            name="Smart Score(偏多)",
            mode="lines",
            line=dict(color=BULL_COLOR, width=2.0),
            fill="tozeroy",
            fillcolor="rgba(0, 192, 135, 0.18)",
            customdata=custom_price,
            hovertemplate=(
                "时间 %{x|%Y-%m-%d %H:%M:%S}"
                "<br>BTC %{customdata:,.2f} USDT"
                "<br>Smart Score %{y:.0f}"
                "<extra></extra>"
            ),
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
            customdata=custom_price,
            hovertemplate=(
                "时间 %{x|%Y-%m-%d %H:%M:%S}"
                "<br>BTC %{customdata:,.2f} USDT"
                "<br>Smart Score %{y:.0f}"
                "<extra></extra>"
            ),
            xaxis="x",
            yaxis="y5",
        )
    )

    add_reference_line(fig, 0, "y2", ZERO_LINE_COLOR, dash="solid")
    add_reference_line(fig, 0, "y3", ZERO_LINE_COLOR, dash="solid")
    add_reference_line(fig, float(options["pain_threshold"]), "y3", WARNING_COLOR)
    add_reference_line(fig, -float(options["pain_threshold"]), "y3", WARNING_COLOR)
    add_reference_line(fig, float(options["crowding_high"]), "y4", WARNING_COLOR)
    add_reference_line(fig, float(options["crowding_low"]), "y4", WARNING_COLOR)
    add_reference_line(fig, 60, "y5", BULL_COLOR)
    add_reference_line(fig, 25, "y5", BULL_COLOR, dash="dash")
    add_reference_line(fig, 0, "y5", ZERO_LINE_COLOR, dash="solid")
    add_reference_line(fig, -25, "y5", BEAR_COLOR, dash="dash")
    add_reference_line(fig, -60, "y5", BEAR_COLOR)

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

    fig.update_layout(
        height=1120,
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=dict(color="#C9D1D9"),
        margin=dict(l=14, r=24, t=28, b=40),
        hovermode="x unified",
        hoversubplots="axis",
        barmode="relative",
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

    return fig


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

build_status_cards(display_df)

main_figure = build_main_figure(display_df, options)
st.plotly_chart(main_figure, use_container_width=True, config={"displaylogo": False})

if options["show_event_table"]:
    st.subheader("最近信号事件")
    table_df = build_signal_table(display_df)
    if table_df.empty:
        st.info("当前范围内还没有触发信号。")
    else:
        st.dataframe(table_df, use_container_width=True)
