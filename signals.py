from typing import Dict, List

import numpy as np
import pandas as pd

from features import get_series, has_col


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


def get_trade_meaning(signal_type: str) -> str:
    meaning_map = {
        "潜在反弹": "空头拥挤且获利较厚，不宜追空，等待价格确认",
        "空头轧空": "空头开始回补，反弹可能加速",
        "潜在回落": "多头拥挤且追高，谨防回落",
        "多头踩踏": "多头亏损扩大并减仓，趋势可能继续下行",
    }
    return meaning_map.get(signal_type, "结合价格确认后再决策")
