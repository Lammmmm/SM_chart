from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from signals import maybe_get_event_col


STRATEGY_NAME = "Smart Money Reversal Confirmation Strategy V1"
DIRECTION_OPTIONS = ("只做多", "只做空", "多空都做")
EXIT_MODE_OPTIONS = ("固定止盈止损", "反向信号", "混合退出")


@dataclass(frozen=True)
class StrategyConfig:
    trading_direction: str = "多空都做"
    confirmation_window: int = 5
    confirmation_required_count: int = 2
    trade_cooldown_bars: int = 10
    exit_mode: str = "混合退出"
    stop_loss_pct: float = 0.006
    take_profit_pct: float = 0.009


def build_strategy_signals(df: pd.DataFrame, params: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    required_count = int(params.get("confirmation_required_count", 2))

    bottom_event_col = maybe_get_event_col(out, "bottom_watch")
    short_squeeze_event_col = maybe_get_event_col(out, "short_squeeze_watch")
    top_event_col = maybe_get_event_col(out, "top_watch")
    long_liquidation_event_col = maybe_get_event_col(out, "long_liquidation_watch")

    bottom_event = out[bottom_event_col].fillna(False).astype(bool)
    short_squeeze_event = out[short_squeeze_event_col].fillna(False).astype(bool)
    top_event = out[top_event_col].fillna(False).astype(bool)
    long_liquidation_event = out[long_liquidation_event_col].fillna(False).astype(bool)

    out["long_setup_event"] = bottom_event | short_squeeze_event
    out["short_setup_event"] = top_event | long_liquidation_event

    out["long_setup_source"] = np.select(
        [short_squeeze_event, bottom_event],
        [short_squeeze_event_col, bottom_event_col],
        default="",
    )
    out["short_setup_source"] = np.select(
        [long_liquidation_event, top_event],
        [long_liquidation_event_col, top_event_col],
        default="",
    )

    out["long_setup_label"] = np.select(
        [short_squeeze_event, bottom_event],
        ["short_squeeze", "bottom_watch"],
        default="",
    )
    out["short_setup_label"] = np.select(
        [long_liquidation_event, top_event],
        ["long_liquidation", "top_watch"],
        default="",
    )

    out["long_setup_priority"] = np.select(
        [short_squeeze_event, bottom_event],
        [1, 3],
        default=99,
    )
    out["short_setup_priority"] = np.select(
        [long_liquidation_event, top_event],
        [2, 4],
        default=99,
    )

    out["long_confirm_count"] = (
        (out["current_price"] > out["price_ma_20"]).astype(int)
        + (out["smart_score"] > -25).astype(int)
        + (out["net_pos_flow_smooth"] > 0).astype(int)
    )
    out["short_confirm_count"] = (
        (out["current_price"] < out["price_ma_20"]).astype(int)
        + (out["smart_score"] < 25).astype(int)
        + (out["net_pos_flow_smooth"] < 0).astype(int)
    )
    out["long_confirmation_pass"] = out["long_confirm_count"] >= required_count
    out["short_confirmation_pass"] = out["short_confirm_count"] >= required_count

    strong_downtrend = (out["smart_score"] <= -60) & (out["current_price"] < out["price_ma_20"])
    strong_uptrend = (out["smart_score"] >= 60) & (out["current_price"] > out["price_ma_20"])
    out["long_entry_blocked"] = bottom_event & (~short_squeeze_event) & strong_downtrend
    out["short_entry_blocked"] = top_event & (~long_liquidation_event) & strong_uptrend

    out["long_reverse_exit_signal"] = (
        top_event
        | long_liquidation_event
        | (out["smart_score"] < -60)
        | (out["current_price"] < out["price_ma_20"])
    )
    out["short_reverse_exit_signal"] = (
        bottom_event
        | short_squeeze_event
        | (out["smart_score"] > 60)
        | (out["current_price"] > out["price_ma_20"])
    )

    out["strategy_signal"] = np.select(
        [
            short_squeeze_event,
            long_liquidation_event,
            bottom_event,
            top_event,
            out["long_confirmation_pass"],
            out["short_confirmation_pass"],
        ],
        [
            "short_squeeze_event",
            "long_liquidation_event",
            "bottom_event",
            "top_event",
            "long_confirmation_pass",
            "short_confirmation_pass",
        ],
        default="flat",
    )
    return out
