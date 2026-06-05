from __future__ import annotations

from typing import Dict

import pandas as pd

from market_structure32 import (
    add_structure32_columns,
    build_structure32_figure,
    install_structure32_streamlit_hook,
    suppress_repeated_events,
)


SIGNAL_META = {
    "structure32_watch": {
        "label": "32结构买卖点",
        "reason_col": "structure32_reason",
        "priority": 0,
    }
}


def add_signal_columns(df: pd.DataFrame, params: Dict[str, object] | None = None) -> pd.DataFrame:
    return add_structure32_columns(df, params or {}, suppress_repeated_events)


def get_trade_meaning(signal_type: str) -> str:
    meaning_map = {
        "32结构买卖点": "由价格、多空成本线、多空仓位五维组合触发；强度1-5仅表示结构强弱，需结合趋势确认",
    }
    return meaning_map.get(signal_type, "仅作为盘口结构提示，不代表确定性交易指令。")


__all__ = [
    "SIGNAL_META",
    "add_signal_columns",
    "build_structure32_figure",
    "get_trade_meaning",
    "install_structure32_streamlit_hook",
]
