from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

STRUCTURE32_WINDOW = 5
PRICE_MOVE_THRESHOLD = 0.001
AVG_ENTRY_MOVE_THRESHOLD = 0.001
POSITION_MOVE_THRESHOLD = 0.005

CONFLICT_LOOKBACK = 10
CONFIRM_LOOKBACK = 5
TRADE_SIGNAL_COOLDOWN = 10
FAILED_CONFIRM_THRESHOLD = 0.003

_LAST_DF: Optional[pd.DataFrame] = None
_RENDERED = False
_ORIGINAL_PLOTLY_CHART = None


def has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()


def suppress_repeated_events(event_series: pd.Series, cooldown_bars: int) -> pd.Series:
    cooldown_bars = max(int(cooldown_bars), 0)
    event_series = event_series.fillna(False).astype(bool)
    result = pd.Series(False, index=event_series.index, dtype="bool")
    last_trigger_idx = None

    for pos, triggered in enumerate(event_series.to_numpy()):
        if not triggered:
            continue
        if last_trigger_idx is None or pos - last_trigger_idx > cooldown_bars:
            result.iloc[pos] = True
            last_trigger_idx = pos

    return result


# 价格方向, 多头均价方向, 多头仓位方向, 空头均价方向, 空头仓位方向
# stance: long = 多头机会提示, short = 空头机会提示, neutral = 观望提示
_CASE_ROWS = [
    (1, "跌", "涨", "升", "涨", "升", "多头高成本接盘 + 空头高位加仓", "short", 5, "多头接盘质量差，空头在更安全成本区继续扩张。"),
    (2, "跌", "涨", "升", "涨", "降", "多头不健康接盘 + 空头部分止盈", "short", 4, "多头逆势接盘仍脆弱，空头止盈使短线可能有反抽。"),
    (3, "跌", "涨", "升", "跌", "升", "多头接飞刀 + 空头低位追空", "short", 4, "多头接盘质量差，空头追空增加波动，需防先杀多后反抽。"),
    (4, "跌", "涨", "升", "跌", "降", "多头不健康接盘 + 高位空头止盈", "neutral", 3, "价格仍弱，但空头止盈增加，偏下跌中后段观察。"),
    (5, "跌", "涨", "降", "涨", "升", "多头撤退 + 空头增强", "short", 5, "低成本多头撤退，空头主动增强，是较干净的偏空结构。"),
    (6, "跌", "涨", "降", "涨", "降", "多空降仓但多头结构变差", "short", 3, "多空同时降杠杆，但多头成本结构恶化，下跌动能可能放缓。"),
    (7, "跌", "涨", "降", "跌", "升", "多头败退 + 空头追杀", "short", 4, "多头退出，空头低位追击，短线偏空但追空风险上升。"),
    (8, "跌", "涨", "降", "跌", "降", "多空同时去杠杆 + 多头质量下降", "neutral", 2, "合约推动力减弱，多头结构未修复，偏空衰减。"),
    (9, "跌", "跌", "升", "涨", "升", "低位多头承接 vs 高质量空头加仓", "neutral", 3, "多头低位承接与空头扩张对抗，等待价格确认。"),
    (10, "跌", "跌", "升", "涨", "降", "多头低位接货 + 空头撤退", "long", 4, "低位承接出现，空头开始撤退，若价格止跌则偏修复。"),
    (11, "跌", "跌", "升", "跌", "升", "多空低位同时加杠杆", "neutral", 3, "双方低位加杠杆，方向不明，容易高波动。"),
    (12, "跌", "跌", "升", "跌", "降", "低位多头承接 + 空头止盈", "long", 4, "多头低位接货，空头兑现利润，反弹条件改善。"),
    (13, "跌", "跌", "降", "涨", "升", "多头去杠杆 + 空头继续增强", "short", 4, "多头压力释放一部分，但空头仍主导。"),
    (14, "跌", "跌", "降", "涨", "降", "多空去杠杆 + 下跌动能衰减", "neutral", 2, "多空同时降仓，杠杆风险下降，等待新方向。"),
    (15, "跌", "跌", "降", "跌", "升", "多头撤退 + 空头低位追空", "short", 3, "短线仍偏弱，但空头成本下降后反抽风险变高。"),
    (16, "跌", "跌", "降", "跌", "降", "全市场去杠杆式下跌", "neutral", 2, "价格下跌但双方都降仓，常见于杠杆降温。"),
    (17, "涨", "涨", "升", "涨", "升", "多头追涨 + 空头高位阻击", "long", 3, "多头推动上涨，空头防守，偏多但波动较大。"),
    (18, "涨", "涨", "升", "涨", "降", "多头追涨 + 空头回补", "long", 4, "多头加仓与空头回补共振，上涨较强但注意追高。"),
    (19, "涨", "涨", "升", "跌", "升", "多头进攻 + 空头逆势硬扛", "long", 4, "空头在不利位置硬扛，若继续上涨容易形成回补。"),
    (20, "涨", "涨", "升", "跌", "降", "多头推升 + 空头逐步撤离", "long", 4, "多头进攻，空头撤退，趋势偏多但多头成本抬高。"),
    (21, "涨", "涨", "降", "涨", "升", "多头获利撤退 + 空头高位阻击", "short", 3, "上涨中低成本多头撤退，空头阻击，需警惕阶段顶部。"),
    (22, "涨", "涨", "降", "涨", "降", "上涨中多空都降仓", "neutral", 2, "上涨缺少新增杠杆确认，持续性一般。"),
    (23, "涨", "涨", "降", "跌", "升", "多头止盈 + 空头低位硬扛", "neutral", 3, "多头撤退与空头硬扛并存，分歧较大。"),
    (24, "涨", "涨", "降", "跌", "降", "多空退出式反弹", "neutral", 2, "反弹可能由降仓和回补推动，趋势质量一般。"),
    (25, "涨", "跌", "升", "涨", "升", "真实多头进场 + 空头高位阻击", "long", 5, "低成本多头主动进场，同时空头高位防守，偏多且有挤压潜力。"),
    (26, "涨", "跌", "升", "涨", "降", "真实多头进场 + 空头回补", "long", 5, "低成本多头进场与空头回补共振，是较健康的修复结构。"),
    (27, "涨", "跌", "升", "跌", "升", "健康多头进攻 + 空头逆势加杠杆", "long", 5, "低成本多头推动上涨，空头逆势加仓，挤压潜力强。"),
    (28, "涨", "跌", "升", "跌", "降", "真实多头进场 + 空头撤退", "long", 4, "多头成本改善并加仓，空头撤退，趋势修复偏稳。"),
    (29, "涨", "跌", "降", "涨", "升", "价格上涨但多头未扩张 + 空头加仓", "neutral", 2, "价格上涨但多头未主动扩张，上方阻力增加。"),
    (30, "涨", "跌", "降", "涨", "降", "价格上涨 + 多空降低杠杆", "neutral", 2, "杠杆结构改善但进攻性一般，等待接力。"),
    (31, "涨", "跌", "降", "跌", "升", "空头逆势硬扛 + 多头未追涨", "long", 3, "空头硬扛形成挤压条件，但多头主动性不足。"),
    (32, "涨", "跌", "降", "跌", "降", "去杠杆修复反弹", "neutral", 2, "上涨伴随双方降仓，属于修复反弹，持续性取决于后续新仓位。"),
]

CASE_MAP: Dict[Tuple[str, str, str, str, str], Dict[str, object]] = {
    (price_dir, long_avg_dir, long_pos_dir, short_avg_dir, short_pos_dir): {
        "case_no": case_no,
        "name": case_name,
        "stance": stance,
        "strength": strength,
        "interpretation": interpretation,
    }
    for (
        case_no,
        price_dir,
        long_avg_dir,
        long_pos_dir,
        short_avg_dir,
        short_pos_dir,
        case_name,
        stance,
        strength,
        interpretation,
    ) in _CASE_ROWS
}

RAW_ACTION_TEXT = {"long": "买点提示", "short": "卖点提示", "neutral": "观望提示"}
RAW_LEVEL_PREFIX = {"long": "买", "short": "卖", "neutral": "观望"}
TREND_TEXT = {"downtrend": "下降趋势", "uptrend": "上升趋势", "range": "震荡区间"}
CONFLICT_TEXT = {
    "no_conflict": "无明显冲突",
    "conflict_zone": "冲突区",
    "dominant_short_preserved": "近端空头主导",
    "dominant_long_preserved": "近端多头主导",
    "suppressed_by_dominant_short": "被更强空头结构压制",
    "suppressed_by_dominant_long": "被更强多头结构压制",
    "mixed_conflict": "多空混战",
    "same_direction_cooldown": "同向冷却中",
    "opposite_signal_cooldown": "反向冷却冲突",
    "replaced_by_stronger_same_direction": "已被更强同向信号替代",
}
CONFIRMATION_TEXT = {
    "confirmed_long": "已确认多头",
    "waiting_long": "等待多头确认",
    "failed_long": "多头确认失败",
    "confirmed_short": "已确认空头",
    "waiting_short": "等待空头确认",
    "failed_short": "空头确认失败",
    "range_observation": "区间观察",
    "neutral_structure": "中性结构",
    "not_applicable": "无需确认",
}

RAW_TRACE_STYLE = {
    "long": {"name": "原始多头结构（可选）", "color": "rgba(16, 185, 129, 0.35)", "symbol": "triangle-up"},
    "short": {"name": "原始空头结构（可选）", "color": "rgba(246, 70, 93, 0.35)", "symbol": "triangle-down"},
    "neutral": {"name": "原始观望结构（可选）", "color": "rgba(240, 185, 11, 0.30)", "symbol": "circle"},
}
TRADE_TRACE_STYLE = {
    "confirmed_long": {"name": "确认多头机会", "color": "#00C087", "symbol": "triangle-up"},
    "confirmed_short": {"name": "确认空头机会", "color": "#F6465D", "symbol": "triangle-down"},
    "observe": {"name": "反弹/回调/区间观察", "color": "#F0B90B", "symbol": "circle"},
    "conflict": {"name": "冲突区", "color": "#9CA3AF", "symbol": "x"},
}


def _pct_change(series: pd.Series, window: int) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    return clean.pct_change(max(int(window), 1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _direction(value: float, threshold: float, up: str, down: str) -> str:
    if pd.isna(value) or not np.isfinite(value):
        return "平"
    if float(value) > threshold:
        return up
    if float(value) < -threshold:
        return down
    return "平"


def _pct_text(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "--"
    return f"{float(numeric) * 100:+.2f}%"


def _raw_level_label(stance: object, strength: object) -> str:
    numeric = pd.to_numeric(strength, errors="coerce")
    if pd.isna(numeric):
        return ""
    level = int(max(min(float(numeric), 5), 0))
    if level <= 0:
        return ""
    return f"{RAW_LEVEL_PREFIX.get(str(stance), '观望')}{level}"


def _trade_level_label(action: str, strength: object) -> str:
    numeric = pd.to_numeric(strength, errors="coerce")
    level = 0 if pd.isna(numeric) else int(max(min(float(numeric), 5), 0))
    if action == "确认多头机会":
        return f"买{level}" if level > 0 else "买"
    if action == "确认空头机会":
        return f"卖{level}" if level > 0 else "卖"
    if action == "冲突区，观望":
        return "冲突"
    if level > 0:
        return f"观{level}"
    return "观"


def _resolve_avg_col(df: pd.DataFrame, primary: str, fallback: str) -> Optional[str]:
    if has_col(df, primary):
        return primary
    if has_col(df, fallback):
        return fallback
    return None


def _prepare_price_context(df: pd.DataFrame) -> None:
    price = pd.to_numeric(df["current_price"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["current_price"] = price
    df["price_ma_20"] = price.rolling(20, min_periods=1).mean()
    df["price_ma_60"] = price.rolling(60, min_periods=1).mean()
    df["price_ret_15"] = price.pct_change(15).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["price_ret_30"] = price.pct_change(30).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["structure32_recent_high_5"] = price.shift(1).rolling(CONFIRM_LOOKBACK, min_periods=1).max()
    df["structure32_recent_low_5"] = price.shift(1).rolling(CONFIRM_LOOKBACK, min_periods=1).min()


def _build_empty_columns(df: pd.DataFrame, reason: str) -> pd.DataFrame:
    out = df.copy()
    out["structure32_key"] = ""
    out["structure32_case_no"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["structure32_name"] = "无完整32结构"
    out["structure32_stance"] = "neutral"
    out["structure32_action"] = "无信号"
    out["structure32_strength"] = 0
    out["structure32_reason"] = reason
    out["structure32_level_label"] = ""
    out["structure32_watch"] = False
    out["structure32_watch_event"] = False
    out["structure32_interpretation"] = reason
    out["structure32_trend_state"] = "range"
    out["structure32_trade_action"] = "无交易提示"
    out["structure32_trade_strength"] = 0
    out["structure32_trade_reason"] = reason
    out["structure32_is_actionable"] = False
    out["structure32_conflict_state"] = "no_conflict"
    out["structure32_confirmation_state"] = "not_applicable"
    out["structure32_trade_event"] = False
    out["structure32_trade_direction"] = "neutral"
    out["structure32_trade_label"] = ""
    return out


def _trend_state_for_row(price: float, ma20: float, ma60: float, ret30: float) -> str:
    if ((price < ma20) and (ma20 < ma60)) or (ret30 < -0.005):
        return "downtrend"
    if ((price > ma20) and (ma20 > ma60)) or (ret30 > 0.005):
        return "uptrend"
    return "range"


def _compute_trend_state(df: pd.DataFrame) -> pd.Series:
    states: List[str] = []
    for _, row in df.iterrows():
        states.append(
            _trend_state_for_row(
                float(pd.to_numeric(row.get("current_price"), errors="coerce") or 0.0),
                float(pd.to_numeric(row.get("price_ma_20"), errors="coerce") or 0.0),
                float(pd.to_numeric(row.get("price_ma_60"), errors="coerce") or 0.0),
                float(pd.to_numeric(row.get("price_ret_30"), errors="coerce") or 0.0),
            )
        )
    return pd.Series(states, index=df.index, dtype="object")


def _long_confirmed(df: pd.DataFrame, pos: int) -> bool:
    price = pd.to_numeric(df.iloc[pos]["current_price"], errors="coerce")
    ma20 = pd.to_numeric(df.iloc[pos]["price_ma_20"], errors="coerce")
    recent_high = pd.to_numeric(df.iloc[pos]["structure32_recent_high_5"], errors="coerce")
    return bool((pd.notna(price) and pd.notna(ma20) and price > ma20) or (pd.notna(price) and pd.notna(recent_high) and price > recent_high))


def _short_confirmed(df: pd.DataFrame, pos: int) -> bool:
    price = pd.to_numeric(df.iloc[pos]["current_price"], errors="coerce")
    ma20 = pd.to_numeric(df.iloc[pos]["price_ma_20"], errors="coerce")
    recent_low = pd.to_numeric(df.iloc[pos]["structure32_recent_low_5"], errors="coerce")
    return bool((pd.notna(price) and pd.notna(ma20) and price < ma20) or (pd.notna(price) and pd.notna(recent_low) and price < recent_low))


def _evaluate_confirmation(
    df: pd.DataFrame,
    pos: int,
    stance: str,
    confirm_lookback: int,
    failed_confirm_threshold: float,
) -> Tuple[str, str]:
    if stance not in {"long", "short"}:
        return "neutral_structure", "原始结构属于观望型结构，不进入方向确认流程。"

    signal_price = pd.to_numeric(df.iloc[pos]["current_price"], errors="coerce")
    if pd.isna(signal_price):
        if stance == "long":
            return "waiting_long", "当前价格缺失，无法完成多头确认判断。"
        return "waiting_short", "当前价格缺失，无法完成空头确认判断。"

    end_pos = min(len(df) - 1, pos + max(int(confirm_lookback), 1))
    for future_pos in range(pos, end_pos + 1):
        future_price = pd.to_numeric(df.iloc[future_pos]["current_price"], errors="coerce")
        if stance == "long":
            if _long_confirmed(df, future_pos):
                return "confirmed_long", f"原始多头结构出现后 {future_pos - pos} 根内，价格重新站上 MA20 或突破最近 5 根高点。"
            if pd.notna(future_price) and future_price <= float(signal_price) * (1 - float(failed_confirm_threshold)):
                return "failed_long", f"原始多头结构出现后，价格继续跌破信号价格 {float(failed_confirm_threshold) * 100:.1f}% 以上，多头确认失败。"
        else:
            if _short_confirmed(df, future_pos):
                return "confirmed_short", f"原始空头结构出现后 {future_pos - pos} 根内，价格重新跌破 MA20 或跌破最近 5 根低点。"
            if pd.notna(future_price) and future_price >= float(signal_price) * (1 + float(failed_confirm_threshold)):
                return "failed_short", f"原始空头结构出现后，价格继续上涨超过信号价格 {float(failed_confirm_threshold) * 100:.1f}%，空头确认失败。"

    if stance == "long":
        return "waiting_long", f"原始多头结构出现后 {max(int(confirm_lookback), 1)} 根内仍未站上 MA20 或突破最近 5 根高点，继续等待确认。"
    return "waiting_short", f"原始空头结构出现后 {max(int(confirm_lookback), 1)} 根内仍未跌破 MA20 或最近 5 根低点，继续等待确认。"


def _window_event_rows(df: pd.DataFrame, pos: int, lookback: int) -> pd.DataFrame:
    start_pos = max(0, pos - lookback + 1)
    window = df.iloc[start_pos : pos + 1].copy()
    if "structure32_watch_event" not in window.columns:
        return window.iloc[0:0].copy()
    return window[window["structure32_watch_event"].fillna(False)].copy()


def _evaluate_conflict(
    df: pd.DataFrame,
    pos: int,
    stance: str,
    trend_state: str,
    conflict_lookback: int,
) -> Tuple[str, str]:
    event_rows = _window_event_rows(df, pos, conflict_lookback)
    long_rows = event_rows[event_rows["structure32_stance"] == "long"]
    short_rows = event_rows[event_rows["structure32_stance"] == "short"]

    if long_rows.empty or short_rows.empty:
        return "no_conflict", "最近 10 根内没有出现明显的多空对打结构。"

    long_strength = int(pd.to_numeric(long_rows["structure32_strength"], errors="coerce").fillna(0).max())
    short_strength = int(pd.to_numeric(short_rows["structure32_strength"], errors="coerce").fillna(0).max())
    diff = abs(long_strength - short_strength)

    if diff < 2:
        return (
            "conflict_zone",
            "最近 10 根内多空结构反复切换，且两边最高强度差小于 2，说明盘口处于噪音区或剧烈博弈区，不适合频繁反手。",
        )

    if trend_state == "downtrend" and short_strength > long_strength:
        if stance == "short":
            return "dominant_short_preserved", "最近 10 根内空头结构强度明显更高，且当前趋势偏空，优先保留空头方向。"
        return "suppressed_by_dominant_short", "最近 10 根内空头结构更强且趋势偏空，本次多头原始结构被压制，不直接升级为交易动作。"

    if trend_state == "uptrend" and long_strength > short_strength:
        if stance == "long":
            return "dominant_long_preserved", "最近 10 根内多头结构强度明显更高，且当前趋势偏多，优先保留多头方向。"
        return "suppressed_by_dominant_long", "最近 10 根内多头结构更强且趋势偏多，本次空头原始结构被压制，不直接升级为交易动作。"

    return "mixed_conflict", "最近 10 根内多空结构虽然有强弱差，但趋势并未给出单边优势，仍按冲突区处理。"


def _base_trade_decision(stance: str, trend_state: str, confirmation_state: str, raw_strength: int) -> Tuple[str, int, bool, str, str]:
    observation_strength = max(raw_strength - 2, 1)
    waiting_strength = max(raw_strength - 1, 1)

    if stance == "neutral":
        return (
            "区间观察",
            observation_strength,
            False,
            "neutral",
            "原始结构本身就是观望型组合，不给强交易提示，只保留结构观察价值。",
        )

    if trend_state == "downtrend":
        if stance == "long":
            if confirmation_state == "confirmed_long":
                return (
                    "确认多头机会",
                    raw_strength,
                    True,
                    "long",
                    "虽然大趋势仍偏空，但价格已经重新站上 MA20 或突破最近 5 根高点，原始多头结构由反弹观察升级为确认多头机会。",
                )
            return (
                "反弹观察",
                observation_strength,
                False,
                "long",
                "当前处于 downtrend，原始多头结构先降级为反弹观察，只有重新站上 MA20 或突破最近 5 根高点后才升级为确认多头机会。",
            )

        if confirmation_state == "confirmed_short":
            return (
                "确认空头机会",
                raw_strength,
                True,
                "short",
                "原始空头结构与 downtrend 一致，且价格重新跌破 MA20 或最近 5 根低点，升级为确认空头机会。",
            )
        if confirmation_state == "failed_short":
            return (
                "区间观察",
                observation_strength,
                False,
                "short",
                "原始空头结构顺势，但信号后价格反弹超过 0.3%，说明延续性不足，先退回区间观察。",
            )
        return (
            "空头机会（待确认）",
            waiting_strength,
            False,
            "short",
            "原始空头结构与 downtrend 一致，但按确认机制仍需等待价格继续跌破 MA20 或最近 5 根低点。",
        )

    if trend_state == "uptrend":
        if stance == "short":
            if confirmation_state == "confirmed_short":
                return (
                    "确认空头机会",
                    raw_strength,
                    True,
                    "short",
                    "虽然大趋势仍偏多，但价格已经跌破 MA20 或最近 5 根低点，原始空头结构由回调观察升级为确认空头机会。",
                )
            return (
                "回调观察",
                observation_strength,
                False,
                "short",
                "当前处于 uptrend，原始空头结构先降级为回调观察，只有跌破 MA20 或最近 5 根低点后才升级为确认空头机会。",
            )

        if confirmation_state == "confirmed_long":
            return (
                "确认多头机会",
                raw_strength,
                True,
                "long",
                "原始多头结构与 uptrend 一致，且价格重新站上 MA20 或最近 5 根高点，升级为确认多头机会。",
            )
        if confirmation_state == "failed_long":
            return (
                "区间观察",
                observation_strength,
                False,
                "long",
                "原始多头结构顺势，但信号后价格回落超过 0.3%，说明延续性不足，先退回区间观察。",
            )
        return (
            "多头机会（待确认）",
            waiting_strength,
            False,
            "long",
            "原始多头结构与 uptrend 一致，但按确认机制仍需等待价格继续站上 MA20 或最近 5 根高点。",
        )

    return (
        "区间观察",
        observation_strength,
        False,
        stance,
        "当前市场处于 range，原始 long / short 结构都只保留为区间观察，不直接给强交易提示。",
    )


def _trade_score(action: str, strength: int, actionable: bool) -> int:
    if action == "确认多头机会" or action == "确认空头机会":
        return 30 + int(strength) + (5 if actionable else 0)
    if action in {"多头机会（待确认）", "空头机会（待确认）"}:
        return 20 + int(strength)
    if action in {"反弹观察", "回调观察", "区间观察"}:
        return 10 + int(strength)
    if action == "冲突区，观望":
        return 1
    return 0


def _compose_raw_reason(case: Dict[str, object], key: Tuple[str, str, str, str, str]) -> str:
    return (
        f"{_raw_level_label(case['stance'], case['strength'])}｜{RAW_ACTION_TEXT.get(str(case['stance']), '观望提示')}｜"
        f"强度 {int(case['strength'])}/5，数字越大越可靠、越强｜"
        f"第{int(case['case_no'])}种：{case['name']}。"
        f"结构=价格{key[0]} + 多头均价{key[1]} + 多头仓位{key[2]} + 空头均价{key[3]} + 空头仓位{key[4]}。"
        f"原理：{case['interpretation']}"
    )


def _compose_trade_reason(row: pd.Series, trade_action: str, trade_strength: int, trend_reason: str, conflict_reason: str, confirmation_reason: str, decision_reason: str) -> str:
    return (
        f"原始结构第{int(row['structure32_case_no'])}种：{row['structure32_name']}，"
        f"原始方向={RAW_ACTION_TEXT.get(str(row['structure32_stance']), '观望提示')}，"
        f"原始强度={int(row['structure32_strength'])}/5。"
        f" 当前趋势={TREND_TEXT.get(str(row['structure32_trend_state']), row['structure32_trend_state'])}，"
        f"冲突状态={CONFLICT_TEXT.get(str(row['structure32_conflict_state']), row['structure32_conflict_state'])}，"
        f"确认状态={CONFIRMATION_TEXT.get(str(row['structure32_confirmation_state']), row['structure32_confirmation_state'])}。"
        f" 最终交易提示={trade_action}，最终强度={trade_strength}/5。"
        f" 趋势判断：{trend_reason}"
        f" 冲突过滤：{conflict_reason}"
        f" 确认机制：{confirmation_reason}"
        f" 最终解释：{decision_reason}"
    )


def _build_hover_data(rows: pd.DataFrame) -> np.ndarray:
    payload: List[List[object]] = []
    for _, row in rows.iterrows():
        payload.append(
            [
                row.get("structure32_case_no", ""),
                row.get("structure32_name", ""),
                RAW_ACTION_TEXT.get(str(row.get("structure32_stance", "neutral")), "观望提示"),
                row.get("structure32_strength", 0),
                TREND_TEXT.get(str(row.get("structure32_trend_state", "range")), "震荡区间"),
                CONFLICT_TEXT.get(str(row.get("structure32_conflict_state", "no_conflict")), "无明显冲突"),
                CONFIRMATION_TEXT.get(str(row.get("structure32_confirmation_state", "not_applicable")), "无需确认"),
                row.get("structure32_trade_action", ""),
                row.get("structure32_trade_strength", 0),
                row.get("structure32_reason", ""),
                row.get("structure32_trade_reason", ""),
                _pct_text(row.get("structure32_price_change_pct")),
                _pct_text(row.get("structure32_long_avg_change_pct")),
                _pct_text(row.get("structure32_long_pos_change_pct")),
                _pct_text(row.get("structure32_short_avg_change_pct")),
                _pct_text(row.get("structure32_short_pos_change_pct")),
            ]
        )
    return np.array(payload, dtype=object)


def _trade_trace_group(action: str) -> str:
    if action == "确认多头机会":
        return "confirmed_long"
    if action == "确认空头机会":
        return "confirmed_short"
    if action == "冲突区，观望":
        return "conflict"
    return "observe"


def add_structure32_columns(
    df: pd.DataFrame,
    params: Optional[Dict[str, object]] = None,
    suppress_func: Optional[Callable[[pd.Series, int], pd.Series]] = None,
) -> pd.DataFrame:
    out = df.copy()
    params = params or {}
    suppress = suppress_func or suppress_repeated_events

    long_avg_col = _resolve_avg_col(out, "long_avg_entry", "long_avg_price")
    short_avg_col = _resolve_avg_col(out, "short_avg_entry", "short_avg_price")

    missing: List[str] = []
    if not has_col(out, "current_price"):
        missing.append("current_price")
    if long_avg_col is None:
        missing.append("long_avg_entry")
    if not has_col(out, "long_pos_usdt"):
        missing.append("long_pos_usdt")
    if short_avg_col is None:
        missing.append("short_avg_entry")
    if not has_col(out, "short_pos_usdt"):
        missing.append("short_pos_usdt")
    if missing:
        return _build_empty_columns(out, f"32结构未启用：缺少字段 {', '.join(missing)}")

    _prepare_price_context(out)

    window = int(params.get("structure32_window", STRUCTURE32_WINDOW))
    price_threshold = float(params.get("structure32_price_move_threshold", PRICE_MOVE_THRESHOLD))
    avg_threshold = float(params.get("structure32_avg_entry_move_threshold", AVG_ENTRY_MOVE_THRESHOLD))
    pos_threshold = float(params.get("structure32_position_move_threshold", POSITION_MOVE_THRESHOLD))
    raw_event_cooldown = int(params.get("event_cooldown_bars", 5))
    conflict_lookback = int(params.get("structure32_conflict_lookback", CONFLICT_LOOKBACK))
    confirm_lookback = int(params.get("structure32_confirm_lookback", CONFIRM_LOOKBACK))
    trade_signal_cooldown = int(params.get("structure32_trade_signal_cooldown", TRADE_SIGNAL_COOLDOWN))
    failed_confirm_threshold = float(
        params.get("structure32_failed_confirm_threshold", FAILED_CONFIRM_THRESHOLD)
    )

    out["structure32_price_change_pct"] = _pct_change(out["current_price"], window)
    out["structure32_long_avg_change_pct"] = _pct_change(out[long_avg_col], window)
    out["structure32_long_pos_change_pct"] = _pct_change(out["long_pos_usdt"], window)
    out["structure32_short_avg_change_pct"] = _pct_change(out[short_avg_col], window)
    out["structure32_short_pos_change_pct"] = _pct_change(out["short_pos_usdt"], window)

    out["structure32_price_dir"] = out["structure32_price_change_pct"].map(lambda value: _direction(value, price_threshold, "涨", "跌"))
    out["structure32_long_avg_dir"] = out["structure32_long_avg_change_pct"].map(lambda value: _direction(value, avg_threshold, "涨", "跌"))
    out["structure32_long_pos_dir"] = out["structure32_long_pos_change_pct"].map(lambda value: _direction(value, pos_threshold, "升", "降"))
    out["structure32_short_avg_dir"] = out["structure32_short_avg_change_pct"].map(lambda value: _direction(value, avg_threshold, "涨", "跌"))
    out["structure32_short_pos_dir"] = out["structure32_short_pos_change_pct"].map(lambda value: _direction(value, pos_threshold, "升", "降"))

    keys = list(
        zip(
            out["structure32_price_dir"],
            out["structure32_long_avg_dir"],
            out["structure32_long_pos_dir"],
            out["structure32_short_avg_dir"],
            out["structure32_short_pos_dir"],
        )
    )
    cases = [CASE_MAP.get(tuple(key)) for key in keys]

    out["structure32_key"] = ["_".join(key) if case else "" for key, case in zip(keys, cases)]
    out["structure32_case_no"] = pd.Series(
        [case["case_no"] if case else pd.NA for case in cases],
        index=out.index,
        dtype="Int64",
    )
    out["structure32_name"] = [str(case["name"]) if case else "无明显32结构" for case in cases]
    out["structure32_stance"] = [str(case["stance"]) if case else "neutral" for case in cases]
    out["structure32_action"] = [RAW_ACTION_TEXT.get(str(case["stance"]), "观望提示") if case else "无信号" for case in cases]
    out["structure32_strength"] = [int(case["strength"]) if case else 0 for case in cases]
    out["structure32_interpretation"] = [
        str(case["interpretation"]) if case else "五个维度中至少一个维度未超过阈值。"
        for case in cases
    ]
    out["structure32_level_label"] = [
        _raw_level_label(case["stance"], case["strength"]) if case else ""
        for case in cases
    ]
    out["structure32_reason"] = [
        _compose_raw_reason(case, key) if case else "无明显32结构：价格、多空成本或仓位变化未全部超过阈值。"
        for case, key in zip(cases, keys)
    ]

    out["structure32_watch"] = pd.Series([case is not None for case in cases], index=out.index, dtype="bool")
    first_seen = out["structure32_watch"] & out["structure32_key"].ne(out["structure32_key"].shift(1))
    out["structure32_watch_event"] = suppress(first_seen, raw_event_cooldown)

    out["structure32_trend_state"] = _compute_trend_state(out)
    out["structure32_trade_action"] = "无交易提示"
    out["structure32_trade_strength"] = 0
    out["structure32_trade_reason"] = "当前没有新的最终交易提示。"
    out["structure32_is_actionable"] = False
    out["structure32_conflict_state"] = "no_conflict"
    out["structure32_confirmation_state"] = "not_applicable"
    out["structure32_trade_event"] = False
    out["structure32_trade_direction"] = "neutral"
    out["structure32_trade_label"] = ""

    raw_positions = np.flatnonzero(out["structure32_watch"].to_numpy())
    for pos in raw_positions:
        idx = out.index[pos]
        row = out.iloc[pos].copy()
        stance = str(row["structure32_stance"])
        raw_strength = int(row["structure32_strength"])
        trend_state = str(row["structure32_trend_state"])

        confirmation_state, confirmation_reason = _evaluate_confirmation(
            out,
            pos,
            stance,
            confirm_lookback=confirm_lookback,
            failed_confirm_threshold=failed_confirm_threshold,
        )
        out.at[idx, "structure32_confirmation_state"] = confirmation_state

        conflict_state, conflict_reason = _evaluate_conflict(
            out,
            pos,
            stance,
            trend_state,
            conflict_lookback=conflict_lookback,
        )
        out.at[idx, "structure32_conflict_state"] = conflict_state

        trade_action, trade_strength, is_actionable, trade_direction, decision_reason = _base_trade_decision(
            stance=stance,
            trend_state=trend_state,
            confirmation_state=confirmation_state,
            raw_strength=raw_strength,
        )

        if conflict_state == "conflict_zone":
            trade_action = "冲突区，观望"
            trade_strength = 0
            is_actionable = False
            trade_direction = "neutral"
            decision_reason = "短时间内多空结构反复切换且强度接近，最终进入冲突区观察，不适合频繁反手。"
        elif conflict_state == "suppressed_by_dominant_short":
            trade_action = "冲突区，观望"
            trade_strength = 0
            is_actionable = False
            trade_direction = "neutral"
            decision_reason = "虽然当前出现多头原始结构，但近端更强的空头结构与下降趋势一致，本次多头信号被压制。"
        elif conflict_state == "suppressed_by_dominant_long":
            trade_action = "冲突区，观望"
            trade_strength = 0
            is_actionable = False
            trade_direction = "neutral"
            decision_reason = "虽然当前出现空头原始结构，但近端更强的多头结构与上升趋势一致，本次空头信号被压制。"
        elif conflict_state == "mixed_conflict":
            trade_action = "冲突区，观望"
            trade_strength = 0
            is_actionable = False
            trade_direction = "neutral"
            decision_reason = "最近 10 根内多空结构都有显著出现，但趋势没有给出单边优势，最终按混战区处理。"

        out.at[idx, "structure32_trade_action"] = trade_action
        out.at[idx, "structure32_trade_strength"] = int(trade_strength)
        out.at[idx, "structure32_is_actionable"] = bool(is_actionable)
        out.at[idx, "structure32_trade_direction"] = trade_direction
        out.at[idx, "structure32_trade_label"] = _trade_level_label(trade_action, trade_strength)
        out.at[idx, "structure32_trade_reason"] = _compose_trade_reason(
            row=out.loc[idx],
            trade_action=trade_action,
            trade_strength=int(trade_strength),
            trend_reason="当前价格、MA20、MA60 与 30 根收益率共同决定趋势状态。",
            conflict_reason=conflict_reason,
            confirmation_reason=confirmation_reason,
            decision_reason=decision_reason,
        )

    trade_event_series = pd.Series(False, index=out.index, dtype="bool")
    directional_kept: List[int] = []
    neutral_kept: List[int] = []
    candidate_positions = np.flatnonzero(out["structure32_watch_event"].to_numpy())

    def _position_score(position: int) -> int:
        row = out.iloc[position]
        return _trade_score(
            action=str(row["structure32_trade_action"]),
            strength=int(pd.to_numeric(row["structure32_trade_strength"], errors="coerce") or 0),
            actionable=bool(row["structure32_is_actionable"]),
        )

    for pos in candidate_positions:
        idx = out.index[pos]
        action = str(out.at[idx, "structure32_trade_action"])
        if action == "无交易提示":
            continue

        raw_side = str(out.at[idx, "structure32_stance"])
        direction = str(out.at[idx, "structure32_trade_direction"])
        candidate_side = raw_side if raw_side in {"long", "short"} and direction != "neutral" else "neutral"

        if candidate_side in {"long", "short"}:
            opposite_side = "short" if candidate_side == "long" else "long"
            recent_same = [
                kept_pos
                for kept_pos in directional_kept
                if pos - kept_pos <= trade_signal_cooldown and str(out.iloc[kept_pos]["structure32_trade_direction"]) == candidate_side
            ]
            recent_opposite = [
                kept_pos
                for kept_pos in directional_kept
                if pos - kept_pos <= trade_signal_cooldown and str(out.iloc[kept_pos]["structure32_trade_direction"]) == opposite_side
            ]

            if recent_opposite:
                out.at[idx, "structure32_trade_action"] = "冲突区，观望"
                out.at[idx, "structure32_trade_strength"] = 0
                out.at[idx, "structure32_is_actionable"] = False
                out.at[idx, "structure32_trade_direction"] = "neutral"
                out.at[idx, "structure32_trade_label"] = "冲突"
                out.at[idx, "structure32_conflict_state"] = "opposite_signal_cooldown"
                out.at[idx, "structure32_trade_reason"] = (
                    str(out.at[idx, "structure32_trade_reason"])
                    + " 另外，10 根内已经出现相反方向的最终交易提示，为避免立即反手，本次信号进入冲突区观察。"
                )
                trade_event_series.at[idx] = True
                continue

            if recent_same:
                strongest_pos = max(recent_same, key=_position_score)
                if _position_score(pos) > _position_score(strongest_pos):
                    prev_idx = out.index[strongest_pos]
                    trade_event_series.at[prev_idx] = False
                    out.at[prev_idx, "structure32_is_actionable"] = False
                    out.at[prev_idx, "structure32_conflict_state"] = "replaced_by_stronger_same_direction"
                    out.at[prev_idx, "structure32_trade_reason"] = (
                        str(out.at[prev_idx, "structure32_trade_reason"])
                        + " 后续 10 根内出现了更强的同向交易提示，本次提示已被替代。"
                    )
                    directional_kept.remove(strongest_pos)
                else:
                    out.at[idx, "structure32_is_actionable"] = False
                    out.at[idx, "structure32_conflict_state"] = "same_direction_cooldown"
                    out.at[idx, "structure32_trade_reason"] = (
                        str(out.at[idx, "structure32_trade_reason"])
                        + " 同方向 10 根冷却区间内已有更强提示，本次不再单独高亮显示。"
                    )
                    continue

            trade_event_series.at[idx] = True
            directional_kept.append(pos)
            continue

        recent_neutral = [
            kept_pos
            for kept_pos in neutral_kept
            if pos - kept_pos <= trade_signal_cooldown
            and str(out.iloc[kept_pos]["structure32_trade_action"]) == action
        ]
        if recent_neutral:
            strongest_pos = max(recent_neutral, key=_position_score)
            if _position_score(pos) > _position_score(strongest_pos):
                prev_idx = out.index[strongest_pos]
                trade_event_series.at[prev_idx] = False
                out.at[prev_idx, "structure32_conflict_state"] = "replaced_by_stronger_same_direction"
                out.at[prev_idx, "structure32_trade_reason"] = (
                    str(out.at[prev_idx, "structure32_trade_reason"])
                    + " 后续 10 根内出现了更强的同类观察提示，本次提示已被替代。"
                )
                neutral_kept.remove(strongest_pos)
            else:
                out.at[idx, "structure32_trade_reason"] = (
                    str(out.at[idx, "structure32_trade_reason"])
                    + " 同类观察提示在 10 根冷却区间内已经出现，本次不再重复高亮。"
                )
                continue

        trade_event_series.at[idx] = True
        neutral_kept.append(pos)

    out["structure32_trade_event"] = trade_event_series
    return out


def build_structure32_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty or "current_price" not in df.columns:
        return fig

    plot_df = df.copy().sort_values("timestamp")
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["current_price"],
            name="BTC 独立走势",
            mode="lines",
            line=dict(color="#E6EDF3", width=2.2),
            hovertemplate="时间 %{x|%Y-%m-%d %H:%M:%S}<br>BTC %{y:,.2f} USDT<extra></extra>",
        )
    )

    raw_events = plot_df[plot_df.get("structure32_watch_event", False).fillna(False)].copy()
    for stance, style in RAW_TRACE_STYLE.items():
        rows = raw_events[raw_events["structure32_stance"] == stance]
        if rows.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=rows["timestamp"],
                y=rows["current_price"],
                name=style["name"],
                mode="markers",
                visible="legendonly",
                marker=dict(color=style["color"], size=8, symbol=style["symbol"]),
                hoverinfo="skip",
                hovertemplate=None,
            )
        )

    trade_events = plot_df[plot_df.get("structure32_trade_event", False).fillna(False)].copy()
    for group_name, style in TRADE_TRACE_STYLE.items():
        rows = trade_events[trade_events["structure32_trade_action"].map(_trade_trace_group) == group_name]
        if rows.empty:
            continue
        show_text = group_name in {"confirmed_long", "confirmed_short", "conflict"}
        fig.add_trace(
            go.Scatter(
                x=rows["timestamp"],
                y=rows["current_price"],
                name=style["name"],
                mode="markers+text" if show_text else "markers",
                text=rows["structure32_trade_label"].fillna("").astype(str).tolist() if show_text else None,
                textposition="top center" if show_text else None,
                textfont=dict(color=style["color"], size=11),
                marker=dict(
                    color=style["color"],
                    size=12 if group_name.startswith("confirmed") else 10,
                    symbol=style["symbol"],
                    line=dict(color="#0B0E11", width=1),
                ),
                hoverinfo="skip",
                hovertemplate=None,
            )
        )

    fig.update_layout(
        height=520,
        paper_bgcolor="#0B0E11",
        plot_bgcolor="#0B0E11",
        font=dict(color="#C9D1D9"),
        margin=dict(l=14, r=24, t=26, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickformat="%m-%d %H:%M",
            hoverformat="%Y-%m-%d %H:%M:%S",
        ),
        yaxis=dict(
            title="BTC 独立走势 + 32结构交易过滤提示",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
    )
    return fig


def get_structure32_status_messages(plot_df: pd.DataFrame) -> Tuple[str, str, str]:
    if plot_df.empty:
        return (
            "最近一次原始32结构：当前范围内还没有触发 32 结构。",
            "最近一次最终交易提示：当前范围内还没有通过过滤器的交易提示。",
            "当前最新状态：暂无可用于判断的结构数据。",
        )

    raw_event_rows = plot_df[plot_df.get("structure32_watch_event", False).fillna(False)].copy()
    if not raw_event_rows.empty:
        latest_raw = raw_event_rows.iloc[-1]
        recent_raw_message = "最近一次原始32结构：" + str(
            latest_raw.get("structure32_reason", "当前范围内还没有触发原始 32 结构。")
        )
    else:
        recent_raw_message = "最近一次原始32结构：当前范围内还没有触发 32 结构。"

    trade_event_rows = plot_df[plot_df.get("structure32_trade_event", False).fillna(False)].copy()
    if not trade_event_rows.empty:
        latest_trade = trade_event_rows.iloc[-1]
        recent_trade_message = "最近一次最终交易提示：" + str(
            latest_trade.get("structure32_trade_reason", "当前范围内还没有通过过滤器的交易提示。")
        )
    else:
        recent_trade_message = "最近一次最终交易提示：当前范围内还没有通过趋势、冲突、确认和冷却过滤的交易提示。"

    latest = plot_df.iloc[-1]
    latest_trend = TREND_TEXT.get(str(latest.get("structure32_trend_state", "range")), "震荡区间")
    if bool(latest.get("structure32_watch", False)):
        current_status_message = (
            "当前最新状态："
            + str(latest.get("structure32_reason", "当前没有完整触发32结构。"))
            + f"｜最终判定={latest.get('structure32_trade_action', '无交易提示')}"
            + f"｜趋势={latest_trend}"
            + f"｜确认={CONFIRMATION_TEXT.get(str(latest.get('structure32_confirmation_state', 'not_applicable')), '无需确认')}"
            + f"｜冲突={CONFLICT_TEXT.get(str(latest.get('structure32_conflict_state', 'no_conflict')), '无明显冲突')}"
        )
    else:
        current_status_message = f"当前最新状态：暂未形成完整32结构｜趋势={latest_trend}"

    return recent_raw_message, recent_trade_message, current_status_message


def render_structure32_section(
    st_module,
    plot_df: pd.DataFrame,
    chart_renderer: Callable[..., object],
    *,
    chart_key: Optional[str] = None,
) -> None:
    recent_raw_message, recent_trade_message, current_status_message = get_structure32_status_messages(plot_df)

    chart_col, message_col = st_module.columns([1, 1], gap="large")

    with chart_col:
        chart_kwargs = {
            "use_container_width": True,
            "config": {"displaylogo": False},
        }
        if chart_key is not None:
            chart_kwargs["key"] = chart_key
        chart_renderer(build_structure32_figure(plot_df), **chart_kwargs)

    with message_col:
        st_module.markdown("**结构消息**")
        st_module.caption("左侧展示图表；右侧集中展示最近结构、最终交易提示和当前状态。")

        raw_box = st_module.container(border=True)
        raw_box.markdown("**最近一次原始32结构**")
        raw_box.write(recent_raw_message)

        trade_box = st_module.container(border=True)
        trade_box.markdown("**最近一次最终交易提示**")
        trade_box.write(recent_trade_message)

        current_box = st_module.container(border=True)
        current_box.markdown("**当前最新状态**")
        current_box.write(current_status_message)


def install_structure32_streamlit_hook(df: pd.DataFrame) -> None:
    global _LAST_DF, _RENDERED, _ORIGINAL_PLOTLY_CHART

    _LAST_DF = df.copy()
    _RENDERED = False
    if _ORIGINAL_PLOTLY_CHART is not None:
        return

    try:
        import streamlit as st
    except Exception:
        return

    _ORIGINAL_PLOTLY_CHART = st.plotly_chart

    def _patched_plotly_chart(fig, *args, **kwargs):
        global _RENDERED

        result = _ORIGINAL_PLOTLY_CHART(fig, *args, **kwargs)
        if _RENDERED or _LAST_DF is None or _LAST_DF.empty or not getattr(fig, "data", None):
            return result

        yaxis_title = str(getattr(getattr(getattr(fig.layout, "yaxis", None), "title", None), "text", ""))
        if yaxis_title != "价格战术图":
            return result

        first_trace = fig.data[0]
        plot_df = _LAST_DF.copy()
        try:
            x_values = pd.to_datetime(pd.Series(first_trace.x), errors="coerce").dropna()
            if not x_values.empty and "timestamp" in plot_df.columns:
                plot_df = plot_df[
                    (plot_df["timestamp"] >= x_values.min()) & (plot_df["timestamp"] <= x_values.max())
                ].copy()
        except Exception:
            pass

        if plot_df.empty:
            return result

        _RENDERED = True
        st.subheader("32种盘口结构提示图")
        st.caption(
            "保留原始 32 结构作为盘口观察层；图上默认高亮的是经过趋势、冲突、确认、冷却过滤后的最终交易提示。"
            " 原始结构点可在图例中手动打开查看。"
        )
        render_structure32_section(
            st,
            plot_df,
            _ORIGINAL_PLOTLY_CHART,
        )
        return result

    st.plotly_chart = _patched_plotly_chart
