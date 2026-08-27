from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


RAW_CACHE_PATH = Path("smart_money_cache.json")
CONFIG_PATH = Path("smart_money_analysis_config.json")
DERIVED_DIR = Path("data/derived")
PROCESSED_PATH = DERIVED_DIR / "smart_money_processed.parquet"
EVENTS_PATH = DERIVED_DIR / "smart_money_events.parquet"
COHORTS_PATH = DERIVED_DIR / "smart_money_daily_cohorts.parquet"
BACKTEST_PATH = DERIVED_DIR / "smart_money_backtest.csv"
SUMMARY_PATH = DERIVED_DIR / "smart_money_summary.json"
CHINESE_PROCESSED_PATH = DERIVED_DIR / "聪明钱逐条分析_中文.parquet"
CHINESE_EVENTS_PATH = DERIVED_DIR / "聪明钱事件表_中文.parquet"
CHINESE_COHORTS_PATH = DERIVED_DIR / "聪明钱批次表_中文.parquet"
CHINESE_BACKTEST_PATH = DERIVED_DIR / "聪明钱回测结果_中文.csv"
CHINESE_REPORT_PATH = DERIVED_DIR / "聪明钱分析报告_中文.md"

STATE_ZH = {
    "UNAVAILABLE": "不可用",
    "REFRESHING": "名单刷新期",
    "STABILIZING": "稳定观察期",
    "BASELINE_BUILDING": "基准构建期",
    "ACTIVE_COHORT": "批次有效期",
}
SIDE_ZH = {"LONG": "多头", "SHORT": "空头"}
RESPONSE_ZH = {"ADD": "加仓", "HOLD": "持仓", "REDUCE": "减仓"}
EVENT_TYPE_ZH = {
    "LONG_NEW_LOSS_ADD": "多头新亏损后加仓",
    "LONG_NEW_LOSS_HOLD": "多头新亏损后持仓",
    "LONG_NEW_LOSS_REDUCE": "多头新亏损后减仓",
    "SHORT_NEW_LOSS_ADD": "空头新亏损后加仓",
    "SHORT_NEW_LOSS_HOLD": "空头新亏损后持仓",
    "SHORT_NEW_LOSS_REDUCE": "空头新亏损后减仓",
    "LONG_RECOVERY": "多头恢复盈利",
    "SHORT_RECOVERY": "空头恢复盈利",
}
QUALITY_ZH = {
    "OK": "正常",
    "MISSING": "字段缺失",
    "TIMESTAMP_OUT_OF_ORDER": "时间戳乱序",
    "TIMESTAMP_DUPLICATE": "时间戳重复",
    "INVALID_LONG_POSITION": "多头仓位异常为零",
    "INVALID_SHORT_POSITION": "空头仓位异常为零",
    "TRADER_COUNT_ZERO_ANOMALY": "交易者人数异常为零",
    "TRADER_COUNT_INCONSISTENT": "交易者总人数不一致",
    "POSITION_TOTAL_INCONSISTENT": "总仓位金额不一致",
    "COHORT_REFRESH": "名单批次刷新",
}
STATUS_ZH = {
    "OPEN": "进行中",
    "RECOVERED": "已恢复盈利",
    "COHORT_REFRESH": "因名单刷新而结束",
}

COLUMN_ZH = {
    "collectionId": "数据集合编号", "collectionName": "数据集合名称",
    "created": "创建时间", "updated": "更新时间", "id": "原始记录编号",
    "symbol": "交易品种", "timestamp": "时间戳", "current_price": "比特币当前价格",
    "funding_rate": "资金费率", "long_avg_price": "多头平均开仓价",
    "short_avg_price": "空头平均开仓价", "long_pnl_ratio": "币安原始多头盈亏比",
    "short_pnl_ratio": "币安原始空头盈亏比", "long_pos_usdt": "多头仓位金额",
    "short_pos_usdt": "空头仓位金额", "total_pos_usdt": "总仓位金额",
    "long_traders": "多头交易者人数", "short_traders": "空头交易者人数",
    "total_traders": "交易者总人数", "long_unrealized_pnl": "多头未实现盈亏",
    "short_unrealized_pnl": "空头未实现盈亏", "ls_ratio": "多空人数比",
    "data_quality_flag": "数据质量标记", "cohort_id": "名单批次编号",
    "analysis_state": "分析状态", "cohort_refresh_detected": "是否检测到名单刷新",
    "cohort_refresh_score": "名单刷新评分", "is_refresh_window": "是否处于名单刷新期",
    "is_stabilization_window": "是否处于稳定观察期",
    "minutes_since_cohort_start": "批次开始后分钟数",
    "cohort_refresh_start": "名单刷新开始时间", "cohort_refresh_end": "名单刷新结束时间",
    "cohort_stable_at": "名单稳定确认时间", "cohort_baseline_start": "基准构建开始时间",
    "cohort_baseline_end": "基准构建结束时间", "signal_eligible": "是否允许产生信号",
    "out_of_sample": "是否为样本外数据", "baseline_price": "批次基准价格",
    "baseline_long_pos": "批次基准多头仓位", "baseline_short_pos": "批次基准空头仓位",
    "baseline_long_avg_price": "批次基准多头平均开仓价",
    "baseline_short_avg_price": "批次基准空头平均开仓价",
    "baseline_long_traders": "批次基准多头人数",
    "baseline_short_traders": "批次基准空头人数",
    "baseline_long_unrealized_pnl": "批次基准多头未实现盈亏",
    "baseline_short_unrealized_pnl": "批次基准空头未实现盈亏",
    "baseline_ls_ratio": "批次基准多空人数比",
    "gross_position": "多空总仓位", "net_exposure": "净风险敞口",
    "net_exposure_share": "净风险敞口占比", "cohort_net_flow": "批次净风险流",
    "long_position_change_pct": "批次内多头仓位变化率",
    "short_position_change_pct": "批次内空头仓位变化率",
    "avg_long_position": "每位多头平均仓位", "avg_short_position": "每位空头平均仓位",
    "avg_position_ratio": "多空人均仓位比", "long_directional_return": "多头方向收益",
    "short_directional_return": "空头方向收益", "long_loss_depth": "多头亏损深度",
    "short_loss_depth": "空头亏损深度", "long_avg_entry_change": "批次内多头平均开仓价变化",
    "short_avg_entry_change": "批次内空头平均开仓价变化",
    "ls_ratio_change_within_cohort": "批次内多空人数比变化",
    "long_existing_loss_at_cohort_start": "多头是否在批次开始时已经亏损",
    "short_existing_loss_at_cohort_start": "空头是否在批次开始时已经亏损",
    "long_new_loss_event": "是否触发多头新亏损事件",
    "short_new_loss_event": "是否触发空头新亏损事件",
    "long_recovery_event": "是否触发多头恢复盈利",
    "short_recovery_event": "是否触发空头恢复盈利",
    "long_loss_event_id": "多头亏损事件编号", "short_loss_event_id": "空头亏损事件编号",
    "long_loss_start_timestamp": "多头亏损开始时间",
    "short_loss_start_timestamp": "空头亏损开始时间",
    "long_loss_start_price": "多头亏损开始价格", "short_loss_start_price": "空头亏损开始价格",
    "long_loss_start_position": "多头亏损开始仓位",
    "short_loss_start_position": "空头亏损开始仓位",
    "long_loss_start_avg_entry": "多头亏损开始平均开仓价",
    "short_loss_start_avg_entry": "空头亏损开始平均开仓价",
    "long_loss_start_net_exposure": "多头亏损开始时净风险敞口",
    "short_loss_start_net_exposure": "空头亏损开始时净风险敞口",
    "long_loss_duration_minutes": "多头亏损持续分钟数",
    "short_loss_duration_minutes": "空头亏损持续分钟数",
    "long_position_change_since_loss": "多头亏损后仓位变化率",
    "short_position_change_since_loss": "空头亏损后仓位变化率",
    "long_avg_entry_change_since_loss": "多头亏损后平均开仓价变化",
    "short_avg_entry_change_since_loss": "空头亏损后平均开仓价变化",
    "long_loss_response": "多头亏损后操作", "short_loss_response": "空头亏损后操作",
    "event_markers": "事件标记", "event_id": "事件编号", "side": "方向",
    "event_type": "事件类型", "loss_start_timestamp": "亏损开始时间",
    "loss_end_timestamp": "亏损结束时间", "loss_start_price": "亏损开始价格",
    "loss_end_price": "亏损结束价格", "loss_start_position": "亏损开始仓位",
    "max_position": "事件期间最大仓位", "min_position": "事件期间最小仓位",
    "loss_start_avg_entry": "亏损开始平均开仓价",
    "loss_end_avg_entry": "亏损结束平均开仓价",
    "loss_start_net_exposure": "亏损开始时净风险敞口",
    "max_loss_depth": "最大亏损深度", "max_loss_duration": "最长亏损分钟数",
    "max_position_add_pct": "最大加仓比例", "max_position_reduce_pct": "最大减仓比例",
    "final_response": "最终操作", "recovered": "是否恢复盈利",
    "recovery_event_type": "恢复盈利事件类型", "event_status": "事件状态",
    "cohort_net_flow_at_start": "事件开始时批次净风险流",
    "cohort_net_flow_at_end": "事件结束时批次净风险流",
    "loss_depth_bucket": "亏损深度分组", "loss_duration_bucket": "亏损时长分组",
    "position_response_bucket": "仓位操作分组", "cohort_start": "批次开始时间",
    "cohort_end": "批次结束时间", "active_at": "批次信号启用时间",
    "refresh_score": "刷新评分", "record_count": "记录数",
    "signal_eligible_records": "可产生信号的记录数", "final_state": "最终状态",
    "dataset": "数据范围", "segment": "分组维度", "segment_value": "分组值",
    "horizon": "未来周期", "sample_count": "样本数",
    "mean_forward_return": "未来收益均值", "median_forward_return": "未来收益中位数",
    "win_rate": "正收益比例", "p25": "第25百分位", "p75": "第75百分位",
    "bootstrap_ci_low": "自助法95%置信区间下限",
    "bootstrap_ci_high": "自助法95%置信区间上限",
}
for _hours in (1, 2, 4, 6, 12, 24):
    COLUMN_ZH[f"btc_return_{_hours}h"] = f"比特币过去{_hours}小时收益"
    COLUMN_ZH[f"cohort_net_flow_change_{_hours}h"] = f"批次净风险流过去{_hours}小时变化"
    COLUMN_ZH[f"divergence_{_hours}h"] = f"过去{_hours}小时价格资金流背离"
    COLUMN_ZH[f"forward_return_{_hours}h"] = f"事件后{_hours}小时比特币收益"

NUMERIC_COLUMNS = [
    "current_price", "funding_rate", "long_avg_price", "short_avg_price",
    "long_pos_usdt", "short_pos_usdt", "total_pos_usdt", "long_traders",
    "short_traders", "total_traders", "long_unrealized_pnl",
    "short_unrealized_pnl", "long_pnl_ratio", "short_pnl_ratio", "ls_ratio",
]
CORE_REQUIRED_COLUMNS = [
    "current_price", "long_avg_price", "short_avg_price", "long_pos_usdt",
    "short_pos_usdt", "long_traders", "short_traders", "total_traders",
]
DEFAULT_CONFIG: dict[str, Any] = {
    "model_freeze_date": "2026-08-27T01:42:59Z",
    "cohort": {
        "total_trader_change_threshold": 0.08,
        "side_trader_change_threshold": 0.12,
        "position_change_confirmation": 0.25,
        "stabilization_minutes": 30,
        "stable_total_trader_change": 0.02,
        "stable_side_trader_change": 0.03,
        "baseline_minutes": 30,
    },
    "loss": {
        "clean_baseline_threshold": -0.002,
        "existing_loss_threshold": -0.005,
        "loss_entry_threshold": -0.005,
        "recovery_threshold": 0.0,
        "add_threshold": 0.03,
        "reduce_threshold": -0.03,
        "cooldown_minutes": 60,
        "depth_bucket_edges": [0.005, 0.01, 0.02, 0.03, 0.05],
    },
    "divergence": {
        "windows_hours": [1, 2, 4, 6],
        "flat_price_threshold": 0.01,
        "flow_threshold": 0.05,
    },
    "quality": {
        "position_total_tolerance": 0.02,
        "trader_total_tolerance": 1.0,
    },
    "backtest": {
        "forward_hours": [1, 4, 12, 24],
        "bootstrap_samples": 2000,
        "random_seed": 20260827,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_analysis_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return deepcopy(DEFAULT_CONFIG)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    section = payload.get("smart_money_analysis", {})
    if not isinstance(section, dict):
        raise ValueError(f"{path}: smart_money_analysis 必须是 object")
    config = _deep_merge(DEFAULT_CONFIG, section)
    if pd.isna(pd.to_datetime(config["model_freeze_date"], utc=True, errors="coerce")):
        raise ValueError("model_freeze_date 必须是带时区的有效时间")
    return config


def load_raw_records(path: Path = RAW_CACHE_PATH) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload, {"record_count": len(payload)}
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{path} 必须是 records 数组或包含 records 的 object")
    return payload["records"], payload


def _safe_abs_change(current: Any, previous: Any) -> float:
    if pd.isna(current) or pd.isna(previous) or float(previous) <= 0:
        return math.nan
    return abs(float(current) / float(previous) - 1.0)


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0:
        return math.nan
    return float(numerator) / float(denominator)


def _quality_flags(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[list[str], pd.Series]:
    tolerance = config["quality"]
    duplicate = frame["timestamp"].duplicated(keep=False) & frame["timestamp"].notna()
    all_flags: list[str] = []
    signal_ok: list[bool] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        flags: list[str] = []
        if pd.isna(row["timestamp"]) or any(pd.isna(row.get(column)) for column in CORE_REQUIRED_COLUMNS):
            flags.append("MISSING")
        if bool(row.get("_timestamp_out_of_order", False)):
            flags.append("TIMESTAMP_OUT_OF_ORDER")
        if bool(duplicate.iloc[position]):
            flags.append("TIMESTAMP_DUPLICATE")
        long_invalid = (
            not pd.isna(row["long_pos_usdt"]) and row["long_pos_usdt"] == 0
            and ((not pd.isna(row["long_traders"]) and row["long_traders"] > 0)
                 or (not pd.isna(row["long_unrealized_pnl"]) and row["long_unrealized_pnl"] != 0))
        )
        short_invalid = (
            not pd.isna(row["short_pos_usdt"]) and row["short_pos_usdt"] == 0
            and ((not pd.isna(row["short_traders"]) and row["short_traders"] > 0)
                 or (not pd.isna(row["short_unrealized_pnl"]) and row["short_unrealized_pnl"] != 0))
        )
        if long_invalid:
            flags.append("INVALID_LONG_POSITION")
        if short_invalid:
            flags.append("INVALID_SHORT_POSITION")
        trader_values = [row[name] for name in ("total_traders", "long_traders", "short_traders")]
        if all(not pd.isna(value) for value in trader_values):
            total, long_count, short_count = map(float, trader_values)
            if total <= 0 or long_count <= 0 or short_count <= 0:
                flags.append("TRADER_COUNT_ZERO_ANOMALY")
            elif abs(total - long_count - short_count) > float(tolerance["trader_total_tolerance"]):
                flags.append("TRADER_COUNT_INCONSISTENT")
        position_values = [row[name] for name in ("total_pos_usdt", "long_pos_usdt", "short_pos_usdt")]
        if all(not pd.isna(value) for value in position_values):
            total_pos, long_pos, short_pos = map(float, position_values)
            denominator = max(abs(total_pos), abs(long_pos + short_pos), 1.0)
            if abs(total_pos - long_pos - short_pos) / denominator > float(tolerance["position_total_tolerance"]):
                flags.append("POSITION_TOTAL_INCONSISTENT")
        blocking = {
            "MISSING", "TIMESTAMP_DUPLICATE", "INVALID_LONG_POSITION",
            "INVALID_SHORT_POSITION", "TRADER_COUNT_ZERO_ANOMALY",
            "TRADER_COUNT_INCONSISTENT", "POSITION_TOTAL_INCONSISTENT",
        }
        all_flags.append("|".join(flags) if flags else "OK")
        signal_ok.append(not any(flag in blocking for flag in flags))
    return all_flags, pd.Series(signal_ok, index=frame.index, dtype=bool)


def prepare_raw_frame(records: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    for column in NUMERIC_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "timestamp" not in frame:
        frame["timestamp"] = pd.NaT
    original_timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["_source_order"] = np.arange(len(frame))
    frame["_timestamp_out_of_order"] = original_timestamp.diff().dt.total_seconds().lt(0).fillna(False)
    frame["timestamp"] = original_timestamp
    frame = frame.sort_values(["timestamp", "_source_order"], kind="stable", na_position="last").reset_index(drop=True)
    frame["data_quality_flag"], frame["_quality_signal_ok"] = _quality_flags(frame, config)
    return frame


BASELINE_COLUMNS = {
    "price": "current_price",
    "long_pos": "long_pos_usdt",
    "short_pos": "short_pos_usdt",
    "long_avg_price": "long_avg_price",
    "short_avg_price": "short_avg_price",
    "long_traders": "long_traders",
    "short_traders": "short_traders",
    "long_unrealized_pnl": "long_unrealized_pnl",
    "short_unrealized_pnl": "short_unrealized_pnl",
    "ls_ratio": "ls_ratio",
}


def _build_baseline(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    if not rows:
        return None
    data = pd.DataFrame(rows)
    baseline = {
        target: float(pd.to_numeric(data[source], errors="coerce").median())
        for target, source in BASELINE_COLUMNS.items()
    }
    if any(pd.isna(baseline[name]) or baseline[name] <= 0
           for name in ("price", "long_pos", "short_pos", "long_avg_price", "short_avg_price")):
        return None
    return baseline


def _blank_features() -> dict[str, Any]:
    numeric = [
        "minutes_since_cohort_start", "baseline_price", "baseline_long_pos",
        "baseline_short_pos", "baseline_long_avg_price", "baseline_short_avg_price",
        "baseline_long_traders", "baseline_short_traders",
        "baseline_long_unrealized_pnl", "baseline_short_unrealized_pnl",
        "gross_position", "net_exposure", "net_exposure_share", "cohort_net_flow",
        "long_position_change_pct", "short_position_change_pct",
        "avg_long_position", "avg_short_position", "avg_position_ratio",
        "long_directional_return", "short_directional_return",
        "long_loss_depth", "short_loss_depth", "long_avg_entry_change",
        "short_avg_entry_change", "ls_ratio_change_within_cohort",
    ]
    result = {name: math.nan for name in numeric}
    result.update({
        "cohort_id": None, "analysis_state": "UNAVAILABLE",
        "cohort_refresh_detected": False, "cohort_refresh_score": 0,
        "is_refresh_window": False, "is_stabilization_window": False,
        "cohort_refresh_start": pd.NaT, "cohort_refresh_end": pd.NaT,
        "cohort_stable_at": pd.NaT, "cohort_baseline_start": pd.NaT,
        "cohort_baseline_end": pd.NaT, "signal_eligible": False,
        "out_of_sample": False, "long_existing_loss_at_cohort_start": False,
        "short_existing_loss_at_cohort_start": False,
    })
    return result


def reconstruct_cohorts(raw_frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw_frame.empty:
        return raw_frame.copy(), pd.DataFrame()
    cfg = config["cohort"]
    freeze = pd.to_datetime(config["model_freeze_date"], utc=True)
    feature_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    state = "UNAVAILABLE"
    cohort_id: str | None = None
    sequence = 0
    cohort_start = refresh_start = refresh_end = stable_at = stable_run_start = None
    baseline_start = baseline_end = None
    baseline_rows: list[dict[str, Any]] = []
    baseline: dict[str, float] | None = None
    prior_detector: pd.Series | None = None
    prior_detector_time: pd.Timestamp | None = None
    prior_timestamp: pd.Timestamp | None = None
    cohort_summary: dict[str, Any] | None = None

    def start_cohort(timestamp: pd.Timestamp, detected: bool, score: int) -> None:
        nonlocal state, cohort_id, sequence, cohort_start, refresh_start, refresh_end
        nonlocal stable_at, stable_run_start, baseline_start, baseline_end
        nonlocal baseline_rows, baseline, cohort_summary
        if cohort_summary is not None:
            cohort_summary["cohort_end"] = prior_timestamp
            cohort_summary["final_state"] = state
            cohort_rows.append(cohort_summary)
        sequence += 1
        cohort_id = f"cohort_{timestamp:%Y%m%d}_{sequence:03d}"
        cohort_start = timestamp
        refresh_start = timestamp if detected else None
        refresh_end = stable_at = stable_run_start = baseline_end = None
        baseline_start = None if detected else timestamp
        baseline_rows, baseline = [], None
        state = "REFRESHING" if detected else "BASELINE_BUILDING"
        cohort_summary = {
            "cohort_id": cohort_id, "cohort_start": timestamp, "cohort_end": pd.NaT,
            "cohort_refresh_start": refresh_start, "cohort_refresh_end": pd.NaT,
            "cohort_stable_at": pd.NaT, "cohort_baseline_start": baseline_start,
            "cohort_baseline_end": pd.NaT, "active_at": pd.NaT,
            "refresh_score": score, "record_count": 0, "signal_eligible_records": 0,
            "out_of_sample": bool(timestamp >= freeze), "final_state": state,
        }

    for index, row in raw_frame.iterrows():
        timestamp = row["timestamp"]
        output = _blank_features()
        if pd.isna(timestamp):
            feature_rows.append(output)
            continue
        trader_consistent = (
            row["total_traders"] > 0 and row["long_traders"] > 0 and row["short_traders"] > 0
            and abs(row["total_traders"] - row["long_traders"] - row["short_traders"])
            <= config["quality"]["trader_total_tolerance"]
        )
        changes = {name: math.nan for name in ("total", "long", "short", "long_pos", "short_pos")}
        if trader_consistent and prior_detector is not None:
            changes = {
                "total": _safe_abs_change(row["total_traders"], prior_detector["total_traders"]),
                "long": _safe_abs_change(row["long_traders"], prior_detector["long_traders"]),
                "short": _safe_abs_change(row["short_traders"], prior_detector["short_traders"]),
                "long_pos": _safe_abs_change(row["long_pos_usdt"], prior_detector["long_pos_usdt"]),
                "short_pos": _safe_abs_change(row["short_pos_usdt"], prior_detector["short_pos_usdt"]),
            }
        total_jump = not pd.isna(changes["total"]) and changes["total"] >= cfg["total_trader_change_threshold"]
        long_jump = not pd.isna(changes["long"]) and changes["long"] >= cfg["side_trader_change_threshold"]
        short_jump = not pd.isna(changes["short"]) and changes["short"] >= cfg["side_trader_change_threshold"]
        score = 2 * int(total_jump) + int(long_jump) + int(short_jump)
        score += int(not pd.isna(changes["long_pos"]) and changes["long_pos"] >= cfg["position_change_confirmation"])
        score += int(not pd.isna(changes["short_pos"]) and changes["short_pos"] >= cfg["position_change_confirmation"])
        refresh_candidate = bool(trader_consistent and (total_jump or long_jump or short_jump))
        stable_pair = bool(
            trader_consistent and prior_detector is not None
            and changes["total"] <= cfg["stable_total_trader_change"]
            and changes["long"] <= cfg["stable_side_trader_change"]
            and changes["short"] <= cfg["stable_side_trader_change"]
        )
        detected_now = False
        if cohort_id is None:
            start_cohort(timestamp, False, 0)
        elif state == "ACTIVE_COHORT" and refresh_candidate:
            start_cohort(timestamp, True, score)
            detected_now = True
        elif state in {"REFRESHING", "STABILIZING"}:
            if not stable_pair:
                state, stable_run_start = "REFRESHING", None
            else:
                if state == "REFRESHING" or stable_run_start is None:
                    state = "STABILIZING"
                    stable_run_start = prior_detector_time or timestamp
                    refresh_end = stable_run_start
                if (timestamp - stable_run_start).total_seconds() / 60 >= cfg["stabilization_minutes"]:
                    stable_at = baseline_start = timestamp
                    baseline_rows = []
                    state = "BASELINE_BUILDING"
                    if cohort_summary is not None:
                        cohort_summary.update({
                            "cohort_refresh_end": refresh_end,
                            "cohort_stable_at": stable_at,
                            "cohort_baseline_start": baseline_start,
                        })
        if state == "BASELINE_BUILDING" and bool(row["_quality_signal_ok"]):
            baseline_rows.append({source: row[source] for source in BASELINE_COLUMNS.values()})
            baseline_start = baseline_start or timestamp
            if (timestamp - baseline_start).total_seconds() / 60 >= cfg["baseline_minutes"]:
                candidate = _build_baseline(baseline_rows)
                if candidate is not None:
                    baseline, baseline_end, state = candidate, timestamp, "ACTIVE_COHORT"
                    if cohort_summary is not None:
                        cohort_summary.update({
                            "cohort_baseline_start": baseline_start,
                            "cohort_baseline_end": baseline_end,
                            "active_at": timestamp,
                            **{f"baseline_{name}": value for name, value in baseline.items()},
                        })
        output.update({
            "cohort_id": cohort_id, "analysis_state": state,
            "cohort_refresh_detected": detected_now, "cohort_refresh_score": score,
            "is_refresh_window": state == "REFRESHING",
            "is_stabilization_window": state == "STABILIZING",
            "minutes_since_cohort_start": (timestamp - cohort_start).total_seconds() / 60,
            "cohort_refresh_start": refresh_start, "cohort_refresh_end": refresh_end,
            "cohort_stable_at": stable_at, "cohort_baseline_start": baseline_start,
            "cohort_baseline_end": baseline_end, "out_of_sample": bool(timestamp >= freeze),
        })
        gross = row["long_pos_usdt"] + row["short_pos_usdt"]
        net = row["long_pos_usdt"] - row["short_pos_usdt"]
        long_return = _safe_ratio(row["current_price"], row["long_avg_price"]) - 1
        short_return = _safe_ratio(row["short_avg_price"], row["current_price"]) - 1
        avg_long = _safe_ratio(row["long_pos_usdt"], row["long_traders"])
        avg_short = _safe_ratio(row["short_pos_usdt"], row["short_traders"])
        output.update({
            "gross_position": gross, "net_exposure": net,
            "net_exposure_share": _safe_ratio(net, gross) if gross > 0 else math.nan,
            "avg_long_position": avg_long, "avg_short_position": avg_short,
            "avg_position_ratio": _safe_ratio(avg_long, avg_short),
            "long_directional_return": long_return, "short_directional_return": short_return,
            "long_loss_depth": max(0.0, -long_return) if not pd.isna(long_return) else math.nan,
            "short_loss_depth": max(0.0, -short_return) if not pd.isna(short_return) else math.nan,
        })
        if baseline is not None:
            output.update({
                "baseline_price": baseline["price"],
                "baseline_long_pos": baseline["long_pos"],
                "baseline_short_pos": baseline["short_pos"],
                "baseline_long_avg_price": baseline["long_avg_price"],
                "baseline_short_avg_price": baseline["short_avg_price"],
                "baseline_long_traders": baseline["long_traders"],
                "baseline_short_traders": baseline["short_traders"],
                "baseline_long_unrealized_pnl": baseline["long_unrealized_pnl"],
                "baseline_short_unrealized_pnl": baseline["short_unrealized_pnl"],
                "cohort_net_flow": _safe_ratio(
                    net - (baseline["long_pos"] - baseline["short_pos"]),
                    baseline["long_pos"] + baseline["short_pos"],
                ),
                "long_position_change_pct": _safe_ratio(row["long_pos_usdt"], baseline["long_pos"]) - 1,
                "short_position_change_pct": _safe_ratio(row["short_pos_usdt"], baseline["short_pos"]) - 1,
                "long_avg_entry_change": _safe_ratio(row["long_avg_price"], baseline["long_avg_price"]) - 1,
                "short_avg_entry_change": _safe_ratio(row["short_avg_price"], baseline["short_avg_price"]) - 1,
                "ls_ratio_change_within_cohort": _safe_ratio(row["ls_ratio"], baseline["ls_ratio"]) - 1,
            })
            baseline_long_return = _safe_ratio(baseline["price"], baseline["long_avg_price"]) - 1
            baseline_short_return = _safe_ratio(baseline["short_avg_price"], baseline["price"]) - 1
            output["long_existing_loss_at_cohort_start"] = (
                baseline_long_return < config["loss"]["existing_loss_threshold"]
            )
            output["short_existing_loss_at_cohort_start"] = (
                baseline_short_return < config["loss"]["existing_loss_threshold"]
            )
        output["signal_eligible"] = bool(
            state == "ACTIVE_COHORT" and row["_quality_signal_ok"] and baseline is not None
        )
        if detected_now:
            old_flag = str(raw_frame.at[index, "data_quality_flag"])
            raw_frame.at[index, "data_quality_flag"] = (
                "COHORT_REFRESH" if old_flag == "OK" else f"{old_flag}|COHORT_REFRESH"
            )
        if cohort_summary is not None:
            cohort_summary["record_count"] += 1
            cohort_summary["signal_eligible_records"] += int(output["signal_eligible"])
            cohort_summary["final_state"] = state
        feature_rows.append(output)
        if trader_consistent:
            prior_detector, prior_detector_time = row, timestamp
        prior_timestamp = timestamp
    if cohort_summary is not None:
        cohort_summary["cohort_end"] = prior_timestamp
        cohort_summary["final_state"] = state
        cohort_rows.append(cohort_summary)
    processed = pd.concat([raw_frame.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)
    return processed, pd.DataFrame(cohort_rows)


def add_divergence_features(processed: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = processed.copy()
    cfg = config["divergence"]
    for hours in cfg["windows_hours"]:
        result[f"btc_return_{hours}h"] = np.nan
        result[f"cohort_net_flow_change_{hours}h"] = np.nan
        result[f"divergence_{hours}h"] = None
    active = result[result["signal_eligible"] & result["timestamp"].notna()]
    for _, group in active.groupby("cohort_id", sort=False):
        indices = group.index.to_numpy()
        times = group["timestamp"].astype("int64").to_numpy()
        prices = group["current_price"].to_numpy(dtype=float)
        flows = group["cohort_net_flow"].to_numpy(dtype=float)
        for hours in cfg["windows_hours"]:
            delta_ns = int(pd.Timedelta(hours=float(hours)).value)
            for position, current_ns in enumerate(times):
                past = int(np.searchsorted(times, current_ns - delta_ns, side="right") - 1)
                if past < 0 or prices[past] <= 0 or not np.isfinite(flows[past]):
                    continue
                price_return = prices[position] / prices[past] - 1
                flow_change = flows[position] - flows[past]
                row_index = indices[position]
                result.at[row_index, f"btc_return_{hours}h"] = price_return
                result.at[row_index, f"cohort_net_flow_change_{hours}h"] = flow_change
                if abs(price_return) <= cfg["flat_price_threshold"] and flow_change >= cfg["flow_threshold"]:
                    result.at[row_index, f"divergence_{hours}h"] = "BULLISH_SM_DIVERGENCE"
                elif abs(price_return) <= cfg["flat_price_threshold"] and flow_change <= -cfg["flow_threshold"]:
                    result.at[row_index, f"divergence_{hours}h"] = "BEARISH_SM_DIVERGENCE"
    return result


def _response(change: float, config: dict[str, Any]) -> str:
    if change >= config["add_threshold"]:
        return "ADD"
    if change <= config["reduce_threshold"]:
        return "REDUCE"
    return "HOLD"


def _loss_depth_bucket(value: float, edges: list[float]) -> str | None:
    if pd.isna(value):
        return None
    if value < edges[0]:
        return "<0.5%"
    labels = ["0.5%-1%", "1%-2%", "2%-3%", "3%-5%"]
    for left, right, label in zip(edges[:-1], edges[1:], labels):
        if left <= value < right:
            return label
    return ">5%"


def _duration_bucket(minutes: float) -> str | None:
    if pd.isna(minutes):
        return None
    for upper, label in [
        (30, "<30m"), (60, "30-60m"), (120, "1-2h"), (240, "2-4h"),
        (480, "4-8h"), (720, "8-12h"),
    ]:
        if minutes < upper:
            return label
    return ">12h"


def _position_bucket(change: float) -> str | None:
    if pd.isna(change):
        return None
    if change <= -0.10:
        return "reduce >10%"
    if change <= -0.03:
        return "reduce 3-10%"
    if change < 0.03:
        return "hold"
    if change < 0.10:
        return "add 3-10%"
    if change < 0.20:
        return "add 10-20%"
    return "add >20%"


def build_loss_events(processed: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = processed.copy()
    cfg = config["loss"]
    for side in ("long", "short"):
        result[f"{side}_new_loss_event"] = False
        result[f"{side}_recovery_event"] = False
        result[f"{side}_loss_event_id"] = None
        result[f"{side}_loss_start_timestamp"] = pd.Series(
            pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
        )
        for suffix in [
            "loss_start_price", "loss_start_position", "loss_start_avg_entry",
            "loss_start_net_exposure", "loss_duration_minutes",
            "position_change_since_loss", "avg_entry_change_since_loss",
        ]:
            result[f"{side}_{suffix}"] = np.nan
        result[f"{side}_loss_response"] = None
    result["event_markers"] = ""

    active: dict[str, dict[str, Any] | None] = {"long": None, "short": None}
    cooldown_until: dict[str, pd.Timestamp | None] = {"long": None, "short": None}
    counters: defaultdict[tuple[str, str], int] = defaultdict(int)
    completed: list[dict[str, Any]] = []
    previous_cohort: str | None = None
    last_active_row: dict[str, pd.Series | None] = {"long": None, "short": None}

    def decorate(event: dict[str, Any]) -> None:
        event["event_type"] = f"{event['side']}_NEW_LOSS_{event['final_response']}"
        event["loss_depth_bucket"] = _loss_depth_bucket(event["max_loss_depth"], cfg["depth_bucket_edges"])
        event["loss_duration_bucket"] = _duration_bucket(event["max_loss_duration"])
        final_change = (
            event["max_position_add_pct"] if event["final_response"] == "ADD"
            else event["max_position_reduce_pct"] if event["final_response"] == "REDUCE"
            else 0.0
        )
        event["position_response_bucket"] = _position_bucket(final_change)

    def finalize(
        side: str,
        event: dict[str, Any],
        row: pd.Series | None,
        recovered: bool,
        status: str,
    ) -> None:
        if row is not None:
            timestamp = row["timestamp"]
            position = row[f"{side}_pos_usdt"]
            event["loss_end_timestamp"] = timestamp
            event["loss_end_price"] = row["current_price"]
            event["loss_end_avg_entry"] = row[f"{side}_avg_price"]
            event["cohort_net_flow_at_end"] = row["cohort_net_flow"]
            event["final_response"] = _response(
                _safe_ratio(position, event["loss_start_position"]) - 1, cfg
            )
            event["max_loss_duration"] = max(
                event["max_loss_duration"],
                (timestamp - event["loss_start_timestamp"]).total_seconds() / 60,
            )
        event["recovered"] = recovered
        event["recovery_event_type"] = f"{side.upper()}_RECOVERY" if recovered else None
        event["event_status"] = status
        decorate(event)
        completed.append(event)

    for index, row in result.iterrows():
        timestamp, cohort_id = row["timestamp"], row["cohort_id"]
        if pd.isna(timestamp):
            continue
        if previous_cohort is not None and cohort_id != previous_cohort:
            for side in ("long", "short"):
                if active[side] is not None:
                    finalize(side, active[side], last_active_row[side], False, "COHORT_REFRESH")
                    active[side] = None
                cooldown_until[side] = None
                last_active_row[side] = None
        previous_cohort = cohort_id
        if not bool(row["signal_eligible"]):
            continue
        markers: list[str] = []
        for side in ("long", "short"):
            directional_return = row[f"{side}_directional_return"]
            position = row[f"{side}_pos_usdt"]
            avg_entry = row[f"{side}_avg_price"]
            baseline_return = (
                _safe_ratio(row["baseline_price"], row["baseline_long_avg_price"]) - 1
                if side == "long"
                else _safe_ratio(row["baseline_short_avg_price"], row["baseline_price"]) - 1
            )
            event = active[side]
            if event is None:
                clean = (
                    not bool(row[f"{side}_existing_loss_at_cohort_start"])
                    and baseline_return >= cfg["clean_baseline_threshold"]
                )
                cooled_down = cooldown_until[side] is None or timestamp >= cooldown_until[side]
                if clean and cooled_down and directional_return <= cfg["loss_entry_threshold"]:
                    date_key = timestamp.strftime("%Y%m%d")
                    counters[(side, date_key)] += 1
                    event_id = f"{side.upper()}_{date_key}_{counters[(side, date_key)]:03d}"
                    event = {
                        "event_id": event_id, "cohort_id": cohort_id, "side": side.upper(),
                        "event_type": f"{side.upper()}_NEW_LOSS_HOLD",
                        "loss_start_timestamp": timestamp, "loss_end_timestamp": pd.NaT,
                        "loss_start_price": row["current_price"], "loss_end_price": np.nan,
                        "loss_start_position": position, "max_position": position,
                        "min_position": position, "loss_start_avg_entry": avg_entry,
                        "loss_end_avg_entry": np.nan,
                        "loss_start_net_exposure": row["net_exposure"],
                        "max_loss_depth": row[f"{side}_loss_depth"],
                        "max_loss_duration": 0.0, "max_position_add_pct": 0.0,
                        "max_position_reduce_pct": 0.0, "final_response": "HOLD",
                        "recovered": False, "recovery_event_type": None,
                        "event_status": "OPEN",
                        "cohort_net_flow_at_start": row["cohort_net_flow"],
                        "cohort_net_flow_at_end": np.nan,
                        "out_of_sample": bool(row["out_of_sample"]),
                    }
                    active[side] = event
                    result.at[index, f"{side}_new_loss_event"] = True
                    markers.append(f"{side.upper()}_NEW_LOSS_HOLD")
            if event is not None:
                duration = (timestamp - event["loss_start_timestamp"]).total_seconds() / 60
                position_change = _safe_ratio(position, event["loss_start_position"]) - 1
                avg_entry_change = _safe_ratio(avg_entry, event["loss_start_avg_entry"]) - 1
                response = _response(position_change, cfg)
                event["max_position"] = max(event["max_position"], position)
                event["min_position"] = min(event["min_position"], position)
                event["max_loss_depth"] = max(event["max_loss_depth"], row[f"{side}_loss_depth"])
                event["max_loss_duration"] = max(event["max_loss_duration"], duration)
                event["max_position_add_pct"] = max(event["max_position_add_pct"], position_change)
                event["max_position_reduce_pct"] = min(event["max_position_reduce_pct"], position_change)
                event["final_response"] = response
                result.at[index, f"{side}_loss_event_id"] = event["event_id"]
                result.at[index, f"{side}_loss_start_timestamp"] = event["loss_start_timestamp"]
                result.at[index, f"{side}_loss_start_price"] = event["loss_start_price"]
                result.at[index, f"{side}_loss_start_position"] = event["loss_start_position"]
                result.at[index, f"{side}_loss_start_avg_entry"] = event["loss_start_avg_entry"]
                result.at[index, f"{side}_loss_start_net_exposure"] = event["loss_start_net_exposure"]
                result.at[index, f"{side}_loss_duration_minutes"] = duration
                result.at[index, f"{side}_position_change_since_loss"] = position_change
                result.at[index, f"{side}_avg_entry_change_since_loss"] = avg_entry_change
                result.at[index, f"{side}_loss_response"] = response
                if directional_return >= cfg["recovery_threshold"]:
                    result.at[index, f"{side}_recovery_event"] = True
                    markers.append(f"{side.upper()}_RECOVERY")
                    finalize(side, event, row, True, "RECOVERED")
                    active[side] = None
                    cooldown_until[side] = timestamp + pd.Timedelta(minutes=cfg["cooldown_minutes"])
            last_active_row[side] = row
        result.at[index, "event_markers"] = "|".join(markers)
    for side in ("long", "short"):
        if active[side] is not None:
            decorate(active[side])
            completed.append(active[side])
    return result, pd.DataFrame(completed)


def add_forward_returns(
    events: pd.DataFrame, processed: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    result = events.copy()
    for hours in config["backtest"]["forward_hours"]:
        result[f"forward_return_{hours}h"] = np.nan
    if result.empty:
        return result
    prices = (
        processed.loc[
            processed["timestamp"].notna() & processed["current_price"].gt(0),
            ["timestamp", "current_price"],
        ]
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
    )
    timestamps = prices["timestamp"].astype("int64").to_numpy()
    price_values = prices["current_price"].to_numpy(dtype=float)
    for index, event in result.iterrows():
        for hours in config["backtest"]["forward_hours"]:
            target = event["loss_start_timestamp"] + pd.Timedelta(hours=hours)
            future_index = int(np.searchsorted(timestamps, target.value, side="left"))
            if future_index < len(timestamps):
                result.at[index, f"forward_return_{hours}h"] = (
                    price_values[future_index] / event["loss_start_price"] - 1
                )
    return result


def _bootstrap_mean_ci(
    values: np.ndarray, sample_count: int, rng: np.random.Generator
) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    draws = rng.choice(values, size=(sample_count, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def build_backtest(events: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "dataset", "event_type", "segment", "segment_value", "horizon", "sample_count",
        "mean_forward_return", "median_forward_return", "win_rate", "p25", "p75",
        "bootstrap_ci_low", "bootstrap_ci_high",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    rng = np.random.default_rng(config["backtest"]["random_seed"])
    rows: list[dict[str, Any]] = []
    datasets = [
        ("exploratory_historical", events[~events["out_of_sample"]]),
        ("out_of_sample", events[events["out_of_sample"]]),
    ]
    for dataset_name, dataset in datasets:
        for event_type, group in dataset.groupby("event_type", sort=True):
            segments: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", group)]
            for segment in [
                "loss_depth_bucket", "loss_duration_bucket", "position_response_bucket"
            ]:
                segments.extend(
                    (segment, str(value), subgroup)
                    for value, subgroup in group.dropna(subset=[segment]).groupby(segment, sort=True)
                )
            for segment, segment_value, subgroup in segments:
                for hours in config["backtest"]["forward_hours"]:
                    values = subgroup[f"forward_return_{hours}h"].dropna().to_numpy(dtype=float)
                    low, high = _bootstrap_mean_ci(
                        values, config["backtest"]["bootstrap_samples"], rng
                    )
                    rows.append({
                        "dataset": dataset_name, "event_type": event_type,
                        "segment": segment, "segment_value": segment_value,
                        "horizon": f"{hours}h", "sample_count": len(values),
                        "mean_forward_return": float(np.mean(values)) if len(values) else math.nan,
                        "median_forward_return": float(np.median(values)) if len(values) else math.nan,
                        "win_rate": float(np.mean(values > 0)) if len(values) else math.nan,
                        "p25": float(np.quantile(values, 0.25)) if len(values) else math.nan,
                        "p75": float(np.quantile(values, 0.75)) if len(values) else math.nan,
                        "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                    })
    return pd.DataFrame(rows, columns=columns)


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _quality_counts(values: Iterable[str]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for joined in values:
        counter.update(str(joined).split("|"))
    return dict(sorted(counter.items()))


def _translate_joined(value: Any, mapping: dict[str, str]) -> Any:
    if pd.isna(value) or value == "":
        return value
    return "|".join(mapping.get(item, item) for item in str(value).split("|"))


def _chinese_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    replacements = {
        "analysis_state": STATE_ZH,
        "final_state": STATE_ZH,
        "side": SIDE_ZH,
        "event_type": EVENT_TYPE_ZH,
        "recovery_event_type": EVENT_TYPE_ZH,
        "final_response": RESPONSE_ZH,
        "long_loss_response": RESPONSE_ZH,
        "short_loss_response": RESPONSE_ZH,
        "event_status": STATUS_ZH,
        "dataset": {
            "exploratory_historical": "历史探索数据",
            "out_of_sample": "样本外验证数据",
        },
        "segment": {
            "overall": "总体",
            "loss_depth_bucket": "亏损深度",
            "loss_duration_bucket": "亏损时长",
            "position_response_bucket": "仓位操作",
        },
        "divergence_1h": {
            "BULLISH_SM_DIVERGENCE": "看多型价格资金流背离",
            "BEARISH_SM_DIVERGENCE": "看空型价格资金流背离",
        },
        "divergence_2h": {
            "BULLISH_SM_DIVERGENCE": "看多型价格资金流背离",
            "BEARISH_SM_DIVERGENCE": "看空型价格资金流背离",
        },
        "divergence_4h": {
            "BULLISH_SM_DIVERGENCE": "看多型价格资金流背离",
            "BEARISH_SM_DIVERGENCE": "看空型价格资金流背离",
        },
        "divergence_6h": {
            "BULLISH_SM_DIVERGENCE": "看多型价格资金流背离",
            "BEARISH_SM_DIVERGENCE": "看空型价格资金流背离",
        },
    }
    for column, mapping in replacements.items():
        if column in result:
            result[column] = result[column].replace(mapping)
    if "data_quality_flag" in result:
        result["data_quality_flag"] = result["data_quality_flag"].map(
            lambda value: _translate_joined(value, QUALITY_ZH)
        )
    if "event_markers" in result:
        result["event_markers"] = result["event_markers"].map(
            lambda value: _translate_joined(value, EVENT_TYPE_ZH)
        )
    for column in ("cohort_id",):
        if column in result:
            result[column] = result[column].astype("string").str.replace(
                "cohort_", "批次_", regex=False
            )
    for column in ("event_id", "long_loss_event_id", "short_loss_event_id"):
        if column in result:
            result[column] = (
                result[column].astype("string")
                .str.replace("LONG_", "多头_", regex=False)
                .str.replace("SHORT_", "空头_", regex=False)
            )
    if "horizon" in result:
        result["horizon"] = result["horizon"].astype("string").str.replace(
            "h", "小时", regex=False
        )
    if "segment_value" in result:
        result["segment_value"] = result["segment_value"].replace({
            "all": "全部",
            "hold": "持仓",
            "add 3-10%": "加仓3%至10%",
            "add 10-20%": "加仓10%至20%",
            "add >20%": "加仓超过20%",
            "reduce 3-10%": "减仓3%至10%",
            "reduce >10%": "减仓超过10%",
            "<30m": "不足30分钟",
            "30-60m": "30至60分钟",
            "1-2h": "1至2小时",
            "2-4h": "2至4小时",
            "4-8h": "4至8小时",
            "8-12h": "8至12小时",
            ">12h": "超过12小时",
        })
    return result.rename(columns={column: COLUMN_ZH.get(column, column) for column in result.columns})


def _format_percent(value: Any) -> str:
    return "-" if pd.isna(value) else f"{float(value):+.2%}"


def _build_chinese_report(
    summary: dict[str, Any], events: pd.DataFrame, backtest: pd.DataFrame
) -> str:
    lines = [
        "# 聪明钱名单批次分析报告",
        "",
        "## 数据与样本边界",
        "",
        f"- 原始记录数：{summary['raw_records']:,}",
        f"- 有效记录数：{summary['valid_records']:,}",
        f"- 无效记录数：{summary['invalid_records']:,}",
        f"- 识别名单批次数：{summary['cohorts_detected']}",
        f"- 名单刷新次数：{summary['refresh_windows']}",
        f"- 模型冻结时间：{summary['model_freeze_date']}",
        f"- 样本外事件数：{summary['oos_events']}",
        "- 冻结时间之前的全部结果只属于历史探索，不能称为样本外结论。",
        "",
        "## 数据质量",
        "",
    ]
    for flag, count in summary["quality_counts"].items():
        lines.append(f"- {QUALITY_ZH.get(flag, flag)}：{count}")
    lines.extend([
        "",
        "## 新亏损事件",
        "",
        f"- 多头新亏损事件：{summary['new_long_loss_events']}",
        f"- 空头新亏损事件：{summary['new_short_loss_events']}",
    ])
    for key, count in summary["response_counts"].items():
        side, response = key.split("_", 1)
        lines.append(f"- {SIDE_ZH.get(side, side)}{RESPONSE_ZH.get(response, response)}：{count}")
    recovered = int(events["recovered"].sum()) if not events.empty else 0
    lines.extend([
        f"- 已恢复盈利：{recovered}",
        f"- 尚未恢复或因名单刷新结束：{len(events) - recovered}",
        "",
        "## 历史探索回测",
        "",
        "空头事件仍使用比特币原始收益，没有乘以 -1。",
        "",
        "| 事件 | 周期 | 样本数 | 平均收益 | 中位数收益 | 正收益比例 | 自助法95%置信区间 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    overall = backtest[
        (backtest["dataset"] == "exploratory_historical")
        & (backtest["segment"] == "overall")
    ]
    for _, row in overall.iterrows():
        interval = (
            f"{_format_percent(row['bootstrap_ci_low'])} 至 "
            f"{_format_percent(row['bootstrap_ci_high'])}"
        )
        lines.append(
            f"| {EVENT_TYPE_ZH.get(row['event_type'], row['event_type'])} "
            f"| {str(row['horizon']).replace('h', '小时')} | {int(row['sample_count'])} "
            f"| {_format_percent(row['mean_forward_return'])} "
            f"| {_format_percent(row['median_forward_return'])} "
            f"| {_format_percent(row['win_rate'])} | {interval} |"
        )
    lines.extend([
        "",
        "## 前十个名单刷新时间（UTC）",
        "",
        *[f"- {timestamp}" for timestamp in summary["refresh_timestamps"]],
        "",
        "当前样本较少，所有统计只用于探索，不构成交易结论。",
        "",
    ])
    return "\n".join(lines)


def run_pipeline(
    raw_path: Path = RAW_CACHE_PATH,
    config_path: Path = CONFIG_PATH,
    output_dir: Path = DERIVED_DIR,
    write_outputs: bool = True,
) -> dict[str, Any]:
    config = load_analysis_config(config_path)
    records, raw_metadata = load_raw_records(raw_path)
    raw = prepare_raw_frame(records, config)
    processed, cohorts = reconstruct_cohorts(raw, config)
    processed = add_divergence_features(processed, config)
    processed, events = build_loss_events(processed, config)
    events = add_forward_returns(events, processed, config)
    backtest = build_backtest(events, config)
    refresh_rows = processed[processed["cohort_refresh_detected"]]
    response_counts = (
        events.groupby(["side", "final_response"]).size().to_dict()
        if not events.empty else {}
    )
    summary = {
        "raw_file": str(raw_path.resolve()),
        "raw_file_unchanged": True,
        "raw_records": len(records),
        "valid_records": int(processed["_quality_signal_ok"].sum()),
        "invalid_records": int((~processed["_quality_signal_ok"]).sum()),
        "model_freeze_date": config["model_freeze_date"],
        "cohorts_detected": int(len(cohorts)),
        "refresh_windows": int(len(refresh_rows)),
        "refresh_timestamps": [
            timestamp.isoformat() for timestamp in refresh_rows["timestamp"].head(10)
        ],
        "quality_counts": _quality_counts(processed["data_quality_flag"]),
        "new_long_loss_events": int((events["side"] == "LONG").sum()) if not events.empty else 0,
        "new_short_loss_events": int((events["side"] == "SHORT").sum()) if not events.empty else 0,
        "response_counts": {
            f"{side}_{response}": int(count)
            for (side, response), count in response_counts.items()
        },
        "oos_events": int(events["out_of_sample"].sum()) if not events.empty else 0,
        "source_latest_timestamp": raw_metadata.get("latest_timestamp"),
    }
    summary["中文摘要"] = {
        "原始记录数": summary["raw_records"],
        "有效记录数": summary["valid_records"],
        "无效记录数": summary["invalid_records"],
        "模型冻结时间": summary["model_freeze_date"],
        "识别名单批次数": summary["cohorts_detected"],
        "名单刷新次数": summary["refresh_windows"],
        "前十个名单刷新时间": summary["refresh_timestamps"],
        "数据质量统计": {
            QUALITY_ZH.get(flag, flag): count
            for flag, count in summary["quality_counts"].items()
        },
        "多头新亏损事件数": summary["new_long_loss_events"],
        "空头新亏损事件数": summary["new_short_loss_events"],
        "亏损后操作统计": {
            f"{SIDE_ZH.get(key.split('_', 1)[0], key)}"
            f"{RESPONSE_ZH.get(key.split('_', 1)[1], '')}": count
            for key, count in summary["response_counts"].items()
        },
        "样本外事件数": summary["oos_events"],
        "原始数据最新时间": summary["source_latest_timestamp"],
        "原始历史文件是否保持不变": True,
    }
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_path = output_dir / PROCESSED_PATH.name
        events_path = output_dir / EVENTS_PATH.name
        cohorts_path = output_dir / COHORTS_PATH.name
        backtest_path = output_dir / BACKTEST_PATH.name
        summary_path = output_dir / SUMMARY_PATH.name
        internal = ["_source_order", "_timestamp_out_of_order", "_quality_signal_ok"]
        _atomic_parquet(processed.drop(columns=internal), processed_path)
        _atomic_parquet(events, events_path)
        _atomic_parquet(cohorts, cohorts_path)
        backtest.to_csv(backtest_path, index=False)
        _atomic_parquet(
            _chinese_frame(processed.drop(columns=internal)),
            output_dir / CHINESE_PROCESSED_PATH.name,
        )
        _atomic_parquet(
            _chinese_frame(events), output_dir / CHINESE_EVENTS_PATH.name
        )
        _atomic_parquet(
            _chinese_frame(cohorts), output_dir / CHINESE_COHORTS_PATH.name
        )
        _chinese_frame(backtest).to_csv(
            output_dir / CHINESE_BACKTEST_PATH.name,
            index=False,
            encoding="utf-8-sig",
        )
        (output_dir / CHINESE_REPORT_PATH.name).write_text(
            _build_chinese_report(summary, events, backtest),
            encoding="utf-8",
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, default=_json_default)
    print(
        "聪明钱分析处理完成 | "
        f"原始记录={summary['raw_records']} 有效记录={summary['valid_records']} "
        f"无效记录={summary['invalid_records']} 名单批次={summary['cohorts_detected']} "
        f"名单刷新={summary['refresh_windows']} "
        f"多头新亏损事件={summary['new_long_loss_events']} "
        f"空头新亏损事件={summary['new_short_loss_events']} "
        f"样本外事件={summary['oos_events']}"
    )
    return {
        "processed": processed, "events": events, "cohorts": cohorts,
        "backtest": backtest, "summary": summary,
    }


def print_report(report_path: Path = CHINESE_REPORT_PATH) -> None:
    if not report_path.exists():
        raise FileNotFoundError("尚无中文分析结果，请先运行 process-smart-money")
    print(report_path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="按币安聪明钱名单批次进行数据处理、事件研究和回测"
    )
    parser.add_argument(
        "command",
        choices=[
            "process-smart-money", "backtest-smart-money", "report-smart-money",
            "处理聪明钱", "回测聪明钱", "查看聪明钱报告",
        ],
        help="处理命令会依次完成：逐条处理、批次识别、事件生成、回测和中文报告",
    )
    args = parser.parse_args(argv)
    if args.command in {
        "process-smart-money", "backtest-smart-money", "处理聪明钱", "回测聪明钱"
    }:
        run_pipeline()
    print_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
