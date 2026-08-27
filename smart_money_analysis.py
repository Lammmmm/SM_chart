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
LOSS_EPISODES_PATH = DERIVED_DIR / "smart_money_loss_episodes.parquet"
ACTION_EVENTS_PATH = DERIVED_DIR / "smart_money_action_events.parquet"
CHINESE_LOSS_EPISODES_PATH = DERIVED_DIR / "聪明钱亏损过程表_中文.parquet"
CHINESE_ACTION_EVENTS_PATH = DERIVED_DIR / "聪明钱动作事件表_中文.parquet"
ACTION_COMPARISON_PATH = DERIVED_DIR / "smart_money_action_definition_comparison.csv"
CHINESE_ACTION_COMPARISON_PATH = DERIVED_DIR / "聪明钱动作定义对照_中文.csv"
ANALYSIS_LOGIC_VERSION = "2.1"
OBSOLETE_ACTIVE_REUSE_WINDOW_HOURS = 18

STATE_ZH = {
    "UNAVAILABLE": "不可用",
    "REFRESHING": "名单刷新期",
    "STABILIZING": "稳定观察期",
    "BASELINE_BUILDING": "基准构建期",
    "ACTIVE_COHORT": "批次有效期",
    "COHORT_EXPIRED": "批次已过期",
}
SIDE_ZH = {"LONG": "多头", "SHORT": "空头"}
RESPONSE_ZH = {"ADD": "加仓", "HOLD": "持仓", "REDUCE": "减仓"}
EVENT_TYPE_ZH = {
    "LONG_NEW_LOSS": "多头首次进入新亏损",
    "SHORT_NEW_LOSS": "空头首次进入新亏损",
    "LONG_LOSS_ADD": "多头亏损后首次明显加仓",
    "LONG_LOSS_REDUCE": "多头亏损后首次明显减仓",
    "SHORT_LOSS_ADD": "空头亏损后首次明显加仓",
    "SHORT_LOSS_REDUCE": "空头亏损后首次明显减仓",
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
    "COHORT_CHANGED": "因推断批次变化而结束",
    "COHORT_EXPIRED": "因推断批次过期而结束",
    "DATA_END": "数据结束",
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
    "hard_refresh_trigger": "是否触发人数硬刷新",
    "structural_refresh_trigger": "是否触发结构刷新",
    "structural_refresh_score": "结构刷新评分",
    "structural_anomalous_fields": "结构异常字段",
    "structural_anomalous_field_count": "结构异常字段数",
    "refresh_time_prior_supported": "历史刷新时间分布是否支持",
    "cohort_confidence": "推断批次可信度",
    "cohort_definition": "批次定义", "refresh_reason": "刷新原因",
    "expired_records": "过期状态记录数", "expired_at": "批次过期时间",
    "data_gap_detected": "是否检测到长时间数据中断",
    "cohort_expired_now": "是否在此刻过期",
    "backtest_primary": "是否进入主要回测",
    "analysis_logic_version": "分析逻辑版本",
    "is_stabilization_window": "是否处于稳定观察期",
    "minutes_since_cohort_start": "批次开始后分钟数",
    "cohort_refresh_start": "名单刷新开始时间", "cohort_refresh_end": "名单刷新结束时间",
    "cohort_stable_at": "名单稳定确认时间", "cohort_baseline_start": "基准构建开始时间",
    "cohort_baseline_end": "基准构建结束时间", "signal_eligible": "是否允许产生信号",
    "out_of_sample": "是否为修正版逻辑样本外数据",
    "logic_valid_from": "修正版逻辑生效时间", "baseline_price": "批次基准价格",
    "baseline_long_pos": "批次基准多头仓位", "baseline_short_pos": "批次基准空头仓位",
    "baseline_long_avg_price": "批次基准多头平均开仓价",
    "baseline_short_avg_price": "批次基准空头平均开仓价",
    "baseline_long_traders": "批次基准多头人数",
    "baseline_short_traders": "批次基准空头人数",
    "baseline_long_unrealized_pnl": "批次基准多头未实现盈亏",
    "baseline_short_unrealized_pnl": "批次基准空头未实现盈亏",
    "baseline_ls_ratio": "批次基准多空人数比",
    "gross_position": "多空USDT名义总仓位（仅描述）",
    "net_exposure": "净USDT名义风险敞口（仅描述）",
    "net_exposure_share": "净名义风险敞口占比",
    "long_qty_proxy": "多头BTC等价仓位代理",
    "short_qty_proxy": "空头BTC等价仓位代理",
    "gross_qty_proxy": "多空BTC等价总仓位代理",
    "net_qty_proxy": "净BTC等价仓位代理",
    "baseline_long_qty_proxy": "基准多头BTC等价仓位代理",
    "baseline_short_qty_proxy": "基准空头BTC等价仓位代理",
    "baseline_gross_qty_proxy": "基准多空BTC等价总仓位代理",
    "baseline_net_qty_proxy": "基准净BTC等价仓位代理",
    "cohort_net_flow": "批次净USDT名义风险流（兼容字段，仅描述）",
    "cohort_net_notional_flow": "批次净USDT名义风险流（仅描述）",
    "cohort_net_qty_flow": "聪明钱净BTC等价风险流",
    "long_position_change_pct": "批次内多头USDT名义仓位变化率（仅描述）",
    "short_position_change_pct": "批次内空头USDT名义仓位变化率（仅描述）",
    "long_qty_change_pct": "批次内多头BTC等价仓位变化率",
    "short_qty_change_pct": "批次内空头BTC等价仓位变化率",
    "avg_long_position": "每位多头平均USDT名义仓位（仅描述）",
    "avg_short_position": "每位空头平均USDT名义仓位（仅描述）",
    "avg_position_ratio": "多空人均USDT名义仓位比（仅描述）",
    "avg_long_qty_proxy_per_trader": "每位多头平均BTC等价仓位代理",
    "avg_short_qty_proxy_per_trader": "每位空头平均BTC等价仓位代理",
    "avg_qty_proxy_ratio": "多空人均BTC等价仓位代理比",
    "long_directional_return": "多头方向收益",
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
    "event_family": "事件类别", "event_timestamp": "事件发生时间",
    "event_price": "事件发生价格", "episode_id": "亏损过程编号",
    "action_event_id": "动作事件编号", "action_type": "动作类型",
    "action_timestamp": "动作发生时间", "action_price": "动作发生价格",
    "action_position": "动作发生时USDT名义仓位（仅描述）",
    "action_avg_entry": "动作发生时平均开仓价",
    "position_change_since_loss": "亏损后USDT名义仓位变化率（兼容字段，仅描述）",
    "position_notional_change_since_loss": "亏损后USDT名义仓位变化率（仅描述）",
    "position_qty_change_since_loss": "亏损后BTC等价仓位变化率",
    "loss_start_position_qty_proxy": "亏损开始时BTC等价仓位代理",
    "action_position_qty_proxy": "动作发生时BTC等价仓位代理",
    "net_qty_proxy_at_action": "动作发生时净BTC等价仓位代理",
    "loss_depth_at_action": "动作发生时亏损深度",
    "loss_duration_at_action": "动作发生时亏损分钟数",
    "cohort_net_flow_at_action": "动作发生时净USDT名义风险流（兼容字段，仅描述）",
    "cohort_net_notional_flow_at_action": "动作发生时净USDT名义风险流（仅描述）",
    "cohort_net_qty_flow_at_action": "动作发生时净BTC等价风险流",
    "net_exposure_at_action": "动作发生时净风险敞口",
    "avg_entry_change_since_loss": "亏损后平均开仓价变化",
    "action_sequence": "动作顺序", "recovery_timestamp": "恢复盈利时间",
    "recovery_price": "恢复盈利价格", "episode_end_timestamp": "亏损过程结束时间",
    "episode_end_reason": "亏损过程结束原因",
    "event_type": "事件类型", "loss_start_timestamp": "亏损开始时间",
    "loss_end_timestamp": "亏损结束时间", "loss_start_price": "亏损开始价格",
    "loss_end_price": "亏损结束价格",
    "loss_start_position": "亏损开始USDT名义仓位（仅描述）",
    "max_position": "事件期间最大仓位", "min_position": "事件期间最小仓位",
    "loss_start_avg_entry": "亏损开始平均开仓价",
    "loss_end_avg_entry": "亏损结束平均开仓价",
    "loss_start_net_exposure": "亏损开始时净风险敞口",
    "max_loss_depth": "最大亏损深度", "max_loss_duration": "最长亏损分钟数",
    "max_position_add_pct": "USDT名义仓位最大增加比例（仅描述）",
    "max_position_reduce_pct": "USDT名义仓位最大减少比例（仅描述）",
    "max_qty_add_pct": "BTC等价仓位最大增加比例",
    "max_qty_reduce_pct": "BTC等价仓位最大减少比例",
    "max_position_qty_proxy": "过程内最大BTC等价仓位代理",
    "min_position_qty_proxy": "过程内最小BTC等价仓位代理",
    "cohort_net_notional_flow_at_start": "过程开始时净USDT名义风险流（仅描述）",
    "cohort_net_notional_flow_at_end": "过程结束时净USDT名义风险流（仅描述）",
    "cohort_net_qty_flow_at_start": "过程开始时净BTC等价风险流",
    "cohort_net_qty_flow_at_end": "过程结束时净BTC等价风险流",
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
    "sensitivity": "批次可信度口径", "unconditional_return": "无条件基准收益",
    "difference_vs_unconditional": "相对无条件基准的超额收益",
    "directional_success_rate": "方向正确比例", "sample_size_warning": "样本量提示",
    "final_response_note": "最终操作说明",
    "active_refresh_new_segment": "是否因有效批次后刷新而强制新分段",
    "old_rule_would_reuse_active_segment": "旧规则是否会错误复用有效批次编号",
    "comparison_status": "新旧动作定义对照结果",
    "old_notional_action_timestamp": "旧USDT名义动作时间",
    "new_qty_action_timestamp": "新BTC等价数量动作时间",
    "timestamp_difference_minutes": "新旧动作时间差分钟数",
    "long_action_event": "是否触发多头首次动作事件",
    "short_action_event": "是否触发空头首次动作事件",
    "long_action_type": "多头首次动作类型",
    "short_action_type": "空头首次动作类型",
    "long_action_event_id": "多头首次动作事件编号",
    "short_action_event_id": "空头首次动作事件编号",
    "long_loss_start_position_qty_proxy": "多头亏损开始时BTC等价仓位代理",
    "short_loss_start_position_qty_proxy": "空头亏损开始时BTC等价仓位代理",
    "long_position_notional_change_since_loss": "多头亏损后USDT名义仓位变化率（仅描述）",
    "short_position_notional_change_since_loss": "空头亏损后USDT名义仓位变化率（仅描述）",
    "long_position_qty_change_since_loss": "多头亏损后BTC等价仓位变化率",
    "short_position_qty_change_since_loss": "空头亏损后BTC等价仓位变化率",
}
_STRUCTURAL_FIELD_ZH = {
    "total_traders": "交易者总人数",
    "long_traders": "多头交易者人数",
    "short_traders": "空头交易者人数",
    "long_avg_price": "多头平均开仓价",
    "short_avg_price": "空头平均开仓价",
    "long_qty_proxy": "多头BTC等价仓位代理",
    "short_qty_proxy": "空头BTC等价仓位代理",
    "long_unrealized_pnl": "多头未实现盈亏",
    "short_unrealized_pnl": "空头未实现盈亏",
    "ls_ratio": "多空人数比",
}
for _field, _field_zh in _STRUCTURAL_FIELD_ZH.items():
    COLUMN_ZH[f"structural_z_{_field}"] = f"{_field_zh}结构异常稳健分数"
for _hours in (1, 2, 4, 6, 12, 24):
    COLUMN_ZH[f"btc_return_{_hours}h"] = f"比特币过去{_hours}小时收益"
    COLUMN_ZH[f"cohort_net_flow_change_{_hours}h"] = f"批次净风险流过去{_hours}小时变化"
    COLUMN_ZH[f"cohort_net_notional_flow_change_{_hours}h"] = f"净USDT名义风险流过去{_hours}小时变化（仅描述）"
    COLUMN_ZH[f"cohort_net_qty_flow_change_{_hours}h"] = f"净BTC等价风险流过去{_hours}小时变化"
    COLUMN_ZH[f"divergence_{_hours}h"] = f"过去{_hours}小时价格资金流背离"
    COLUMN_ZH[f"forward_return_{_hours}h"] = f"事件后{_hours}小时比特币收益"
    COLUMN_ZH[f"forward_price_timestamp_{_hours}h"] = f"事件后{_hours}小时价格采样时间"
    COLUMN_ZH[f"forward_price_delay_minutes_{_hours}h"] = f"事件后{_hours}小时价格延迟分钟数"
for _prefix, _action_name in (("first_add", "首次加仓"), ("first_reduce", "首次减仓")):
    COLUMN_ZH[f"{_prefix}_timestamp"] = f"{_action_name}时间"
    COLUMN_ZH[f"{_prefix}_price"] = f"{_action_name}时比特币价格"
    COLUMN_ZH[f"{_prefix}_position"] = f"{_action_name}时仓位"
    COLUMN_ZH[f"{_prefix}_avg_entry"] = f"{_action_name}时平均开仓价"
    COLUMN_ZH[f"{_prefix}_loss_depth"] = f"{_action_name}时亏损深度"
    COLUMN_ZH[f"{_prefix}_loss_duration"] = f"{_action_name}时亏损分钟数"
    COLUMN_ZH[f"{_prefix}_net_exposure"] = f"{_action_name}时净风险敞口"
    COLUMN_ZH[f"{_prefix}_cohort_net_flow"] = f"{_action_name}时批次净风险流"
    COLUMN_ZH[f"{_prefix}_position_qty_proxy"] = f"{_action_name}时BTC等价仓位代理"
    COLUMN_ZH[f"{_prefix}_cohort_net_qty_flow"] = f"{_action_name}时净BTC等价风险流"

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
    "logic_valid_from": "2026-08-27T07:26:03Z",
    "cohort": {
        "total_trader_change_threshold": 0.08,
        "side_trader_change_threshold": 0.12,
        "position_change_confirmation": 0.25,
        "stabilization_minutes": 30,
        "stable_total_trader_change": 0.02,
        "stable_side_trader_change": 0.03,
        "baseline_minutes": 30,
        "max_cohort_age_hours": 26,
        "max_data_gap_minutes": 120,
        "initial_cohort_confidence": "LOW",
        "structural_window_hours": 24,
        "structural_min_history": 24,
        "structural_z_threshold": 6.0,
        "structural_score_threshold": 6.0,
        "structural_min_anomalous_fields": 3,
        "structural_mad_epsilon": 1e-9,
        "refresh_prior_tolerance_minutes": 120,
        "refresh_prior_min_samples": 3,
        "structural_min_abs_change": {
            "total_traders": 0.01,
            "long_traders": 0.015,
            "short_traders": 0.015,
            "long_avg_price": 0.005,
            "short_avg_price": 0.005,
            "long_qty_proxy": 0.10,
            "short_qty_proxy": 0.10,
            "long_unrealized_pnl": 0.25,
            "short_unrealized_pnl": 0.25,
            "ls_ratio": 0.05,
        },
        "structural_weights": {
            "total_traders": 3.0,
            "long_traders": 2.0,
            "short_traders": 2.0,
            "long_avg_price": 2.0,
            "short_avg_price": 2.0,
            "long_qty_proxy": 1.0,
            "short_qty_proxy": 1.0,
            "long_unrealized_pnl": 0.5,
            "short_unrealized_pnl": 0.5,
            "ls_ratio": 1.0,
        },
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
        "max_forward_price_delay_minutes": 15,
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
    if pd.isna(pd.to_datetime(config["logic_valid_from"], utc=True, errors="coerce")):
        raise ValueError("logic_valid_from 必须是带时区的有效时间")
    return config


def _out_of_sample_start(config: dict[str, Any]) -> pd.Timestamp:
    return max(
        pd.to_datetime(config["model_freeze_date"], utc=True),
        pd.to_datetime(config["logic_valid_from"], utc=True),
    )


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
    valid_price = frame["current_price"].gt(0)
    frame["long_qty_proxy"] = frame["long_pos_usdt"].div(frame["current_price"]).where(valid_price)
    frame["short_qty_proxy"] = frame["short_pos_usdt"].div(frame["current_price"]).where(valid_price)
    frame["data_quality_flag"], frame["_quality_signal_ok"] = _quality_flags(frame, config)
    return frame


BASELINE_COLUMNS = {
    "price": "current_price",
    "long_pos": "long_pos_usdt",
    "short_pos": "short_pos_usdt",
    "long_qty_proxy": "long_qty_proxy",
    "short_qty_proxy": "short_qty_proxy",
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
    baseline["gross_qty_proxy"] = baseline["long_qty_proxy"] + baseline["short_qty_proxy"]
    baseline["net_qty_proxy"] = baseline["long_qty_proxy"] - baseline["short_qty_proxy"]
    if any(pd.isna(baseline[name]) or baseline[name] <= 0
           for name in (
               "price", "long_pos", "short_pos", "long_qty_proxy", "short_qty_proxy",
               "long_avg_price", "short_avg_price",
           )):
        return None
    return baseline


def _blank_features() -> dict[str, Any]:
    numeric = [
        "minutes_since_cohort_start", "baseline_price", "baseline_long_pos",
        "baseline_short_pos", "baseline_long_avg_price", "baseline_short_avg_price",
        "baseline_long_traders", "baseline_short_traders",
        "baseline_long_unrealized_pnl", "baseline_short_unrealized_pnl",
        "gross_position", "net_exposure", "net_exposure_share", "cohort_net_flow",
        "cohort_net_notional_flow", "gross_qty_proxy", "net_qty_proxy",
        "cohort_net_qty_flow",
        "baseline_long_qty_proxy", "baseline_short_qty_proxy",
        "baseline_gross_qty_proxy", "baseline_net_qty_proxy",
        "long_position_change_pct", "short_position_change_pct",
        "long_qty_change_pct", "short_qty_change_pct",
        "avg_long_position", "avg_short_position", "avg_position_ratio",
        "avg_long_qty_proxy_per_trader", "avg_short_qty_proxy_per_trader",
        "avg_qty_proxy_ratio",
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


def _obsolete_v1_reconstruct_cohorts(raw_frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        result[f"cohort_net_notional_flow_change_{hours}h"] = np.nan
        result[f"cohort_net_qty_flow_change_{hours}h"] = np.nan
        result[f"divergence_{hours}h"] = None
    active = result[result["signal_eligible"] & result["timestamp"].notna()]
    for _, group in active.groupby("cohort_id", sort=False):
        indices = group.index.to_numpy()
        times = group["timestamp"].astype("int64").to_numpy()
        prices = group["current_price"].to_numpy(dtype=float)
        notional_flows = group["cohort_net_notional_flow"].to_numpy(dtype=float)
        qty_flows = group["cohort_net_qty_flow"].to_numpy(dtype=float)
        for hours in cfg["windows_hours"]:
            delta_ns = int(pd.Timedelta(hours=float(hours)).value)
            for position, current_ns in enumerate(times):
                past = int(np.searchsorted(times, current_ns - delta_ns, side="right") - 1)
                if (
                    past < 0 or prices[past] <= 0
                    or not np.isfinite(qty_flows[past])
                    or not np.isfinite(qty_flows[position])
                ):
                    continue
                price_return = prices[position] / prices[past] - 1
                qty_flow_change = qty_flows[position] - qty_flows[past]
                row_index = indices[position]
                result.at[row_index, f"btc_return_{hours}h"] = price_return
                result.at[row_index, f"cohort_net_qty_flow_change_{hours}h"] = qty_flow_change
                if np.isfinite(notional_flows[past]) and np.isfinite(notional_flows[position]):
                    notional_change = notional_flows[position] - notional_flows[past]
                    result.at[row_index, f"cohort_net_flow_change_{hours}h"] = notional_change
                    result.at[row_index, f"cohort_net_notional_flow_change_{hours}h"] = notional_change
                if abs(price_return) <= cfg["flat_price_threshold"] and qty_flow_change >= cfg["flow_threshold"]:
                    result.at[row_index, f"divergence_{hours}h"] = "BULLISH_SM_DIVERGENCE"
                elif abs(price_return) <= cfg["flat_price_threshold"] and qty_flow_change <= -cfg["flow_threshold"]:
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


def _obsolete_v1_build_loss_events(processed: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            "loss_start_position_qty_proxy",
            "loss_start_net_exposure", "loss_duration_minutes",
            "position_change_since_loss", "position_notional_change_since_loss",
            "position_qty_change_since_loss", "avg_entry_change_since_loss",
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


def _obsolete_v1_add_forward_returns(
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


def _obsolete_v1_build_backtest(events: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
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


STRUCTURAL_FIELDS = (
    "total_traders", "long_traders", "short_traders",
    "long_avg_price", "short_avg_price", "long_qty_proxy", "short_qty_proxy",
    "long_unrealized_pnl", "short_unrealized_pnl", "ls_ratio",
)
STRUCTURAL_PRIMARY_FIELDS = {
    "total_traders", "long_traders", "short_traders",
    "long_avg_price", "short_avg_price",
}
STRUCTURAL_TRADER_FIELDS = {"total_traders", "long_traders", "short_traders"}


def _signed_relative_change(current: Any, previous: Any) -> float:
    if pd.isna(current) or pd.isna(previous):
        return math.nan
    denominator = max(abs(float(previous)), 1e-12)
    return (float(current) - float(previous)) / denominator


def _causal_robust_z(
    current_delta: float,
    history: list[tuple[pd.Timestamp, float]],
    timestamp: pd.Timestamp,
    config: dict[str, Any],
) -> float:
    if pd.isna(current_delta):
        return math.nan
    window_start = timestamp - pd.Timedelta(hours=config["structural_window_hours"])
    past = [
        value for past_timestamp, value in history
        if window_start <= past_timestamp < timestamp and not pd.isna(value)
    ]
    if len(past) < int(config["structural_min_history"]):
        return math.nan
    median = float(np.median(past))
    mad = float(np.median(np.abs(np.asarray(past) - median)))
    scale = 1.4826 * mad + float(config["structural_mad_epsilon"])
    return abs(float(current_delta) - median) / scale


def _time_prior_supported(
    timestamp: pd.Timestamp,
    high_refresh_minutes: list[int],
    config: dict[str, Any],
) -> bool:
    minimum = int(config["refresh_prior_min_samples"])
    if minimum == 0:
        return True
    if len(high_refresh_minutes) < minimum:
        return False
    minute = timestamp.hour * 60 + timestamp.minute
    tolerance = float(config["refresh_prior_tolerance_minutes"])
    return any(
        min(abs(minute - historical), 1440 - abs(minute - historical)) <= tolerance
        for historical in high_refresh_minutes
    )


def _v2_blank_features(config: dict[str, Any]) -> dict[str, Any]:
    result = _blank_features()
    result.update({
        "analysis_logic_version": ANALYSIS_LOGIC_VERSION,
        "hard_refresh_trigger": False,
        "structural_refresh_trigger": False,
        "structural_refresh_score": 0.0,
        "structural_anomalous_fields": "",
        "structural_anomalous_field_count": 0,
        "refresh_time_prior_supported": False,
        "cohort_confidence": "LOW",
        "backtest_primary": False,
        "data_gap_detected": False,
        "cohort_expired_now": False,
        "active_refresh_new_segment": False,
        "old_rule_would_reuse_active_segment": False,
        "logic_valid_from": config["logic_valid_from"],
    })
    for field in config["cohort"]["structural_weights"]:
        result[f"structural_z_{field}"] = math.nan
    return result


def reconstruct_cohorts(
    raw_frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Causally reconstruct inferred stable sample windows, never account-level cohorts."""
    if raw_frame.empty:
        return raw_frame.copy(), pd.DataFrame()
    cfg = config["cohort"]
    oos_start = _out_of_sample_start(config)
    feature_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    delta_history: dict[str, list[tuple[pd.Timestamp, float]]] = {
        field: [] for field in cfg["structural_weights"]
    }
    high_refresh_minutes: list[int] = []
    last_refresh_timestamp: pd.Timestamp | None = None
    state = "UNAVAILABLE"
    cohort_id: str | None = None
    confidence = str(cfg["initial_cohort_confidence"]).upper()
    sequence = 0
    cohort_start = refresh_start = refresh_end = stable_at = stable_run_start = None
    baseline_start = baseline_end = None
    baseline_rows: list[dict[str, Any]] = []
    baseline: dict[str, float] | None = None
    prior_detector: pd.Series | None = None
    prior_detector_time: pd.Timestamp | None = None
    prior_timestamp: pd.Timestamp | None = None
    cohort_summary: dict[str, Any] | None = None

    def start_cohort(
        timestamp: pd.Timestamp,
        *,
        detected: bool,
        new_confidence: str,
        score: float,
        reason: str,
        initial_state: str | None = None,
    ) -> None:
        nonlocal state, cohort_id, confidence, sequence, cohort_start
        nonlocal refresh_start, refresh_end, stable_at, stable_run_start
        nonlocal baseline_start, baseline_end, baseline_rows, baseline, cohort_summary
        if cohort_summary is not None:
            cohort_summary["cohort_end"] = prior_timestamp
            cohort_summary["final_state"] = state
            cohort_rows.append(cohort_summary)
        sequence += 1
        cohort_id = f"cohort_{timestamp:%Y%m%d}_{sequence:03d}"
        confidence = new_confidence
        cohort_start = timestamp
        refresh_start = timestamp if detected else None
        refresh_end = stable_at = stable_run_start = baseline_end = None
        baseline_start = timestamp if not detected and initial_state != "UNAVAILABLE" else None
        baseline_rows, baseline = [], None
        state = initial_state or ("REFRESHING" if detected else "BASELINE_BUILDING")
        cohort_summary = {
            "analysis_logic_version": ANALYSIS_LOGIC_VERSION,
            "cohort_id": cohort_id,
            "cohort_definition": "INFERRED_STABLE_SAMPLE_WINDOW",
            "cohort_confidence": confidence,
            "cohort_start": timestamp,
            "cohort_end": pd.NaT,
            "cohort_refresh_start": refresh_start,
            "cohort_refresh_end": pd.NaT,
            "cohort_stable_at": pd.NaT,
            "cohort_baseline_start": baseline_start,
            "cohort_baseline_end": pd.NaT,
            "active_at": pd.NaT,
            "refresh_reason": reason,
            "refresh_score": score,
            "record_count": 0,
            "signal_eligible_records": 0,
            "expired_records": 0,
            "out_of_sample": bool(timestamp >= oos_start),
            "logic_valid_from": config["logic_valid_from"],
            "final_state": state,
        }

    def reset_refresh_in_place(
        timestamp: pd.Timestamp,
        *,
        new_confidence: str,
        score: float,
        reason: str,
    ) -> bool:
        """Merge staged jumps into one refresh episode while rebuilding its baseline."""
        nonlocal state, confidence, refresh_start, refresh_end, stable_at, stable_run_start
        nonlocal baseline_start, baseline_end, baseline_rows, baseline, cohort_summary
        confidence_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        upgraded_to_high = (
            confidence != "HIGH" and confidence_rank[new_confidence] >= confidence_rank["HIGH"]
        )
        if confidence_rank[new_confidence] > confidence_rank[confidence]:
            confidence = new_confidence
        state = "REFRESHING"
        refresh_start = refresh_start or timestamp
        refresh_end = stable_at = stable_run_start = baseline_start = baseline_end = None
        baseline_rows, baseline = [], None
        if cohort_summary is not None:
            cohort_summary.update({
                "cohort_confidence": confidence,
                "refresh_reason": reason,
                "refresh_score": max(float(cohort_summary["refresh_score"]), float(score)),
                "cohort_refresh_start": refresh_start,
                "cohort_refresh_end": pd.NaT,
                "cohort_stable_at": pd.NaT,
                "cohort_baseline_start": pd.NaT,
                "cohort_baseline_end": pd.NaT,
                "active_at": pd.NaT,
            })
        return upgraded_to_high

    for index, row in raw_frame.iterrows():
        timestamp = row["timestamp"]
        output = _v2_blank_features(config)
        if pd.isna(timestamp):
            feature_rows.append(output)
            continue

        gap_detected = bool(
            prior_timestamp is not None
            and (timestamp - prior_timestamp).total_seconds() / 60
            > float(cfg["max_data_gap_minutes"])
        )
        if gap_detected:
            start_cohort(
                timestamp,
                detected=False,
                new_confidence="LOW",
                score=0.0,
                reason="DATA_GAP",
                initial_state="UNAVAILABLE",
            )
            prior_detector = None
            prior_detector_time = None
            for history in delta_history.values():
                history.clear()

        trader_consistent = bool(
            row["total_traders"] > 0
            and row["long_traders"] > 0
            and row["short_traders"] > 0
            and abs(row["total_traders"] - row["long_traders"] - row["short_traders"])
            <= config["quality"]["trader_total_tolerance"]
        )
        deltas = {field: math.nan for field in cfg["structural_weights"]}
        if trader_consistent and prior_detector is not None:
            for field in cfg["structural_weights"]:
                deltas[field] = _signed_relative_change(row[field], prior_detector[field])
        absolute = {field: abs(value) if not pd.isna(value) else math.nan for field, value in deltas.items()}
        total_jump = (
            not pd.isna(absolute["total_traders"])
            and absolute["total_traders"] >= cfg["total_trader_change_threshold"]
        )
        long_jump = (
            not pd.isna(absolute["long_traders"])
            and absolute["long_traders"] >= cfg["side_trader_change_threshold"]
        )
        short_jump = (
            not pd.isna(absolute["short_traders"])
            and absolute["short_traders"] >= cfg["side_trader_change_threshold"]
        )
        hard_trigger = bool(trader_consistent and (total_jump or long_jump or short_jump))

        z_scores = {
            field: _causal_robust_z(deltas[field], delta_history[field], timestamp, cfg)
            for field in cfg["structural_weights"]
        }
        minimum_changes = cfg["structural_min_abs_change"]
        anomalous = [
            field for field, z_value in z_scores.items()
            if (
                not pd.isna(z_value)
                and z_value >= cfg["structural_z_threshold"]
                and not pd.isna(absolute[field])
                and absolute[field] >= float(minimum_changes[field])
            )
        ]
        structural_score = sum(float(cfg["structural_weights"][field]) for field in anomalous)
        trader_anomalies = len(set(anomalous) & STRUCTURAL_TRADER_FIELDS)
        primary_anomalies = len(set(anomalous) & STRUCTURAL_PRIMARY_FIELDS)
        diverse_primary_support = trader_anomalies >= 1 or primary_anomalies >= 2
        structural_trigger = bool(
            not hard_trigger
            and structural_score >= cfg["structural_score_threshold"]
            and len(anomalous) >= int(cfg["structural_min_anomalous_fields"])
            and diverse_primary_support
        )
        prior_supported = _time_prior_supported(timestamp, high_refresh_minutes, cfg)
        refresh_candidate = hard_trigger or structural_trigger
        trigger_confidence = (
            "HIGH" if hard_trigger else "MEDIUM" if structural_trigger and prior_supported else "LOW"
        )
        refresh_score = (
            2 * int(total_jump) + int(long_jump) + int(short_jump) + structural_score
        )
        stable_pair = bool(
            trader_consistent
            and prior_detector is not None
            and absolute["total_traders"] <= cfg["stable_total_trader_change"]
            and absolute["long_traders"] <= cfg["stable_side_trader_change"]
            and absolute["short_traders"] <= cfg["stable_side_trader_change"]
        )

        detected_now = False
        expired_now = False
        active_refresh_new_segment = False
        old_rule_would_reuse_active_segment = False
        if cohort_id is None:
            start_cohort(
                timestamp,
                detected=False,
                new_confidence=confidence,
                score=0.0,
                reason="INITIAL_DATA",
            )
        elif refresh_candidate:
            reason = "HARD_TRADER_JUMP" if hard_trigger else "STRUCTURAL_BREAK"
            if state in {"REFRESHING", "STABILIZING", "BASELINE_BUILDING"}:
                upgraded_to_high = reset_refresh_in_place(
                    timestamp,
                    new_confidence=trigger_confidence,
                    score=refresh_score,
                    reason=reason,
                )
                if upgraded_to_high:
                    high_refresh_minutes.append(timestamp.hour * 60 + timestamp.minute)
            else:
                active_refresh_new_segment = state == "ACTIVE_COHORT"
                old_rule_would_reuse_active_segment = bool(
                    active_refresh_new_segment
                    and last_refresh_timestamp is not None
                    and (timestamp - last_refresh_timestamp).total_seconds() / 3600
                    < OBSOLETE_ACTIVE_REUSE_WINDOW_HOURS
                )
                start_cohort(
                    timestamp,
                    detected=True,
                    new_confidence=trigger_confidence,
                    score=refresh_score,
                    reason=reason,
                )
                last_refresh_timestamp = timestamp
                detected_now = True
                if hard_trigger:
                    high_refresh_minutes.append(timestamp.hour * 60 + timestamp.minute)
        elif (
            state == "ACTIVE_COHORT"
            and cohort_start is not None
            and (timestamp - cohort_start).total_seconds() / 3600
            > float(cfg["max_cohort_age_hours"])
        ):
            state = "COHORT_EXPIRED"
            expired_now = True
            baseline = None
            if cohort_summary is not None:
                cohort_summary["expired_at"] = timestamp
        elif state in {"REFRESHING", "STABILIZING", "UNAVAILABLE"}:
            if not stable_pair:
                if state != "UNAVAILABLE":
                    state = "REFRESHING"
                stable_run_start = None
            else:
                if state in {"REFRESHING", "UNAVAILABLE"} or stable_run_start is None:
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

        reliable = confidence in {"HIGH", "MEDIUM"}
        comparable = bool(
            state == "ACTIVE_COHORT"
            and reliable
            and row["_quality_signal_ok"]
            and baseline is not None
        )
        output.update({
            "cohort_id": cohort_id,
            "analysis_state": state,
            "cohort_confidence": confidence,
            "cohort_refresh_detected": detected_now,
            "cohort_refresh_score": refresh_score,
            "hard_refresh_trigger": hard_trigger,
            "structural_refresh_trigger": structural_trigger,
            "structural_refresh_score": structural_score,
            "structural_anomalous_fields": "|".join(anomalous),
            "structural_anomalous_field_count": len(anomalous),
            "refresh_time_prior_supported": prior_supported,
            "is_refresh_window": state == "REFRESHING",
            "is_stabilization_window": state == "STABILIZING",
            "minutes_since_cohort_start": (
                (timestamp - cohort_start).total_seconds() / 60 if cohort_start else math.nan
            ),
            "cohort_refresh_start": refresh_start,
            "cohort_refresh_end": refresh_end,
            "cohort_stable_at": stable_at,
            "cohort_baseline_start": baseline_start,
            "cohort_baseline_end": baseline_end,
            "signal_eligible": comparable,
            "backtest_primary": comparable,
            "out_of_sample": bool(timestamp >= oos_start),
            "logic_valid_from": config["logic_valid_from"],
            "data_gap_detected": gap_detected,
            "cohort_expired_now": expired_now,
            "active_refresh_new_segment": active_refresh_new_segment,
            "old_rule_would_reuse_active_segment": old_rule_would_reuse_active_segment,
            **{f"structural_z_{field}": value for field, value in z_scores.items()},
        })
        gross = row["long_pos_usdt"] + row["short_pos_usdt"]
        net = row["long_pos_usdt"] - row["short_pos_usdt"]
        long_qty_proxy = row["long_qty_proxy"]
        short_qty_proxy = row["short_qty_proxy"]
        gross_qty_proxy = long_qty_proxy + short_qty_proxy
        net_qty_proxy = long_qty_proxy - short_qty_proxy
        long_return = _safe_ratio(row["current_price"], row["long_avg_price"]) - 1
        short_return = _safe_ratio(row["short_avg_price"], row["current_price"]) - 1
        output.update({
            "gross_position": gross,
            "net_exposure": net,
            "net_exposure_share": _safe_ratio(net, gross) if gross > 0 else math.nan,
            "gross_qty_proxy": gross_qty_proxy,
            "net_qty_proxy": net_qty_proxy,
            "avg_long_position": _safe_ratio(row["long_pos_usdt"], row["long_traders"]),
            "avg_short_position": _safe_ratio(row["short_pos_usdt"], row["short_traders"]),
            "avg_long_qty_proxy_per_trader": _safe_ratio(long_qty_proxy, row["long_traders"]),
            "avg_short_qty_proxy_per_trader": _safe_ratio(short_qty_proxy, row["short_traders"]),
            "long_directional_return": long_return,
            "short_directional_return": short_return,
            "long_loss_depth": max(0.0, -long_return) if not pd.isna(long_return) else math.nan,
            "short_loss_depth": max(0.0, -short_return) if not pd.isna(short_return) else math.nan,
        })
        output["avg_position_ratio"] = _safe_ratio(
            output["avg_long_position"], output["avg_short_position"]
        )
        output["avg_qty_proxy_ratio"] = _safe_ratio(
            output["avg_long_qty_proxy_per_trader"],
            output["avg_short_qty_proxy_per_trader"],
        )
        if comparable and baseline is not None:
            notional_flow = _safe_ratio(
                net - (baseline["long_pos"] - baseline["short_pos"]),
                baseline["long_pos"] + baseline["short_pos"],
            )
            qty_flow = (
                _safe_ratio(
                    net_qty_proxy - baseline["net_qty_proxy"],
                    baseline["gross_qty_proxy"],
                )
                if baseline["gross_qty_proxy"] > 0 else math.nan
            )
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
                "baseline_long_qty_proxy": baseline["long_qty_proxy"],
                "baseline_short_qty_proxy": baseline["short_qty_proxy"],
                "baseline_gross_qty_proxy": baseline["gross_qty_proxy"],
                "baseline_net_qty_proxy": baseline["net_qty_proxy"],
                "cohort_net_flow": notional_flow,
                "cohort_net_notional_flow": notional_flow,
                "cohort_net_qty_flow": qty_flow,
                "long_position_change_pct": _safe_ratio(row["long_pos_usdt"], baseline["long_pos"]) - 1,
                "short_position_change_pct": _safe_ratio(row["short_pos_usdt"], baseline["short_pos"]) - 1,
                "long_qty_change_pct": _safe_ratio(long_qty_proxy, baseline["long_qty_proxy"]) - 1,
                "short_qty_change_pct": _safe_ratio(short_qty_proxy, baseline["short_qty_proxy"]) - 1,
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

        if detected_now:
            old_flag = str(raw_frame.at[index, "data_quality_flag"])
            raw_frame.at[index, "data_quality_flag"] = (
                "COHORT_REFRESH" if old_flag == "OK" else f"{old_flag}|COHORT_REFRESH"
            )
        if cohort_summary is not None:
            cohort_summary["record_count"] += 1
            cohort_summary["signal_eligible_records"] += int(comparable)
            cohort_summary["expired_records"] += int(state == "COHORT_EXPIRED")
            cohort_summary["final_state"] = state
        feature_rows.append(output)

        if trader_consistent:
            if prior_detector is not None:
                for field in cfg["structural_weights"]:
                    if not pd.isna(deltas[field]):
                        delta_history[field].append((timestamp, deltas[field]))
                    cutoff = timestamp - pd.Timedelta(hours=cfg["structural_window_hours"])
                    delta_history[field] = [
                        item for item in delta_history[field] if item[0] >= cutoff
                    ]
            prior_detector, prior_detector_time = row, timestamp
        prior_timestamp = timestamp

    if cohort_summary is not None:
        cohort_summary["cohort_end"] = prior_timestamp
        cohort_summary["final_state"] = state
        cohort_rows.append(cohort_summary)
    processed = pd.concat([raw_frame.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)
    return processed, pd.DataFrame(cohort_rows)


def build_loss_event_tables(
    processed: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build causal loss episodes and first-threshold action events."""
    result = processed.copy()
    cfg = config["loss"]
    for side in ("long", "short"):
        result[f"{side}_new_loss_event"] = False
        result[f"{side}_recovery_event"] = False
        result[f"{side}_action_event"] = False
        result[f"{side}_action_type"] = None
        result[f"{side}_action_event_id"] = None
        result[f"{side}_loss_event_id"] = None
        result[f"{side}_loss_start_timestamp"] = pd.Series(
            pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
        )
        for suffix in [
            "loss_start_price", "loss_start_position",
            "loss_start_position_qty_proxy", "loss_start_avg_entry",
            "loss_start_net_exposure", "loss_duration_minutes",
            "position_change_since_loss", "position_notional_change_since_loss",
            "position_qty_change_since_loss", "avg_entry_change_since_loss",
        ]:
            result[f"{side}_{suffix}"] = np.nan
        result[f"{side}_loss_response"] = None
    result["event_markers"] = ""

    active: dict[str, dict[str, Any] | None] = {"long": None, "short": None}
    cooldown_until: dict[str, pd.Timestamp | None] = {"long": None, "short": None}
    counters: defaultdict[tuple[str, str], int] = defaultdict(int)
    episodes: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    previous_cohort: str | None = None
    last_active_row: dict[str, pd.Series | None] = {"long": None, "short": None}

    def finalize(
        side: str,
        episode: dict[str, Any],
        row: pd.Series | None,
        *,
        recovered: bool,
        reason: str,
    ) -> None:
        if row is not None:
            timestamp = row["timestamp"]
            position = row[f"{side}_pos_usdt"]
            qty_proxy = row[f"{side}_qty_proxy"]
            notional_change = _safe_ratio(position, episode["loss_start_position"]) - 1
            qty_change = _safe_ratio(
                qty_proxy, episode["loss_start_position_qty_proxy"]
            ) - 1
            episode["episode_end_timestamp"] = timestamp
            episode["loss_end_timestamp"] = timestamp
            episode["loss_end_price"] = row["current_price"]
            episode["loss_end_avg_entry"] = row[f"{side}_avg_price"]
            episode["cohort_net_flow_at_end"] = row["cohort_net_flow"]
            episode["cohort_net_notional_flow_at_end"] = row["cohort_net_notional_flow"]
            episode["cohort_net_qty_flow_at_end"] = row["cohort_net_qty_flow"]
            episode["final_response"] = _response(qty_change, cfg)
            episode["max_loss_duration"] = max(
                episode["max_loss_duration"],
                (timestamp - episode["loss_start_timestamp"]).total_seconds() / 60,
            )
        episode["episode_end_reason"] = reason
        episode["event_status"] = "RECOVERED" if recovered else reason
        episode["recovered"] = recovered
        episode["action_sequence"] = "→".join(episode.pop("_action_order")) or "NONE"
        episode.pop("_seen_add", None)
        episode.pop("_seen_reduce", None)
        episode["loss_depth_bucket"] = _loss_depth_bucket(
            episode["max_loss_depth"], cfg["depth_bucket_edges"]
        )
        episode["loss_duration_bucket"] = _duration_bucket(
            episode["max_loss_duration"]
        )
        episode["position_response_bucket"] = _position_bucket(
            episode["max_qty_add_pct"]
            if episode["final_response"] == "ADD"
            else episode["max_qty_reduce_pct"]
            if episode["final_response"] == "REDUCE"
            else 0.0
        )
        episodes.append(episode)

    for index, row in result.iterrows():
        timestamp, cohort_id = row["timestamp"], row["cohort_id"]
        if pd.isna(timestamp):
            continue
        if previous_cohort is not None and cohort_id != previous_cohort:
            for side in ("long", "short"):
                if active[side] is not None:
                    finalize(
                        side, active[side], last_active_row[side],
                        recovered=False, reason="COHORT_CHANGED",
                    )
                    active[side] = None
                cooldown_until[side] = None
                last_active_row[side] = None
        previous_cohort = cohort_id
        markers: list[str] = []

        if not bool(row["signal_eligible"]):
            if row["analysis_state"] in {
                "REFRESHING", "STABILIZING", "BASELINE_BUILDING",
                "COHORT_EXPIRED", "UNAVAILABLE",
            }:
                for side in ("long", "short"):
                    if active[side] is not None:
                        finalize(
                            side, active[side], last_active_row[side],
                            recovered=False, reason=str(row["analysis_state"]),
                        )
                        active[side] = None
            continue

        for side in ("long", "short"):
            directional_return = row[f"{side}_directional_return"]
            position = row[f"{side}_pos_usdt"]
            qty_proxy = row[f"{side}_qty_proxy"]
            avg_entry = row[f"{side}_avg_price"]
            baseline_return = (
                _safe_ratio(row["baseline_price"], row["baseline_long_avg_price"]) - 1
                if side == "long"
                else _safe_ratio(row["baseline_short_avg_price"], row["baseline_price"]) - 1
            )
            episode = active[side]
            if episode is None:
                clean = (
                    not bool(row[f"{side}_existing_loss_at_cohort_start"])
                    and baseline_return >= cfg["clean_baseline_threshold"]
                )
                cooled_down = (
                    cooldown_until[side] is None or timestamp >= cooldown_until[side]
                )
                if clean and cooled_down and directional_return <= cfg["loss_entry_threshold"]:
                    date_key = timestamp.strftime("%Y%m%d")
                    counters[(side, date_key)] += 1
                    episode_id = (
                        f"{side.upper()}_{date_key}_{counters[(side, date_key)]:03d}"
                    )
                    episode = {
                        "analysis_logic_version": ANALYSIS_LOGIC_VERSION,
                        "event_family": "LOSS_EPISODE",
                        "event_id": episode_id,
                        "episode_id": episode_id,
                        "event_type": f"{side.upper()}_NEW_LOSS",
                        "event_timestamp": timestamp,
                        "event_price": row["current_price"],
                        "cohort_id": cohort_id,
                        "cohort_confidence": row["cohort_confidence"],
                        "side": side.upper(),
                        "loss_start_timestamp": timestamp,
                        "loss_start_price": row["current_price"],
                        "loss_start_position": position,
                        "loss_start_position_qty_proxy": qty_proxy,
                        "loss_start_avg_entry": avg_entry,
                        "loss_start_net_exposure": row["net_exposure"],
                        "max_position": position,
                        "min_position": position,
                        "max_position_qty_proxy": qty_proxy,
                        "min_position_qty_proxy": qty_proxy,
                        "max_loss_depth": row[f"{side}_loss_depth"],
                        "max_loss_duration": 0.0,
                        "max_position_add_pct": 0.0,
                        "max_position_reduce_pct": 0.0,
                        "max_qty_add_pct": 0.0,
                        "max_qty_reduce_pct": 0.0,
                        "first_add_timestamp": pd.NaT,
                        "first_add_price": np.nan,
                        "first_add_position": np.nan,
                        "first_add_position_qty_proxy": np.nan,
                        "first_add_avg_entry": np.nan,
                        "first_add_loss_depth": np.nan,
                        "first_add_loss_duration": np.nan,
                        "first_add_net_exposure": np.nan,
                        "first_add_cohort_net_flow": np.nan,
                        "first_add_cohort_net_qty_flow": np.nan,
                        "first_reduce_timestamp": pd.NaT,
                        "first_reduce_price": np.nan,
                        "first_reduce_position": np.nan,
                        "first_reduce_position_qty_proxy": np.nan,
                        "first_reduce_avg_entry": np.nan,
                        "first_reduce_loss_depth": np.nan,
                        "first_reduce_loss_duration": np.nan,
                        "first_reduce_net_exposure": np.nan,
                        "first_reduce_cohort_net_flow": np.nan,
                        "first_reduce_cohort_net_qty_flow": np.nan,
                        "recovery_timestamp": pd.NaT,
                        "recovery_price": np.nan,
                        "episode_end_timestamp": pd.NaT,
                        "episode_end_reason": "OPEN",
                        "loss_end_timestamp": pd.NaT,
                        "loss_end_price": np.nan,
                        "loss_end_avg_entry": np.nan,
                        "cohort_net_flow_at_start": row["cohort_net_flow"],
                        "cohort_net_flow_at_end": np.nan,
                        "cohort_net_notional_flow_at_start": row["cohort_net_notional_flow"],
                        "cohort_net_notional_flow_at_end": np.nan,
                        "cohort_net_qty_flow_at_start": row["cohort_net_qty_flow"],
                        "cohort_net_qty_flow_at_end": np.nan,
                        "final_response": "HOLD",
                        "final_response_note": "DESCRIPTIVE_ONLY_QUANTITY_BASED_NOT_FOR_CAUSAL_BACKTEST",
                        "recovered": False,
                        "event_status": "OPEN",
                        "out_of_sample": bool(row["out_of_sample"]),
                        "_action_order": [],
                        "_seen_add": False,
                        "_seen_reduce": False,
                    }
                    active[side] = episode
                    result.at[index, f"{side}_new_loss_event"] = True
                    markers.append(f"{side.upper()}_NEW_LOSS")

            if episode is not None:
                duration = (
                    timestamp - episode["loss_start_timestamp"]
                ).total_seconds() / 60
                notional_change = (
                    _safe_ratio(position, episode["loss_start_position"]) - 1
                )
                qty_change = (
                    _safe_ratio(qty_proxy, episode["loss_start_position_qty_proxy"]) - 1
                )
                avg_entry_change = (
                    _safe_ratio(avg_entry, episode["loss_start_avg_entry"]) - 1
                )
                response = _response(qty_change, cfg)
                episode["max_position"] = max(episode["max_position"], position)
                episode["min_position"] = min(episode["min_position"], position)
                episode["max_position_qty_proxy"] = max(
                    episode["max_position_qty_proxy"], qty_proxy
                )
                episode["min_position_qty_proxy"] = min(
                    episode["min_position_qty_proxy"], qty_proxy
                )
                episode["max_loss_depth"] = max(
                    episode["max_loss_depth"], row[f"{side}_loss_depth"]
                )
                episode["max_loss_duration"] = max(
                    episode["max_loss_duration"], duration
                )
                episode["max_position_add_pct"] = max(
                    episode["max_position_add_pct"], notional_change
                )
                episode["max_position_reduce_pct"] = min(
                    episode["max_position_reduce_pct"], notional_change
                )
                episode["max_qty_add_pct"] = max(
                    episode["max_qty_add_pct"], qty_change
                )
                episode["max_qty_reduce_pct"] = min(
                    episode["max_qty_reduce_pct"], qty_change
                )
                episode["final_response"] = response
                result.at[index, f"{side}_loss_event_id"] = episode["episode_id"]
                result.at[index, f"{side}_loss_start_timestamp"] = episode["loss_start_timestamp"]
                result.at[index, f"{side}_loss_start_price"] = episode["loss_start_price"]
                result.at[index, f"{side}_loss_start_position"] = episode["loss_start_position"]
                result.at[index, f"{side}_loss_start_position_qty_proxy"] = episode["loss_start_position_qty_proxy"]
                result.at[index, f"{side}_loss_start_avg_entry"] = episode["loss_start_avg_entry"]
                result.at[index, f"{side}_loss_start_net_exposure"] = episode["loss_start_net_exposure"]
                result.at[index, f"{side}_loss_duration_minutes"] = duration
                result.at[index, f"{side}_position_change_since_loss"] = notional_change
                result.at[index, f"{side}_position_notional_change_since_loss"] = notional_change
                result.at[index, f"{side}_position_qty_change_since_loss"] = qty_change
                result.at[index, f"{side}_avg_entry_change_since_loss"] = avg_entry_change
                result.at[index, f"{side}_loss_response"] = response

                for action_type, threshold_met in (
                    ("ADD", qty_change >= cfg["add_threshold"]),
                    ("REDUCE", qty_change <= cfg["reduce_threshold"]),
                ):
                    seen_key = f"_seen_{action_type.lower()}"
                    if not episode[seen_key] and threshold_met:
                        episode[seen_key] = True
                        episode["_action_order"].append(action_type)
                        action_event_id = f"{episode['episode_id']}_{action_type}"
                        action = {
                            "analysis_logic_version": ANALYSIS_LOGIC_VERSION,
                            "event_family": "ACTION_EVENT",
                            "event_id": action_event_id,
                            "action_event_id": action_event_id,
                            "episode_id": episode["episode_id"],
                            "cohort_id": cohort_id,
                            "cohort_confidence": row["cohort_confidence"],
                            "side": side.upper(),
                            "event_type": f"{side.upper()}_LOSS_{action_type}",
                            "action_type": action_type,
                            "event_timestamp": timestamp,
                            "event_price": row["current_price"],
                            "action_timestamp": timestamp,
                            "action_price": row["current_price"],
                            "action_position": position,
                            "action_position_qty_proxy": qty_proxy,
                            "action_avg_entry": avg_entry,
                            "loss_start_position": episode["loss_start_position"],
                            "loss_start_position_qty_proxy": episode["loss_start_position_qty_proxy"],
                            "position_change_since_loss": notional_change,
                            "position_notional_change_since_loss": notional_change,
                            "position_qty_change_since_loss": qty_change,
                            "avg_entry_change_since_loss": avg_entry_change,
                            "loss_depth_at_action": row[f"{side}_loss_depth"],
                            "loss_duration_at_action": duration,
                            "cohort_net_flow_at_action": row["cohort_net_flow"],
                            "cohort_net_notional_flow_at_action": row["cohort_net_notional_flow"],
                            "cohort_net_qty_flow_at_action": row["cohort_net_qty_flow"],
                            "net_exposure_at_action": row["net_exposure"],
                            "net_qty_proxy_at_action": row["net_qty_proxy"],
                            "loss_start_timestamp": episode["loss_start_timestamp"],
                            "loss_start_price": episode["loss_start_price"],
                            "out_of_sample": bool(row["out_of_sample"]),
                        }
                        actions.append(action)
                        prefix = "first_add" if action_type == "ADD" else "first_reduce"
                        episode[f"{prefix}_timestamp"] = timestamp
                        episode[f"{prefix}_price"] = row["current_price"]
                        episode[f"{prefix}_position"] = position
                        episode[f"{prefix}_position_qty_proxy"] = qty_proxy
                        episode[f"{prefix}_avg_entry"] = avg_entry
                        episode[f"{prefix}_loss_depth"] = row[f"{side}_loss_depth"]
                        episode[f"{prefix}_loss_duration"] = duration
                        episode[f"{prefix}_net_exposure"] = row["net_exposure"]
                        episode[f"{prefix}_cohort_net_flow"] = row["cohort_net_flow"]
                        episode[f"{prefix}_cohort_net_qty_flow"] = row["cohort_net_qty_flow"]
                        result.at[index, f"{side}_action_event"] = True
                        result.at[index, f"{side}_action_type"] = action_type
                        result.at[index, f"{side}_action_event_id"] = action_event_id
                        markers.append(f"{side.upper()}_LOSS_{action_type}")

                if directional_return >= cfg["recovery_threshold"]:
                    result.at[index, f"{side}_recovery_event"] = True
                    markers.append(f"{side.upper()}_RECOVERY")
                    episode["recovery_timestamp"] = timestamp
                    episode["recovery_price"] = row["current_price"]
                    recoveries.append({
                        "analysis_logic_version": ANALYSIS_LOGIC_VERSION,
                        "event_family": "RECOVERY_EVENT",
                        "event_id": f"{episode['episode_id']}_RECOVERY",
                        "episode_id": episode["episode_id"],
                        "cohort_id": cohort_id,
                        "cohort_confidence": row["cohort_confidence"],
                        "side": side.upper(),
                        "event_type": f"{side.upper()}_RECOVERY",
                        "event_timestamp": timestamp,
                        "event_price": row["current_price"],
                        "recovery_timestamp": timestamp,
                        "recovery_price": row["current_price"],
                        "out_of_sample": bool(row["out_of_sample"]),
                    })
                    finalize(
                        side, episode, row, recovered=True, reason="RECOVERED"
                    )
                    active[side] = None
                    cooldown_until[side] = timestamp + pd.Timedelta(
                        minutes=cfg["cooldown_minutes"]
                    )
            last_active_row[side] = row
        result.at[index, "event_markers"] = "|".join(markers)

    for side in ("long", "short"):
        if active[side] is not None:
            finalize(
                side, active[side], last_active_row[side],
                recovered=False, reason="DATA_END",
            )
    episode_frame = pd.DataFrame(episodes)
    action_frame = pd.DataFrame(actions)
    recovery_frame = pd.DataFrame(recoveries)
    combined = pd.concat(
        [episode_frame, action_frame, recovery_frame],
        ignore_index=True,
        sort=False,
    )
    if not combined.empty:
        combined = combined.sort_values(
            ["event_timestamp", "event_family", "event_id"], kind="stable"
        ).reset_index(drop=True)
    return result, episode_frame, action_frame, combined


def build_loss_events(
    processed: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper; causal users should use build_loss_event_tables."""
    result, _, _, events = build_loss_event_tables(processed, config)
    return result, events


def build_action_definition_comparison(
    episodes: pd.DataFrame,
    actions: pd.DataFrame,
    processed: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compare obsolete USDT-notional actions with current quantity-proxy actions."""
    columns = [
        "analysis_logic_version", "side", "episode_id", "action_type",
        "loss_start_timestamp", "old_notional_action_timestamp",
        "new_qty_action_timestamp", "timestamp_difference_minutes",
        "comparison_status",
    ]
    if episodes.empty:
        return pd.DataFrame(columns=columns)
    action_times = {
        (str(row["episode_id"]), str(row["action_type"])): row["action_timestamp"]
        for _, row in actions.iterrows()
    } if not actions.empty else {}
    rows: list[dict[str, Any]] = []
    for _, episode in episodes.iterrows():
        side = str(episode["side"])
        side_lower = side.lower()
        start = episode["loss_start_timestamp"]
        end = episode["episode_end_timestamp"]
        timeline = processed[
            processed["cohort_id"].eq(episode["cohort_id"])
            & processed["signal_eligible"]
            & processed["timestamp"].ge(start)
            & processed["timestamp"].le(end)
        ]
        notional_change = (
            timeline[f"{side_lower}_pos_usdt"]
            .div(float(episode["loss_start_position"]))
            .sub(1.0)
        )
        for action_type, mask in (
            ("ADD", notional_change.ge(config["loss"]["add_threshold"])),
            ("REDUCE", notional_change.le(config["loss"]["reduce_threshold"])),
        ):
            matching = timeline.loc[mask, "timestamp"]
            old_timestamp = matching.iloc[0] if not matching.empty else pd.NaT
            new_timestamp = action_times.get(
                (str(episode["episode_id"]), action_type), pd.NaT
            )
            if pd.isna(old_timestamp) and pd.isna(new_timestamp):
                status = "NEITHER"
                difference = math.nan
            elif not pd.isna(old_timestamp) and pd.isna(new_timestamp):
                status = "DISAPPEARED"
                difference = math.nan
            elif pd.isna(old_timestamp) and not pd.isna(new_timestamp):
                status = "NEW"
                difference = math.nan
            else:
                difference = (new_timestamp - old_timestamp).total_seconds() / 60
                status = "EXACT_MATCH" if difference == 0 else "TIMESTAMP_CHANGED"
            rows.append({
                "analysis_logic_version": ANALYSIS_LOGIC_VERSION,
                "side": side,
                "episode_id": episode["episode_id"],
                "action_type": action_type,
                "loss_start_timestamp": start,
                "old_notional_action_timestamp": old_timestamp,
                "new_qty_action_timestamp": new_timestamp,
                "timestamp_difference_minutes": difference,
                "comparison_status": status,
            })
    return pd.DataFrame(rows, columns=columns)


def add_forward_returns(
    events: pd.DataFrame,
    processed: pd.DataFrame,
    config: dict[str, Any],
    event_timestamp_col: str = "event_timestamp",
    event_price_col: str = "event_price",
) -> pd.DataFrame:
    result = events.copy()
    horizons = config["backtest"]["forward_hours"]
    for hours in horizons:
        result[f"forward_return_{hours}h"] = np.nan
        result[f"forward_price_timestamp_{hours}h"] = pd.Series(
            pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
        )
        result[f"forward_price_delay_minutes_{hours}h"] = np.nan
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
    tolerance = float(config["backtest"]["max_forward_price_delay_minutes"])
    for index, event in result.iterrows():
        event_timestamp = event.get(event_timestamp_col)
        event_price = event.get(event_price_col)
        if pd.isna(event_timestamp) or pd.isna(event_price) or float(event_price) <= 0:
            continue
        for hours in horizons:
            target = event_timestamp + pd.Timedelta(hours=hours)
            future_index = int(np.searchsorted(timestamps, target.value, side="left"))
            if future_index >= len(timestamps):
                continue
            observed_at = pd.Timestamp(timestamps[future_index], tz="UTC")
            delay = (observed_at - target).total_seconds() / 60
            if not 0 <= delay <= tolerance:
                continue
            result.at[index, f"forward_return_{hours}h"] = (
                price_values[future_index] / float(event_price) - 1
            )
            result.at[index, f"forward_price_timestamp_{hours}h"] = observed_at
            result.at[index, f"forward_price_delay_minutes_{hours}h"] = delay
    return result


def _forward_returns_for_controls(
    controls: pd.DataFrame,
    processed: pd.DataFrame,
    hours: float,
    tolerance_minutes: float,
) -> np.ndarray:
    prices = (
        processed.loc[
            processed["timestamp"].notna() & processed["current_price"].gt(0),
            ["timestamp", "current_price"],
        ]
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
    )
    timestamps = prices["timestamp"].astype("int64").to_numpy()
    values = prices["current_price"].to_numpy(dtype=float)
    returns: list[float] = []
    for _, row in controls.iterrows():
        target = row["timestamp"] + pd.Timedelta(hours=hours)
        future_index = int(np.searchsorted(timestamps, target.value, side="left"))
        if future_index >= len(timestamps):
            continue
        observed_at = pd.Timestamp(timestamps[future_index], tz="UTC")
        delay = (observed_at - target).total_seconds() / 60
        if 0 <= delay <= tolerance_minutes and row["current_price"] > 0:
            returns.append(values[future_index] / row["current_price"] - 1)
    return np.asarray(returns, dtype=float)


def build_backtest(
    events: pd.DataFrame, processed: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    columns = [
        "analysis_logic_version", "dataset", "sensitivity", "event_type",
        "segment", "segment_value", "horizon", "sample_count",
        "mean_forward_return", "median_forward_return", "win_rate", "p25", "p75",
        "bootstrap_ci_low", "bootstrap_ci_high", "unconditional_return",
        "difference_vs_unconditional", "directional_success_rate",
        "sample_size_warning",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    research = events[
        events["event_family"].isin(["LOSS_EPISODE", "ACTION_EVENT"])
    ].copy()
    if research.empty:
        return pd.DataFrame(columns=columns)
    rng = np.random.default_rng(config["backtest"]["random_seed"])
    rows: list[dict[str, Any]] = []
    sensitivities = [
        ("conservative_high_only", {"HIGH"}),
        ("standard_high_medium", {"HIGH", "MEDIUM"}),
    ]
    datasets = [
        ("exploratory_historical", False),
        ("out_of_sample", True),
    ]
    all_event_times = set(research["event_timestamp"].dropna())
    for dataset_name, is_oos in datasets:
        for sensitivity, allowed_confidence in sensitivities:
            selected = research[
                (research["out_of_sample"] == is_oos)
                & research["cohort_confidence"].isin(allowed_confidence)
            ]
            for event_type, group in selected.groupby("event_type", sort=True):
                cohort_ids = set(group["cohort_id"].dropna())
                controls = processed[
                    (processed["signal_eligible"])
                    & (processed["out_of_sample"] == is_oos)
                    & processed["cohort_confidence"].isin(allowed_confidence)
                    & processed["cohort_id"].isin(cohort_ids)
                    & ~processed["timestamp"].isin(all_event_times)
                ]
                for hours in config["backtest"]["forward_hours"]:
                    values = group[f"forward_return_{hours}h"].dropna().to_numpy(dtype=float)
                    low, high = _bootstrap_mean_ci(
                        values, config["backtest"]["bootstrap_samples"], rng
                    )
                    control_values = _forward_returns_for_controls(
                        controls,
                        processed,
                        hours,
                        config["backtest"]["max_forward_price_delay_minutes"],
                    )
                    baseline_mean = (
                        float(np.mean(control_values)) if len(control_values) else math.nan
                    )
                    event_mean = float(np.mean(values)) if len(values) else math.nan
                    directional_success = math.nan
                    if event_type == "LONG_LOSS_ADD" and len(values):
                        directional_success = float(np.mean(values > 0))
                    elif event_type == "SHORT_LOSS_ADD" and len(values):
                        directional_success = float(np.mean(values < 0))
                    sample_warning = (
                        "VERY_SMALL" if len(values) < 20
                        else "SMALL" if len(values) < 50
                        else "ADEQUATE"
                    )
                    rows.append({
                        "analysis_logic_version": ANALYSIS_LOGIC_VERSION,
                        "dataset": dataset_name,
                        "sensitivity": sensitivity,
                        "event_type": event_type,
                        "segment": "overall",
                        "segment_value": "all",
                        "horizon": f"{hours}h",
                        "sample_count": len(values),
                        "mean_forward_return": event_mean,
                        "median_forward_return": (
                            float(np.median(values)) if len(values) else math.nan
                        ),
                        "win_rate": float(np.mean(values > 0)) if len(values) else math.nan,
                        "p25": float(np.quantile(values, 0.25)) if len(values) else math.nan,
                        "p75": float(np.quantile(values, 0.75)) if len(values) else math.nan,
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "unconditional_return": baseline_mean,
                        "difference_vs_unconditional": (
                            event_mean - baseline_mean
                            if not pd.isna(event_mean) and not pd.isna(baseline_mean)
                            else math.nan
                        ),
                        "directional_success_rate": directional_success,
                        "sample_size_warning": sample_warning,
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
        "episode_end_reason": STATUS_ZH,
        "action_type": RESPONSE_ZH,
        "cohort_confidence": {"HIGH": "高", "MEDIUM": "中", "LOW": "低"},
        "event_family": {
            "LOSS_EPISODE": "新亏损过程",
            "ACTION_EVENT": "首次动作事件",
            "RECOVERY_EVENT": "恢复盈利事件",
        },
        "sensitivity": {
            "conservative_high_only": "保守口径（仅高可信批次）",
            "standard_high_medium": "标准口径（高+中可信批次）",
        },
        "sample_size_warning": {
            "VERY_SMALL": "样本极少",
            "SMALL": "样本偏少",
            "ADEQUATE": "样本量尚可",
        },
        "cohort_definition": {
            "INFERRED_STABLE_SAMPLE_WINDOW": "推断的稳定样本窗口",
        },
        "refresh_reason": {
            "INITIAL_DATA": "历史数据起点",
            "HARD_TRADER_JUMP": "交易者人数明显跳变",
            "STRUCTURAL_BREAK": "多字段结构突变",
            "DATA_GAP": "长时间数据中断",
        },
        "action_sequence": {
            "ADD": "加仓",
            "REDUCE": "减仓",
            "ADD→REDUCE": "先加仓→后减仓",
            "REDUCE→ADD": "先减仓→后加仓",
            "NONE": "无明显动作",
        },
        "comparison_status": {
            "EXACT_MATCH": "新旧动作时间完全一致",
            "DISAPPEARED": "旧名义动作在数量口径下消失",
            "NEW": "数量口径新增动作",
            "TIMESTAMP_CHANGED": "新旧动作触发时间改变",
            "NEITHER": "两种口径均未触发",
        },
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
            result[column] = result[column].map(lambda value: mapping.get(value, value))
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
    for column in (
        "event_id", "episode_id", "action_event_id",
        "long_loss_event_id", "short_loss_event_id",
        "long_action_event_id", "short_action_event_id",
    ):
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
    episodes = (
        events[events["event_family"] == "LOSS_EPISODE"]
        if not events.empty else pd.DataFrame()
    )
    lines = [
        "# 聪明钱推断名单批次分析报告",
        "",
        "## 研究边界与逻辑版本",
        "",
        f"- 分析逻辑版本：{summary['analysis_logic_version']}",
        "- 这里的批次是“推断的稳定样本窗口”，不是可追踪账户组成的真实固定名单。",
        "- 币安没有提供可完整跨日追踪的聪明钱账户身份集合，因此系统宁可放弃不确定数据，也不假设名单连续。",
        "- v1 动作回测已作废：旧分类使用未来动作，却从亏损开始时计算收益。",
        "- 旧 2.0 动作定义也已作废：直接使用 USDT 名义仓位会受到 BTC 价格机械变化污染。",
        f"- 原始记录数：{summary['raw_records']:,}",
        f"- 有效记录数：{summary['valid_records']:,}",
        f"- 无效记录数：{summary['invalid_records']:,}",
        f"- 推断名单批次数：{summary['cohorts_detected']}",
        f"- 高可信批次：{summary['high_cohorts']}",
        f"- 中可信批次：{summary['medium_cohorts']}",
        f"- 低可信批次：{summary['low_cohorts']}（默认不进入研究信号）",
        f"- 批次过期状态记录数：{summary['expired_period_records']}",
        f"- 名单刷新次数：{summary['refresh_windows']}",
        f"- 模型冻结时间：{summary['model_freeze_date']}",
        f"- 2.1 修正版逻辑生效时间：{summary['logic_valid_from']}",
        f"- 样本外事件数：{summary['oos_events']}",
        "- 只有同时晚于模型冻结时间和修正版逻辑生效时间的数据，才属于当前逻辑的样本外验证。",
        "",
        "## 为什么不能直接用USDT仓位金额判断加仓",
        "",
        "- USDT 名义仓位同时受到真实风险数量和 BTC 价格影响；即使持有数量完全不变，BTC 上涨也会机械推高名义金额。",
        "- 动作事件现在使用“USDT 名义仓位 ÷ 当时 BTC 价格”得到 BTC 等价数量代理，再判断是否越过 ±3% 门槛。",
        "- 该数量只是从聚合名义金额反推的 exposure proxy，不是 Binance 提供的账户级精确合约张数。",
        "- USDT 名义仓位和名义净流继续保留用于展示，但不再触发加仓、减仓、结构刷新或主要背离信号。",
        "",
        "## 批次分段与新旧动作定义对照",
        "",
        f"- ACTIVE 后检测到刷新并强制新建批次：{summary['active_refresh_new_segments']} 次",
        f"- 其中旧 18 小时复用规则本会错误沿用旧编号：{summary['old_rule_active_segment_reuses']} 次",
        f"- 旧名义 ADD：{summary['old_notional_action_counts'].get('ADD', 0)}；新数量 ADD：{summary['new_qty_action_counts'].get('ADD', 0)}",
        f"- 旧名义 REDUCE：{summary['old_notional_action_counts'].get('REDUCE', 0)}；新数量 REDUCE：{summary['new_qty_action_counts'].get('REDUCE', 0)}",
        f"- 完全一致：{summary['action_comparison_counts'].get('EXACT_MATCH', 0)}；消失：{summary['action_comparison_counts'].get('DISAPPEARED', 0)}；新增：{summary['action_comparison_counts'].get('NEW', 0)}；时间改变：{summary['action_comparison_counts'].get('TIMESTAMP_CHANGED', 0)}",
        *[
            f"- 时间改变示例：{example['side']} {example['episode_id']} {example['action_type']}；"
            f"旧名义时间 {example['old_notional_action_timestamp']}；"
            f"新数量时间 {example['new_qty_action_timestamp']}；"
            f"相差 {example['timestamp_difference_minutes']:.2f} 分钟"
            for example in summary["action_timestamp_changed_examples"]
        ],
        "",
        "## 数据质量",
        "",
    ]
    for flag, count in summary["quality_counts"].items():
        lines.append(f"- {QUALITY_ZH.get(flag, flag)}：{count}")
    lines.extend([
        "",
        "## 新亏损过程与首次动作事件",
        "",
        f"- 多头新亏损事件：{summary['new_long_loss_events']}",
        f"- 空头新亏损事件：{summary['new_short_loss_events']}",
    ])
    for key, count in summary["response_counts"].items():
        side, response = key.split("_", 1)
        lines.append(f"- {SIDE_ZH.get(side, side)}{RESPONSE_ZH.get(response, response)}：{count}")
    recovered = int(episodes["recovered"].sum()) if not episodes.empty else 0
    lines.extend([
        f"- 已恢复盈利：{recovered}",
        f"- 尚未恢复或因批次变化结束：{len(episodes) - recovered}",
        "",
        "## 历史探索回测（因果事件时点）",
        "",
        "- 新亏损收益从首次进入新亏损的时刻开始。",
        "- 加仓/减仓由 BTC 等价数量代理第一次越过动作阈值触发，收益从该动作时刻开始。",
        "- 未来目标价格只接受目标时刻之后 15 分钟内的第一条记录。",
        "- 空头事件仍保存比特币原始收益，没有乘以 -1。",
        "",
        "| 可信度口径 | 事件 | 周期 | 样本 | 平均收益 | 中位数 | 正收益率 | 无条件基准 | 超额收益 | 方向正确率 | 95%置信区间 | 样本提示 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
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
        warning_label = {
            "VERY_SMALL": "样本极少",
            "SMALL": "样本偏少",
            "ADEQUATE": "样本量尚可",
        }.get(row["sample_size_warning"], row["sample_size_warning"])
        lines.append(
            f"| {'仅高可信' if row['sensitivity'] == 'conservative_high_only' else '高+中可信'} "
            f"| {EVENT_TYPE_ZH.get(row['event_type'], row['event_type'])} "
            f"| {str(row['horizon']).replace('h', '小时')} | {int(row['sample_count'])} "
            f"| {_format_percent(row['mean_forward_return'])} "
            f"| {_format_percent(row['median_forward_return'])} "
            f"| {_format_percent(row['win_rate'])} "
            f"| {_format_percent(row['unconditional_return'])} "
            f"| {_format_percent(row['difference_vs_unconditional'])} "
            f"| {_format_percent(row['directional_success_rate'])} "
            f"| {interval} "
            f"| {warning_label} |"
        )
    lines.extend([
        "",
        "## 高可信刷新时间（协调世界时）",
        "",
        *[f"- {timestamp}" for timestamp in summary["high_refresh_timestamps"]],
        "",
        "## 中可信刷新时间（协调世界时）",
        "",
        *[f"- {timestamp}" for timestamp in summary["medium_refresh_timestamps"]],
        "",
        "## 高可信刷新小时分布（协调世界时）",
        "",
        *[
            f"- {hour}:00：{count} 次"
            for hour, count in summary["high_refresh_utc_hour_histogram"].items()
        ],
        "",
        "## 结论表述限制",
        "",
        "- 样本数少于 20 一律标记为“样本极少”；20 至 49 标记为“样本偏少”。",
        "- 置信区间跨过零时，只能称为尚无统计证据。",
        "- 本报告不会自动把任何结果称为有效策略或强交易信号。",
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
    processed, episodes, actions, events = build_loss_event_tables(processed, config)
    events = add_forward_returns(events, processed, config)
    episodes = events[events["event_family"] == "LOSS_EPISODE"].copy()
    actions = events[events["event_family"] == "ACTION_EVENT"].copy()
    action_comparison = build_action_definition_comparison(
        episodes, actions, processed, config
    )
    backtest = build_backtest(events, processed, config)
    refresh_rows = processed[processed["cohort_refresh_detected"]]
    action_counts = (
        actions.groupby(["side", "action_type"]).size().to_dict()
        if not actions.empty else {}
    )
    confidence_counts = (
        cohorts["cohort_confidence"].value_counts().to_dict()
        if not cohorts.empty else {}
    )
    high_refresh = cohorts[
        cohorts["cohort_refresh_start"].notna()
        & cohorts["cohort_confidence"].eq("HIGH")
    ]
    medium_refresh = cohorts[
        cohorts["cohort_refresh_start"].notna()
        & cohorts["cohort_confidence"].eq("MEDIUM")
    ]
    high_hour_histogram = (
        high_refresh["cohort_refresh_start"].dt.hour.value_counts().sort_index().to_dict()
        if not high_refresh.empty else {}
    )
    old_notional_counts = {
        action_type: int(count)
        for action_type, count in action_comparison.loc[
            action_comparison["old_notional_action_timestamp"].notna()
        ].groupby("action_type").size().items()
    }
    new_qty_counts = {
        action_type: int(count)
        for action_type, count in action_comparison.loc[
            action_comparison["new_qty_action_timestamp"].notna()
        ].groupby("action_type").size().items()
    }
    comparison_counts = {
        str(status): int(count)
        for status, count in action_comparison["comparison_status"].value_counts().items()
        if status != "NEITHER"
    }
    changed_examples = action_comparison[
        action_comparison["comparison_status"].eq("TIMESTAMP_CHANGED")
    ].head(10)
    summary = {
        "analysis_logic_version": ANALYSIS_LOGIC_VERSION,
        "previous_action_backtest_status": (
            "OBSOLETE_LOOKAHEAD_BIAS_AND_NOTIONAL_PRICE_CONTAMINATION"
        ),
        "raw_file": str(raw_path.resolve()),
        "raw_file_unchanged": True,
        "raw_records": len(records),
        "valid_records": int(processed["_quality_signal_ok"].sum()),
        "invalid_records": int((~processed["_quality_signal_ok"]).sum()),
        "model_freeze_date": config["model_freeze_date"],
        "logic_valid_from": config["logic_valid_from"],
        "cohorts_detected": int(len(cohorts)),
        "high_cohorts": int(confidence_counts.get("HIGH", 0)),
        "medium_cohorts": int(confidence_counts.get("MEDIUM", 0)),
        "low_cohorts": int(confidence_counts.get("LOW", 0)),
        "expired_period_records": int((processed["analysis_state"] == "COHORT_EXPIRED").sum()),
        "refresh_windows": int(len(refresh_rows)),
        "active_refresh_new_segments": int(processed["active_refresh_new_segment"].sum()),
        "old_rule_active_segment_reuses": int(
            processed["old_rule_would_reuse_active_segment"].sum()
        ),
        "refresh_timestamps": [
            timestamp.isoformat() for timestamp in refresh_rows["timestamp"].head(10)
        ],
        "quality_counts": _quality_counts(processed["data_quality_flag"]),
        "high_refresh_timestamps": [
            value.isoformat() for value in high_refresh["cohort_refresh_start"]
        ],
        "medium_refresh_timestamps": [
            value.isoformat() for value in medium_refresh["cohort_refresh_start"]
        ],
        "high_refresh_utc_hour_histogram": {
            str(int(hour)): int(count) for hour, count in high_hour_histogram.items()
        },
        "new_long_loss_events": int((episodes["side"] == "LONG").sum()) if not episodes.empty else 0,
        "new_short_loss_events": int((episodes["side"] == "SHORT").sum()) if not episodes.empty else 0,
        "action_counts": {
            f"{side}_{response}": int(count)
            for (side, response), count in action_counts.items()
        },
        "response_counts": {
            f"{side}_{response}": int(count)
            for (side, response), count in action_counts.items()
        },
        "old_notional_action_counts": old_notional_counts,
        "new_qty_action_counts": new_qty_counts,
        "action_comparison_counts": comparison_counts,
        "action_timestamp_changed_examples": [
            {
                "side": row["side"],
                "episode_id": row["episode_id"],
                "action_type": row["action_type"],
                "old_notional_action_timestamp": row["old_notional_action_timestamp"].isoformat(),
                "new_qty_action_timestamp": row["new_qty_action_timestamp"].isoformat(),
                "timestamp_difference_minutes": float(row["timestamp_difference_minutes"]),
            }
            for _, row in changed_examples.iterrows()
        ],
        "oos_events": int(
            events.loc[
                events["event_family"].isin(["LOSS_EPISODE", "ACTION_EVENT"]),
                "out_of_sample",
            ].sum()
        ) if not events.empty else 0,
        "source_latest_timestamp": raw_metadata.get("latest_timestamp"),
    }
    summary["中文摘要"] = {
        "原始记录数": summary["raw_records"],
        "有效记录数": summary["valid_records"],
        "无效记录数": summary["invalid_records"],
        "模型冻结时间": summary["model_freeze_date"],
        "修正版逻辑生效时间": summary["logic_valid_from"],
        "识别名单批次数": summary["cohorts_detected"],
        "高可信批次数": summary["high_cohorts"],
        "中可信批次数": summary["medium_cohorts"],
        "低可信批次数": summary["low_cohorts"],
        "批次过期记录数": summary["expired_period_records"],
        "名单刷新次数": summary["refresh_windows"],
        "有效批次后刷新强制新分段数": summary["active_refresh_new_segments"],
        "旧规则本会复用有效批次编号数": summary["old_rule_active_segment_reuses"],
        "前十个名单刷新时间": summary["refresh_timestamps"],
        "数据质量统计": {
            QUALITY_ZH.get(flag, flag): count
            for flag, count in summary["quality_counts"].items()
        },
        "多头新亏损事件数": summary["new_long_loss_events"],
        "空头新亏损事件数": summary["new_short_loss_events"],
        "首次动作事件统计": {
            f"{SIDE_ZH.get(key.split('_', 1)[0], key)}"
            f"{RESPONSE_ZH.get(key.split('_', 1)[1], '')}": count
            for key, count in summary["response_counts"].items()
        },
        "样本外事件数": summary["oos_events"],
        "原始数据最新时间": summary["source_latest_timestamp"],
        "原始历史文件是否保持不变": True,
        "分析逻辑版本": ANALYSIS_LOGIC_VERSION,
        "旧动作回测状态": "已作废：前视偏差及USDT名义仓位价格污染",
        "旧USDT名义动作统计": summary["old_notional_action_counts"],
        "新BTC等价数量动作统计": summary["new_qty_action_counts"],
        "新旧动作对照统计": summary["action_comparison_counts"],
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
        _atomic_parquet(episodes, output_dir / LOSS_EPISODES_PATH.name)
        _atomic_parquet(actions, output_dir / ACTION_EVENTS_PATH.name)
        backtest.to_csv(backtest_path, index=False)
        action_comparison.to_csv(
            output_dir / ACTION_COMPARISON_PATH.name, index=False
        )
        _atomic_parquet(
            _chinese_frame(processed.drop(columns=internal)),
            output_dir / CHINESE_PROCESSED_PATH.name,
        )
        _atomic_parquet(
            _chinese_frame(events), output_dir / CHINESE_EVENTS_PATH.name
        )
        _atomic_parquet(
            _chinese_frame(episodes),
            output_dir / CHINESE_LOSS_EPISODES_PATH.name,
        )
        _atomic_parquet(
            _chinese_frame(actions),
            output_dir / CHINESE_ACTION_EVENTS_PATH.name,
        )
        _atomic_parquet(
            _chinese_frame(cohorts), output_dir / CHINESE_COHORTS_PATH.name
        )
        _chinese_frame(backtest).to_csv(
            output_dir / CHINESE_BACKTEST_PATH.name,
            index=False,
            encoding="utf-8-sig",
        )
        _chinese_frame(action_comparison).to_csv(
            output_dir / CHINESE_ACTION_COMPARISON_PATH.name,
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
        f"逻辑版本={ANALYSIS_LOGIC_VERSION} "
        f"原始记录={summary['raw_records']} 有效记录={summary['valid_records']} "
        f"无效记录={summary['invalid_records']} 名单批次={summary['cohorts_detected']} "
        f"名单刷新={summary['refresh_windows']} "
        f"ACTIVE后新分段={summary['active_refresh_new_segments']} "
        f"高/中/低可信批次={summary['high_cohorts']}/{summary['medium_cohorts']}/{summary['low_cohorts']} "
        f"多头新亏损事件={summary['new_long_loss_events']} "
        f"空头新亏损事件={summary['new_short_loss_events']} "
        f"首次动作事件={len(actions)} "
        f"样本外事件={summary['oos_events']}"
    )
    return {
        "processed": processed, "events": events, "episodes": episodes,
        "actions": actions, "cohorts": cohorts, "backtest": backtest,
        "action_comparison": action_comparison,
        "summary": summary,
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
