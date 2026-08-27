from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from smart_money_analysis import (
    DEFAULT_CONFIG,
    add_divergence_features,
    add_forward_returns,
    build_loss_event_tables,
    prepare_raw_frame,
    reconstruct_cohorts,
)


BASE = pd.Timestamp("2026-01-01T09:00:00Z")


def record(
    minute: int,
    *,
    price: float = 101.0,
    long_avg: float = 100.0,
    short_avg: float = 102.0,
    long_pos: float = 1_000.0,
    short_pos: float = 1_000.0,
    long_traders: int = 50,
    short_traders: int = 50,
) -> dict[str, object]:
    return {
        "timestamp": (BASE + pd.Timedelta(minutes=minute)).isoformat(),
        "current_price": price,
        "long_avg_price": long_avg,
        "short_avg_price": short_avg,
        "long_pos_usdt": long_pos,
        "short_pos_usdt": short_pos,
        "total_pos_usdt": long_pos + short_pos,
        "long_traders": long_traders,
        "short_traders": short_traders,
        "total_traders": long_traders + short_traders,
        "long_unrealized_pnl": 1.0,
        "short_unrealized_pnl": 1.0,
        "ls_ratio": long_traders / short_traders,
    }


def config() -> dict[str, object]:
    value = deepcopy(DEFAULT_CONFIG)
    value["model_freeze_date"] = "2030-01-01T00:00:00Z"
    value["backtest"]["bootstrap_samples"] = 50
    value["cohort"]["initial_cohort_confidence"] = "HIGH"
    value["cohort"]["max_cohort_age_hours"] = 1_000
    value["cohort"]["structural_min_history"] = 10_000
    return value


def process(records: list[dict[str, object]], cfg=None):
    cfg = cfg or config()
    raw = prepare_raw_frame(records, cfg)
    processed, cohorts = reconstruct_cohorts(raw, cfg)
    processed = add_divergence_features(processed, cfg)
    processed, episodes, actions, events = build_loss_event_tables(processed, cfg)
    return processed, cohorts, episodes, actions, events


def baseline_records(**overrides):
    return [record(minute, **overrides) for minute in range(0, 35, 5)]


def long_loss_timeline(
    *,
    action: str | None = "ADD",
    include_until: int = 300,
    second_action: bool = False,
) -> list[dict[str, object]]:
    rows = baseline_records(price=101.0, long_avg=100.0)
    for minute in range(35, include_until + 1, 5):
        kwargs = {
            "price": 99.3,
            "long_avg": 110.0,
            "long_pos": 1_000.0,
        }
        if minute == 120:
            kwargs["price"] = 101.286
        if minute >= 240 and action == "ADD":
            kwargs["long_pos"] = 1_050.0
            kwargs["price"] = 100.0
        if minute >= 240 and action == "REDUCE":
            kwargs["long_pos"] = 950.0
            kwargs["price"] = 100.0
        if second_action and minute >= 270:
            kwargs["long_pos"] = 950.0
        if minute == 300:
            kwargs["price"] = 103.0 if action == "ADD" else 97.0
        rows.append(record(minute, **kwargs))
    return rows


def test_normal_position_jump_is_not_refresh():
    records = baseline_records()
    records.append(record(35, long_pos=1_500, short_pos=700))
    processed, cohorts, *_ = process(records)
    assert len(cohorts) == 1
    assert not processed.iloc[-1]["cohort_refresh_detected"]


def test_hard_trader_jump_detects_refresh():
    records = baseline_records()
    records.append(record(35, long_traders=1_500, short_traders=1_500))
    processed, cohorts, *_ = process(records)
    assert len(cohorts) == 2
    assert processed.iloc[-1]["hard_refresh_trigger"]
    assert processed.iloc[-1]["cohort_confidence"] == "HIGH"
    assert processed.iloc[-1]["active_refresh_new_segment"]
    assert processed.iloc[-1]["cohort_id"] != processed.iloc[-2]["cohort_id"]


def test_continuous_refresh_creates_one_new_cohort():
    records = baseline_records()
    records.extend([
        record(35, long_traders=140, short_traders=140),
        record(40, long_traders=155, short_traders=155),
        record(45, long_traders=152, short_traders=153),
        record(50, long_traders=153, short_traders=153),
    ])
    processed, cohorts, *_ = process(records)
    assert len(cohorts) == 2
    assert processed["cohort_refresh_detected"].sum() == 1


def test_active_refresh_within_old_18h_window_starts_new_cohort():
    records = baseline_records()
    records.append(record(35, long_traders=150, short_traders=150))
    records.extend([
        record(minute, long_traders=150, short_traders=150)
        for minute in range(40, 100, 5)
    ])
    records.append(record(100, long_traders=200, short_traders=200))
    processed, cohorts, *_ = process(records)
    assert processed.iloc[-2]["analysis_state"] == "ACTIVE_COHORT"
    assert processed.iloc[-1]["active_refresh_new_segment"]
    assert processed.iloc[-1]["old_rule_would_reuse_active_segment"]
    assert processed.iloc[-1]["cohort_id"] != processed.iloc[-2]["cohort_id"]
    assert len(cohorts) == 3


def test_cross_cohort_delta_is_null():
    records = baseline_records()
    records.append(record(35, long_pos=3_000, long_traders=150, short_traders=150))
    processed, *_ = process(records)
    assert pd.isna(processed.iloc[-1]["long_position_change_pct"])
    assert not processed.iloc[-1]["signal_eligible"]


def test_existing_loss_does_not_create_new_loss():
    records = baseline_records(price=98.0, long_avg=100.0)
    records.append(record(35, price=97.0, long_avg=100.0))
    processed, _, episodes, _, _ = process(records)
    assert processed.iloc[-1]["long_existing_loss_at_cohort_start"]
    assert episodes.empty or not (episodes["side"] == "LONG").any()


def test_new_loss_episode_has_causal_type():
    processed, _, episodes, _, _ = process(long_loss_timeline(action=None, include_until=60))
    assert processed["long_new_loss_event"].sum() == 1
    episode = episodes[episodes["side"] == "LONG"].iloc[0]
    assert episode["event_type"] == "LONG_NEW_LOSS"
    assert episode["event_timestamp"] == BASE + pd.Timedelta(minutes=35)


def test_add_forward_return_starts_at_add_timestamp():
    cfg = config()
    processed, _, _, actions, events = process(long_loss_timeline(action="ADD"), cfg)
    events = add_forward_returns(events, processed, cfg)
    loss = events[events["event_type"] == "LONG_NEW_LOSS"].iloc[0]
    add = events[events["event_type"] == "LONG_LOSS_ADD"].iloc[0]
    assert loss["event_timestamp"] == BASE + pd.Timedelta(minutes=35)
    assert add["event_timestamp"] == BASE + pd.Timedelta(minutes=240)
    assert np.isclose(add["forward_return_1h"], 0.03)
    assert not np.isclose(add["forward_return_1h"], loss["forward_return_1h"])


def test_acceptance_example_loss_10_add_13_uses_separate_clocks():
    cfg = config()
    price_rows = [
        record(60, price=100.0),   # 10:00 首次亏损
        record(120, price=102.0),  # 11:00
        record(240, price=100.0),  # 13:00 首次加仓
        record(300, price=103.0),  # 14:00
    ]
    processed = prepare_raw_frame(price_rows, cfg)
    events = pd.DataFrame([
        {
            "event_type": "LONG_NEW_LOSS",
            "event_timestamp": BASE + pd.Timedelta(minutes=60),
            "event_price": 100.0,
        },
        {
            "event_type": "LONG_LOSS_ADD",
            "event_timestamp": BASE + pd.Timedelta(minutes=240),
            "event_price": 100.0,
        },
    ])
    result = add_forward_returns(events, processed, cfg).set_index("event_type")
    assert np.isclose(result.loc["LONG_NEW_LOSS", "forward_return_1h"], 0.02)
    assert np.isclose(result.loc["LONG_LOSS_ADD", "forward_return_1h"], 0.03)
    assert result.loc["LONG_LOSS_ADD", "forward_price_timestamp_1h"] == (
        BASE + pd.Timedelta(minutes=300)
    )


def test_reduce_forward_return_starts_at_reduce_timestamp():
    cfg = config()
    processed, _, _, _, events = process(long_loss_timeline(action="REDUCE"), cfg)
    events = add_forward_returns(events, processed, cfg)
    reduce_event = events[events["event_type"] == "LONG_LOSS_REDUCE"].iloc[0]
    assert reduce_event["event_timestamp"] == BASE + pd.Timedelta(minutes=240)
    assert np.isclose(reduce_event["forward_return_1h"], -0.03)


def test_future_add_does_not_change_past_rows():
    prefix = long_loss_timeline(action=None, include_until=180)
    full = long_loss_timeline(action="ADD", include_until=300)
    prefix_processed, *_ = process(prefix)
    full_processed, *_ = process(full)
    causal_columns = [
        "cohort_id", "analysis_state", "cohort_confidence",
        "cohort_refresh_detected", "structural_refresh_score",
        "signal_eligible", "baseline_price", "cohort_net_flow",
        "cohort_net_qty_flow", "long_position_qty_change_since_loss",
        "long_new_loss_event", "long_action_event", "long_action_type",
        "long_action_event_id", "long_loss_response", "event_markers",
    ]
    pd.testing.assert_frame_equal(
        prefix_processed[causal_columns].reset_index(drop=True),
        full_processed.iloc[: len(prefix)][causal_columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_same_episode_add_triggers_once():
    _, _, episodes, actions, _ = process(long_loss_timeline(action="ADD"))
    long_adds = actions[
        (actions["side"] == "LONG") & (actions["action_type"] == "ADD")
    ]
    assert len(long_adds) == 1
    assert episodes.loc[episodes["side"] == "LONG", "action_sequence"].iloc[0] == "ADD"


def test_same_episode_can_add_then_reduce():
    _, _, episodes, actions, _ = process(
        long_loss_timeline(action="ADD", second_action=True)
    )
    long_actions = actions[actions["side"] == "LONG"]
    assert list(long_actions["action_type"]) == ["ADD", "REDUCE"]
    assert episodes.loc[episodes["side"] == "LONG", "action_sequence"].iloc[0] == "ADD→REDUCE"


def test_forward_price_beyond_tolerance_is_nan():
    cfg = config()
    event_time = BASE + pd.Timedelta(minutes=60)
    events = pd.DataFrame([{
        "event_timestamp": event_time,
        "event_price": 100.0,
    }])
    price_rows = [
        record(60, price=100.0),
        record(140, price=102.0),
    ]
    processed = prepare_raw_frame(price_rows, cfg)
    result = add_forward_returns(events, processed, cfg)
    assert pd.isna(result.iloc[0]["forward_return_1h"])


def test_refresh_during_baseline_building_resets_baseline():
    records = [record(minute) for minute in range(0, 20, 5)]
    records.append(record(20, long_traders=100, short_traders=100))
    processed, cohorts, *_ = process(records)
    # 基准构建期内的连续跳变仍属于同一轮刷新，但旧基准样本必须全部清空。
    assert len(cohorts) == 1
    assert processed.iloc[-1]["analysis_state"] == "REFRESHING"
    assert pd.isna(processed.iloc[-1]["baseline_price"])


def test_data_gap_invalidates_cohort():
    records = baseline_records()
    records.append(record(200))
    processed, cohorts, *_ = process(records)
    assert len(cohorts) == 2
    assert processed.iloc[-1]["data_gap_detected"]
    assert processed.iloc[-1]["analysis_state"] == "UNAVAILABLE"
    assert not processed.iloc[-1]["signal_eligible"]


def test_cohort_age_expiry_disables_signal():
    cfg = config()
    cfg["cohort"]["max_cohort_age_hours"] = 26
    cfg["cohort"]["max_data_gap_minutes"] = 2_000
    records = baseline_records()
    records.append(record(1_565))
    processed, *_ = process(records, cfg)
    assert processed.iloc[-1]["analysis_state"] == "COHORT_EXPIRED"
    assert not processed.iloc[-1]["signal_eligible"]


def structural_config():
    cfg = config()
    cfg["cohort"]["structural_min_history"] = 5
    cfg["cohort"]["refresh_prior_min_samples"] = 0
    return cfg


def structural_history():
    return [record(minute) for minute in range(0, 40, 5)]


def test_position_only_jump_does_not_trigger_structural_refresh():
    records = structural_history()
    records.append(record(40, long_pos=1_600, short_pos=1_600))
    processed, cohorts, *_ = process(records, structural_config())
    assert not processed.iloc[-1]["structural_refresh_trigger"]
    assert len(cohorts) == 1


def test_trader_and_avg_entry_anomalies_trigger_structural_refresh():
    records = structural_history()
    records.append(record(
        40,
        long_traders=51,
        short_traders=51,
        long_avg=110,
        short_avg=92,
    ))
    processed, cohorts, *_ = process(records, structural_config())
    assert processed.iloc[-1]["structural_refresh_trigger"]
    assert processed.iloc[-1]["cohort_confidence"] == "MEDIUM"
    assert len(cohorts) == 2
    assert processed.iloc[-1]["cohort_id"] != processed.iloc[-2]["cohort_id"]
    assert not processed.iloc[-1]["signal_eligible"]
    assert pd.isna(processed.iloc[-1]["long_qty_change_pct"])
    assert pd.isna(processed.iloc[-1]["cohort_net_qty_flow"])


def test_structural_detector_is_prefix_causal():
    cfg = structural_config()
    prefix = structural_history()
    prefix.append(record(40, long_traders=51, short_traders=51, long_avg=110, short_avg=92))
    future = [record(minute) for minute in range(45, 90, 5)]
    prefix_processed, *_ = process(prefix, cfg)
    full_processed, *_ = process(prefix + future, cfg)
    columns = [
        "structural_refresh_score", "structural_anomalous_fields",
        "structural_refresh_trigger", "hard_refresh_trigger",
        "cohort_refresh_detected", "cohort_confidence", "analysis_state",
    ]
    pd.testing.assert_frame_equal(
        prefix_processed[columns].reset_index(drop=True),
        full_processed.iloc[: len(prefix)][columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_price_only_long_notional_increase_does_not_create_add():
    rows = [
        record(minute, price=110.0, long_avg=110.0, long_pos=1_100.0)
        for minute in range(0, 35, 5)
    ]
    rows.extend([
        record(35, price=100.0, long_avg=110.0, long_pos=1_000.0),
        record(40, price=103.0, long_avg=110.0, long_pos=1_030.0),
    ])
    processed, _, _, actions, _ = process(rows)
    assert np.isclose(processed.iloc[-1]["long_position_change_since_loss"], 0.03)
    assert abs(processed.iloc[-1]["long_position_qty_change_since_loss"]) < 1e-12
    assert not (
        actions["event_type"].eq("LONG_LOSS_ADD").any()
        if not actions.empty else False
    )


def test_price_only_short_notional_increase_does_not_create_add():
    rows = [
        record(minute, price=100.0, short_avg=100.0, short_pos=1_000.0)
        for minute in range(0, 35, 5)
    ]
    rows.extend([
        record(35, price=103.0, short_avg=100.0, short_pos=1_030.0),
        record(40, price=107.0, short_avg=100.0, short_pos=1_070.0),
    ])
    processed, _, _, actions, _ = process(rows)
    assert processed.iloc[-1]["short_position_change_since_loss"] > 0.03
    assert abs(processed.iloc[-1]["short_position_qty_change_since_loss"]) < 1e-12
    assert not (
        actions["event_type"].eq("SHORT_LOSS_ADD").any()
        if not actions.empty else False
    )


def test_real_quantity_increase_creates_add_at_first_crossing():
    rows = [
        record(minute, price=110.0, long_avg=110.0, long_pos=1_100.0)
        for minute in range(0, 35, 5)
    ]
    rows.extend([
        record(35, price=100.0, long_avg=110.0, long_pos=1_000.0),
        record(40, price=103.0, long_avg=110.0, long_pos=1_061.93),
    ])
    _, _, _, actions, _ = process(rows)
    add = actions[actions["event_type"].eq("LONG_LOSS_ADD")].iloc[0]
    assert add["action_timestamp"] == BASE + pd.Timedelta(minutes=40)
    assert np.isclose(add["position_qty_change_since_loss"], 0.031)


def test_price_change_does_not_create_quantity_flow():
    rows = [
        record(minute, price=100.0, long_pos=1_000.0, short_pos=500.0)
        for minute in range(0, 35, 5)
    ]
    rows.append(record(35, price=110.0, long_pos=1_100.0, short_pos=550.0))
    processed, *_ = process(rows)
    latest = processed.iloc[-1]
    assert latest["cohort_net_notional_flow"] != 0
    assert abs(latest["cohort_net_qty_flow"]) < 1e-12


def test_quantity_baseline_is_median_of_row_ratios():
    prices = [100.0, 100.0, 100.0, 200.0, 200.0, 200.0, 200.0]
    positions = [1_000.0, 1_000.0, 4_000.0, 1_000.0, 1_000.0, 4_000.0, 4_000.0]
    rows = [
        record(index * 5, price=price, long_pos=position)
        for index, (price, position) in enumerate(zip(prices, positions))
    ]
    processed, *_ = process(rows)
    latest = processed.iloc[-1]
    assert np.isclose(latest["baseline_long_qty_proxy"], 10.0)
    assert not np.isclose(
        latest["baseline_long_qty_proxy"],
        latest["baseline_long_pos"] / latest["baseline_price"],
    )


def test_real_quantity_change_creates_expected_quantity_flow():
    rows = [
        record(minute, price=100.0, long_pos=1_000.0, short_pos=500.0)
        for minute in range(0, 35, 5)
    ]
    rows.append(record(35, price=100.0, long_pos=1_100.0, short_pos=500.0))
    processed, *_ = process(rows)
    assert np.isclose(processed.iloc[-1]["cohort_net_qty_flow"], 1 / 15)


def test_low_structural_refresh_after_active_starts_low_cohort():
    cfg = structural_config()
    cfg["cohort"]["refresh_prior_min_samples"] = 3
    rows = structural_history()
    rows.append(record(40, long_traders=51, short_traders=51, long_avg=110, short_avg=92))
    processed, cohorts, *_ = process(rows, cfg)
    assert len(cohorts) == 2
    assert processed.iloc[-1]["cohort_confidence"] == "LOW"
    assert processed.iloc[-1]["cohort_id"] != processed.iloc[-2]["cohort_id"]
    assert not processed.iloc[-1]["signal_eligible"]


def test_divergence_never_uses_previous_cohort():
    cfg = config()
    frame = pd.DataFrame([
        {
            "timestamp": BASE,
            "cohort_id": "cohort_a",
            "signal_eligible": True,
            "current_price": 100.0,
            "cohort_net_notional_flow": 0.0,
            "cohort_net_qty_flow": 0.0,
        },
        {
            "timestamp": BASE + pd.Timedelta(hours=4),
            "cohort_id": "cohort_b",
            "signal_eligible": True,
            "current_price": 100.0,
            "cohort_net_notional_flow": 0.2,
            "cohort_net_qty_flow": 0.2,
        },
    ])
    result = add_divergence_features(frame, cfg)
    assert pd.isna(result.iloc[-1]["cohort_net_qty_flow_change_4h"])
    assert result.iloc[-1]["divergence_4h"] is None


def test_out_of_sample_waits_for_logic_valid_from():
    cfg = config()
    cfg["model_freeze_date"] = (BASE + pd.Timedelta(minutes=10)).isoformat()
    cfg["logic_valid_from"] = (BASE + pd.Timedelta(minutes=40)).isoformat()
    rows = baseline_records() + [record(35), record(40), record(45)]
    processed, *_ = process(rows, cfg)
    assert not processed.loc[processed["timestamp"].eq(BASE + pd.Timedelta(minutes=35)), "out_of_sample"].iloc[0]
    assert processed.loc[processed["timestamp"].eq(BASE + pd.Timedelta(minutes=40)), "out_of_sample"].iloc[0]
