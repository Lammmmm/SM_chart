from __future__ import annotations

from copy import deepcopy

import pandas as pd

from smart_money_analysis import (
    DEFAULT_CONFIG,
    add_divergence_features,
    build_loss_events,
    prepare_raw_frame,
    reconstruct_cohorts,
)


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
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(minutes=minute)
    return {
        "timestamp": timestamp.isoformat(),
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
    return value


def process(records: list[dict[str, object]]):
    cfg = config()
    raw = prepare_raw_frame(records, cfg)
    processed, cohorts = reconstruct_cohorts(raw, cfg)
    processed = add_divergence_features(processed, cfg)
    processed, events = build_loss_events(processed, cfg)
    return processed, cohorts, events


def baseline_records(**overrides):
    return [record(minute, **overrides) for minute in range(0, 35, 5)]


def test_normal_trading_position_jump_is_not_refresh():
    records = baseline_records()
    records.append(record(35, long_pos=1_500, short_pos=700))
    processed, cohorts, _ = process(records)
    assert len(cohorts) == 1
    assert not processed.iloc[-1]["cohort_refresh_detected"]
    assert processed.iloc[-1]["analysis_state"] == "ACTIVE_COHORT"


def test_trader_count_jump_detects_refresh():
    records = baseline_records()
    records.append(record(35, long_traders=1_500, short_traders=1_500))
    processed, cohorts, _ = process(records)
    assert len(cohorts) == 2
    assert processed.iloc[-1]["cohort_refresh_detected"]
    assert processed.iloc[-1]["analysis_state"] == "REFRESHING"


def test_multi_step_refresh_creates_only_one_new_cohort():
    records = baseline_records()
    counts = [(35, 140, 140), (40, 155, 155), (45, 152, 153), (50, 153, 153), (55, 153, 154)]
    records.extend(
        record(minute, long_traders=long_count, short_traders=short_count)
        for minute, long_count, short_count in counts
    )
    processed, cohorts, _ = process(records)
    assert len(cohorts) == 2
    assert processed["cohort_refresh_detected"].sum() == 1


def test_cross_cohort_position_delta_is_null_until_new_baseline():
    records = baseline_records(long_pos=1_000)
    records.append(record(35, long_pos=3_000, long_traders=150, short_traders=150))
    processed, _, _ = process(records)
    refresh_row = processed.iloc[-1]
    assert refresh_row["cohort_refresh_detected"]
    assert pd.isna(refresh_row["long_position_change_pct"])
    assert not refresh_row["signal_eligible"]


def test_existing_loss_does_not_create_new_loss_event():
    records = baseline_records(price=98.0, long_avg=100.0)
    records.append(record(35, price=97.0, long_avg=100.0))
    processed, _, events = process(records)
    assert processed.iloc[-1]["long_existing_loss_at_cohort_start"]
    assert not processed["long_new_loss_event"].any()
    assert events.empty or not (events["side"] == "LONG").any()


def test_clean_baseline_then_loss_creates_new_loss_event():
    records = baseline_records(price=101.0, long_avg=100.0)
    records.append(record(35, price=99.3, long_avg=100.0))
    processed, _, events = process(records)
    assert processed.iloc[-1]["long_new_loss_event"]
    assert len(events[events["side"] == "LONG"]) == 1


def test_loss_add_is_classified():
    records = baseline_records(price=101.0, long_avg=100.0)
    records.extend([
        record(35, price=99.3, long_avg=100.0, long_pos=1_000),
        record(40, price=99.0, long_avg=99.9, long_pos=1_100, short_pos=1_000),
    ])
    processed, _, events = process(records)
    assert processed.iloc[-1]["long_loss_response"] == "ADD"
    assert events.loc[events["side"] == "LONG", "final_response"].iloc[0] == "ADD"


def test_loss_reduce_is_classified():
    records = baseline_records(price=101.0, long_avg=100.0)
    records.extend([
        record(35, price=99.3, long_avg=100.0, long_pos=1_000),
        record(40, price=99.0, long_avg=100.1, long_pos=900, short_pos=1_000),
    ])
    processed, _, events = process(records)
    assert processed.iloc[-1]["long_loss_response"] == "REDUCE"
    assert events.loc[events["side"] == "LONG", "final_response"].iloc[0] == "REDUCE"


def test_features_do_not_change_when_future_rows_are_appended():
    prefix = baseline_records()
    prefix.extend([
        record(35, long_traders=100, short_traders=100),
        record(40, long_traders=101, short_traders=100),
    ])
    future = [
        record(minute, long_traders=101, short_traders=100)
        for minute in range(45, 125, 5)
    ]
    prefix_processed, _, _ = process(prefix)
    full_processed, _, _ = process(prefix + future)
    causal_columns = [
        "cohort_id", "analysis_state", "cohort_refresh_detected",
        "cohort_refresh_score", "signal_eligible", "baseline_price",
        "cohort_net_flow", "long_new_loss_event", "short_new_loss_event",
    ]
    pd.testing.assert_frame_equal(
        prefix_processed[causal_columns].reset_index(drop=True),
        full_processed.iloc[: len(prefix)][causal_columns].reset_index(drop=True),
        check_dtype=False,
    )
