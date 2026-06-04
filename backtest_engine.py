from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies import StrategyConfig, build_strategy_signals


@dataclass
class PendingSetup:
    direction: str
    signal_name: str
    event_index: int
    event_time: pd.Timestamp
    expiry_index: int
    priority: int


@dataclass
class PositionState:
    trade_id: int
    direction: str
    entry_index: int
    entry_time: pd.Timestamp
    entry_price: float
    raw_entry_price: float
    size: float
    notional: float
    entry_fee: float
    entry_slippage_cost: float
    entry_signal: str
    smart_score_entry: float
    long_percent_entry: float
    short_percent_entry: float
    long_pain_entry: float
    short_pain_entry: float
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0


def _safe_float(value, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(numeric) else float(numeric)


def _direction_allowed(trading_direction: str, direction: str) -> bool:
    if trading_direction == "多空都做":
        return True
    if trading_direction == "只做多":
        return direction == "long"
    if trading_direction == "只做空":
        return direction == "short"
    return True


def _entry_fill_price(direction: str, raw_price: float, slippage_rate: float) -> float:
    if direction == "long":
        return raw_price * (1 + slippage_rate)
    return raw_price * (1 - slippage_rate)


def _exit_fill_price(direction: str, raw_price: float, slippage_rate: float) -> float:
    if direction == "long":
        return raw_price * (1 - slippage_rate)
    return raw_price * (1 + slippage_rate)


def _unrealized_pnl(position: PositionState, current_price: float) -> float:
    if position.direction == "long":
        return (current_price - position.entry_price) * position.size
    return (position.entry_price - current_price) * position.size


def _mark_position_excursion(position: PositionState, current_price: float) -> None:
    unrealized = _unrealized_pnl(position, current_price)
    excursion = 0.0 if position.notional <= 0 else unrealized / position.notional
    position.max_favorable_excursion = max(position.max_favorable_excursion, excursion)
    position.max_adverse_excursion = min(position.max_adverse_excursion, excursion)


def _build_setup_candidates(
    row: pd.Series,
    bar_index: int,
    confirmation_window: int,
    trading_direction: str,
) -> List[PendingSetup]:
    setups: List[PendingSetup] = []

    if (
        _direction_allowed(trading_direction, "long")
        and bool(row.get("long_setup_event", False))
        and not bool(row.get("long_entry_blocked", False))
    ):
        setups.append(
            PendingSetup(
                direction="long",
                signal_name=str(row.get("long_setup_source", "long_setup_event")),
                event_index=bar_index,
                event_time=row["timestamp"],
                expiry_index=bar_index + max(int(confirmation_window), 1),
                priority=int(_safe_float(row.get("long_setup_priority", 99), 99)),
            )
        )

    if (
        _direction_allowed(trading_direction, "short")
        and bool(row.get("short_setup_event", False))
        and not bool(row.get("short_entry_blocked", False))
    ):
        setups.append(
            PendingSetup(
                direction="short",
                signal_name=str(row.get("short_setup_source", "short_setup_event")),
                event_index=bar_index,
                event_time=row["timestamp"],
                expiry_index=bar_index + max(int(confirmation_window), 1),
                priority=int(_safe_float(row.get("short_setup_priority", 99), 99)),
            )
        )

    return setups


def _upsert_pending_setup(existing: List[PendingSetup], new_setup: PendingSetup) -> List[PendingSetup]:
    remaining = [setup for setup in existing if setup.direction != new_setup.direction]
    same_direction = [setup for setup in existing if setup.direction == new_setup.direction]
    if not same_direction:
        remaining.append(new_setup)
        return remaining

    current = same_direction[0]
    if new_setup.priority < current.priority or new_setup.event_index >= current.event_index:
        remaining.append(new_setup)
    else:
        remaining.append(current)
    return remaining


def _choose_confirmed_setup(
    confirmed_setups: List[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    if not confirmed_setups:
        return None

    if len(confirmed_setups) == 1:
        return confirmed_setups[0]

    confirmed_setups = sorted(
        confirmed_setups,
        key=lambda item: (item["priority"], -item["confirm_count"], -item["event_index"]),
    )
    best = confirmed_setups[0]
    second = confirmed_setups[1]
    if (best["priority"], best["confirm_count"]) == (second["priority"], second["confirm_count"]):
        return None
    return best


def _determine_exit_reason(
    row: pd.Series,
    position: PositionState,
    exit_mode: str,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> Optional[str]:
    current_price = _safe_float(row.get("current_price", 0), 0.0)
    if current_price <= 0:
        return None

    if position.direction == "long":
        stop_hit = current_price <= position.entry_price * (1 - stop_loss_pct)
        target_hit = current_price >= position.entry_price * (1 + take_profit_pct)
        reverse_signal_hit = bool(row.get("long_reverse_exit_signal", False))
    else:
        stop_hit = current_price >= position.entry_price * (1 + stop_loss_pct)
        target_hit = current_price <= position.entry_price * (1 - take_profit_pct)
        reverse_signal_hit = bool(row.get("short_reverse_exit_signal", False))

    if exit_mode in {"固定止盈止损", "混合退出"}:
        if stop_hit:
            return "stop_loss"
        if target_hit:
            return "take_profit"

    if exit_mode in {"反向信号", "混合退出"} and reverse_signal_hit:
        return "reverse_signal"

    return None


def run_backtest(
    df: pd.DataFrame,
    strategy_params: Dict[str, object],
    engine_params: Dict[str, object],
) -> Dict[str, pd.DataFrame]:
    if df.empty:
        empty_bars = df.copy()
        return {
            "bars": empty_bars,
            "equity_curve": pd.DataFrame(columns=["timestamp", "equity", "drawdown", "position"]),
            "trade_log": pd.DataFrame(),
        }

    params = StrategyConfig(
        trading_direction=str(strategy_params.get("trading_direction", "多空都做")),
        confirmation_window=int(strategy_params.get("confirmation_window", 5)),
        confirmation_required_count=int(strategy_params.get("confirmation_required_count", 2)),
        trade_cooldown_bars=int(strategy_params.get("trade_cooldown_bars", 10)),
        exit_mode=str(strategy_params.get("exit_mode", "混合退出")),
        stop_loss_pct=float(strategy_params.get("stop_loss_pct", 0.006)),
        take_profit_pct=float(strategy_params.get("take_profit_pct", 0.009)),
    )

    bars = build_strategy_signals(df, strategy_params).reset_index(drop=True).copy()
    for col in [
        "entry_long",
        "exit_long",
        "entry_short",
        "exit_short",
    ]:
        bars[col] = False
    bars["entry_marker_price"] = np.nan
    bars["exit_marker_price"] = np.nan
    bars["position"] = "flat"
    bars["equity"] = np.nan
    bars["drawdown"] = np.nan

    initial_capital = float(engine_params.get("initial_capital", 10000.0))
    position_size_pct = float(engine_params.get("position_size_pct", 1.0))
    fee_rate = float(engine_params.get("fee_rate", 0.0004))
    slippage_rate = float(engine_params.get("slippage_rate", 0.0002))

    realized_equity = initial_capital
    peak_equity = initial_capital
    cooldown_until_index = -1
    pending_entry: Optional[Dict[str, object]] = None
    pending_setups: List[PendingSetup] = []
    open_position: Optional[PositionState] = None
    trade_logs: List[Dict[str, object]] = []
    equity_records: List[Dict[str, object]] = []
    trade_id = 0

    for i in range(len(bars)):
        row = bars.iloc[i]
        current_price = _safe_float(row.get("current_price", 0), 0.0)

        if pending_entry and pending_entry["entry_index"] == i and open_position is None:
            entry_price = _entry_fill_price(
                pending_entry["direction"],
                current_price,
                slippage_rate,
            )
            notional = max(realized_equity, 0.0) * max(position_size_pct, 0.0)
            if current_price > 0 and entry_price > 0 and notional > 0:
                trade_id += 1
                size = notional / entry_price
                entry_fee = notional * fee_rate
                entry_slippage_cost = abs(entry_price - current_price) * size
                realized_equity -= entry_fee
                open_position = PositionState(
                    trade_id=trade_id,
                    direction=pending_entry["direction"],
                    entry_index=i,
                    entry_time=row["timestamp"],
                    entry_price=entry_price,
                    raw_entry_price=current_price,
                    size=size,
                    notional=notional,
                    entry_fee=entry_fee,
                    entry_slippage_cost=entry_slippage_cost,
                    entry_signal=str(pending_entry["signal_name"]),
                    smart_score_entry=_safe_float(row.get("smart_score", 0), 0.0),
                    long_percent_entry=_safe_float(row.get("long_percent_smooth", 0), 0.0),
                    short_percent_entry=_safe_float(row.get("short_percent_smooth", 0), 0.0),
                    long_pain_entry=_safe_float(row.get("long_pain_smooth", 0), 0.0),
                    short_pain_entry=_safe_float(row.get("short_pain_smooth", 0), 0.0),
                )
                if open_position.direction == "long":
                    bars.at[i, "entry_long"] = True
                else:
                    bars.at[i, "entry_short"] = True
                bars.at[i, "entry_marker_price"] = entry_price
            pending_entry = None

        pending_setups = [setup for setup in pending_setups if i <= setup.expiry_index]

        if open_position is None and pending_entry is None and i > cooldown_until_index:
            new_setups = _build_setup_candidates(
                row=row,
                bar_index=i,
                confirmation_window=params.confirmation_window,
                trading_direction=params.trading_direction,
            )
            for setup in new_setups:
                pending_setups = _upsert_pending_setup(pending_setups, setup)

            confirmed_candidates: List[Dict[str, object]] = []
            for setup in pending_setups:
                if i <= setup.event_index or i > setup.expiry_index:
                    continue

                if setup.direction == "long":
                    confirm_pass = bool(row.get("long_confirmation_pass", False))
                    confirm_count = int(_safe_float(row.get("long_confirm_count", 0), 0))
                else:
                    confirm_pass = bool(row.get("short_confirmation_pass", False))
                    confirm_count = int(_safe_float(row.get("short_confirm_count", 0), 0))

                if confirm_pass:
                    confirmed_candidates.append(
                        {
                            "direction": setup.direction,
                            "signal_name": setup.signal_name,
                            "event_index": setup.event_index,
                            "priority": setup.priority,
                            "confirm_count": confirm_count,
                        }
                    )

            chosen_setup = _choose_confirmed_setup(confirmed_candidates)
            if chosen_setup and i + 1 < len(bars):
                pending_entry = {
                    "direction": chosen_setup["direction"],
                    "signal_name": chosen_setup["signal_name"],
                    "entry_index": i + 1,
                }
                pending_setups = []

        if open_position is not None:
            _mark_position_excursion(open_position, current_price)

            if i > open_position.entry_index:
                exit_reason = _determine_exit_reason(
                    row=row,
                    position=open_position,
                    exit_mode=params.exit_mode,
                    stop_loss_pct=params.stop_loss_pct,
                    take_profit_pct=params.take_profit_pct,
                )
                if exit_reason is not None:
                    exit_price = _exit_fill_price(open_position.direction, current_price, slippage_rate)
                    exit_fee = exit_price * open_position.size * fee_rate
                    exit_slippage_cost = abs(exit_price - current_price) * open_position.size
                    if open_position.direction == "long":
                        gross_pnl = (exit_price - open_position.entry_price) * open_position.size
                    else:
                        gross_pnl = (open_position.entry_price - exit_price) * open_position.size
                    net_pnl = gross_pnl - open_position.entry_fee - exit_fee
                    realized_equity += gross_pnl - exit_fee
                    holding_bars = i - open_position.entry_index

                    trade_logs.append(
                        {
                            "trade_id": open_position.trade_id,
                            "direction": open_position.direction,
                            "entry_time": open_position.entry_time,
                            "exit_time": row["timestamp"],
                            "entry_price": open_position.entry_price,
                            "exit_price": exit_price,
                            "size": open_position.size,
                            "notional": open_position.notional,
                            "entry_fee": open_position.entry_fee,
                            "exit_fee": exit_fee,
                            "total_fee": open_position.entry_fee + exit_fee,
                            "slippage_cost": open_position.entry_slippage_cost + exit_slippage_cost,
                            "gross_pnl": gross_pnl,
                            "net_pnl": net_pnl,
                            "return_pct": 0.0 if open_position.notional <= 0 else net_pnl / open_position.notional,
                            "holding_bars": holding_bars,
                            "entry_signal": open_position.entry_signal,
                            "exit_reason": exit_reason,
                            "smart_score_entry": open_position.smart_score_entry,
                            "smart_score_exit": _safe_float(row.get("smart_score", 0), 0.0),
                            "long_percent_entry": open_position.long_percent_entry,
                            "short_percent_entry": open_position.short_percent_entry,
                            "long_pain_entry": open_position.long_pain_entry,
                            "short_pain_entry": open_position.short_pain_entry,
                            "max_favorable_excursion": open_position.max_favorable_excursion,
                            "max_adverse_excursion": open_position.max_adverse_excursion,
                        }
                    )

                    if open_position.direction == "long":
                        bars.at[i, "exit_long"] = True
                    else:
                        bars.at[i, "exit_short"] = True
                    bars.at[i, "exit_marker_price"] = exit_price

                    open_position = None
                    cooldown_until_index = i + params.trade_cooldown_bars
                    pending_entry = None
                    pending_setups = []

        if open_position is not None:
            unrealized = _unrealized_pnl(open_position, current_price)
            equity = realized_equity + unrealized
            position_label = open_position.direction
        else:
            equity = realized_equity
            position_label = "flat"

        peak_equity = max(peak_equity, equity)
        drawdown = 0.0 if peak_equity <= 0 else (equity / peak_equity) - 1.0

        bars.at[i, "position"] = position_label
        bars.at[i, "equity"] = equity
        bars.at[i, "drawdown"] = drawdown
        equity_records.append(
            {
                "timestamp": row["timestamp"],
                "equity": equity,
                "drawdown": drawdown,
                "position": position_label,
            }
        )

    if open_position is not None and len(bars) > 0:
        last_index = len(bars) - 1
        last_row = bars.iloc[last_index]
        current_price = _safe_float(last_row.get("current_price", 0), 0.0)
        exit_price = _exit_fill_price(open_position.direction, current_price, slippage_rate)
        exit_fee = exit_price * open_position.size * fee_rate
        exit_slippage_cost = abs(exit_price - current_price) * open_position.size
        if open_position.direction == "long":
            gross_pnl = (exit_price - open_position.entry_price) * open_position.size
        else:
            gross_pnl = (open_position.entry_price - exit_price) * open_position.size
        net_pnl = gross_pnl - open_position.entry_fee - exit_fee
        realized_equity += gross_pnl - exit_fee
        holding_bars = last_index - open_position.entry_index

        trade_logs.append(
            {
                "trade_id": open_position.trade_id,
                "direction": open_position.direction,
                "entry_time": open_position.entry_time,
                "exit_time": last_row["timestamp"],
                "entry_price": open_position.entry_price,
                "exit_price": exit_price,
                "size": open_position.size,
                "notional": open_position.notional,
                "entry_fee": open_position.entry_fee,
                "exit_fee": exit_fee,
                "total_fee": open_position.entry_fee + exit_fee,
                "slippage_cost": open_position.entry_slippage_cost + exit_slippage_cost,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "return_pct": 0.0 if open_position.notional <= 0 else net_pnl / open_position.notional,
                "holding_bars": holding_bars,
                "entry_signal": open_position.entry_signal,
                "exit_reason": "end_of_data",
                "smart_score_entry": open_position.smart_score_entry,
                "smart_score_exit": _safe_float(last_row.get("smart_score", 0), 0.0),
                "long_percent_entry": open_position.long_percent_entry,
                "short_percent_entry": open_position.short_percent_entry,
                "long_pain_entry": open_position.long_pain_entry,
                "short_pain_entry": open_position.short_pain_entry,
                "max_favorable_excursion": open_position.max_favorable_excursion,
                "max_adverse_excursion": open_position.max_adverse_excursion,
            }
        )

        if open_position.direction == "long":
            bars.at[last_index, "exit_long"] = True
        else:
            bars.at[last_index, "exit_short"] = True
        bars.at[last_index, "exit_marker_price"] = exit_price
        bars.at[last_index, "position"] = "flat"
        bars.at[last_index, "equity"] = realized_equity
        peak_equity = max(peak_equity, realized_equity)
        bars.at[last_index, "drawdown"] = 0.0 if peak_equity <= 0 else (realized_equity / peak_equity) - 1.0

        if equity_records:
            equity_records[-1] = {
                "timestamp": last_row["timestamp"],
                "equity": realized_equity,
                "drawdown": bars.at[last_index, "drawdown"],
                "position": "flat",
            }

    trade_log_df = pd.DataFrame(trade_logs)
    if not trade_log_df.empty:
        trade_log_df["entry_time"] = pd.to_datetime(trade_log_df["entry_time"])
        trade_log_df["exit_time"] = pd.to_datetime(trade_log_df["exit_time"])

    equity_curve_df = pd.DataFrame(equity_records)
    if not equity_curve_df.empty:
        equity_curve_df["timestamp"] = pd.to_datetime(equity_curve_df["timestamp"])

    return {
        "bars": bars,
        "equity_curve": equity_curve_df,
        "trade_log": trade_log_df,
    }
