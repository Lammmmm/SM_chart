from typing import Dict

import numpy as np
import pandas as pd


def calculate_backtest_metrics(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    initial_capital: float,
) -> Dict[str, float]:
    initial_capital = float(initial_capital)

    if equity_curve.empty:
        return {
            "final_equity": initial_capital,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_return_per_trade": 0.0,
            "trade_count": 0,
            "long_trade_count": 0,
            "short_trade_count": 0,
            "long_win_rate": 0.0,
            "short_win_rate": 0.0,
            "average_holding_bars": 0.0,
            "max_consecutive_losses": 0,
            "total_fees": 0.0,
            "total_slippage_cost": 0.0,
            "exposure_ratio": 0.0,
        }

    final_equity = float(pd.to_numeric(equity_curve["equity"], errors="coerce").dropna().iloc[-1])
    total_return_pct = 0.0 if initial_capital <= 0 else (final_equity / initial_capital - 1.0) * 100
    max_drawdown_pct = abs(
        float(pd.to_numeric(equity_curve["drawdown"], errors="coerce").fillna(0.0).min()) * 100
    )
    exposure_ratio = float((equity_curve["position"] != "flat").mean() * 100)

    if trade_log.empty:
        return {
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_return_per_trade": 0.0,
            "trade_count": 0,
            "long_trade_count": 0,
            "short_trade_count": 0,
            "long_win_rate": 0.0,
            "short_win_rate": 0.0,
            "average_holding_bars": 0.0,
            "max_consecutive_losses": 0,
            "total_fees": 0.0,
            "total_slippage_cost": 0.0,
            "exposure_ratio": exposure_ratio,
        }

    pnl = pd.to_numeric(trade_log["net_pnl"], errors="coerce").fillna(0.0)
    positive = pnl[pnl > 0]
    negative = pnl[pnl < 0]

    trade_count = int(len(trade_log))
    win_rate = float((pnl > 0).mean() * 100)
    gross_profit = float(positive.sum())
    gross_loss = float(abs(negative.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    avg_win = float(positive.mean()) if not positive.empty else 0.0
    avg_loss = float(negative.mean()) if not negative.empty else 0.0
    avg_return_per_trade = float(pd.to_numeric(trade_log["return_pct"], errors="coerce").fillna(0.0).mean() * 100)

    long_trades = trade_log[trade_log["direction"] == "long"]
    short_trades = trade_log[trade_log["direction"] == "short"]
    long_trade_count = int(len(long_trades))
    short_trade_count = int(len(short_trades))
    long_win_rate = (
        float((pd.to_numeric(long_trades["net_pnl"], errors="coerce").fillna(0.0) > 0).mean() * 100)
        if long_trade_count > 0
        else 0.0
    )
    short_win_rate = (
        float((pd.to_numeric(short_trades["net_pnl"], errors="coerce").fillna(0.0) > 0).mean() * 100)
        if short_trade_count > 0
        else 0.0
    )

    average_holding_bars = float(pd.to_numeric(trade_log["holding_bars"], errors="coerce").fillna(0.0).mean())

    max_consecutive_losses = 0
    current_loss_streak = 0
    for value in pnl.to_list():
        if value < 0:
            current_loss_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)
        else:
            current_loss_streak = 0

    total_fees = float(pd.to_numeric(trade_log["total_fee"], errors="coerce").fillna(0.0).sum())
    total_slippage_cost = float(
        pd.to_numeric(trade_log["slippage_cost"], errors="coerce").fillna(0.0).sum()
    )

    return {
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_return_per_trade": avg_return_per_trade,
        "trade_count": trade_count,
        "long_trade_count": long_trade_count,
        "short_trade_count": short_trade_count,
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate,
        "average_holding_bars": average_holding_bars,
        "max_consecutive_losses": max_consecutive_losses,
        "total_fees": total_fees,
        "total_slippage_cost": total_slippage_cost,
        "exposure_ratio": exposure_ratio,
    }
