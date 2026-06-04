from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtest_engine import run_backtest
from backtest_metrics import calculate_backtest_metrics
from data_client import fetch_smart_money_data
from features import add_derived_metrics
from signals import add_signal_columns
from strategies import EXIT_MODE_OPTIONS, STRATEGY_NAME
from ui_shell import apply_common_style, render_console_header


BULL_COLOR = "#00C087"
BEAR_COLOR = "#F6465D"
PRICE_COLOR = "#E6EDF3"
MA20_COLOR = "#F0B90B"
MA60_COLOR = "#4F7CFF"
BACKGROUND = "#0B0E11"

TIME_RANGE_OPTIONS = {
    "最近 24 小时": pd.Timedelta(hours=24),
    "最近 3 天": pd.Timedelta(days=3),
    "最近 7 天": pd.Timedelta(days=7),
    "全部": None,
}


def filter_by_range(df: pd.DataFrame, range_label: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    delta = TIME_RANGE_OPTIONS.get(range_label)
    if delta is None:
        return df.copy()

    cutoff = df["timestamp"].max() - delta
    return df[df["timestamp"] >= cutoff].copy().reset_index(drop=True)


@st.cache_data(ttl=60)
def load_raw_data() -> pd.DataFrame:
    return fetch_smart_money_data()


def build_sidebar_options() -> Dict[str, object]:
    st.sidebar.header("回测参数")

    st.sidebar.subheader("基础数据")
    range_label = st.sidebar.selectbox("数据范围", list(TIME_RANGE_OPTIONS.keys()), index=1)
    smooth_window = st.sidebar.slider("smooth_window", 1, 30, 5)
    z_window = st.sidebar.slider("z_window", 20, 240, 60, step=5)
    price_move_threshold = st.sidebar.slider(
        "price_move_threshold",
        min_value=0.001,
        max_value=0.030,
        value=0.005,
        step=0.001,
        format="%.3f",
    )
    pain_threshold = st.sidebar.slider(
        "pain_threshold",
        min_value=0.005,
        max_value=0.150,
        value=0.030,
        step=0.005,
        format="%.3f",
    )
    crowding_high = st.sidebar.slider("crowding_high", 55, 95, 70)
    crowding_low = st.sidebar.slider("crowding_low", 5, 45, 30)
    flow_z_threshold = st.sidebar.slider(
        "flow_z_threshold",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        format="%.1f",
    )
    event_cooldown_bars = st.sidebar.slider("event_cooldown_bars", 1, 30, 5)

    st.sidebar.subheader("策略参数")
    trading_direction = st.sidebar.selectbox("交易方向", ["只做多", "只做空", "多空都做"], index=2)
    confirmation_window = st.sidebar.slider("confirmation_window", 1, 20, 5)
    confirmation_required_count = st.sidebar.slider("confirmation_required_count", 1, 3, 2)
    trade_cooldown_bars = st.sidebar.slider("trade_cooldown_bars", 0, 50, 10)
    exit_mode = st.sidebar.selectbox("exit_mode", list(EXIT_MODE_OPTIONS), index=2)
    stop_loss_pct = st.sidebar.slider(
        "stop_loss_pct",
        min_value=0.001,
        max_value=0.050,
        value=0.006,
        step=0.001,
        format="%.3f",
    )
    take_profit_pct = st.sidebar.slider(
        "take_profit_pct",
        min_value=0.001,
        max_value=0.100,
        value=0.009,
        step=0.001,
        format="%.3f",
    )

    st.sidebar.subheader("成本参数")
    initial_capital = st.sidebar.number_input("initial_capital", min_value=100.0, value=10000.0, step=100.0)
    position_size_pct = st.sidebar.slider(
        "position_size_pct",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.1,
        format="%.1f",
    )
    fee_rate = st.sidebar.slider(
        "fee_rate",
        min_value=0.0,
        max_value=0.005,
        value=0.0004,
        step=0.0001,
        format="%.4f",
    )
    slippage_rate = st.sidebar.slider(
        "slippage_rate",
        min_value=0.0,
        max_value=0.005,
        value=0.0002,
        step=0.0001,
        format="%.4f",
    )

    if crowding_low >= crowding_high:
        st.sidebar.warning("crowding_low 应小于 crowding_high，程序将自动收紧区间。")
        crowding_low = max(5, crowding_high - 5)

    return {
        "range_label": range_label,
        "smooth_window": smooth_window,
        "z_window": z_window,
        "price_move_threshold": price_move_threshold,
        "pain_threshold": pain_threshold,
        "crowding_high": crowding_high,
        "crowding_low": crowding_low,
        "flow_z_threshold": flow_z_threshold,
        "event_cooldown_bars": event_cooldown_bars,
        "trading_direction": trading_direction,
        "confirmation_window": confirmation_window,
        "confirmation_required_count": confirmation_required_count,
        "trade_cooldown_bars": trade_cooldown_bars,
        "exit_mode": exit_mode,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "initial_capital": initial_capital,
        "position_size_pct": position_size_pct,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
    }


def render_metric_cards(metrics: Dict[str, float]) -> None:
    cards = [
        ("最终权益", f"{metrics['final_equity']:,.2f}"),
        ("总收益", f"{metrics['total_return_pct']:.2f}%"),
        ("最大回撤", f"{metrics['max_drawdown_pct']:.2f}%"),
        ("胜率", f"{metrics['win_rate']:.2f}%"),
        ("Profit Factor", "∞" if metrics["profit_factor"] == float("inf") else f"{metrics['profit_factor']:.2f}"),
        ("交易次数", f"{int(metrics['trade_count'])}"),
    ]
    for column, (label, value) in zip(st.columns(6), cards):
        column.metric(label, value)


def render_metric_details(metrics: Dict[str, float]) -> None:
    with st.expander("展开查看完整绩效", expanded=False):
        detail_cards = [
            ("多单次数", int(metrics["long_trade_count"])),
            ("空单次数", int(metrics["short_trade_count"])),
            ("多单胜率", f"{metrics['long_win_rate']:.2f}%"),
            ("空单胜率", f"{metrics['short_win_rate']:.2f}%"),
            ("平均持仓", f"{metrics['average_holding_bars']:.2f} 根"),
            ("平均单笔收益", f"{metrics['avg_return_per_trade']:.2f}%"),
            ("平均盈利", f"{metrics['avg_win']:.2f}"),
            ("平均亏损", f"{metrics['avg_loss']:.2f}"),
            ("最大连续亏损", int(metrics["max_consecutive_losses"])),
            ("总手续费", f"{metrics['total_fees']:.2f}"),
            ("总滑点成本", f"{metrics['total_slippage_cost']:.2f}"),
            ("资金暴露率", f"{metrics['exposure_ratio']:.2f}%"),
        ]
        for column, (label, value) in zip(st.columns(4) * 3, detail_cards):
            column.metric(label, value)


def build_equity_figure(equity_curve: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=equity_curve["timestamp"],
            y=equity_curve["equity"],
            name="权益曲线",
            line=dict(color=BULL_COLOR, width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=equity_curve["timestamp"],
            y=equity_curve["drawdown"] * 100,
            name="回撤曲线",
            line=dict(color=BEAR_COLOR, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(246,70,93,0.18)",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        height=420,
        template="plotly_dark",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.02, x=0),
        title="净值曲线与回撤",
    )
    fig.update_yaxes(title_text="权益", secondary_y=False)
    fig.update_yaxes(title_text="回撤(%)", secondary_y=True)
    return fig


def build_price_trade_figure(bars: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bars["timestamp"],
            y=bars["current_price"],
            mode="lines",
            name="BTC 价格",
            line=dict(color=PRICE_COLOR, width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bars["timestamp"],
            y=bars["price_ma_20"],
            mode="lines",
            name="MA20",
            line=dict(color=MA20_COLOR, width=1.3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bars["timestamp"],
            y=bars["price_ma_60"],
            mode="lines",
            name="MA60",
            line=dict(color=MA60_COLOR, width=1.2, dash="dot"),
        )
    )

    long_entries = bars[bars["entry_long"].fillna(False)]
    long_exits = bars[bars["exit_long"].fillna(False)]
    short_entries = bars[bars["entry_short"].fillna(False)]
    short_exits = bars[bars["exit_short"].fillna(False)]

    if not long_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=long_entries["timestamp"],
                y=long_entries["entry_marker_price"],
                mode="markers",
                name="开多点",
                marker=dict(color=BULL_COLOR, size=10, symbol="triangle-up"),
            )
        )
    if not long_exits.empty:
        fig.add_trace(
            go.Scatter(
                x=long_exits["timestamp"],
                y=long_exits["exit_marker_price"],
                mode="markers",
                name="平多点",
                marker=dict(color=BULL_COLOR, size=9, symbol="circle"),
            )
        )
    if not short_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=short_entries["timestamp"],
                y=short_entries["entry_marker_price"],
                mode="markers",
                name="开空点",
                marker=dict(color=BEAR_COLOR, size=10, symbol="triangle-down"),
            )
        )
    if not short_exits.empty:
        fig.add_trace(
            go.Scatter(
                x=short_exits["timestamp"],
                y=short_exits["exit_marker_price"],
                mode="markers",
                name="平空点",
                marker=dict(color=BEAR_COLOR, size=9, symbol="circle"),
            )
        )

    fig.update_layout(
        height=520,
        template="plotly_dark",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.02, x=0),
        title="BTC 价格与交易点",
        xaxis_title="时间",
        yaxis_title="价格",
    )
    return fig


def render_strategy_notes(options: Dict[str, object]) -> None:
    st.markdown(
        f"""
        **策略说明**

        当前策略：`{STRATEGY_NAME}`

        - `event` 只表示预警，不直接成交。
        - 出现预警后，会在后续 `confirmation_window = {options['confirmation_window']}` 根数据内等待确认。
        - 满足至少 `confirmation_required_count = {options['confirmation_required_count']}` 个确认条件后，**下一根数据** 才允许入场。
        - 回测已纳入手续费 `fee_rate = {options['fee_rate']:.4f}` 和滑点 `slippage_rate = {options['slippage_rate']:.4f}`。
        - 当前退出模式：`{options['exit_mode']}`。
        """,
    )


def render_backtest_page(embedded_in_console: bool = False) -> None:
    apply_common_style()
    render_console_header(
        "统一控制台 / 策略回测" if embedded_in_console else "独立页面 / 策略回测",
        "📈 Smart Money 策略回测平台",
        "回测仅用于研究 Smart Money 信号的统计价值，不构成自动交易建议。",
    )
    if embedded_in_console:
        st.caption("当前在统一控制台内查看回测页，侧栏顶部可随时切回实时雷达。")

    options = build_sidebar_options()

    with st.spinner("正在加载 Smart Money 数据并运行回测..."):
        raw_df = load_raw_data()

    if raw_df.empty:
        st.warning("暂无可用数据，请检查 PocketBase 地址、网络连接或数据采集状态。")
        st.stop()

    pipeline_params = {
        "smooth_window": options["smooth_window"],
        "z_window": options["z_window"],
        "price_move_threshold": options["price_move_threshold"],
        "pain_threshold": options["pain_threshold"],
        "crowding_high": options["crowding_high"],
        "crowding_low": options["crowding_low"],
        "flow_z_threshold": options["flow_z_threshold"],
        "event_cooldown_bars": options["event_cooldown_bars"],
    }
    feature_df = add_derived_metrics(raw_df, pipeline_params)
    signal_df = add_signal_columns(feature_df, pipeline_params)
    backtest_df = filter_by_range(signal_df, str(options["range_label"]))

    if backtest_df.empty or len(backtest_df) < 20:
        st.warning("当前回测范围内数据太少，请扩大时间范围后重试。")
        st.stop()

    strategy_params = {
        "trading_direction": options["trading_direction"],
        "confirmation_window": options["confirmation_window"],
        "confirmation_required_count": options["confirmation_required_count"],
        "trade_cooldown_bars": options["trade_cooldown_bars"],
        "exit_mode": options["exit_mode"],
        "stop_loss_pct": options["stop_loss_pct"],
        "take_profit_pct": options["take_profit_pct"],
    }
    engine_params = {
        "initial_capital": options["initial_capital"],
        "position_size_pct": options["position_size_pct"],
        "fee_rate": options["fee_rate"],
        "slippage_rate": options["slippage_rate"],
    }

    results = run_backtest(backtest_df, strategy_params=strategy_params, engine_params=engine_params)
    bars = results["bars"]
    equity_curve = results["equity_curve"]
    trade_log = results["trade_log"]
    metrics = calculate_backtest_metrics(
        equity_curve=equity_curve,
        trade_log=trade_log,
        initial_capital=float(options["initial_capital"]),
    )

    render_metric_cards(metrics)
    render_metric_details(metrics)

    if not equity_curve.empty:
        st.plotly_chart(build_equity_figure(equity_curve), use_container_width=True)

    if not bars.empty:
        st.plotly_chart(build_price_trade_figure(bars), use_container_width=True)

    st.subheader("交易明细表")
    if trade_log.empty:
        st.info("当前参数下没有成交记录。可以放宽确认条件或扩大数据范围再观察。")
    else:
        display_trade_log = trade_log.copy()
        display_trade_log["entry_time"] = pd.to_datetime(display_trade_log["entry_time"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        display_trade_log["exit_time"] = pd.to_datetime(display_trade_log["exit_time"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        display_trade_log["return_pct"] = pd.to_numeric(display_trade_log["return_pct"], errors="coerce") * 100
        numeric_round_cols = [
            "entry_price",
            "exit_price",
            "size",
            "notional",
            "entry_fee",
            "exit_fee",
            "total_fee",
            "slippage_cost",
            "gross_pnl",
            "net_pnl",
            "return_pct",
            "smart_score_entry",
            "smart_score_exit",
            "long_percent_entry",
            "short_percent_entry",
            "long_pain_entry",
            "short_pain_entry",
            "max_favorable_excursion",
            "max_adverse_excursion",
        ]
        for col in numeric_round_cols:
            if col in display_trade_log.columns:
                display_trade_log[col] = pd.to_numeric(display_trade_log[col], errors="coerce").round(4)
        st.dataframe(display_trade_log, use_container_width=True)

    render_strategy_notes(options)
