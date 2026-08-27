from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from smart_money_analysis import (
    EVENTS_PATH,
    PROCESSED_PATH,
    SUMMARY_PATH,
    run_pipeline,
)

EVENT_TYPE_ZH = {
    "LONG_NEW_LOSS_ADD": "多头新亏损后加仓",
    "LONG_NEW_LOSS_HOLD": "多头新亏损后持仓",
    "LONG_NEW_LOSS_REDUCE": "多头新亏损后减仓",
    "SHORT_NEW_LOSS_ADD": "空头新亏损后加仓",
    "SHORT_NEW_LOSS_HOLD": "空头新亏损后持仓",
    "SHORT_NEW_LOSS_REDUCE": "空头新亏损后减仓",
}
SIDE_ZH = {"long": "多头", "short": "空头"}
RESPONSE_ZH = {"ADD": "加仓", "HOLD": "持仓", "REDUCE": "减仓"}


def _read_derived() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    processed = pd.read_parquet(PROCESSED_PATH)
    events = pd.read_parquet(EVENTS_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    processed["timestamp"] = pd.to_datetime(processed["timestamp"], utc=True)
    if not events.empty:
        events["loss_start_timestamp"] = pd.to_datetime(
            events["loss_start_timestamp"], utc=True
        )
    return processed, events, summary


def _filter_period(frame: pd.DataFrame, period_label: str) -> pd.DataFrame:
    if frame.empty or period_label == "all":
        return frame
    delta = {
        "24H": pd.Timedelta(hours=24),
        "1W": pd.Timedelta(weeks=1),
        "1M": pd.Timedelta(days=30),
    }[period_label]
    return frame[frame["timestamp"] >= frame["timestamp"].max() - delta]


def _base_layout(figure: go.Figure, title: str) -> go.Figure:
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=430,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return figure


def render_cohort_analysis_section(st: Any, period_label: str) -> None:
    st.divider()
    st.subheader("按名单批次划分的聪明钱分析")
    st.caption(
        "原始多空人数比、仓位和币安盈亏数据仅作描述；资金流、亏损事件和信号全部限制在同一名单批次内。"
        "名单刷新期、稳定观察期和基准构建期一律不产生交易信号。"
    )
    if st.button("重新生成批次、事件和回测", key="rebuild_cohort_analysis"):
        with st.spinner("正在从只读历史数据重建分析结果……"):
            run_pipeline()
        st.success("分析结果已更新；原始历史文件没有被改写。")
    if not all(path.exists() for path in (PROCESSED_PATH, EVENTS_PATH, SUMMARY_PATH)):
        st.info("尚无批次分析结果。请运行：python smart_money_analysis.py process-smart-money")
        return
    try:
        processed, events, summary = _read_derived()
    except Exception as exc:
        st.warning(f"读取分析结果失败：{exc}")
        return

    cards = st.columns(6)
    cards[0].metric("识别批次数", summary["cohorts_detected"])
    cards[1].metric("名单刷新次数", summary["refresh_windows"])
    cards[2].metric("无效记录", summary["invalid_records"])
    cards[3].metric("多头新亏损", summary["new_long_loss_events"])
    cards[4].metric("空头新亏损", summary["new_short_loss_events"])
    cards[5].metric("样本外事件", summary["oos_events"])
    st.caption(
        f"模型冻结时间：{summary['model_freeze_date']}；此前结果均为历史探索结果，"
        "只有此时间之后新采集的数据才属于真正的样本外验证。"
    )

    chart = _filter_period(processed[processed["timestamp"].notna()], period_label)
    price_figure = go.Figure()
    price_figure.add_trace(go.Scatter(
        x=chart["timestamp"], y=chart["current_price"], name="比特币价格",
        mode="lines", line=dict(color="#111827", width=1.8),
    ))
    refresh = chart[chart["cohort_refresh_detected"]]
    price_figure.add_trace(go.Scatter(
        x=refresh["timestamp"], y=refresh["current_price"], name="名单批次刷新",
        mode="markers", marker=dict(color="#dc2626", symbol="line-ns", size=18),
        text=refresh["cohort_id"].str.replace("cohort_", "批次_", regex=False),
    ))
    st.plotly_chart(
        _base_layout(price_figure, "图表一 · 比特币价格与名单批次刷新"),
        width="stretch", key="cohort_price_refresh",
    )

    active = chart[chart["signal_eligible"]]
    flow_figure = go.Figure()
    for column, name, color in [
        ("long_position_change_pct", "多头仓位变化", "#ef4444"),
        ("short_position_change_pct", "空头仓位变化", "#10b981"),
        ("cohort_net_flow", "批次净风险流", "#2563eb"),
    ]:
        flow_figure.add_trace(go.Scatter(
            x=active["timestamp"], y=active[column], name=name,
            mode="lines", line=dict(color=color, width=1.5),
            connectgaps=False,
        ))
    flow_figure.update_yaxes(tickformat=".1%")
    st.plotly_chart(
        _base_layout(flow_figure, "图表二 · 同一批次内的仓位与净风险流"),
        width="stretch", key="cohort_flows",
    )

    loss_figure = go.Figure()
    for side, color in [("long", "#ef4444"), ("short", "#10b981")]:
        loss_figure.add_trace(go.Scatter(
            x=chart["timestamp"], y=chart[f"{side}_directional_return"],
            name=f"{SIDE_ZH[side]}方向收益", mode="lines",
            line=dict(color=color, width=1.4),
        ))
        for response, symbol in [("ADD", "triangle-up"), ("REDUCE", "triangle-down")]:
            points = chart[chart[f"{side}_loss_response"] == response].drop_duplicates(
                [f"{side}_loss_event_id", f"{side}_loss_response"]
            )
            loss_figure.add_trace(go.Scatter(
                x=points["timestamp"], y=points[f"{side}_directional_return"],
                name=f"{SIDE_ZH[side]}{RESPONSE_ZH[response]}", mode="markers",
                marker=dict(symbol=symbol, size=9, color=color),
            ))
        recovery = chart[chart[f"{side}_recovery_event"]]
        loss_figure.add_trace(go.Scatter(
            x=recovery["timestamp"], y=recovery[f"{side}_directional_return"],
            name=f"{SIDE_ZH[side]}恢复盈利", mode="markers",
            marker=dict(symbol="circle-open", size=10, color=color),
        ))
    loss_figure.add_hline(y=-0.005, line_dash="dot", line_color="#64748b")
    loss_figure.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    loss_figure.update_yaxes(tickformat=".1%")
    st.plotly_chart(
        _base_layout(loss_figure, "图表三 · 方向收益、亏损后操作与恢复盈利"),
        width="stretch", key="cohort_losses",
    )

    if events.empty:
        st.info("当前阈值下尚无新亏损事件。")
        return
    event_types = sorted(events["event_type"].dropna().unique())
    selected = st.selectbox(
        "事件研究类型",
        event_types,
        format_func=lambda value: EVENT_TYPE_ZH.get(value, value),
        key="cohort_event_study_type",
    )
    study = events[events["event_type"] == selected]
    horizons = [0, 1, 4, 12, 24]
    mean_values = [0.0] + [
        study[f"forward_return_{hours}h"].mean() for hours in horizons[1:]
    ]
    median_values = [0.0] + [
        study[f"forward_return_{hours}h"].median() for hours in horizons[1:]
    ]
    event_figure = go.Figure()
    event_figure.add_trace(go.Scatter(
        x=horizons, y=mean_values, name="比特币平均收益",
        mode="lines+markers", line=dict(color="#2563eb"),
    ))
    event_figure.add_trace(go.Scatter(
        x=horizons, y=median_values, name="比特币收益中位数",
        mode="lines+markers", line=dict(color="#7c3aed"),
    ))
    event_figure.update_xaxes(title="事件发生后的小时数", tickvals=horizons)
    event_figure.update_yaxes(tickformat=".1%")
    st.plotly_chart(
        _base_layout(
            event_figure,
            f"图表四 · {EVENT_TYPE_ZH.get(selected, selected)}事件研究（样本数={len(study)}）",
        ),
        width="stretch", key="cohort_event_study",
    )
    st.caption(
        "空头事件同样保存比特币原始收益，没有乘以 -1。当前历史样本只用于探索，不能作为样本外结论。"
    )
