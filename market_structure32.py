from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

STRUCTURE32_WINDOW = 5
PRICE_MOVE_THRESHOLD = 0.001
AVG_ENTRY_MOVE_THRESHOLD = 0.001
POSITION_MOVE_THRESHOLD = 0.005

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

    for i, triggered in enumerate(event_series.to_numpy()):
        if not triggered:
            continue
        if last_trigger_idx is None or i - last_trigger_idx > cooldown_bars:
            result.iloc[i] = True
            last_trigger_idx = i

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
    (p, la, lp, sa, sp): {
        "case_no": no,
        "name": name,
        "stance": stance,
        "strength": strength,
        "interpretation": interpretation,
    }
    for no, p, la, lp, sa, sp, name, stance, strength, interpretation in _CASE_ROWS
}

STANCE_TEXT = {"long": "买点提示", "short": "卖点提示", "neutral": "观望提示"}
SIGNAL_LEVEL_PREFIX = {"long": "买", "short": "卖", "neutral": "观望"}
STANCE_COLOR = {"long": "#00C087", "short": "#F6465D", "neutral": "#F0B90B"}
STANCE_SYMBOL = {"long": "triangle-up", "short": "triangle-down", "neutral": "circle"}


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


def _level_label(stance: object, strength: object) -> str:
    numeric = pd.to_numeric(strength, errors="coerce")
    if pd.isna(numeric):
        return ""
    level = int(max(min(float(numeric), 5), 0))
    if level <= 0:
        return ""
    return f"{SIGNAL_LEVEL_PREFIX.get(str(stance), '观望')}{level}"


def _empty_columns(df: pd.DataFrame, reason: str) -> pd.DataFrame:
    df["structure32_key"] = ""
    df["structure32_case_no"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df["structure32_name"] = "无完整32结构"
    df["structure32_stance"] = "neutral"
    df["structure32_action"] = "无信号"
    df["structure32_strength"] = 0
    df["structure32_level_label"] = ""
    df["structure32_interpretation"] = reason
    df["structure32_reason"] = reason
    df["structure32_watch"] = False
    df["structure32_watch_event"] = False
    return df


def _resolve_avg_col(df: pd.DataFrame, primary: str, fallback: str) -> Optional[str]:
    if has_col(df, primary):
        return primary
    if has_col(df, fallback):
        return fallback
    return None


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
        return _empty_columns(out, f"32结构未启用：缺少字段 {', '.join(missing)}")

    window = int(params.get("structure32_window", STRUCTURE32_WINDOW))
    price_threshold = float(params.get("structure32_price_move_threshold", PRICE_MOVE_THRESHOLD))
    avg_threshold = float(params.get("structure32_avg_entry_move_threshold", AVG_ENTRY_MOVE_THRESHOLD))
    pos_threshold = float(params.get("structure32_position_move_threshold", POSITION_MOVE_THRESHOLD))
    cooldown = int(params.get("event_cooldown_bars", 5))

    out["structure32_price_change_pct"] = _pct_change(out["current_price"], window)
    out["structure32_long_avg_change_pct"] = _pct_change(out[long_avg_col], window)
    out["structure32_long_pos_change_pct"] = _pct_change(out["long_pos_usdt"], window)
    out["structure32_short_avg_change_pct"] = _pct_change(out[short_avg_col], window)
    out["structure32_short_pos_change_pct"] = _pct_change(out["short_pos_usdt"], window)

    out["structure32_price_dir"] = out["structure32_price_change_pct"].map(lambda v: _direction(v, price_threshold, "涨", "跌"))
    out["structure32_long_avg_dir"] = out["structure32_long_avg_change_pct"].map(lambda v: _direction(v, avg_threshold, "涨", "跌"))
    out["structure32_long_pos_dir"] = out["structure32_long_pos_change_pct"].map(lambda v: _direction(v, pos_threshold, "升", "降"))
    out["structure32_short_avg_dir"] = out["structure32_short_avg_change_pct"].map(lambda v: _direction(v, avg_threshold, "涨", "跌"))
    out["structure32_short_pos_dir"] = out["structure32_short_pos_change_pct"].map(lambda v: _direction(v, pos_threshold, "升", "降"))

    keys = list(zip(
        out["structure32_price_dir"],
        out["structure32_long_avg_dir"],
        out["structure32_long_pos_dir"],
        out["structure32_short_avg_dir"],
        out["structure32_short_pos_dir"],
    ))
    cases = [CASE_MAP.get(tuple(key)) for key in keys]

    out["structure32_key"] = ["_".join(key) if case else "" for key, case in zip(keys, cases)]
    out["structure32_case_no"] = pd.Series([case["case_no"] if case else pd.NA for case in cases], index=out.index, dtype="Int64")
    out["structure32_name"] = [str(case["name"]) if case else "无明显32结构" for case in cases]
    out["structure32_stance"] = [str(case["stance"]) if case else "neutral" for case in cases]
    out["structure32_action"] = [STANCE_TEXT.get(str(case["stance"]), "观望提示") if case else "无信号" for case in cases]
    out["structure32_strength"] = [int(case["strength"]) if case else 0 for case in cases]
    out["structure32_level_label"] = [_level_label(case["stance"], case["strength"]) if case else "" for case in cases]
    out["structure32_interpretation"] = [str(case["interpretation"]) if case else "五个维度中至少一个维度未超过阈值。" for case in cases]
    out["structure32_reason"] = [
        (
            f"{_level_label(case['stance'], case['strength'])}｜{STANCE_TEXT.get(str(case['stance']), '观望提示')}｜"
            f"强度 {int(case['strength'])}/5，数字越大越可靠、越强｜"
            f"第{int(case['case_no'])}种：{case['name']}。"
            f"结构=价格{key[0]} + 多头均价{key[1]} + 多头仓位{key[2]} + 空头均价{key[3]} + 空头仓位{key[4]}。"
            f"原理：{case['interpretation']}"
        )
        if case
        else "无明显32结构：价格、多空成本或仓位变化未全部超过阈值。"
        for case, key in zip(cases, keys)
    ]
    out["structure32_watch"] = pd.Series([case is not None for case in cases], index=out.index, dtype="bool")
    first_seen = out["structure32_watch"] & out["structure32_key"].ne(out["structure32_key"].shift(1))
    out["structure32_watch_event"] = suppress(first_seen, cooldown)
    return out


def _hover_data(rows: pd.DataFrame) -> np.ndarray:
    payload: List[List[object]] = []
    for _, row in rows.iterrows():
        payload.append([
            row.get("structure32_case_no", ""),
            row.get("structure32_name", ""),
            row.get("structure32_action", ""),
            row.get("structure32_strength", 0),
            row.get("structure32_level_label", ""),
            _pct_text(row.get("structure32_price_change_pct")),
            _pct_text(row.get("structure32_long_avg_change_pct")),
            _pct_text(row.get("structure32_long_pos_change_pct")),
            _pct_text(row.get("structure32_short_avg_change_pct")),
            _pct_text(row.get("structure32_short_pos_change_pct")),
            row.get("structure32_interpretation", ""),
        ])
    return np.array(payload, dtype=object)


def build_structure32_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty or "current_price" not in df.columns:
        return fig
    plot_df = df.copy().sort_values("timestamp")
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["current_price"],
        name="BTC 独立走势",
        mode="lines",
        line=dict(color="#E6EDF3", width=2.2),
        hovertemplate="时间 %{x|%Y-%m-%d %H:%M:%S}<br>BTC %{y:,.2f} USDT<extra></extra>",
    ))

    events = plot_df[plot_df.get("structure32_watch_event", False).fillna(False)].copy()
    for stance, label in [("long", "买点"), ("short", "卖点"), ("neutral", "观望")]:
        rows = events[events["structure32_stance"] == stance]
        if rows.empty:
            continue
        fig.add_trace(go.Scatter(
            x=rows["timestamp"],
            y=rows["current_price"],
            name=label,
            mode="markers+text",
            text=rows["structure32_level_label"].fillna("").astype(str).tolist(),
            textposition="top center",
            textfont=dict(color=STANCE_COLOR[stance], size=11),
            marker=dict(color=STANCE_COLOR[stance], size=11, symbol=STANCE_SYMBOL[stance], line=dict(color="#0B0E11", width=1)),
            customdata=_hover_data(rows),
            hovertemplate=(
                "时间 %{x|%Y-%m-%d %H:%M:%S}"
                "<br>BTC %{y:,.2f} USDT"
                "<br>第几种结构 第%{customdata[0]}种"
                "<br>结构名称 %{customdata[1]}"
                "<br>提示方向 %{customdata[2]}"
                "<br>信号等级 %{customdata[4]}"
                "<br>信号强度 %{customdata[3]}/5（数字越大越可靠）"
                "<br>价格变化率 %{customdata[5]}"
                "<br>多头均价变化率 %{customdata[6]}"
                "<br>多头仓位变化率 %{customdata[7]}"
                "<br>空头均价变化率 %{customdata[8]}"
                "<br>空头仓位变化率 %{customdata[9]}"
                "<br>背后原理解释 %{customdata[10]}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        height=680,
        paper_bgcolor="#0B0E11",
        plot_bgcolor="#0B0E11",
        font=dict(color="#C9D1D9"),
        margin=dict(l=14, r=24, t=26, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, tickformat="%m-%d %H:%M", hoverformat="%Y-%m-%d %H:%M:%S"),
        yaxis=dict(title="BTC 独立走势 + 32结构提示", showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False),
    )
    return fig


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
                plot_df = plot_df[(plot_df["timestamp"] >= x_values.min()) & (plot_df["timestamp"] <= x_values.max())].copy()
        except Exception:
            pass
        if plot_df.empty:
            return result

        _RENDERED = True
        st.subheader("32种盘口结构提示图")
        st.caption("按首次出现的 价格涨跌 + 多空均价涨跌 + 多空仓位升降 组合标记。图上显示为 买1/卖1/观望1 等等级标签，数字越大表示结构越强、可靠度越高。")
        _ORIGINAL_PLOTLY_CHART(build_structure32_figure(plot_df), use_container_width=True, config={"displaylogo": False})
        latest = plot_df.iloc[-1]
        st.info(str(latest.get("structure32_reason", "当前没有完整触发32结构。")))
        return result

    st.plotly_chart = _patched_plotly_chart
