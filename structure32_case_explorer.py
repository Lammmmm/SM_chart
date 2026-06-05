from __future__ import annotations

from typing import Any

import pandas as pd


def _case_map() -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    try:
        from market_structure32 import CASE_MAP
        return CASE_MAP
    except Exception:
        return {}


def _match_stats(df: pd.DataFrame, key: str) -> tuple[int, str, str]:
    if df.empty or "structure32_key" not in df.columns:
        return 0, "--", "--"
    rows = df[df["structure32_key"].astype(str) == key]
    if rows.empty:
        return 0, "--", "--"
    latest = rows.iloc[-1]
    try:
        time_text = pd.to_datetime(latest.get("timestamp")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        time_text = str(latest.get("timestamp", "--"))
    price = pd.to_numeric(latest.get("current_price"), errors="coerce")
    price_text = "--" if pd.isna(price) else f"{float(price):,.2f}"
    return len(rows), time_text, price_text


def render_structure32_case_explorer(st, structure_df: pd.DataFrame) -> None:
    cases = _case_map()
    with st.container(border=True):
        st.markdown("### 32种盘口结构解释器")
        st.caption("自由选择五个维度，查看该结构背后的资金含义、风险点和后续观察重点。")
        c1, c2, c3, c4, c5 = st.columns(5)
        price = c1.radio("价格", ["涨", "跌"], horizontal=True, key="s32x_price")
        long_avg = c2.radio("多头均价", ["涨", "跌"], horizontal=True, key="s32x_long_avg")
        long_pos = c3.radio("多头仓位", ["升", "降"], horizontal=True, key="s32x_long_pos")
        short_avg = c4.radio("空头均价", ["涨", "跌"], horizontal=True, key="s32x_short_avg")
        short_pos = c5.radio("空头仓位", ["升", "降"], horizontal=True, key="s32x_short_pos")
        key_tuple = (price, long_avg, long_pos, short_avg, short_pos)
        key = "_".join(key_tuple)
        case = cases.get(key_tuple)
        if not case:
            st.warning("没有找到这个组合，请检查 32 结构字典。")
            return
        stance = {"long": "偏多观察", "short": "偏空观察", "neutral": "分歧观察"}.get(str(case.get("stance")), "分歧观察")
        count, latest_time, latest_price = _match_stats(structure_df, key)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("结构编号", f"第 {case.get('case_no')} 种")
        m2.metric("结构倾向", stance)
        m3.metric("结构强度", f"{case.get('strength')}/5")
        m4.metric("当前范围出现次数", str(count))
        st.markdown(f"**结构名称：** {case.get('name')}")
        st.markdown(f"**基础解释：** {case.get('interpretation')}")
        left, right = st.columns(2)
        with left:
            st.markdown("**逐项拆解**")
            if price == "涨":
                st.markdown("- 价格上涨表示当前窗口上行动能或空头回补占优。")
            else:
                st.markdown("- 价格下跌表示当前窗口卖压、空头进攻或多头出清压力占优。")
            if long_avg == "涨":
                st.markdown("- 多头均价上涨表示场内多头平均成本抬高，可能是低成本多头离场，也可能是高成本追多或补仓。")
            else:
                st.markdown("- 多头均价下跌表示多头成本中枢下移，可能是低位多头进入，也可能是高成本多头被清理。")
            if long_pos == "升":
                st.markdown("- 多头仓位上升表示多头敞口增加；只有在成本下降并且价格企稳时才偏健康。")
            else:
                st.markdown("- 多头仓位下降表示多头正在降风险、止盈、止损或被动出清。")
            if short_avg == "涨":
                st.markdown("- 空头均价上涨表示空头平均成本抬高，安全垫相对变厚。")
            else:
                st.markdown("- 空头均价下跌表示空头成本中枢下移，常见于低位追空或高位空头止盈后剩余低成本空头。")
            if short_pos == "升":
                st.markdown("- 空头仓位上升表示空头敞口增加，可能是主动进攻或防守反弹。")
            else:
                st.markdown("- 空头仓位下降表示空头正在止盈、止损或回补。")
        with right:
            st.markdown("**组合后的核心含义**")
            if price == "跌" and long_avg == "涨" and long_pos == "升":
                st.markdown("- 重点风险：价格下跌时多头成本上升且仓位增加，通常不是健康低吸，更像高成本接盘或低成本多头离场后的被动抬均价。")
            if price == "跌" and long_avg == "跌" and long_pos == "升":
                st.markdown("- 潜在承接：价格下跌时多头成本下降且仓位增加，说明低位接货存在；若继续破低，这些承接会变成新的被套盘。")
            if price == "涨" and long_avg == "跌" and long_pos == "升":
                st.markdown("- 偏健康上涨：价格上涨、低成本多头增加、多头均价下降，说明可能有真实低位多头进入。")
            if short_avg == "跌" and short_pos == "升":
                st.markdown("- 空头追空：空头均价下降且仓位增加，短线可能压制价格，但一旦跌不动会积累挤压风险。")
            if long_pos == "升" and short_pos == "升":
                st.markdown("- 多空双边加杠杆，代表分歧扩大，后续波动通常上升。")
            if long_pos == "降" and short_pos == "降":
                st.markdown("- 多空双边去杠杆，代表合约推动力下降，市场可能进入降温或等待新方向。")
            st.markdown("**当前数据中的最近匹配**")
            st.markdown(f"- 最近出现时间：{latest_time}\n- 最近出现价格：{latest_price}\n- 当前范围出现次数：{count}")
        st.markdown("**观察重点**")
        st.markdown("- 下降趋势里，偏多结构更多是反弹观察；上升趋势里，偏空结构更多是回调观察。\n- 等价格确认，不要只因为结构出现就立即判断方向。\n- 如果短时间相反结构频繁出现，说明盘口在噪音区或剧烈博弈区。")
