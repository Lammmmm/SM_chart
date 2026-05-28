import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import json
import os

# ==========================================
# 1. 配置加载与页面初始化
# ==========================================
config_path = "config.json"
POCKETBASE_URL = "http://YOUR_POCKETBASE_IP:8090"
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        POCKETBASE_URL = config.get("POCKETBASE_URL", POCKETBASE_URL)

st.set_page_config(
    page_title="Smart Money 量化战术雷达",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🎯 Smart Money 量化战术雷达")
st.markdown("巨鲸脉冲与散户绞肉机全景监控面板 (CoinGlass 专业堆叠风格)")

# ==========================================
# 2. 数据获取与战术衍生计算 (Pandas)
# ==========================================
@st.cache_data(ttl=60)
def fetch_and_process_data():
    url = f"{POCKETBASE_URL.rstrip('/')}/api/collections/smart_money_stats/records"
    try:
        items = []
        page = 1
        while True:
            params = {"perPage": 500, "page": page, "sort": "-timestamp", "filter": "current_price > 0"}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            batch = data.get("items", [])
            if not batch:
                break
            items.extend(batch)
            if len(batch) < 500:
                break
            page += 1

        if not items:
            return pd.DataFrame()

        items.reverse()
        df = pd.DataFrame(items)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)

        cols_to_convert = [
            'current_price', 'long_traders', 'short_traders', 'total_traders',
            'long_pos_usdt', 'short_pos_usdt', 'total_pos_usdt',
            'long_unrealized_pnl', 'short_unrealized_pnl', 'funding_rate', 'ls_ratio',
            'long_pnl_ratio', 'short_pnl_ratio'
        ]
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        if 'current_price' in df.columns:
            df = df[df['current_price'] > 0]
        if 'long_traders' in df.columns and 'short_traders' in df.columns:
            df = df[(df['long_traders'] > 0) | (df['short_traders'] > 0)]

        if 'long_pos_usdt' in df.columns and 'short_pos_usdt' in df.columns:
            df['long_pos_delta'] = df['long_pos_usdt'].diff().fillna(0)
            df['short_pos_delta'] = df['short_pos_usdt'].diff().fillna(0) * -1
        else:
            df['long_pos_delta'] = 0
            df['short_pos_delta'] = 0

        if 'short_unrealized_pnl' in df.columns:
            df['short_unrealized_pnl'] = -df['short_unrealized_pnl'].abs()

        return df

    except Exception as e:
        st.error(f"数据拉取或战术计算失败: {e}")
        return pd.DataFrame()

with st.spinner("正在扫描链上战术数据..."):
    df = fetch_and_process_data()

# ==========================================
# 3. 动态堆叠子图渲染
# ==========================================
if not df.empty:
    if 'ls_ratio' in df.columns:
        df['long_percent'] = (df["ls_ratio"] / (df["ls_ratio"] + 1)) * 100
    else:
        df['long_percent'] = 0

    # ══════════════════════════════════════════════════════════════════════
    # 架构：go.Figure() + 手动 domain 分域，所有 trace 共用同一个 xaxis='x'
    #
    # 这是解决"垂直准星贯穿全图"的唯一正确方案。
    # make_subplots 无论如何配置，内部都会生成多个 xaxis 对象（xaxis/xaxis2/...），
    # 导致 spikemode='across' 只能在当前子图内画线，永远无法跨行。
    #
    # 用 domain 手动控制每行的垂直位置，spikemode='across' 才能贯穿全图。
    # ══════════════════════════════════════════════════════════════════════

    # 行高比 [0.4, 0.2, 0.2, 0.2]，间距 0.03，可用高度=1-3×0.03=0.91
    # 从下往上累加：
    D = {
        4: [0.000, 0.182],   # 多头占比（底部）
        3: [0.212, 0.394],   # 未实现盈亏
        2: [0.424, 0.606],   # 多头人数 + 资金
        1: [0.636, 1.000],   # BTC 价格（顶部，最高）
    }
    GC  = "rgba(255,255,255,0.05)"
    ZLC = "rgba(255,255,255,0.15)"

    fig = go.Figure()

    # ── Row 1: BTC 价格 ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["current_price"],
        name="BTC 价格", mode='lines',
        line=dict(color='#00E676', width=2.5),
        hovertemplate="%{y:,.2f} USDT<extra></extra>",
        xaxis='x', yaxis='y'
    ))

    # ── Row 2: 多头人数（y2）+ 多头总资金（y5，副轴）──────────────────
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["long_traders"],
        name="多头人数", mode='lines',
        line=dict(color="#A78BFA", width=1),
        fill='tozeroy', fillcolor='rgba(167,139,250,0.85)',
        hovertemplate="%{y:,.0f} 人<extra></extra>",
        xaxis='x', yaxis='y2'
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["long_pos_usdt"],
        name="多头总资金", mode='lines',
        line=dict(color='#2962FF', width=3),
        hovertemplate="%{y:,.0f} U<extra></extra>",
        xaxis='x', yaxis='y5'
    ))

    # ── Row 3: 未实现盈亏 ───────────────────────────────────────────────
    if 'long_unrealized_pnl' in df.columns and 'short_unrealized_pnl' in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["long_unrealized_pnl"],
            name="多头未实现盈亏", mode='lines',
            line=dict(color="#00E676", width=1),
            fill='tozeroy', fillcolor='rgba(0,230,118,0.85)',
            hovertemplate="%{y:,.0f} U<extra></extra>",
            xaxis='x', yaxis='y3'
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["short_unrealized_pnl"],
            name="空头未实现盈亏", mode='lines',
            line=dict(color="#FF1744", width=1),
            fill='tozeroy', fillcolor='rgba(255,23,68,0.85)',
            hovertemplate="%{y:,.0f} U<extra></extra>",
            xaxis='x', yaxis='y3'
        ))

    # ── Row 4: 比例指标（多头占比 + 盈亏比）──────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["long_percent"],
        name="多头占比", mode='lines',
        line=dict(color="#F59E0B", width=1),
        fill='tozeroy', fillcolor='rgba(245,158,11,0.3)',
        hovertemplate="%{y:.1f}%<extra></extra>",
        xaxis='x', yaxis='y4'
    ))
    if 'long_pnl_ratio' in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["long_pnl_ratio"],
            name="多头盈亏比", mode='lines',
            line=dict(color="#00E676", width=1.5, dash='dot'),
            hovertemplate="多头盈亏比: %{y:.2f}%<extra></extra>",
            xaxis='x', yaxis='y4'
        ))
    if 'short_pnl_ratio' in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["short_pnl_ratio"],
            name="空头盈亏比", mode='lines',
            line=dict(color="#FF1744", width=1.5, dash='dot'),
            hovertemplate="空头盈亏比: %{y:.2f}%<extra></extra>",
            xaxis='x', yaxis='y4'
        ))

    # ── Layout ───────────────────────────────────────────────────────────
    fig.update_layout(
        height=900,
        hovermode="x unified",
        hoversubplots="axis",
        showlegend=False,
        margin=dict(l=10, r=80, t=40, b=50),
        paper_bgcolor="#0b0e11",
        plot_bgcolor="#0b0e11",
        font=dict(color="#848e9c"),
        hoverlabel=dict(bgcolor="rgba(11,14,17,0.95)", bordercolor="#2b3139", font_size=13),

        # ★ 唯一的 xaxis：spikemode='across' 会从图顶画到图底，贯穿所有 domain
        xaxis=dict(
            domain=[0, 1],
            showgrid=True, gridcolor=GC, zeroline=False,
            showspikes=True, spikemode="across",
            spikecolor="#888888", spikethickness=1, spikedash="solid",
            rangeslider_visible=False,
            tickformat="%m月%d日 %H:%M",
            hoverformat="%Y年%m月%d日 %H:%M:%S",
            color="#848e9c",
        ),

        # y (Row 1 - BTC价格)
        yaxis=dict(
            domain=D[1], anchor='x',
            title=dict(text="<b>BTC 价格 (USDT)</b>"),
            showgrid=True, gridcolor=GC, zeroline=False, showspikes=False,
        ),
        # y2 (Row 2 - 多头人数，主轴)
        yaxis2=dict(
            domain=D[2], anchor='x',
            title=dict(text="多头人数"),
            showgrid=False, zeroline=False, showspikes=False,
        ),
        # y3 (Row 3 - 未实现盈亏)
        yaxis3=dict(
            domain=D[3], anchor='x',
            title=dict(text="未实现盈亏"),
            showgrid=True, gridcolor=GC,
            zeroline=True, zerolinecolor=ZLC, showspikes=False,
        ),
        # y4 (Row 4 - 比例指标，最底部)
        yaxis4=dict(
            domain=D[4], anchor='x',
            title=dict(text="比率指标(%)"),
            showgrid=True, gridcolor=GC, zeroline=False, showspikes=False,
        ),
        # y5 (Row 2 副轴 - 多头总资金，右侧)
        # anchor='x' 因为现在只有一个 xaxis，overlaying='y2' 确保画在 Row 2 域内
        yaxis5=dict(
            overlaying='y2', side='right', anchor='x',
            title=dict(text='总资金', font=dict(color='#2962FF')),
            showgrid=False, showspikes=False,
            tickfont=dict(color='#2962FF'),
            tickformat=',.0f', zeroline=False,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("暂无数据，请检查网络或配置。")
