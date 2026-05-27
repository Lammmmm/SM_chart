import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# ==========================================
# 1. 配置加载与页面初始化
# ==========================================
config_path = "config.json"
# 预置占位符，安全加载配置
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
            # 增加 filter 参数，直接在数据库层面过滤价格 <= 0 的异常脏数据
            params = {"perPage": 500, "page": page, "sort": "-timestamp", "filter": "current_price > 0"}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            batch = data.get("items", [])
            if not batch:
                break
            items.extend(batch)
            
            # 如果当前页拉取的数据少于 500 条，说明已经到底了
            if len(batch) < 500:
                break
            page += 1
            
        if not items:
            return pd.DataFrame()
            
        # 1. 翻转为时间正序（从左到右展现）
        items.reverse()
        df = pd.DataFrame(items)
        
        # 2. 时区处理：转换为东八区
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        
        # 将需要的原始字段转换为浮点数
        cols_to_convert = [
            'current_price', 'long_traders', 'short_traders', 'total_traders', 
            'long_pos_usdt', 'short_pos_usdt', 'total_pos_usdt', 
            'long_unrealized_pnl', 'short_unrealized_pnl', 'funding_rate', 'ls_ratio'
        ]
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        # ---------------- 本地内存级异常值清洗 ----------------
        if 'current_price' in df.columns:
            df = df[df['current_price'] > 0] # 剔除解析后仍为0或负数的诡异数据
        if 'long_traders' in df.columns and 'short_traders' in df.columns:
            df = df[(df['long_traders'] > 0) | (df['short_traders'] > 0)] # 剔除交易所宕机导致的多空人数全为0的数据点
        
        # ---------------- 核心战术模型（Pandas 衍生报警数据） ----------------
        
        # 衍生 1：T1 级巨鲸脉冲 (Delta) - 计算持仓资金的变化绝对值
        # 多头为正向，空头翻转为负向 (乘以 -1 确保空头脉冲往 0 轴下方画)
        if 'long_pos_usdt' in df.columns and 'short_pos_usdt' in df.columns:
            df['long_pos_delta'] = df['long_pos_usdt'].diff().fillna(0)
            df['short_pos_delta'] = df['short_pos_usdt'].diff().fillna(0) * -1
        else:
            df['long_pos_delta'] = 0
            df['short_pos_delta'] = 0
            
        # 衍生 2：盈亏双向化 - 将空头未实现盈亏转到 0 轴下方
        if 'short_unrealized_pnl' in df.columns:
            df['short_unrealized_pnl'] = -df['short_unrealized_pnl'].abs()
        
        return df
        
    except Exception as e:
        st.error(f"数据拉取或战术计算失败: {e}")
        return pd.DataFrame()

with st.spinner("正在扫描链上战术数据..."):
    df = fetch_and_process_data()

# ==========================================
# 3. 动态堆叠子图渲染 (1对多联动)
# ==========================================
if not df.empty:
    # 定义 4 行子图，仅在 Row 2 启用次坐标轴 (secondary_y=True)
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True,           # 全局 X 轴完美联动
        vertical_spacing=0.03,       # 紧凑排列
        row_heights=[0.35, 0.25, 0.20, 0.20], # 高度动态分配
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}]
        ]
    )
    
    # ---------------- Row 1: 主图 (价格走势) ----------------
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["current_price"],
            name="BTC 当前价格",
            mode='lines',
            line=dict(color='#00E676', width=2.5) # 极客感亮绿
        ),
        row=1, col=1
    )
    fig.update_yaxes(title_text="<b>BTC 价格 (USDT)</b>", row=1, col=1)

    # ---------------- Row 2: T0 级散户背离监控 ----------------
    # 目的：展示散户人数飙升，但主力总资金未跟上或撤退的诱多诱空陷阱
    
    # 主 Y 轴 (左)：散户人数柱子 (极高可见度配色)
    fig.add_trace(
        go.Bar(
            x=df["timestamp"], y=df["long_traders"],
            name="多头人数",
            marker_color="#A78BFA", # 亮紫色，与深色背景及蓝色折线形成强烈视觉对比
            opacity=0.9
        ),
        row=2, col=1, secondary_y=False
    )
    # 次 Y 轴 (右)：总资金折线 (亮色展示趋势)
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["long_pos_usdt"],
            name="多头总资金",
            mode='lines',
            line=dict(color='#2962FF', width=3) # 加粗亮蓝
        ),
        row=2, col=1, secondary_y=True
    )
    fig.update_yaxes(title_text="多头人数", row=2, col=1, secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="总资金 (USDT)", row=2, col=1, secondary_y=True, showgrid=True)

    # ---------------- Row 3: 轧空绞肉机 ----------------
    # 一眼展示市场大爆仓时的多空血腥程度
    if 'long_unrealized_pnl' in df.columns and 'short_unrealized_pnl' in df.columns:
        # 核心修复：Plotly 原生机制下纯柱状图无法全屏捕获悬停，在此植入一条隐形折线，用来完美吸附鼠标
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df["long_unrealized_pnl"],
                mode='lines', line=dict(color='rgba(0,0,0,0)', width=0),
                hoverinfo='skip', showlegend=False
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=df["timestamp"], y=df["long_unrealized_pnl"],
                name="多头未实现盈亏",
                marker_color="#00E676", # 绿色
                opacity=0.85
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(
                x=df["timestamp"], y=df["short_unrealized_pnl"],
                name="空头未实现盈亏",
                marker_color="#FF1744", # 红色向深渊延伸
                opacity=0.85
            ),
            row=3, col=1
        )
    fig.update_yaxes(title_text="未实现盈亏", row=3, col=1)

    # ---------------- Row 4: 多头账户占比 (0-100%) ----------------
    if 'ls_ratio' in df.columns:
        # 将原始多空比 (比如 1.5) 转换为多头百分比 (比如 60%)
        long_percent = (df["ls_ratio"] / (df["ls_ratio"] + 1)) * 100
        
        # 核心修复：同上，植入隐形折线完美吸附鼠标
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=long_percent,
                mode='lines', line=dict(color='rgba(0,0,0,0)', width=0),
                hoverinfo='skip', showlegend=False
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=df["timestamp"], y=long_percent,
                name="多头占比(%)",
                marker_color="#F59E0B", # 橙色
                opacity=0.9
            ),
            row=4, col=1
        )
    fig.update_yaxes(title_text="多头占比(%)", range=[0, 100], row=4, col=1)

    # ==========================================
    # 全局交互与高级极客暗黑 UI
    # ==========================================
    fig.update_layout(
        height=900,                  # 高度撑满视野
        hovermode="x",               # 核心修复：改为 'x' 模式。悬停时，所有子图同一时间点的数据会同时弹出！
        hoverdistance=-1,            # 强制捕获跨图层悬停
        spikedistance=-1,            # 强制捕获跨图层准星
        showlegend=False,            # 屏蔽图例，专注图形
        barmode='relative',          # 确保正负向的 Bar 在 0 轴两侧正确渲染不遮挡
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0b0e11",     # 纯正暗黑风
        plot_bgcolor="#0b0e11",
        font=dict(color="#848e9c"),
        hoverlabel=dict(
            bgcolor="rgba(11, 14, 17, 0.95)",
            bordercolor="#2b3139",
            font_size=13
        )
    )

    # 统一化极度微弱的网格线与全局贯穿十字准星 (Spike Lines)
    for row in range(1, 5):
        fig.update_xaxes(
            showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, 
            showspikes=True, spikemode="across", spikecolor="#6B7280", spikethickness=1, spikedash="solid",
            row=row, col=1
        )
        # 为主坐标轴添加微弱网格，并开启 Y 轴水平十字准星，组合成完整的 TradingView 十字光标
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(255,255,255,0.05)", 
            zeroline=True, zerolinecolor="rgba(255,255,255,0.15)", 
            showspikes=True, spikemode="across", spikecolor="#6B7280", spikethickness=1, spikedash="solid",
            row=row, col=1, secondary_y=False
        )

    # 强制关闭 RangeSlider (它会导致多子图 x unified 失效)，采用原生的鼠标框选缩放
    fig.update_xaxes(
        rangeslider_visible=False,
        tickformat="%m月%d日 %H:%M",
        hoverformat="%Y年%m月%d日 %H:%M:%S",
        row=4, col=1
    )

    # 渲染页面，自适应浏览器宽度
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("暂无数据，请检查网络或配置。")
