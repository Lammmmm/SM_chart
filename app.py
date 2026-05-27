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
    if 'ls_ratio' in df.columns:
        df['long_percent'] = (df["ls_ratio"] / (df["ls_ratio"] + 1)) * 100
    else:
        df['long_percent'] = 0

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
        row_heights=[0.4, 0.2, 0.2, 0.2], 
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # ---------------- Row 1: 主图 (价格走势) ----------------
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["current_price"],
            name="BTC 价格",
            mode='lines',
            line=dict(color='#00E676', width=2.5),
            hovertemplate="%{y:,.2f} USDT<extra></extra>"
        ),
        row=1, col=1
    )
    fig.update_yaxes(title_text="<b>BTC 价格 (USDT)</b>", row=1, col=1)

    # ---------------- Row 2: 多头人数与总资金 (T0 级散户背离监控) ----------------
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["long_traders"],
            name="多头人数",
            mode='lines',
            line=dict(color="#A78BFA", width=1),
            fill='tozeroy',
            fillcolor='rgba(167, 139, 250, 0.85)',
            hovertemplate="%{y:,.0f} 人<extra></extra>"
        ),
        row=2, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["long_pos_usdt"],
            name="多头总资金",
            mode='lines',
            line=dict(color='#2962FF', width=3),
            hovertemplate="%{y:,.0f} U<extra></extra>"
        ),
        row=2, col=1, secondary_y=True
    )
    fig.update_yaxes(title_text="多头人数", row=2, col=1, secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="总资金", row=2, col=1, secondary_y=True, showgrid=True)

    # ---------------- Row 3: 轧空绞肉机 (面积图) ----------------
    if 'long_unrealized_pnl' in df.columns and 'short_unrealized_pnl' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df["long_unrealized_pnl"],
                name="多头未实现盈亏",
                mode='lines',
                line=dict(color="#00E676", width=1),
                fill='tozeroy',
                fillcolor='rgba(0, 230, 118, 0.85)',
                hovertemplate="%{y:,.0f} U<extra></extra>"
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df["short_unrealized_pnl"],
                name="空头未实现盈亏",
                mode='lines',
                line=dict(color="#FF1744", width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 23, 68, 0.85)',
                hovertemplate="%{y:,.0f} U<extra></extra>"
            ),
            row=3, col=1
        )
    fig.update_yaxes(title_text="未实现盈亏", row=3, col=1)

    # ---------------- Row 4: 多头账户占比 (面积图) ----------------
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["long_percent"],
            name="多头占比",
            mode='lines',
            line=dict(color="#F59E0B", width=1),
            fill='tozeroy',
            fillcolor='rgba(245, 158, 11, 0.9)',
            hovertemplate="%{y:.1f}%<extra></extra>"
        ),
        row=4, col=1
    )
    fig.update_yaxes(title_text="多头占比(%)", range=[0, 100], row=4, col=1)

    # ==========================================
    # 全局交互与高级极客暗黑 UI
    # ==========================================
    fig.update_layout(
        height=900,                  
        hovermode="x unified",       
        hoversubplots="axis",        # 解决跨子图联动与双 Y 轴冲突的唯一核心开关
        showlegend=False,            
        barmode='relative',          
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0b0e11",     
        plot_bgcolor="#0b0e11",
        font=dict(color="#848e9c"),
        hoverlabel=dict(
            bgcolor="rgba(11, 14, 17, 0.95)",
            bordercolor="#2b3139",
            font_size=13
        )
    )

    # 强制所有 X 轴贯穿垂直十字准星，关闭 Y 轴水平准星防止杂乱
    for row in range(1, 5):
        fig.update_xaxes(
            showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, 
            showspikes=True, spikemode="across", spikecolor="#999999", spikethickness=1, spikedash="solid",
            rangeslider_visible=False,
            row=row, col=1
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(255,255,255,0.05)", 
            zeroline=True, zerolinecolor="rgba(255,255,255,0.15)", 
            showspikes=False, 
            row=row, col=1, secondary_y=False
        )

    # 针对 Row 4 的底层 X 轴，特殊格式化时间戳
    fig.update_xaxes(
        tickformat="%m月%d日 %H:%M",
        hoverformat="%Y年%m月%d日 %H:%M:%S",
        row=4, col=1
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("暂无数据，请检查网络或配置。")
