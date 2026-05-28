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
POCKETBASE_URL = "http://YOUR_POCKETBASE_IP:8090"
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        POCKETBASE_URL = config.get("POCKETBASE_URL", POCKETBASE_URL)

st.set_page_config(
    page_title="Smart Money 量化战术雷达",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义全局亮色主题 CSS（隐藏部分原生UI，调整背景）
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Smart Money 宏观网格监控板")
st.markdown("基于独立卡片的指标/价格交叉分析 (Superset 亮色风格)")

# ==========================================
# 2. 数据获取与处理
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
        
        cols = ['current_price','long_traders','short_traders','total_traders',
                'long_pos_usdt','short_pos_usdt','total_pos_usdt',
                'long_unrealized_pnl','short_unrealized_pnl','funding_rate','ls_ratio',
                'long_pnl_ratio', 'short_pnl_ratio']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        if 'current_price' in df.columns:
            df = df[df['current_price'] > 0]
        if 'long_traders' in df.columns and 'short_traders' in df.columns:
            df = df[(df['long_traders'] > 0) | (df['short_traders'] > 0)]
            
        # PnL 转为负值方便可视化
        if 'short_unrealized_pnl' in df.columns:
            df['short_unrealized_pnl'] = -df['short_unrealized_pnl'].abs()
            
        return df
    except Exception as e:
        st.error(f"数据拉取失败: {e}")
        return pd.DataFrame()

with st.spinner("正在扫描链上战术数据..."):
    df = fetch_and_process_data()

# ==========================================
# 3. 独立卡片网格渲染
# ==========================================
if not df.empty:
    metrics_config = [
        {"col": "ls_ratio", "name": "多空人数比 - BTC", "color": "#14b8a6"}, 
        {"col": "long_pos_usdt", "name": "多头持仓总额 - BTC", "color": "#f43f5e"}, 
        {"col": "short_pos_usdt", "name": "空头持仓总额 - BTC", "color": "#10b981"}, 
        {"col": "long_unrealized_pnl", "name": "多头未实现盈亏", "color": "#3b82f6"}, 
        {"col": "short_unrealized_pnl", "name": "空头未实现盈亏", "color": "#8b5cf6"}, 
        {"col": "funding_rate", "name": "资金费率", "color": "#f59e0b"} 
    ]
    
    # 过滤出当前 df 实际拥有的列
    available_metrics = [m for m in metrics_config if m["col"] in df.columns]
    
    # 创建 3 列布局
    cols = st.columns(3)
    
    for i, config in enumerate(available_metrics):
        metric_col = config["col"]
        metric_name = config["name"]
        metric_color = config["color"]
        
        # 颜色转换函数：Hex to RGBA
        def hex_to_rgba(hex_color, alpha=0.2):
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 6:
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                return f'rgba({r},{g},{b},{alpha})'
            return hex_color

        # 1. 创建独立的双 Y 轴图表
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 2. 主 Y 轴：指标区域图
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], 
                y=df[metric_col],
                name=metric_name, 
                mode='lines',
                line=dict(color=metric_color, width=2),
                fill='tozeroy', 
                fillcolor=hex_to_rgba(metric_color, 0.2) if '#' in metric_color else metric_color.replace(')', ', 0.2)').replace('rgb', 'rgba'),
                hovertemplate=f"%{{y:,.4f}}<extra></extra>"
            ),
            secondary_y=False,
        )
        
        # 3. 次 Y 轴：BTC 价格折线图 (深红色)
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], 
                y=df["current_price"],
                name="BTC 价格", 
                mode='lines',
                line=dict(color='#900C3F', width=2),
                hovertemplate="%{y:,.2f} USDT<extra></extra>"
            ),
            secondary_y=True,
        )
        
        # 4. 设置布局和清爽亮色主题
        fig.update_layout(
            title=dict(
                text=f"<b>{metric_name}</b> vs BTC",
                font=dict(size=16, color='#333333'),
                x=0.01,
                y=0.95
            ),
            height=450,
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#333333"),
            hoverlabel=dict(
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="#e5e7eb",
                font=dict(color="#333333", size=13)
            )
        )
        
        # 5. X 轴开启 rangeslider 和样式
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            rangeslider_visible=True,
            rangeslider=dict(thickness=0.08, bgcolor='#f8f9fa'),
            tickformat="%m-%d %H:%M",
            hoverformat="%Y-%m-%d %H:%M:%S",
            color="#6b7280"
        )
        
        # 主副 Y 轴网格线设置 (极浅的水平线)
        fig.update_yaxes(
            showgrid=True, gridcolor="#f3f4f6", zeroline=True, zerolinecolor="#e5e7eb",
            color=metric_color, secondary_y=False,
            showticklabels=True
        )
        fig.update_yaxes(
            showgrid=False, zeroline=False, 
            color="#900C3F", secondary_y=True,
            showticklabels=True
        )
        
        # 将图表渲染到对应的网格列中
        with cols[i % 3]:
            st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("暂无数据，请检查网络或配置。")
