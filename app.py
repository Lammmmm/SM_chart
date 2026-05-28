import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
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
                'long_pnl_ratio', 'short_pnl_ratio', 'long_avg_price', 'short_avg_price']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        if 'current_price' in df.columns:
            df = df[df['current_price'] > 0]
        if 'long_traders' in df.columns and 'short_traders' in df.columns:
            df = df[(df['long_traders'] > 0) | (df['short_traders'] > 0)]
            

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
    ALL_METRICS = [
        {"col": "ls_ratio", "name": "多空人数比", "color": "#14b8a6"}, 
        {"col": "long_pos_usdt", "name": "多头持仓总额", "color": "#f43f5e"}, 
        {"col": "short_pos_usdt", "name": "空头持仓总额", "color": "#10b981"}, 
        {"col": "long_unrealized_pnl", "name": "多头未实现盈亏", "color": "#3b82f6"}, 
        {"col": "short_unrealized_pnl", "name": "空头未实现盈亏", "color": "#8b5cf6"}, 
        {"col": "funding_rate", "name": "资金费率", "color": "#f59e0b"},
        {"col": "long_pnl_ratio", "name": "多头盈亏比", "color": "#10b981"},
        {"col": "short_pnl_ratio", "name": "空头盈亏比", "color": "#f43f5e"},
        {"col": "long_avg_price", "name": "多头开仓均价", "color": "#3b82f6"},
        {"col": "short_avg_price", "name": "空头开仓均价", "color": "#8b5cf6"},
        {"col": "long_traders", "name": "多头人数", "color": "#3b82f6"},
        {"col": "short_traders", "name": "空头人数", "color": "#8b5cf6"},
        {"col": "total_traders", "name": "总人数", "color": "#64748b"},
        {"col": "total_pos_usdt", "name": "总持仓总额", "color": "#0ea5e9"}
    ]
    
    # 过滤出当前 df 实际拥有的列
    available_metrics = [m for m in ALL_METRICS if m["col"] in df.columns]
    
    # 根据用户要求，将默认启动的 6 个指标固定为以下内容（上下对齐，多空对比）
    default_cols = [
        "long_pnl_ratio", "long_traders", "long_avg_price",
        "short_pnl_ratio", "short_traders", "short_avg_price"
    ]
    
    # 计算默认的 24 小时显示区间
    x_max = df["timestamp"].max()
    x_min = x_max - pd.Timedelta(hours=24)
    x_full_min = df["timestamp"].min()
    
    # 创建 3 列布局
    cols = st.columns(3)
    
    for i in range(6):
        with cols[i % 3]:
            # 找到默认指标在下拉菜单中的索引
            target_col = default_cols[i] if i < len(default_cols) else available_metrics[0]["col"]
            def_idx = 0
            for idx, m in enumerate(available_metrics):
                if m["col"] == target_col:
                    def_idx = idx
                    break
                    
            # 渲染独立的下拉菜单 (替换原有的静态标题)
            selected_metric = st.selectbox(
                "指标",
                options=available_metrics,
                format_func=lambda x: f"{x['name']} vs BTC",
                index=def_idx,
                key=f"chart_metric_{i}",
                label_visibility="collapsed"
            )
            
            metric_col = selected_metric["col"]
            metric_name = selected_metric["name"]
            metric_color = selected_metric["color"]
            
            # 基础颜色策略
            base_color = metric_color
            
            # 如果本身是盈亏比，强制把它的基础色覆盖为绿色
            if metric_col in ["long_pnl_ratio", "short_pnl_ratio"]:
                base_color = "#10b981"
            
            # 判断当前图表是否属于多头或空头阵营，从而决定参考哪个盈亏比进行同步变色
            ref_col = None
            if metric_col.startswith("long_") and "long_pnl_ratio" in df.columns:
                ref_col = "long_pnl_ratio"
            elif metric_col.startswith("short_") and "short_pnl_ratio" in df.columns:
                ref_col = "short_pnl_ratio"
                
            # 针对多空阵营的所有指标，根据对应的盈亏比（<0.1 或 >0.9）进行颜色同步预警
            if ref_col:
                colors = []
                for val in df[ref_col]:
                    if pd.isna(val):
                        colors.append(base_color)
                    elif val < 0.1:
                        colors.append("#ef4444") # 红色预警
                    elif val > 0.9:
                        colors.append("#f59e0b") # 亮橙黄高亮
                    else:
                        colors.append(base_color)
                bar_color = colors
                yaxis_color = base_color
            else:
                bar_color = base_color
                yaxis_color = base_color
        
            # 颜色转换函数：Hex to RGBA
            def hex_to_rgba(hex_color, alpha=0.2):
                hex_color = hex_color.lstrip('#')
                if len(hex_color) == 6:
                    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    return f'rgba({r},{g},{b},{alpha})'
                return hex_color

            # 1. 创建独立的双 Y 轴图表
            fig = make_subplots(specs=[[{"secondary_y": True}]])
        
            # 2. 主 Y 轴：指标图表 (均价使用折线图，其余使用柱状图)
            if metric_col.endswith("_avg_price"):
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"], 
                        y=df[metric_col],
                        name=metric_name, 
                        mode='lines',
                        line=dict(color=yaxis_color, width=2),
                        hovertemplate=f"%{{y:,.4f}}<extra></extra>"
                    ),
                    secondary_y=False,
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=df["timestamp"], 
                        y=df[metric_col],
                        name=metric_name, 
                        marker_color=bar_color,
                        opacity=0.75,
                        width=1000 * 60 * 4, # 强制柱子宽度为 4 分钟 (假设5分钟采集一次)，避免时间轴上柱子太细
                        hovertemplate=f"%{{y:,.4f}}<extra></extra>"
                    ),
                    secondary_y=False,
                )
        
            # 3. 次 Y 轴：BTC 价格折线图 (深红色)
            # 如果是均价类指标，则强制将 BTC 价格也画在主 Y 轴上，保证二者比例尺绝对统一
            is_price_metric = metric_col.endswith("_avg_price")
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"], 
                    y=df["current_price"],
                    name="BTC 价格", 
                    mode='lines',
                    line=dict(color='#900C3F', width=2),
                    hovertemplate="%{y:,.2f} USDT<extra></extra>"
                ),
                secondary_y=not is_price_metric,
            )
        
            # 4. 设置布局和清爽亮色主题
            fig.update_layout(
                height=450,
                hovermode="x unified",
                showlegend=False,
                margin=dict(l=10, r=10, t=20, b=10), # 顶部边距调小，因为有了下拉菜单
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
                range=[x_min, x_max], # 默认只显示最近 24 小时，让图表撑大
                rangeslider_visible=True,
                rangeslider=dict(thickness=0.08, bgcolor='#f8f9fa', range=[x_full_min, x_max]), # 下方的导航条仍然保留所有历史数据
                tickformat="%m-%d %H:%M",
                hoverformat="%Y-%m-%d %H:%M:%S",
                color="#6b7280"
            )
        
            # 主副 Y 轴网格线设置 (极浅的水平线)
            fig.update_yaxes(
                showgrid=True, gridcolor="#f3f4f6", zeroline=True, zerolinecolor="#e5e7eb",
                color=yaxis_color, secondary_y=False,
                showticklabels=True
            )
            fig.update_yaxes(
                showgrid=False, zeroline=False, 
                color="#900C3F", secondary_y=True,
                showticklabels=not is_price_metric
            )
        
            # 将图表渲染到当前的上下文中 (已在 with cols[i % 3] 中)
            st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("暂无数据，请检查网络或配置。")

# ==========================================
# 4. 自动化：每 6 分钟自动刷新全局页面
# ==========================================
components.html(
    """
    <script>
    // 360000 毫秒 = 6 分钟
    setTimeout(function(){
        window.parent.location.reload();
    }, 360000);
    </script>
    """,
    height=0,
    width=0,
)