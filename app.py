import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. 页面配置 (必须作为第一个 Streamlit 命令)
# ==========================================
st.set_page_config(
    page_title="Smart Money 投研终端",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. 顶部 UI 与控制栏
# ==========================================
st.title("📈 Smart Money 投研终端")
st.markdown("基于实时链上资金流向与情绪指标监测分析。")

# 定义可用指标映射
METRICS = {
    "多头未实现盈亏比 (long_pnl_ratio)": "long_pnl_ratio",
    "空头未实现盈亏比 (short_pnl_ratio)": "short_pnl_ratio",
    "多空人数比 (ls_ratio)": "ls_ratio",
    "多头持仓总额 (long_pos_usdt)": "long_pos_usdt",
    "资金费率 (funding_rate)": "funding_rate"
}

# 顶部下拉选择框
col1, col2 = st.columns([1, 3])
with col1:
    selected_label = st.selectbox(
        "选择对比副轴指标:",
        options=list(METRICS.keys())
    )
selected_key = METRICS[selected_label]

# ==========================================
# 3. 数据拉取与处理
# ==========================================
@st.cache_data(ttl=60)  # 设置 60 秒数据缓存
def fetch_data():
    # PocketBase REST API 地址
    url = "http://YOUR_POCKETBASE_IP:8090/api/collections/smart_money_stats/records"
    
    # perPage=500 获取 500 条, sort=-timestamp 按时间降序(最新在前)
    params = {
        "perPage": 500,
        "sort": "-timestamp"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        items = data.get("items", [])
        
        if not items:
            return pd.DataFrame()
            
        # 由于 API 是按时间降序返回，我们需要将其翻转为正序（时间从左到右）
        items.reverse()
        
        # 转换为 Pandas DataFrame
        df = pd.DataFrame(items)
        
        # 解析 ISO 8601 格式，默认带 UTC 时区信息
        # 转换为东八区 (Asia/Shanghai) 的本地时间，并移除时区信息供 Plotly 平滑显示
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        
        # 确保关键数值列为浮点数
        numeric_cols = ["current_price"] + list(METRICS.values())
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        return df
        
    except Exception as e:
        st.error(f"获取 API 数据失败: {e}")
        return pd.DataFrame()

# ==========================================
# 4. 图表绘制
# ==========================================
with st.spinner("正在从区块链拉取最新数据..."):
    df = fetch_data()

if not df.empty:
    # 提取短名称作为图例
    short_metric_name = selected_label.split(" ")[0]
    
    # 创建带有双 Y 轴的画布
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # ------------------ 绘制副轴 (右侧, 指标柱状图) ------------------
    # 先画柱状图使其位于底层，防止遮挡价格线
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df[selected_key],
            name=short_metric_name,
            opacity=0.4,             # 增加透明度防遮挡
            marker_color="#3b82f6"   # 蓝色系
        ),
        secondary_y=True,
    )
    
    # ------------------ 绘制主轴 (左侧, 价格折线图) ------------------
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["current_price"],
            name="BTC 当前价格",
            mode='lines',            # 平滑折线图
            line=dict(
                color='#ef4444',     # 红色系，醒目
                width=2.5
            )
        ),
        secondary_y=False,
    )
    
    # ------------------ 图表布局与交互配置 ------------------
    fig.update_layout(
        # 十字准星与悬浮提示数据漫游 (同时显示 X 轴对应的所有数据)
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            orientation="h",         # 水平图例
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        # 背景颜色设为透明以适应 Streamlit 默认暗色主题
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    # 配置 X 轴：开启底部时间滑动条 (Range Slider)
    fig.update_xaxes(
        rangeslider_visible=True,
        showgrid=False,
        tickformat="%m月%d日 %H:%M",
        hoverformat="%Y年%m月%d日 %H:%M:%S"
    )
    
    # 配置左侧主 Y 轴 (价格)
    fig.update_yaxes(
        title_text="<b>BTC 价格 (USDT)</b>", 
        secondary_y=False,
        showgrid=False,
        color="#ef4444"
    )
    
    # 配置右侧副 Y 轴 (指标)
    fig.update_yaxes(
        title_text=f"<b>{short_metric_name}</b>", 
        secondary_y=True,
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.1)",
        griddash="dot",
        color="#3b82f6"
    )

    # ==========================================
    # 5. 渲染页面
    # ==========================================
    # 强制全宽自适应容器渲染
    st.plotly_chart(fig, width="stretch")
    
else:
    st.warning("暂无数据可展示，请检查数据库服务是否正常。")
