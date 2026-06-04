import streamlit as st


COMMON_STYLE = """
<style>
.block-container {
    padding-top: 1.15rem;
    padding-bottom: 1rem;
}
.status-card {
    background: linear-gradient(180deg, #151b23 0%, #0f141b 100%);
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 14px 16px;
    min-height: 92px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.status-label {
    color: #8b949e;
    font-size: 0.82rem;
    margin-bottom: 0.35rem;
}
.status-value {
    font-size: 1.16rem;
    font-weight: 700;
    line-height: 1.35;
}
.status-sub {
    color: #6b7280;
    font-size: 0.76rem;
    margin-top: 0.3rem;
}
.console-hero {
    background: linear-gradient(135deg, rgba(17,22,29,0.96) 0%, rgba(11,14,17,0.98) 100%);
    border: 1px solid #1f2937;
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 1rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.console-kicker {
    color: #8b949e;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.console-title {
    color: #E6EDF3;
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.25;
}
.console-subtitle {
    color: #AAB4C3;
    font-size: 0.92rem;
    margin-top: 0.45rem;
}
[data-testid="stSidebar"] {
    background: #0f141b;
}
</style>
"""


def apply_common_style() -> None:
    st.markdown(COMMON_STYLE, unsafe_allow_html=True)


def render_console_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="console-hero">
            <div class="console-kicker">{kicker}</div>
            <div class="console-title">{title}</div>
            <div class="console-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_switch(default_page: str = "实时雷达", key: str = "workspace_page") -> str:
    page_options = ["实时雷达", "策略回测"]
    default_index = page_options.index(default_page) if default_page in page_options else 0
    st.sidebar.markdown("### 控制台入口")
    selected_page = st.sidebar.radio(
        "功能页面",
        page_options,
        index=default_index,
        key=key,
    )
    st.sidebar.caption("一个入口切换实时雷达与回测，不需要记两套启动命令。")
    st.sidebar.markdown("---")
    return selected_page
