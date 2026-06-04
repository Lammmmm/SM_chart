import streamlit as st

from backtest_page import render_backtest_page


st.set_page_config(
    page_title="Smart Money 策略回测平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


render_backtest_page(embedded_in_console=False)
