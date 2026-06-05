from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from market_structure32 import add_structure32_columns, render_structure32_section
from plotly.subplots import make_subplots
from structure32_case_explorer import render_structure32_case_explorer


CONFIG_PATH = "config.json"
DEFAULT_POCKETBASE_URL = "http://YOUR_POCKETBASE_IP:8090"
LOCAL_CACHE_PATH = Path("smart_money_cache.json")
SYNC_INTERVAL_SECONDS = 5 * 60

NUMERIC_COLUMNS = [
    "current_price",
    "long_traders",
    "short_traders",
    "total_traders",
    "long_pos_usdt",
    "short_pos_usdt",
    "total_pos_usdt",
    "long_unrealized_pnl",
    "short_unrealized_pnl",
    "funding_rate",
    "ls_ratio",
    "long_pnl_ratio",
    "short_pnl_ratio",
    "long_avg_price",
    "short_avg_price",
]

ALL_METRICS = [
    {"col": "ls_ratio", "name": "多空人数比", "color": "#64748b"},
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
    {"col": "total_pos_usdt", "name": "总持仓总额", "color": "#0ea5e9"},
]

DEFAULT_METRIC_COLUMNS = [
    "long_pnl_ratio",
    "long_traders",
    "long_pos_usdt",
    "long_avg_price",
    "short_pnl_ratio",
    "short_traders",
    "short_pos_usdt",
    "short_avg_price",
]


def load_pocketbase_url() -> str:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            config = json.load(file)
            return config.get("POCKETBASE_URL", DEFAULT_POCKETBASE_URL)
    return DEFAULT_POCKETBASE_URL


POCKETBASE_URL = load_pocketbase_url()


def emit_server_log(message: str, **details: Any) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if details:
        detail_text = " | ".join(f"{key}={value}" for key, value in details.items())
        print(f"[SmartMoney {timestamp}] {message} | {detail_text}", flush=True)
    else:
        print(f"[SmartMoney {timestamp}] {message}", flush=True)


def escape_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def fetch_smart_money_records(since: str | None = None) -> list[dict[str, Any]]:
    url = f"{POCKETBASE_URL.rstrip('/')}/api/collections/smart_money_stats/records"
    sort_order = "timestamp" if since else "-timestamp"
    filter_parts = ["current_price > 0"]
    if since:
        filter_parts.append(f'timestamp > "{escape_filter_value(since)}"')

    items: list[dict[str, Any]] = []
    page = 1
    while True:
        response = requests.get(
            url,
            params={
                "perPage": 500,
                "page": page,
                "sort": sort_order,
                "filter": " && ".join(f"({part})" for part in filter_parts),
            },
            timeout=15,
        )
        response.raise_for_status()
        batch = response.json().get("items", [])
        if not batch:
            break
        items.extend(batch)
        if len(batch) < 500:
            break
        page += 1

    if not since:
        items.reverse()
    return items


def sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: str(record.get("timestamp", "")))


def get_record_key(record: dict[str, Any]) -> str:
    return str(record.get("id") or record.get("timestamp") or "")


def merge_records(old_records: list[dict[str, Any]], new_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for record in [*old_records, *new_records]:
        key = get_record_key(record)
        if key:
            merged[key] = record
    return sort_records(list(merged.values()))


def get_latest_timestamp(records: list[dict[str, Any]]) -> str | None:
    if not records:
        return None
    latest = sort_records(records)[-1].get("timestamp")
    return str(latest) if latest else None


def build_cache_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_records = sort_records(records)
    return {
        "schema_version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_timestamp": get_latest_timestamp(sorted_records),
        "record_count": len(sorted_records),
        "records": sorted_records,
    }


def load_local_cache() -> dict[str, Any] | None:
    if not LOCAL_CACHE_PATH.exists():
        return None
    with LOCAL_CACHE_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{LOCAL_CACHE_PATH} 格式错误：根节点不是 JSON object")
    if not isinstance(payload.get("records", []), list):
        raise ValueError(f"{LOCAL_CACHE_PATH} 格式错误：records 不是数组")
    return payload


def save_local_cache(records: list[dict[str, Any]]) -> dict[str, Any]:
    payload = build_cache_payload(records)
    tmp_path = LOCAL_CACHE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, LOCAL_CACHE_PATH)
    return payload


def ensure_local_cache_exists() -> dict[str, Any]:
    payload = load_local_cache()
    if payload is not None:
        return payload

    emit_server_log("本地 JSON 不存在，开始全量拉取数据库")
    records = fetch_smart_money_records()
    payload = save_local_cache(records)
    emit_server_log(
        "本地 JSON 初始化完成",
        file=str(LOCAL_CACHE_PATH),
        rows=payload.get("record_count", 0),
        latest=payload.get("latest_timestamp") or "-",
    )
    return payload


def sync_incremental_records() -> dict[str, Any]:
    payload = load_local_cache()
    if payload is None:
        created_payload = ensure_local_cache_exists()
        created_payload["new_record_count"] = int(created_payload.get("record_count", 0) or 0)
        return created_payload

    old_records = payload.get("records", [])
    latest_timestamp = payload.get("latest_timestamp") or get_latest_timestamp(old_records)

    if not latest_timestamp:
        records = fetch_smart_money_records()
        refreshed_payload = save_local_cache(records)
        refreshed_payload["new_record_count"] = int(refreshed_payload.get("record_count", 0) or 0)
        return refreshed_payload

    new_records = fetch_smart_money_records(since=str(latest_timestamp))
    if not new_records:
        return {
            **payload,
            "new_record_count": 0,
            "sync_checked_at": datetime.now().isoformat(timespec="seconds"),
        }

    merged_records = merge_records(old_records, new_records)
    updated_payload = save_local_cache(merged_records)
    updated_payload["new_record_count"] = len(new_records)
    return updated_payload


class LocalJsonSyncController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.started_at: str | None = None
        self.last_checked_at: str | None = None
        self.last_success_at: str | None = None
        self.last_error_at: str | None = None
        self.last_error: str | None = None
        self.last_new_records = 0
        self.total_new_records = 0
        self.sync_count = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self._thread = threading.Thread(target=self._run_loop, name="smart-money-json-sync", daemon=True)
        self._thread.start()

    def snapshot(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "last_checked_at": self.last_checked_at,
            "last_success_at": self.last_success_at,
            "last_error_at": self.last_error_at,
            "last_error": self.last_error,
            "last_new_records": self.last_new_records,
            "total_new_records": self.total_new_records,
            "sync_count": self.sync_count,
            "thread_alive": bool(self._thread and self._thread.is_alive()),
        }

    def sync_now(self) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            return {"skipped": True, "reason": "同步任务正在运行，本次跳过", **self.snapshot()}

        try:
            self.last_checked_at = datetime.now().isoformat(timespec="seconds")
            payload = sync_incremental_records()
            new_count = int(payload.get("new_record_count", 0) or 0)
            self.last_new_records = new_count
            self.total_new_records += new_count
            self.sync_count += 1
            self.last_success_at = datetime.now().isoformat(timespec="seconds")
            self.last_error = None
            emit_server_log(
                "本地 JSON 增量同步完成",
                new_rows=new_count,
                total_rows=payload.get("record_count", 0),
                latest=payload.get("latest_timestamp") or "-",
            )
            return {
                "skipped": False,
                "new_record_count": new_count,
                "record_count": payload.get("record_count", 0),
                "latest_timestamp": payload.get("latest_timestamp"),
                **self.snapshot(),
            }
        except Exception as exc:
            self.last_error_at = datetime.now().isoformat(timespec="seconds")
            self.last_error = str(exc)
            emit_server_log("本地 JSON 增量同步失败", error=exc)
            return {"skipped": False, "error": str(exc), **self.snapshot()}
        finally:
            self._lock.release()

    def _run_loop(self) -> None:
        while True:
            time.sleep(SYNC_INTERVAL_SECONDS)
            self.sync_now()


@st.cache_resource
def get_sync_controller() -> LocalJsonSyncController:
    controller = LocalJsonSyncController()
    controller.start()
    return controller


def prepare_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "timestamp" not in df.columns:
        return pd.DataFrame()

    parsed_timestamp = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[parsed_timestamp.notna()].copy()
    parsed_timestamp = parsed_timestamp[parsed_timestamp.notna()]
    df["timestamp"] = parsed_timestamp.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    if "current_price" in df.columns:
        df = df[df["current_price"] > 0]
    if "long_traders" in df.columns and "short_traders" in df.columns:
        df = df[(df["long_traders"] > 0) | (df["short_traders"] > 0)]

    return df.sort_values("timestamp").reset_index(drop=True)


def build_chart_figure(
    df: pd.DataFrame,
    metric: dict[str, Any],
    chart_index: int,
    x_min: pd.Timestamp,
    x_max: pd.Timestamp,
) -> go.Figure:
    metric_col = metric["col"]
    metric_name = metric["name"]
    metric_color = metric["color"]
    is_price_metric = metric_col.endswith("_avg_price")
    figure = make_subplots(specs=[[{"secondary_y": True}]])

    if is_price_metric:
        figure.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[metric_col],
                name=metric_name,
                mode="lines",
                line=dict(color=metric_color, width=2),
                hovertemplate="%{y:,.4f}<extra></extra>",
            ),
            secondary_y=False,
        )
    else:
        figure.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df[metric_col],
                name=metric_name,
                marker_color=metric_color,
                opacity=0.75,
                width=1000 * 60 * 4,
                hovertemplate="%{y:,.4f}<extra></extra>",
            ),
            secondary_y=False,
        )

    figure.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["current_price"],
            name="BTC 价格",
            mode="lines",
            line=dict(color="#900C3F", width=2),
            hovertemplate="%{y:,.2f} USDT<extra></extra>",
        ),
        secondary_y=not is_price_metric,
    )

    figure.update_layout(
        height=450,
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#333333"),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#e5e7eb",
            font=dict(color="#333333", size=13),
        ),
        uirevision=f"smart_money_chart_{chart_index}",
        meta={
            "smart_money_sync_group": "long" if chart_index < 4 else "short",
            "smart_money_chart_index": chart_index,
        },
    )

    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        range=[x_min, x_max],
        rangeslider_visible=True,
        rangeslider=dict(thickness=0.08, bgcolor="#f8f9fa", range=[x_min, x_max]),
        tickformat="%m-%d %H:%M",
        hoverformat="%Y-%m-%d %H:%M:%S",
        color="#6b7280",
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor="#f3f4f6",
        zeroline=True,
        zerolinecolor="#e5e7eb",
        color=metric_color,
        secondary_y=False,
        showticklabels=True,
    )
    figure.update_yaxes(
        showgrid=False,
        zeroline=False,
        color="#900C3F",
        secondary_y=True,
        showticklabels=not is_price_metric,
    )
    return figure


def render_dashboard() -> None:
    st.set_page_config(
        page_title="Smart Money 宏观网格监控板",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .stApp { background-color: #f8f9fa; }
            .block-container { padding-top: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📈 Smart Money 宏观网格监控板")
    st.markdown("基于本地 JSON 缓存的指标与 BTC 价格交叉分析")

    controller = get_sync_controller()
    controller.start()

    startup_sync_result: dict[str, Any] = {}
    startup_sync_warning: str | None = None
    startup_sync_error: str | None = None

    with st.spinner("启动中：正在按本地最新时间戳同步数据库增量..."):
        cache_existed_before_startup = LOCAL_CACHE_PATH.exists()
        startup_sync_result = controller.sync_now()
        if startup_sync_result.get("error"):
            if cache_existed_before_startup:
                startup_sync_warning = str(startup_sync_result["error"])
            else:
                startup_sync_error = str(startup_sync_result["error"])

    if startup_sync_error:
        st.error(f"本地 JSON 初始化失败，且无法从数据库拉取全量数据：{startup_sync_error}")
        st.stop()

    if startup_sync_warning:
        st.warning(f"启动增量同步失败，已先使用本地 JSON 绘图：{startup_sync_warning}")
    elif startup_sync_result.get("skipped"):
        st.info(startup_sync_result.get("reason", "同步任务正在运行，本次启动跳过增量同步"))
    else:
        st.caption(
            f"启动同步完成：新增 {startup_sync_result.get('new_record_count', 0)} 条，"
            f"本地总数 {startup_sync_result.get('record_count', '-')}, "
            f"最新时间 {startup_sync_result.get('latest_timestamp') or '-'}。"
        )

    control_col1, control_col2, _ = st.columns([1, 1, 4])
    with control_col1:
        refresh_panel = st.button("刷新面板", type="primary", help="重新读取根目录 smart_money_cache.json，并重绘全部图表")
    with control_col2:
        sync_now = st.button("立即同步数据库", help="立刻按本地 latest_timestamp 拉取新增数据写入 JSON，并在本次运行中重绘图表")

    if sync_now:
        result = controller.sync_now()
        if result.get("error"):
            st.error(f"同步失败：{result['error']}")
        elif result.get("skipped"):
            st.warning(result.get("reason", "同步任务正在运行，本次跳过"))
        else:
            st.success(
                f"同步完成：新增 {result.get('new_record_count', 0)} 条；"
                f"本地总数 {result.get('record_count', '-')}; "
                f"最新时间 {result.get('latest_timestamp') or '-'}"
            )

    if refresh_panel:
        st.toast("已重新读取本地 JSON 并重绘图表", icon="🔄")

    try:
        cache_payload = load_local_cache() or {"records": []}
    except Exception as exc:
        st.error(f"读取本地 JSON 失败：{exc}")
        st.stop()

    records = cache_payload.get("records", [])
    df = prepare_dataframe(records)

    status = controller.snapshot()
    status_cols = st.columns(6)
    status_cols[0].metric("本地记录数", f"{len(records):,}")
    status_cols[1].metric("图表记录数", f"{len(df):,}")
    status_cols[2].metric("最近同步新增", status.get("last_new_records", 0))
    status_cols[3].metric("后台同步轮次", status.get("sync_count", 0))
    status_cols[4].metric("后台线程", "运行中" if status.get("thread_alive") else "未运行")
    status_cols[5].metric("同步间隔", "5 分钟")

    with st.expander("本地 JSON 与后台同步状态", expanded=False):
        st.write(
            {
                "local_json": str(LOCAL_CACHE_PATH.resolve()),
                "cache_updated_at": cache_payload.get("updated_at"),
                "cache_latest_timestamp": cache_payload.get("latest_timestamp"),
                "cache_record_count": cache_payload.get("record_count"),
                "startup_new_records": startup_sync_result.get("new_record_count"),
                "startup_latest_timestamp": startup_sync_result.get("latest_timestamp"),
                "background_started_at": status.get("started_at"),
                "background_last_checked_at": status.get("last_checked_at"),
                "background_last_success_at": status.get("last_success_at"),
                "background_last_error_at": status.get("last_error_at"),
                "background_last_error": status.get("last_error"),
            }
        )

    if df.empty:
        st.warning("暂无数据，请检查数据库、config.json 或本地 smart_money_cache.json。")
        return

    available_metrics = [metric for metric in ALL_METRICS if metric["col"] in df.columns]
    if not available_metrics:
        st.warning("当前数据集中没有可展示的指标。")
        return

    x_min = df["timestamp"].min()
    x_max = df["timestamp"].max()
    st.caption(
        f"当前图表展示全量本地数据：{x_min:%Y-%m-%d %H:%M:%S} 至 {x_max:%Y-%m-%d %H:%M:%S}。"
        "启动和手动同步会先按 latest_timestamp 追加数据库新增数据到本地 JSON，再读取本地文件绘图；"
        "后台 5 分钟同步只更新本地 JSON，不会自动刷新图表。"
    )

    columns = st.columns(4)
    for index, target_column in enumerate(DEFAULT_METRIC_COLUMNS):
        with columns[index % 4]:
            default_index = next(
                (
                    metric_index
                    for metric_index, metric in enumerate(available_metrics)
                    if metric["col"] == target_column
                ),
                0,
            )

            selected_metric = st.selectbox(
                "指标",
                options=available_metrics,
                format_func=lambda metric: f"{metric['name']} vs BTC",
                index=default_index,
                key=f"chart_metric_{index}",
                label_visibility="collapsed",
            )

            figure = build_chart_figure(
                df=df,
                metric=selected_metric,
                chart_index=index,
                x_min=x_min,
                x_max=x_max,
            )
            st.plotly_chart(figure, width="stretch", key=f"smart_money_chart_{index}")

    structure32_df = add_structure32_columns(df, {})

    # 新增的活动框：放在 8 个小图表之后，32 结构大图标题之前。
    render_structure32_case_explorer(st, structure32_df)

    st.subheader("32种盘口结构提示图")
    st.caption(
        "保留原始 32 结构作为盘口观察层；图上默认高亮的是经过趋势、冲突、确认、冷却过滤后的最终交易提示。"
        "原始结构点可在图例中手动打开查看。"
    )
    render_structure32_section(
        st,
        structure32_df,
        st.plotly_chart,
        chart_key="smart_money_structure32_chart",
    )


render_dashboard()
