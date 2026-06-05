from __future__ import annotations

import asyncio
import gc
import json
import os
import time
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
import tornado.web
from plotly.subplots import make_subplots
from streamlit.web.server.server_util import make_url_path_regex


CONFIG_PATH = "config.json"
DEFAULT_POCKETBASE_URL = "http://YOUR_POCKETBASE_IP:8090"
DEFAULT_WINDOW_HOURS = 24
INCREMENTAL_REFRESH_MS = 5 * 60 * 1000
CHART_DATA_API_ROUTE = "api/chart-data"
LOG_PANEL_ANCHOR_ID = "smart-money-log-anchor"

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
    {"col": "total_pos_usdt", "name": "总持仓总额", "color": "#0ea5e9"},
]

DEFAULT_METRIC_COLUMNS = [
    "long_pnl_ratio",
    "long_traders",
    "long_avg_price",
    "short_pnl_ratio",
    "short_traders",
    "short_avg_price",
]

METRIC_CONFIG_BY_COL = {metric["col"]: metric for metric in ALL_METRICS}


def load_pocketbase_url() -> str:
    pocketbase_url = DEFAULT_POCKETBASE_URL
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            config = json.load(file)
            pocketbase_url = config.get("POCKETBASE_URL", pocketbase_url)
    return pocketbase_url


POCKETBASE_URL = load_pocketbase_url()


def emit_server_log(message: str, **details: Any) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if details:
        detail_text = " | ".join(f"{key}={value}" for key, value in details.items())
        print(f"[SmartMoney {timestamp}] {message} | {detail_text}", flush=True)
    else:
        print(f"[SmartMoney {timestamp}] {message}", flush=True)


def get_base_url_path() -> str:
    base_url_path = st.get_option("server.baseUrlPath") or ""
    normalized = base_url_path.strip("/")
    return f"/{normalized}" if normalized else ""


def build_same_port_api_path() -> str:
    return f"{get_base_url_path()}/{CHART_DATA_API_ROUTE}".replace("//", "/")


def escape_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def get_reference_metric(metric_col: str) -> str | None:
    if metric_col.startswith("long_"):
        return "long_pnl_ratio"
    if metric_col.startswith("short_"):
        return "short_pnl_ratio"
    return None


def build_requested_fields(metric: str | None) -> list[str]:
    base_fields = ["timestamp", "current_price"]
    if not metric:
        return ["timestamp", *NUMERIC_COLUMNS]

    if metric not in METRIC_CONFIG_BY_COL:
        raise ValueError(f"不支持的 metric: {metric}")

    fields = [*base_fields, metric]
    ref_col = get_reference_metric(metric)
    if ref_col and ref_col not in fields:
        fields.append(ref_col)
    return fields


def fetch_smart_money_records(
    since: str | None = None,
    fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    url = f"{POCKETBASE_URL.rstrip('/')}/api/collections/smart_money_stats/records"
    sort_order = "timestamp" if since else "-timestamp"
    filter_parts = ["current_price > 0"]
    if since:
        filter_parts.append(f'timestamp > "{escape_filter_value(since)}"')

    items: list[dict[str, Any]] = []
    page = 1
    while True:
        params: dict[str, Any] = {
            "perPage": 500,
            "page": page,
            "sort": sort_order,
            "filter": " && ".join(f"({part})" for part in filter_parts),
        }
        if fields:
            params["fields"] = ",".join(dict.fromkeys(fields))

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("items", [])
        if not batch:
            break

        items.extend(batch)
        if len(batch) < 500:
            break
        page += 1

    if not since:
        items.reverse()

    return items


def prepare_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "timestamp" not in df.columns:
        return pd.DataFrame()

    df["source_timestamp"] = df["timestamp"]
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

    return df.reset_index(drop=True)


def to_chart_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert PocketBase raw records into the same timestamp space used by the first chart render.

    The API cursor still uses source_timestamp/latest_timestamp, but Plotly x values use the
    Asia/Shanghai display timestamp. This prevents mixed UTC/local x values after auto update.
    """
    df = prepare_dataframe(records)
    if df.empty:
        return []

    chart_records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        record: dict[str, Any] = {
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(row["timestamp"], "strftime")
            else str(row["timestamp"]),
            "source_timestamp": row.get("source_timestamp"),
        }
        for column in NUMERIC_COLUMNS:
            if column in row:
                value = row[column]
                if pd.isna(value):
                    record[column] = 0
                else:
                    record[column] = float(value)
        chart_records.append(record)
    return chart_records


@st.cache_data(ttl=60)
def fetch_and_process_data() -> pd.DataFrame:
    started_at = time.perf_counter()
    try:
        records = fetch_smart_money_records()
        df = prepare_dataframe(records)
        emit_server_log(
            "完成首屏历史数据加载",
            rows=len(df),
            latest=df["source_timestamp"].iloc[-1] if not df.empty else "-",
            cost_ms=int((time.perf_counter() - started_at) * 1000),
        )
        return df
    except Exception as exc:
        emit_server_log("首屏历史数据加载失败", error=exc)
        st.error(f"数据拉取失败: {exc}")
        return pd.DataFrame()


def build_chart_data_response(since: str | None, metric: str | None) -> dict[str, Any]:
    started_at = time.perf_counter()
    fields = build_requested_fields(metric)
    records = fetch_smart_money_records(since=since, fields=fields)
    latest_timestamp = records[-1]["timestamp"] if records else since
    chart_records = to_chart_records(records)
    duration_ms = int((time.perf_counter() - started_at) * 1000)

    emit_server_log(
        "处理同端口增量接口请求",
        since=since or "-",
        metric=metric or "all",
        appended_rows=len(chart_records),
        latest=latest_timestamp or "-",
        cost_ms=duration_ms,
    )

    return {
        "has_new_data": bool(chart_records),
        "latest_timestamp": latest_timestamp,
        "append_data": chart_records,
        "metric": metric,
        "server_time": datetime.now().isoformat(timespec="seconds"),
        "record_count": len(chart_records),
    }


class ChartDataRequestHandler(tornado.web.RequestHandler):
    def set_default_headers(self) -> None:
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.set_header("Cache-Control", "no-store, max-age=0")

    async def get(self) -> None:
        since = self.get_query_argument("since", default=None)
        metric = self.get_query_argument("metric", default=None)

        try:
            payload = await asyncio.to_thread(build_chart_data_response, since, metric)
            self.set_status(200)
        except ValueError as exc:
            emit_server_log("同端口增量接口参数错误", since=since or "-", metric=metric or "-", error=exc)
            payload = {
                "has_new_data": False,
                "latest_timestamp": since,
                "append_data": [],
                "error": str(exc),
            }
            self.set_status(400)
        except Exception as exc:
            emit_server_log("同端口增量接口处理失败", since=since or "-", metric=metric or "-", error=exc)
            payload = {
                "has_new_data": False,
                "latest_timestamp": since,
                "append_data": [],
                "error": f"接口处理失败: {exc}",
            }
            self.set_status(500)

        self.finish(json.dumps(payload, ensure_ascii=False))


def is_streamlit_application(app: Any) -> bool:
    rules = getattr(getattr(app, "wildcard_router", None), "rules", [])
    for rule in rules:
        target = getattr(rule, "target", None)
        module_name = getattr(target, "__module__", "")
        if module_name.startswith("streamlit.web.server"):
            return True
    return False


def find_streamlit_application() -> tornado.web.Application | None:
    for obj in gc.get_objects():
        try:
            if isinstance(obj, tornado.web.Application) and is_streamlit_application(obj):
                return obj
        except Exception:
            continue
    return None


def ensure_same_port_api_route() -> bool:
    app = find_streamlit_application()
    if app is None:
        emit_server_log("未找到 Streamlit Tornado 应用，无法注册同端口接口")
        return False

    route_pattern = make_url_path_regex(get_base_url_path(), CHART_DATA_API_ROUTE)
    rules = getattr(app.wildcard_router, "rules", [])
    for rule in rules:
        matcher = getattr(rule, "matcher", None)
        regex = getattr(matcher, "regex", None)
        if getattr(regex, "pattern", None) == route_pattern:
            return True

    url_spec = tornado.web.URLSpec(route_pattern, ChartDataRequestHandler, name="smart_money_chart_data")
    app.wildcard_router.rules.insert(0, url_spec)
    named_handlers = getattr(app, "named_handlers", None)
    if isinstance(named_handlers, dict):
        named_handlers[url_spec.name] = url_spec

    emit_server_log("已注册同端口增量接口", path=build_same_port_api_path())
    return True


def resolve_metric_style(
    df: pd.DataFrame,
    metric_col: str,
    metric_color: str,
) -> tuple[str, str | None, str | list[str]]:
    base_color = "#10b981" if metric_col in {"long_pnl_ratio", "short_pnl_ratio"} else metric_color
    ref_col = get_reference_metric(metric_col)

    if not ref_col or ref_col not in df.columns:
        return base_color, None, base_color

    colors: list[str] = []
    for value in df[ref_col]:
        if pd.isna(value):
            colors.append(base_color)
        elif value < 0.1:
            colors.append("#ef4444")
        elif value > 0.9:
            colors.append("#f59e0b")
        else:
            colors.append(base_color)
    return base_color, ref_col, colors


def build_chart_figure(
    df: pd.DataFrame,
    metric: dict[str, Any],
    chart_index: int,
    x_min: pd.Timestamp,
    x_max: pd.Timestamp,
    x_full_min: pd.Timestamp,
) -> go.Figure:
    metric_col = metric["col"]
    metric_name = metric["name"]
    metric_color = metric["color"]

    base_color, ref_col, bar_color = resolve_metric_style(df, metric_col, metric_color)
    is_price_metric = metric_col.endswith("_avg_price")
    figure = make_subplots(specs=[[{"secondary_y": True}]])

    if is_price_metric:
        figure.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[metric_col],
                name=metric_name,
                mode="lines",
                line=dict(color=base_color, width=2),
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
                marker_color=bar_color,
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
            "smart_money_config": {
                "metric_col": metric_col,
                "ref_col": ref_col,
                "base_color": base_color,
                "default_window_ms": DEFAULT_WINDOW_HOURS * 60 * 60 * 1000,
            }
        },
    )

    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        range=[x_min, x_max],
        rangeslider_visible=True,
        rangeslider=dict(
            thickness=0.08,
            bgcolor="#f8f9fa",
            range=[x_full_min, x_max],
        ),
        tickformat="%m-%d %H:%M",
        hoverformat="%Y-%m-%d %H:%M:%S",
        color="#6b7280",
    )

    figure.update_yaxes(
        showgrid=True,
        gridcolor="#f3f4f6",
        zeroline=True,
        zerolinecolor="#e5e7eb",
        color=base_color,
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


def build_initial_log_entries(
    df: pd.DataFrame,
    rendered_chart_count: int,
    api_ready: bool,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    timestamp = datetime.now().isoformat(timespec="seconds")

    if df.empty:
        entries.append(
            {
                "time": timestamp,
                "level": "warning",
                "message": "页面初始化完成，但当前没有可展示的数据。",
                "details": {"api": build_same_port_api_path()},
            }
        )
    else:
        entries.append(
            {
                "time": timestamp,
                "level": "info",
                "message": "首屏历史数据已完成加载。",
                "details": {
                    "historyRows": int(len(df)),
                    "charts": rendered_chart_count,
                    "latestTimestamp": str(df["source_timestamp"].iloc[-1]),
                },
            }
        )

    entries.append(
        {
            "time": timestamp,
            "level": "success" if api_ready else "error",
            "message": "同端口增量接口已就绪。" if api_ready else "同端口增量接口注册失败。",
            "details": {"api": build_same_port_api_path(), "pollIntervalMs": INCREMENTAL_REFRESH_MS},
        }
    )
    return entries


def render_incremental_refresh_script(
    latest_timestamp: str,
    expected_chart_count: int,
    initial_log_entries: list[dict[str, Any]],
    api_ready: bool,
) -> None:
    payload = {
        "apiPath": build_same_port_api_path(),
        "apiReady": api_ready,
        "initialLatestTimestamp": latest_timestamp,
        "expectedChartCount": expected_chart_count,
        "pollIntervalMs": INCREMENTAL_REFRESH_MS,
        "defaultWindowMs": DEFAULT_WINDOW_HOURS * 60 * 60 * 1000,
        "logAnchorId": LOG_PANEL_ANCHOR_ID,
        "initialLogEntries": initial_log_entries,
    }

    script = """
    <script>
    (function() {
      const payload = __SMART_MONEY_PAYLOAD__;
      const parentWindow = window.parent || window;
      const controllerKey = "__smartMoneyIncrementalRefreshController";
      const STATUS_NODE_ID = "smart-money-refresh-status";
      const LOG_PANEL_ID = "smart-money-log-panel";
      const MAX_LOG_ENTRIES = 18;

      if (parentWindow[controllerKey] && parentWindow[controllerKey].timerId) {
        parentWindow.clearInterval(parentWindow[controllerKey].timerId);
      }

      const state = {
        logEntries: [],
        totalAppendedRecords: 0,
        successfulUpdates: 0,
        lastAttemptAt: null,
        lastSuccessAt: null,
        lastFailureAt: null,
        lastResult: "等待首次轮询"
      };

      const controller = {
        timerId: null,
        lastTimestamp: payload.initialLatestTimestamp,
        isUpdating: false,
        bootstrapped: false
      };
      parentWindow[controllerKey] = controller;

      function getParentDocument() {
        return parentWindow.document || document;
      }

      function getPlotly() {
        return parentWindow.Plotly || window.Plotly;
      }

      function getSmartMoneyCharts() {
        const doc = getParentDocument();
        return Array.from(
          doc.querySelectorAll('[data-testid="stPlotlyChart"] .js-plotly-plot')
        ).filter((chart) => chart && chart.layout && chart.layout.meta && chart.layout.meta.smart_money_config);
      }

      function ensureStatusNode() {
        const doc = getParentDocument();
        let node = doc.getElementById(STATUS_NODE_ID);
        if (!node) {
          node = doc.createElement("div");
          node.id = STATUS_NODE_ID;
          Object.assign(node.style, {
            position: "fixed",
            top: "18px",
            right: "24px",
            zIndex: "9999",
            padding: "8px 12px",
            borderRadius: "10px",
            background: "rgba(239, 68, 68, 0.92)",
            color: "#ffffff",
            fontSize: "13px",
            fontFamily: "system-ui, sans-serif",
            boxShadow: "0 10px 30px rgba(15, 23, 42, 0.18)",
            opacity: "0",
            pointerEvents: "none",
            transform: "translateY(-4px)",
            transition: "opacity 0.2s ease, transform 0.2s ease"
          });
          doc.body.appendChild(node);
        }
        return node;
      }

      function showStatus(message) {
        const node = ensureStatusNode();
        node.textContent = message;
        node.style.opacity = "1";
        node.style.transform = "translateY(0)";
      }

      function hideStatus() {
        const node = ensureStatusNode();
        node.style.opacity = "0";
        node.style.transform = "translateY(-4px)";
      }

      function parseDateMs(value) {
        if (value instanceof Date) {
          const ms = value.getTime();
          return Number.isNaN(ms) ? null : ms;
        }
        if (typeof value === "number") {
          return Number.isFinite(value) ? value : null;
        }
        const parsed = new Date(value).getTime();
        return Number.isNaN(parsed) ? null : parsed;
      }

      function toNumber(value) {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : 0;
      }

      function toTraceArray(value) {
        if (Array.isArray(value)) {
          return value.slice();
        }
        if (!value) {
          return [];
        }
        if (ArrayBuffer.isView(value)) {
          return Array.from(value);
        }
        if (typeof value !== "string" && typeof value[Symbol.iterator] === "function") {
          return Array.from(value);
        }
        if (typeof value === "object" && Number.isFinite(value.length)) {
          return Array.from(value);
        }
        return [];
      }

      function formatMoment(value) {
        if (!value) {
          return "-";
        }
        const date = new Date(value);
        return Number.isNaN(date.getTime())
          ? String(value)
          : date.toLocaleString("zh-CN", {
              hour12: false,
              month: "2-digit",
              day: "2-digit",
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit"
            });
      }

      function buildDetailString(details) {
        if (!details) {
          return "";
        }
        return Object.entries(details)
          .filter(([, value]) => value !== undefined && value !== null && value !== "")
          .map(([key, value]) => `${key}=${value}`)
          .join(" | ");
      }

      function ensureLogPanelHost() {
        const doc = getParentDocument();
        let host = doc.getElementById(payload.logAnchorId);
        if (!host) {
          host = doc.createElement("div");
          host.id = payload.logAnchorId;
          doc.body.appendChild(host);
        }
        return host;
      }

      function levelColors(level) {
        return {
          info: "#2563eb",
          success: "#059669",
          idle: "#64748b",
          warning: "#d97706",
          error: "#dc2626"
        }[level] || "#334155";
      }

      function renderLogPanel() {
        const host = ensureLogPanelHost();
        const entriesHtml = state.logEntries.length
          ? state.logEntries.map((entry) => {
              const detailString = buildDetailString(entry.details);
              return `
                <div style="padding:10px 12px;border-bottom:1px solid #e5e7eb;">
                  <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
                    <span style="font:600 12px/1.4 system-ui,sans-serif;color:${levelColors(entry.level)};text-transform:uppercase;">${entry.level}</span>
                    <span style="font:500 12px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace;color:#64748b;">${formatMoment(entry.time)}</span>
                    <span style="font:600 13px/1.5 system-ui,sans-serif;color:#0f172a;">${entry.message}</span>
                  </div>
                  ${detailString ? `<div style="margin-top:6px;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;color:#475569;word-break:break-word;">${detailString}</div>` : ""}
                </div>
              `;
            }).join("")
          : `<div style="padding:18px 12px;color:#64748b;font:13px/1.6 system-ui,sans-serif;">暂无日志，等待首轮数据更新。</div>`;

        host.innerHTML = `
          <section id="${LOG_PANEL_ID}" style="margin-top:20px;border:1px solid #e5e7eb;border-radius:16px;background:#ffffff;box-shadow:0 8px 24px rgba(15,23,42,0.06);overflow:hidden;">
            <div style="padding:16px 18px 12px 18px;border-bottom:1px solid #e5e7eb;background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);">
              <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;">
                <div>
                  <div style="font:700 16px/1.4 system-ui,sans-serif;color:#0f172a;">运行日志</div>
                  <div style="margin-top:4px;font:13px/1.5 system-ui,sans-serif;color:#64748b;">
                    同端口接口 <span style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace;">${payload.apiPath}</span>，
                    每 5 分钟静默拉取一次增量数据。
                  </div>
                </div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;">
                  <span style="padding:6px 10px;border-radius:999px;background:#eff6ff;color:#1d4ed8;font:600 12px/1 system-ui,sans-serif;">状态: ${state.lastResult}</span>
                  <span style="padding:6px 10px;border-radius:999px;background:#ecfdf5;color:#047857;font:600 12px/1 system-ui,sans-serif;">成功轮次: ${state.successfulUpdates}</span>
                  <span style="padding:6px 10px;border-radius:999px;background:#f8fafc;color:#334155;font:600 12px/1 system-ui,sans-serif;">累计追加: ${state.totalAppendedRecords}</span>
                </div>
              </div>
              <div style="margin-top:10px;display:flex;gap:16px;flex-wrap:wrap;font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;color:#475569;">
                <span>最近尝试: ${formatMoment(state.lastAttemptAt)}</span>
                <span>最近成功: ${formatMoment(state.lastSuccessAt)}</span>
                <span>最近失败: ${formatMoment(state.lastFailureAt)}</span>
                <span>图表数: ${payload.expectedChartCount}</span>
              </div>
            </div>
            <div style="max-height:260px;overflow:auto;background:#fcfcfd;">${entriesHtml}</div>
          </section>
        `;
      }

      function appendLog(level, message, details) {
        const entry = {
          time: new Date().toISOString(),
          level,
          message,
          details: details || {}
        };
        state.logEntries = [entry, ...state.logEntries].slice(0, MAX_LOG_ENTRIES);
        if (level === "error") {
          console.error(`[SmartMoney] ${message}`, details || {});
        } else {
          console.log(`[SmartMoney] ${message}`, details || {});
        }
        renderLogPanel();
      }

      function seedInitialLogs() {
        if (controller.bootstrapped) {
          return;
        }
        payload.initialLogEntries.forEach((entry) => {
          state.logEntries.push(entry);
        });
        state.logEntries = state.logEntries.slice(0, MAX_LOG_ENTRIES);
        state.lastResult = payload.apiReady ? "待轮询" : "接口未就绪";
        renderLogPanel();
        appendLog(
          payload.apiReady ? "success" : "error",
          payload.apiReady
            ? "同端口增量更新已接管后续刷新，不再进行整页 reload。"
            : "同端口增量接口未就绪，当前无法启动静默更新。",
          {
            latestTimestamp: payload.initialLatestTimestamp || "-",
            pollIntervalMs: payload.pollIntervalMs
          }
        );
        controller.bootstrapped = true;
      }

      function shouldFollowLiveWindow(chart, previousLatestMs, defaultWindowMs) {
        const xaxis = chart && chart.layout ? chart.layout.xaxis : null;
        const range = xaxis && Array.isArray(xaxis.range) ? xaxis.range : null;
        if (!range || range.length < 2 || previousLatestMs === null) {
          return false;
        }

        const startMs = parseDateMs(range[0]);
        const endMs = parseDateMs(range[1]);
        if (startMs === null || endMs === null) {
          return false;
        }

        const nearLiveEdge = Math.abs(endMs - previousLatestMs) <= 60 * 1000;
        const keepsDefaultWindow = Math.abs((endMs - startMs) - defaultWindowMs) <= 5 * 60 * 1000;
        return nearLiveEdge && keepsDefaultWindow;
      }

      function buildBarColors(records, config) {
        if (!config.ref_col) {
          return null;
        }

        return records.map((record) => {
          const refValue = Number(record[config.ref_col]);
          if (!Number.isFinite(refValue)) {
            return config.base_color;
          }
          if (refValue < 0.1) {
            return "#ef4444";
          }
          if (refValue > 0.9) {
            return "#f59e0b";
          }
          return config.base_color;
        });
      }

      async function fetchIncrementalPayload(sinceTimestamp) {
        const url = new URL(payload.apiPath, parentWindow.location.origin);
        url.searchParams.set("since", sinceTimestamp);
        const response = await fetch(url.toString(), {
          method: "GET",
          cache: "no-store",
          credentials: "same-origin"
        });

        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(data.error || `HTTP ${response.status}`);
        }
        return data;
      }

      async function appendRecordsToChart(chart, records, plotly) {
        const config = chart.layout.meta.smart_money_config;
        const metricTrace = chart.data[0];
        const priceTrace = chart.data[1];

        if (!metricTrace || !priceTrace || !records.length) {
          return;
        }

        const currentMetricX = toTraceArray(metricTrace.x);
        const previousLatestX = currentMetricX.length ? currentMetricX[currentMetricX.length - 1] : null;
        const previousLatestMs = parseDateMs(previousLatestX);
        const followLiveWindow = shouldFollowLiveWindow(chart, previousLatestMs, config.default_window_ms);

        const newX = records.map((record) => record.timestamp);
        const newMetricY = records.map((record) => toNumber(record[config.metric_col]));
        const newPriceY = records.map((record) => toNumber(record.current_price));

        if (!newX.length) {
          return;
        }

        await plotly.extendTraces(
          chart,
          {
            x: [newX, newX],
            y: [newMetricY, newPriceY]
          },
          [0, 1]
        );

        if (metricTrace.type === "bar") {
          const appendedColors = buildBarColors(records, config);
          if (appendedColors && appendedColors.length) {
            try {
              await plotly.extendTraces(
                chart,
                {
                  "marker.color": [appendedColors]
                },
                [0]
              );
            } catch (error) {
              console.warn("[SmartMoney] marker color append skipped:", error);
            }
          }
        }

        const relayoutPayload = {};
        const xaxis = chart.layout ? chart.layout.xaxis : null;
        const sliderRange = xaxis && xaxis.rangeslider && Array.isArray(xaxis.rangeslider.range)
          ? xaxis.rangeslider.range
          : null;
        const latestPoint = newX[newX.length - 1];

        relayoutPayload["xaxis.rangeslider.range[0]"] =
          sliderRange && sliderRange[0] ? sliderRange[0] : (currentMetricX[0] || newX[0]);
        relayoutPayload["xaxis.rangeslider.range[1]"] = latestPoint;

        if (followLiveWindow) {
          const latestMs = parseDateMs(latestPoint);
          if (latestMs !== null) {
            relayoutPayload["xaxis.range[0]"] = new Date(latestMs - config.default_window_ms);
            relayoutPayload["xaxis.range[1]"] = new Date(latestMs);
          }
        }

        await plotly.relayout(chart, relayoutPayload);
      }

      async function runIncrementalRefresh() {
        if (controller.isUpdating || !controller.lastTimestamp || !payload.apiReady) {
          return;
        }

        const plotly = getPlotly();
        const charts = getSmartMoneyCharts();
        if (!plotly || charts.length < payload.expectedChartCount) {
          return;
        }

        state.lastAttemptAt = new Date().toISOString();
        state.lastResult = "轮询中";
        renderLogPanel();

        controller.isUpdating = true;
        try {
          const responseData = await fetchIncrementalPayload(controller.lastTimestamp);
          const records = Array.isArray(responseData.append_data) ? responseData.append_data : [];

          if (!records.length) {
            state.lastResult = "无新数据";
            hideStatus();
            appendLog("idle", "本轮无新数据，页面保持不动。", {
              since: controller.lastTimestamp,
              latestTimestamp: responseData.latest_timestamp || controller.lastTimestamp
            });
            return;
          }

          for (const chart of charts) {
            await appendRecordsToChart(chart, records, plotly);
          }

          controller.lastTimestamp =
            responseData.latest_timestamp ||
            records[records.length - 1].source_timestamp ||
            controller.lastTimestamp;
          state.successfulUpdates += 1;
          state.totalAppendedRecords += records.length;
          state.lastSuccessAt = new Date().toISOString();
          state.lastResult = `已追加 ${records.length} 条`;
          hideStatus();
          appendLog("success", "增量数据已用 Plotly.extendTraces 追加到图表。", {
            appendedRows: records.length,
            chartsUpdated: charts.length,
            latestTimestamp: controller.lastTimestamp
          });
        } catch (error) {
          state.lastFailureAt = new Date().toISOString();
          state.lastResult = "更新失败";
          console.error("Smart Money incremental update failed:", error);
          showStatus("数据更新失败，稍后重试");
          appendLog("error", "增量更新失败，将在下一轮自动重试。", {
            error: error && error.message ? error.message : String(error),
            since: controller.lastTimestamp
          });
        } finally {
          controller.isUpdating = false;
          renderLogPanel();
        }
      }

      function bootstrap() {
        seedInitialLogs();

        if (!payload.apiReady) {
          showStatus("同端口接口未就绪");
          return;
        }

        if (!payload.initialLatestTimestamp || payload.expectedChartCount < 1) {
          appendLog("warning", "缺少图表或最新时间戳，未启动自动轮询。", {
            latestTimestamp: payload.initialLatestTimestamp || "-",
            charts: payload.expectedChartCount
          });
          return;
        }

        const plotly = getPlotly();
        const charts = getSmartMoneyCharts();
        if (!plotly || charts.length < payload.expectedChartCount) {
          parentWindow.setTimeout(bootstrap, 1000);
          return;
        }

        ensureStatusNode();
        hideStatus();
        controller.timerId = parentWindow.setInterval(runIncrementalRefresh, payload.pollIntervalMs);
        appendLog("info", "日志栏与静默增量更新已启动。", {
          pollIntervalMs: payload.pollIntervalMs,
          charts: payload.expectedChartCount,
          api: payload.apiPath
        });
      }

      bootstrap();
    })();
    </script>
    """.replace("__SMART_MONEY_PAYLOAD__", json.dumps(payload, ensure_ascii=False))

    components.html(script, height=1, width=1)


st.set_page_config(
    page_title="Smart Money 宏观网格监控板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Smart Money 宏观网格监控板")
st.markdown("基于独立卡片的指标与 BTC 价格交叉分析")

api_ready = ensure_same_port_api_route()

with st.spinner("正在加载 Smart Money 数据..."):
    df = fetch_and_process_data()

rendered_chart_count = 0
latest_timestamp = ""

if df.empty:
    st.warning("暂无数据，请检查网络或配置。")
else:
    available_metrics = [metric for metric in ALL_METRICS if metric["col"] in df.columns]
    if not available_metrics:
        st.warning("当前数据集中没有可展示的指标。")
    else:
        x_max = df["timestamp"].max()
        x_min = x_max - pd.Timedelta(hours=DEFAULT_WINDOW_HOURS)
        x_full_min = df["timestamp"].min()
        latest_timestamp = str(df["source_timestamp"].iloc[-1])

        columns = st.columns(3)
        for index in range(6):
            with columns[index % 3]:
                target_column = (
                    DEFAULT_METRIC_COLUMNS[index]
                    if index < len(DEFAULT_METRIC_COLUMNS)
                    else available_metrics[0]["col"]
                )
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
                    x_full_min=x_full_min,
                )
                st.plotly_chart(
                    figure,
                    width="stretch",
                    key=f"smart_money_chart_{index}",
                )
                rendered_chart_count += 1

st.markdown(f'<div id="{LOG_PANEL_ANCHOR_ID}"></div>', unsafe_allow_html=True)

initial_log_entries = build_initial_log_entries(df, rendered_chart_count, api_ready)
render_incremental_refresh_script(
    latest_timestamp=latest_timestamp,
    expected_chart_count=rendered_chart_count,
    initial_log_entries=initial_log_entries,
    api_ready=api_ready,
)
