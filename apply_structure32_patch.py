from pathlib import Path

signals_path = Path("signals.py")
if not signals_path.exists():
    raise SystemExit("找不到 signals.py。请把这个脚本放在 SM_chart 仓库根目录运行。")

text = signals_path.read_text(encoding="utf-8")

if "from market_structure32 import add_structure32_columns, install_structure32_streamlit_hook" not in text:
    text = text.replace(
        "from features import get_series, has_col\n",
        "from features import get_series, has_col\n"
        "from market_structure32 import add_structure32_columns, install_structure32_streamlit_hook\n",
    )

if '"structure32_watch": {"label": "32结构买卖点", "reason_col": "structure32_reason", "priority": 0},' not in text:
    text = text.replace(
        "SIGNAL_META = {\n",
        'SIGNAL_META = {\n'
        '    "structure32_watch": {"label": "32结构买卖点", "reason_col": "structure32_reason", "priority": 0},\n',
    )

needle = """    for col in signal_cols:
        watch_series = out[col].fillna(False).astype(bool)
        out[f"{col}_duration"] = consecutive_true_count(watch_series)
        event_series = watch_series & (~watch_series.shift(1, fill_value=False))
        out[f"{col}_event"] = suppress_repeated_events(event_series, cooldown_bars)

    return out
"""
replacement = """    for col in signal_cols:
        watch_series = out[col].fillna(False).astype(bool)
        out[f"{col}_duration"] = consecutive_true_count(watch_series)
        event_series = watch_series & (~watch_series.shift(1, fill_value=False))
        out[f"{col}_event"] = suppress_repeated_events(event_series, cooldown_bars)

    out = add_structure32_columns(out, params, suppress_repeated_events)
    install_structure32_streamlit_hook(out)

    return out
"""
if "install_structure32_streamlit_hook(out)" not in text:
    if needle not in text:
        raise SystemExit("没有找到 add_signal_columns 末尾插入点。请手动把 add_structure32_columns / install_structure32_streamlit_hook 加到 return out 前。")
    text = text.replace(needle, replacement)

if '"32结构买卖点": "由价格、多空成本线、多空仓位五维组合触发；强度1-5仅表示结构强弱，需结合趋势确认",' not in text:
    text = text.replace(
        "meaning_map = {\n",
        'meaning_map = {\n'
        '        "32结构买卖点": "由价格、多空成本线、多空仓位五维组合触发；强度1-5仅表示结构强弱，需结合趋势确认",\n',
    )

signals_path.write_text(text, encoding="utf-8")
print("已完成 signals.py 集成。下一步运行：streamlit run app.py")
