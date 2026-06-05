# SM_chart 32种盘口结构提示补丁

## 文件说明

- `market_structure32.py`：新增的 32 种结构计算与图表模块。
- `apply_structure32_patch.py`：自动修改 `signals.py` 的集成脚本。

## 使用步骤

1. 把 `market_structure32.py` 和 `apply_structure32_patch.py` 放到 SM_chart 仓库根目录。
2. 在仓库根目录运行：

```bash
python apply_structure32_patch.py
```

3. 启动面板：

```bash
streamlit run app.py
```

## 效果

- 在原主图下方自动新增一个大图表：`32种盘口结构提示图`。
- 图表只有 BTC 单独走势线。
- 当五维结构首次出现时，会在 BTC 折线上标记：
  - 多头机会
  - 空头机会
  - 观望
- 每个标记包含强度 1-5。
- 鼠标悬浮会显示：
  - 第几种结构
  - 结构名称
  - 价格变化
  - 多头均价变化
  - 多头仓位变化
  - 空头均价变化
  - 空头仓位变化
  - 背后原理

## 字段要求

需要这些字段存在：

- current_price
- long_avg_entry
- long_pos_usdt
- short_avg_entry
- short_pos_usdt

你的截图里已有多头/空头均价图，因此大概率字段已经满足。
