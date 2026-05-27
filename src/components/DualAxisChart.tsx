import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import * as echarts from 'echarts/core';

interface DualAxisChartProps {
  times: string[];
  prices: number[];
  metricData: number[]; // 动态切换的副轴数据
  metricName: string;   // 副轴图例名称
}

const DualAxisChart: React.FC<DualAxisChartProps> = ({ times, prices, metricData, metricName }) => {
  // 使用 useMemo 缓存 Option 对象，当入参变化时触发 ECharts 的 setOption 平滑更新
  const option = useMemo(() => {
    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
          crossStyle: { color: '#9ca3af' },
          label: { backgroundColor: '#374151' } // 坐标轴指示器的 Label 颜色
        },
        backgroundColor: 'rgba(17, 24, 39, 0.9)',
        borderColor: '#374151',
        textStyle: { color: '#f3f4f6' },
      },
      legend: {
        data: ['BTC 价格', metricName],
        top: 0,
        textStyle: { color: '#d1d5db', fontSize: 13 },
      },
      grid: {
        left: '2%',
        right: '2%',
        bottom: '12%',
        top: '12%',
        containLabel: true,
      },
      dataZoom: [
        {
          type: 'slider',
          show: true,
          xAxisIndex: [0],
          bottom: 10,
          textStyle: { color: '#9ca3af' },
          borderColor: '#374151',
          fillerColor: 'rgba(59, 130, 246, 0.2)',
        },
        {
          type: 'inside', // 允许滚轮缩放和平移
          xAxisIndex: [0],
        },
      ],
      xAxis: [
        {
          type: 'category',
          data: times,
          axisPointer: { type: 'shadow' },
          axisLine: { lineStyle: { color: '#4b5563' } },
          axisLabel: { color: '#9ca3af' },
        },
      ],
      yAxis: [
        {
          // 主 Y 轴
          type: 'value',
          name: 'BTC 价格',
          position: 'left',
          scale: true, // 自适应刻度区间
          axisLine: { show: true, lineStyle: { color: '#ef4444' } },
          axisLabel: { color: '#ef4444', formatter: '${value}' },
          splitLine: { show: false }, // 隐藏网格线防止凌乱
        },
        {
          // 副 Y 轴
          type: 'value',
          name: metricName,
          position: 'right',
          scale: true, // 支持负值（如资金费率）及自适应
          axisLine: { show: true, lineStyle: { color: '#3b82f6' } },
          axisLabel: { color: '#3b82f6' },
          splitLine: {
            show: true,
            lineStyle: { type: 'dashed', color: 'rgba(75, 85, 99, 0.3)' },
          },
        },
      ],
      series: [
        {
          name: metricName,
          type: 'bar',
          yAxisIndex: 1, // 挂载于右侧副轴
          data: metricData,
          z: 1, // 置于底层
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(59, 130, 246, 0.8)' },
              { offset: 1, color: 'rgba(29, 78, 216, 0.2)' },
            ]),
            borderRadius: [2, 2, 0, 0],
          },
        },
        {
          name: 'BTC 价格',
          type: 'line',
          yAxisIndex: 0, // 挂载于左侧主轴
          data: prices,
          smooth: true,
          z: 2, // 置于顶层
          symbol: 'none',
          lineStyle: {
            color: '#ef4444',
            width: 3,
            shadowColor: 'rgba(239, 68, 68, 0.4)',
            shadowBlur: 10,
          },
          itemStyle: { color: '#ef4444' },
        },
      ],
    };
  }, [times, prices, metricData, metricName]);

  return (
    <ReactECharts
      option={option}
      theme="dark" // 启用深色主题
      style={{ height: '100%', width: '100%', minHeight: '550px' }}
      notMerge={false} // 设为 false 让图表系列在切换时保留渐变动画效果
    />
  );
};

export default DualAxisChart;
