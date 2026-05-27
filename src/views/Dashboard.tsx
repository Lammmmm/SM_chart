import React, { useState, useMemo } from 'react';
import { useSmartData } from '../hooks/useSmartData';
import DualAxisChart from '../components/DualAxisChart';

// 将指标 key 约束为返回数据模型中存在的类型
type MetricKey = 'lsRatios' | 'longPosUsdt' | 'fundingRates';

const metricOptions: { value: MetricKey; label: string }[] = [
  { value: 'lsRatios', label: '多空人数比 (LS Ratio)' },
  { value: 'longPosUsdt', label: '多头持仓总额 (Long Pos USDT)' },
  { value: 'fundingRates', label: '资金费率 (Funding Rate)' },
];

const Dashboard: React.FC = () => {
  // 调用自定义 hook 自动拉取数据
  const { loading, error, data } = useSmartData('BTCUSDT');
  
  // 维护下拉框选中的指标状态
  const [selectedMetric, setSelectedMetric] = useState<MetricKey>('lsRatios');

  // 计算当前选中的数组和显示名称
  const currentMetricData = useMemo(() => data[selectedMetric], [data, selectedMetric]);
  const currentMetricLabel = useMemo(() => {
    return metricOptions.find((opt) => opt.value === selectedMetric)?.label || '';
  }, [selectedMetric]);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4 md:p-8 flex flex-col font-sans">
      
      {/* 顶部控制面板 */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 bg-gray-800 p-5 rounded-xl border border-gray-700 shadow-lg">
        <div className="mb-4 md:mb-0">
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
            Smart Money 数据看板
          </h1>
          <p className="text-sm text-gray-400 mt-1">实时追踪加密市场主力资金流入与情绪指标</p>
        </div>
        
        <div className="flex items-center gap-3 w-full md:w-auto">
          <label htmlFor="metric-select" className="text-sm font-medium text-gray-300 whitespace-nowrap">
            对比副轴指标:
          </label>
          <select
            id="metric-select"
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value as MetricKey)}
            className="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 block w-full md:w-64 p-2.5 outline-none cursor-pointer transition-colors"
          >
            {metricOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* 主图表展示区 */}
      <div className="flex-grow bg-gray-800 rounded-xl border border-gray-700 p-2 md:p-6 shadow-lg relative flex flex-col">
        {/* Loading 状态 */}
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-gray-900/70 backdrop-blur-sm rounded-xl">
            <div className="flex flex-col items-center">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-3"></div>
              <span className="text-blue-400 text-sm tracking-wider">连接区块链数据...</span>
            </div>
          </div>
        )}

        {/* 错误状态 */}
        {error && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-gray-900/90 rounded-xl">
            <div className="text-red-400 flex flex-col items-center">
              <svg className="w-12 h-12 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              <span>数据加载失败: {error.message}</span>
            </div>
          </div>
        )}

        {/* ECharts 封装组件 */}
        <DualAxisChart
          times={data.times}
          prices={data.prices}
          metricData={currentMetricData}
          metricName={currentMetricLabel}
        />
      </div>
    </div>
  );
};

export default Dashboard;
