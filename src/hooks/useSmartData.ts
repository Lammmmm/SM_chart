import { useState, useEffect, useCallback } from 'react';
import { fetchStats } from '../api/pb';

export interface SmartDataState {
  times: string[];
  prices: number[];
  lsRatios: number[];
  longPosUsdt: number[];
  fundingRates: number[];
}

export function useSmartData(symbol: string = 'BTCUSDT') {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);
  
  // 按照 ECharts Series 数据结构要求，将其拆分为多个独立的一维数组
  const [data, setData] = useState<SmartDataState>({
    times: [],
    prices: [],
    lsRatios: [],
    longPosUsdt: [],
    fundingRates: []
  });

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const records = await fetchStats(symbol, 500);
      
      const times: string[] = [];
      const prices: number[] = [];
      const lsRatios: number[] = [];
      const longPosUsdt: number[] = [];
      const fundingRates: number[] = [];

      records.forEach(item => {
        // 格式化时间戳为 MM-DD HH:mm 格式
        const date = new Date(item.timestamp);
        const formattedTime = `${(date.getMonth() + 1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
        
        times.push(formattedTime);
        prices.push(item.current_price || 0);
        lsRatios.push(item.ls_ratio || 0);
        longPosUsdt.push(item.long_pos_usdt || 0);
        fundingRates.push(item.funding_rate || 0);
      });

      setData({ times, prices, lsRatios, longPosUsdt, fundingRates });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('获取数据出现未知错误'));
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  return { loading, error, data, refetch: loadData };
}
