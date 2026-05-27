import PocketBase from 'pocketbase';

// 请替换为你实际的 PocketBase IP
const POCKETBASE_URL = 'http://YOUR_POCKETBASE_IP:8090';
export const pb = new PocketBase(POCKETBASE_URL);

// 定义 Smart Money 数据的数据结构模型
export interface SmartMoneyRecord {
  id: string;
  collectionId: string;
  collectionName: string;
  created: string;
  updated: string;
  timestamp: string;
  current_price: number;
  ls_ratio: number;
  long_pos_usdt: number;
  funding_rate: number;
  long_unrealized_pnl: number;
  [key: string]: any;
}

/**
 * 异步获取 Smart Money 统计数据
 * @param symbol  币种标识 (如果数据库设计有区分的话，默认传入)
 * @param limit   一次性获取的数据条数
 */
export async function fetchStats(symbol: string = 'BTCUSDT', limit: number = 500): Promise<SmartMoneyRecord[]> {
  try {
    const result = await pb.collection('smart_money_stats').getList<SmartMoneyRecord>(1, limit, {
      sort: 'timestamp', // 确保时间正序，以便 ECharts X轴从左到右渲染
      // filter: `symbol = '${symbol}'` // 若表结构中有 symbol，可解除此行注释
    });
    return result.items;
  } catch (error) {
    console.error('获取 PocketBase 数据失败:', error);
    throw error;
  }
}
