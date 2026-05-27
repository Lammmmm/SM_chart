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
export async function fetchStats(symbol: string = 'BTCUSDT'): Promise<SmartMoneyRecord[]> {
  try {
    // 使用 getFullList 获取所有历史数据
    const items = await pb.collection('smart_money_stats').getFullList<SmartMoneyRecord>({
      sort: 'timestamp', // 确保时间正序，以便 ECharts X轴从左到右渲染
      filter: 'current_price > 0', // 数据库层面拦截过滤价格为 0 的异常脏数据
    });
    return items;
  } catch (error) {
    console.error('获取 PocketBase 数据失败:', error);
    throw error;
  }
}
