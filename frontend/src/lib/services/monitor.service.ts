import { apiFetch } from '$lib/utils';
import type { MonitorReport, MonitorUsage } from '$lib/types';

export class MonitorService {
  /**
   * Full report: CPU, RAM, GPUs and the memory of the loaded models by purpose.
   * The server answers from published snapshots, so polling never wakes a sleeping model.
   */
  static async mFetchReport(): Promise<MonitorReport> {
    return apiFetch<MonitorReport>('/monitor', { authOnly: true, cache: 'no-store' });
  }

  /** CPU and GPU usage only (status bar); also answers while the model loads. */
  static async mFetchUsage(): Promise<MonitorUsage> {
    return apiFetch<MonitorUsage>('/monitor/usage', { authOnly: true, cache: 'no-store' });
  }
}
