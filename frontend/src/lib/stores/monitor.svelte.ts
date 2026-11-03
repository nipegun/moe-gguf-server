import { browser } from '$app/environment';
import { MonitorService } from '$lib/services/monitor.service';
import { cMonitorReportIntervalMs, cMonitorUsageIntervalMs } from '$lib/constants/monitor';
import type { MonitorReport, MonitorUsage } from '$lib/types';

/**
 * monitorStore - CPU/GPU usage for the status bar and the full report for the Monitor page.
 *
 * Polling only runs while something is subscribed (reference counted) and pauses while the tab is
 * hidden. While the Monitor page polls the full report, the status bar takes its usage from it
 * instead of sending a second request.
 */
class MonitorStore {
  vUsage = $state<MonitorUsage | null>(null);
  vReport = $state<MonitorReport | null>(null);
  vReportError = $state<string | null>(null);

  private vUsageSubscribers = 0;
  private vReportSubscribers = 0;
  private vUsageTimer: ReturnType<typeof setTimeout> | null = null;
  private vReportTimer: ReturnType<typeof setTimeout> | null = null;

  /** Starts the usage polling; call the returned function to stop it. */
  mSubscribeUsage(): () => void {
    this.vUsageSubscribers++;
    if (this.vUsageSubscribers === 1) {
      void this.mPollUsage();
    }
    return () => {
      this.vUsageSubscribers = Math.max(0, this.vUsageSubscribers - 1);
      if (this.vUsageSubscribers === 0 && this.vUsageTimer) {
        clearTimeout(this.vUsageTimer);
        this.vUsageTimer = null;
      }
    };
  }

  /** Starts the report polling; call the returned function to stop it. */
  mSubscribeReport(): () => void {
    this.vReportSubscribers++;
    if (this.vReportSubscribers === 1) {
      void this.mPollReport();
    }
    return () => {
      this.vReportSubscribers = Math.max(0, this.vReportSubscribers - 1);
      if (this.vReportSubscribers === 0 && this.vReportTimer) {
        clearTimeout(this.vReportTimer);
        this.vReportTimer = null;
      }
    };
  }

  private mIsHidden(): boolean {
    return browser && document.visibilityState === 'hidden';
  }

  private async mPollUsage(): Promise<void> {
    this.vUsageTimer = null;
    if (this.vUsageSubscribers === 0) return;

    // the report already carries the usage while the Monitor page is open
    if (this.vReportSubscribers === 0 && !this.mIsHidden()) {
      try {
        this.vUsage = await MonitorService.mFetchUsage();
      } catch {
        this.vUsage = null;
      }
    }

    if (this.vUsageSubscribers > 0) {
      this.vUsageTimer = setTimeout(() => void this.mPollUsage(), cMonitorUsageIntervalMs);
    }
  }

  private async mPollReport(): Promise<void> {
    this.vReportTimer = null;
    if (this.vReportSubscribers === 0) return;

    if (!this.mIsHidden()) {
      try {
        const vReport = await MonitorService.mFetchReport();
        this.vReport = vReport;
        this.vReportError = null;
        this.vUsage = {
          cpu: {
            usage: vReport.cpu.usage,
            processors: vReport.cpu.processors.map((pProcessor) => ({
              id: pProcessor.id,
              model: pProcessor.model,
              usage: pProcessor.usage
            }))
          },
          gpus: vReport.gpus.map((pGpu) => ({ id: pGpu.id, name: pGpu.name, usage: pGpu.usage }))
        };
      } catch (pError) {
        this.vReportError = pError instanceof Error ? pError.message : String(pError);
      }
    }

    if (this.vReportSubscribers > 0) {
      this.vReportTimer = setTimeout(() => void this.mPollReport(), cMonitorReportIntervalMs);
    }
  }
}

export const cMonitorStore = new MonitorStore();
