import { fGetLanguage } from '$lib/i18n';
import type { MonitorSegmentGroup } from '$lib/constants/monitor';
import type { MonitorGpu, MonitorSegments } from '$lib/types';

const cKiB = 1024;
const cMiB = 1024 * 1024;
const cGiB = 1024 * 1024 * 1024;

/** Binary units, one decimal for GiB, in the number format of the interface language. */
export function fFormatMonitorBytes(pBytes: number): string {
  const vBytes = Number.isFinite(pBytes) ? Math.max(0, pBytes) : 0;
  if (vBytes >= cGiB) {
    const vFormat = new Intl.NumberFormat(fGetLanguage(), {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1
    });
    return `${vFormat.format(vBytes / cGiB)} GiB`;
  }
  const vFormat = new Intl.NumberFormat(fGetLanguage(), { maximumFractionDigits: 0 });
  if (vBytes >= cMiB) {
    return `${vFormat.format(vBytes / cMiB)} MiB`;
  }
  return `${vFormat.format(vBytes / cKiB)} KiB`;
}

export function fFormatMonitorNumber(pValue: number, pDigits = 0): string {
  return new Intl.NumberFormat(fGetLanguage(), {
    minimumFractionDigits: pDigits,
    maximumFractionDigits: pDigits
  }).format(pValue);
}

/** Percentage (0-100) in the format of the interface language ("58%", "58 %"). */
export function fFormatMonitorPercent(pValue: number, pDigits = 0): string {
  return new Intl.NumberFormat(fGetLanguage(), {
    style: 'percent',
    minimumFractionDigits: pDigits,
    maximumFractionDigits: pDigits
  }).format(pValue / 100);
}

/** Bytes of every server segment drawn with the colour of a group. */
export function fMonitorGroupBytes(pSegments: MonitorSegments, pGroup: MonitorSegmentGroup): number {
  let vBytes = 0;
  for (const vKey of pGroup.keys) {
    vBytes += pSegments[vKey] ?? 0;
  }
  return vBytes;
}

/**
 * GPU memory of the models that the kernel keeps outside VRAM (evicted to system memory through GTT),
 * or 0 below the noise of the accounting (64 MiB or 2 % of what was allocated).
 */
export function fMonitorOutsideVram(pGpu: MonitorGpu): number {
  const vKernel = pGpu.kernel;
  if (!vKernel) return 0;
  const vLimit = Math.max(64 * cMiB, vKernel.allocated_bytes / 50);
  return vKernel.outside_vram_bytes > vLimit ? vKernel.outside_vram_bytes : 0;
}
