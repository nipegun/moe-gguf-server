export type ColorLevel = 'ok' | 'warning' | 'critical' | 'neutral';

const cWarningThreshold = 80;
const cCriticalThreshold = 95;

export function fColorLevelFromPercent(pPercent: number | null): ColorLevel {
  if (pPercent === null) return 'neutral';
  if (pPercent >= cCriticalThreshold) return 'critical';
  if (pPercent >= cWarningThreshold) return 'warning';
  return 'ok';
}

export function fColorLevelTextClass(pLevel: ColorLevel): string {
  switch (pLevel) {
    case 'critical':
      return 'text-red-400';
    case 'warning':
      return 'text-amber-400';
    case 'ok':
      return 'text-muted-foreground';
    default:
      return 'text-muted-foreground';
  }
}

export function fColorLevelBgClass(pLevel: ColorLevel): string {
  switch (pLevel) {
    case 'critical':
      return 'bg-red-500';
    case 'warning':
      return 'bg-amber-500';
    case 'ok':
      return 'bg-green-500';
    default:
      return 'bg-muted';
  }
}
