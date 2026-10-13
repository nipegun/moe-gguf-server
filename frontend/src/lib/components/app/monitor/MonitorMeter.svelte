<script lang="ts">
  import { fFormatMonitorPercent } from '$lib/utils/monitor';

  interface Props {
    label: string;
    /** Percentage 0-100, or null when the value is unknown. */
    value: number | null;
    class?: string;
    /** Taller track for the Monitor page; the status bar uses the thin default. */
    size?: 'sm' | 'md';
  }

  let {
    label: vLabel,
    value: vValue,
    class: vClassName = '',
    size: vSize = 'sm'
  }: Props = $props();

  let vPercent = $derived(vValue === null ? 0 : Math.min(100, Math.max(0, vValue)));
</script>

<div
  class="flex min-w-0 items-center gap-2 {vClassName}"
  role="meter"
  aria-label={vLabel}
  aria-valuemin={0}
  aria-valuemax={100}
  aria-valuenow={vValue === null ? undefined : Math.round(vPercent)}
>
  <span class="shrink-0 font-medium text-muted-foreground">{vLabel}</span>

  <!-- green bar filled in red by the usage; an unknown value keeps a neutral track -->
  <span
    class="relative min-w-0 flex-1 overflow-hidden rounded-full {vSize === 'md'
      ? 'h-2.5'
      : 'h-1.5'} {vValue === null ? 'bg-muted' : ''}"
    style={vValue === null ? undefined : 'background-color: var(--monitor-meter-free)'}
  >
    <span
      class="absolute inset-y-0 left-0 transition-[width] duration-500 ease-out"
      style="width: {vPercent}%; background-color: var(--monitor-meter-used)"
    ></span>
  </span>

  <span class="w-[5ch] shrink-0 text-right tabular-nums">
    {vValue === null ? '—' : fFormatMonitorPercent(vPercent)}
  </span>
</div>
