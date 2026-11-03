<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import * as Tooltip from '$lib/components/ui/tooltip';
  import { aMonitorSegmentGroups } from '$lib/constants/monitor';
  import {
    fFormatMonitorBytes,
    fFormatMonitorPercent,
    fMonitorGroupBytes
  } from '$lib/utils/monitor';
  import type { MonitorMemoryPool } from '$lib/types';

  interface Props {
    title: string;
    /** Secondary line under the title, e.g. the device name. */
    subtitle?: string;
    pool: MonitorMemoryPool;
    class?: string;
  }

  let { title: vTitle, subtitle: vSubtitle, pool: vPool, class: vClassName = '' }: Props = $props();

  interface MemoryPart {
    id: string;
    label: string;
    colorVar: string | null;
    bytes: number;
    percent: number;
  }

  let vTotal = $derived(Math.max(vPool.total_bytes, 1));

  // pinned staging buffers and similar crumbs (a few KiB) would only clutter the bar and the legend
  const cMinimumBytes = 1024 * 1024;

  // segments with memory, in the fixed group order; free memory closes the bar as its track
  let aParts = $derived.by<MemoryPart[]>(() => {
    const aResult: MemoryPart[] = [];
    for (const vGroup of aMonitorSegmentGroups) {
      const vBytes = fMonitorGroupBytes(vPool.segments, vGroup);
      if (vBytes >= cMinimumBytes) {
        aResult.push({
          id: vGroup.id,
          label: vGroup.label,
          colorVar: vGroup.colorVar,
          bytes: vBytes,
          percent: (vBytes / vTotal) * 100
        });
      }
    }
    return aResult;
  });

  let vFreePart = $derived<MemoryPart>({
    id: 'free',
    label: fTranslate('text_f411a1fb6275'),
    colorVar: null,
    bytes: vPool.free_bytes,
    percent: (vPool.free_bytes / vTotal) * 100
  });

  let vUsedBytes = $derived(Math.max(0, vPool.total_bytes - vPool.free_bytes));

  function fPartTitle(pPart: MemoryPart): string {
    return fTranslate('text_07e3a63ea0ed', {
      p0: pPart.label,
      p1: `${fFormatMonitorBytes(pPart.bytes)} · ${fFormatMonitorPercent(pPart.percent, 1)}`
    });
  }
</script>

<section class="flex min-w-0 flex-col gap-3 rounded-xl border bg-card p-4 {vClassName}">
  <header class="flex min-w-0 flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
    <div class="min-w-0">
      <h3 class="text-sm font-semibold [overflow-wrap:anywhere]">{vTitle}</h3>

      {#if vSubtitle}
        <p class="text-xs text-muted-foreground [overflow-wrap:anywhere]">{vSubtitle}</p>
      {/if}
    </div>

    <span class="min-w-0 text-xs text-muted-foreground tabular-nums">
      {fTranslate('text_eee4ca253785', {
        p0: fFormatMonitorBytes(vUsedBytes),
        p1: fFormatMonitorBytes(vPool.total_bytes)
      })}
    </span>
  </header>

  <div class="flex h-5 w-full min-w-0 gap-0.5 overflow-hidden rounded-md">
    {#each aParts as vPart (vPart.id)}
      <Tooltip.Root>
        <Tooltip.Trigger
          class="h-full min-w-0 cursor-default"
          style="flex: 0 1 {vPart.percent}%; background-color: var({vPart.colorVar})"
          aria-label={fPartTitle(vPart)}
        />
        <Tooltip.Content>
          <p>{fPartTitle(vPart)}</p>
        </Tooltip.Content>
      </Tooltip.Root>
    {/each}

    {#if vFreePart.bytes > 0}
      <Tooltip.Root>
        <Tooltip.Trigger
          class="h-full min-w-0 flex-1 cursor-default bg-muted"
          aria-label={fPartTitle(vFreePart)}
        />
        <Tooltip.Content>
          <p>{fPartTitle(vFreePart)}</p>
        </Tooltip.Content>
      </Tooltip.Root>
    {/if}
  </div>

  <!-- the legend doubles as the table view: every segment with its size and share, one per line in the order of the bar -->
  <ul class="flex min-w-0 flex-col gap-1 text-xs">
    {#each [...aParts, vFreePart] as vPart (vPart.id)}
      <li class="flex min-w-0 items-center gap-2">
        <span
          class="size-2.5 shrink-0 rounded-sm {vPart.colorVar ? '' : 'bg-muted ring-1 ring-border'}"
          style={vPart.colorVar ? `background-color: var(${vPart.colorVar})` : undefined}
        ></span>

        <span class="min-w-0 flex-1 [overflow-wrap:anywhere]">{vPart.label}</span>

        <span class="shrink-0 text-muted-foreground tabular-nums">
          {fFormatMonitorBytes(vPart.bytes)}
        </span>

        <span class="w-[7ch] shrink-0 text-right text-muted-foreground tabular-nums">
          {fFormatMonitorPercent(vPart.percent, 1)}
        </span>
      </li>
    {/each}
  </ul>
</section>
