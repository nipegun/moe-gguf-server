<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { onMount } from 'svelte';
  import { cRoutes } from '$lib/constants';
  import { cMonitorStore } from '$lib/stores/monitor.svelte';
  import { fFormatMonitorPercent } from '$lib/utils/monitor';
  import MonitorMeter from './MonitorMeter.svelte';

  onMount(() => cMonitorStore.mSubscribeUsage());

  type StatusBarDevice = { key: string; label: string; usage: number | null; details: string };

  // "gpu-1: NVIDIA GeForce RTX 4060 Ti: 58 %", the hover text of one meter
  function fDeviceDetails(pLabel: string, pName: string, pUsage: number | null): string {
    const vUsage =
      pUsage === null ? fTranslate('text_67a926f7008e') : fFormatMonitorPercent(pUsage);
    const vName = pName ? fTranslate('text_07e3a63ea0ed', { p0: pLabel, p1: pName }) : pLabel;
    return fTranslate('text_07e3a63ea0ed', { p0: vName, p1: vUsage });
  }

  // one meter per GPU (gpu-1, gpu-2...) and then one per physical processor (cpu-1, cpu-2...),
  // labelled and ordered like the cards of the Monitor page
  let ldDevices = $derived.by<StatusBarDevice[]>(() => {
    const dUsage = cMonitorStore.vUsage;
    const ldGpus = (dUsage?.gpus ?? []).map((pGpu, pIndex) => ({
      key: `gpu-${pGpu.id}`,
      label: `gpu-${pIndex + 1}`,
      usage: pGpu.usage,
      details: fDeviceDetails(`gpu-${pIndex + 1}`, pGpu.name, pGpu.usage)
    }));
    // without a processor list, one meter with the global usage
    const ldProcessors =
      dUsage && dUsage.cpu.processors.length > 0
        ? dUsage.cpu.processors
        : [{ id: 0, model: '', usage: dUsage?.cpu.usage ?? null }];
    const ldCpus = ldProcessors.map((pProcessor, pIndex) => ({
      key: `cpu-${pProcessor.id}`,
      label: `cpu-${pIndex + 1}`,
      usage: pProcessor.usage,
      details: fDeviceDetails(`cpu-${pIndex + 1}`, pProcessor.model, pProcessor.usage)
    }));
    return [...ldGpus, ...ldCpus];
  });

  let vHasGpus = $derived((cMonitorStore.vUsage?.gpus.length ?? 0) > 0);
</script>

<!-- the meters keep a minimum width; with many devices the bar scrolls sideways instead of squeezing them -->
<a
  href={cRoutes.MONITOR}
  class="scrollbar-hide fixed inset-x-0 bottom-0 z-40 flex items-start justify-end-safe gap-4 overflow-x-auto border-t border-border/60 bg-background/90 px-4 pt-1.5 text-[11px] leading-4 text-foreground backdrop-blur-md transition-colors hover:bg-muted/90 md:gap-6"
  style="height: var(--status-bar-height); padding-bottom: env(safe-area-inset-bottom, 0px)"
  aria-label={fTranslate('text_c31346395732')}
  title={fTranslate('text_c31346395732')}
>
  {#if !vHasGpus}
    <span class="flex max-w-56 min-w-24 flex-1" title={fTranslate('text_e60a0ce3c3fc')}>
      <MonitorMeter label={fTranslate('text_ea49523d744c')} value={null} class="w-full" />
    </span>
  {/if}

  {#each ldDevices as dDevice (dDevice.key)}
    <span class="flex max-w-56 min-w-24 flex-1" title={dDevice.details}>
      <MonitorMeter label={dDevice.label} value={dDevice.usage} class="w-full" />
    </span>
  {/each}
</a>
