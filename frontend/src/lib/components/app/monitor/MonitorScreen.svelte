<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { Activity, Cpu, Gpu, X } from '@lucide/svelte';
  import { fade } from 'svelte/transition';
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { ActionIcon } from '$lib/components/app';
  import { Skeleton } from '$lib/components/ui/skeleton';
  import { cRoutes } from '$lib/constants';
  import { cMonitorStore } from '$lib/stores/monitor.svelte';
  import {
    fFormatMonitorBytes,
    fFormatMonitorNumber,
    fFormatMonitorPercent,
    fMonitorOutsideVram
  } from '$lib/utils/monitor';
  import type { MonitorCpuProcessor } from '$lib/types';
  import MonitorMemoryBar from './MonitorMemoryBar.svelte';
  import MonitorMeter from './MonitorMeter.svelte';

  interface Props {
    class?: string;
  }

  let { class: vClassName = '' }: Props = $props();

  let vReport = $derived(cMonitorStore.vReport);
  let vError = $derived(cMonitorStore.vReportError);

  // one card per physical processor; without a processor list, one card with the global usage
  let vLProcessors = $derived.by<MonitorCpuProcessor[]>(() => {
    if (!vReport) return [];
    if (vReport.cpu.processors.length > 0) return vReport.cpu.processors;
    return [
      {
        id: 0,
        model: '',
        usage: vReport.cpu.usage,
        logical_cpus: vReport.cpu.logical_cpus,
        cores: []
      }
    ];
  });

  // GPUs without memory figures (an Intel GPU that no model uses yet) have no VRAM bar
  let vLMemoryGpus = $derived(vReport ? vReport.gpus.filter((pGpu) => pGpu.total_bytes > 0) : []);

  // the load average belongs to the whole system, so every processor card shows it
  let vLoadAverage = $derived(
    vReport?.cpu.load_average
      ? vReport.cpu.load_average.map((pValue) => fFormatMonitorNumber(pValue, 2)).join(' · ')
      : ''
  );

  let vPreviousRouteId = $state<string | null>(null);

  $effect(() => {
    const cCurrentId = page.route.id;
    return () => {
      vPreviousRouteId = cCurrentId;
    };
  });

  function fHandleClose() {
    if (browser && window.history.length > 1 && vPreviousRouteId !== '/monitor') {
      history.back();
    } else {
      goto(cRoutes.START);
    }
  }

  onMount(() => cMonitorStore.mSubscribeReport());
</script>

<!-- "gpu-1: NVIDIA GeForce RTX 4060 Ti": the device label in bold, then its name -->
{#snippet cardHeader(Icon: typeof Cpu, label: string, title?: string, subtitle?: string)}
  <div class="flex min-w-0 items-start gap-2">
    <Icon class="mt-0.5 size-4 shrink-0 text-muted-foreground" />

    <div class="min-w-0">
      <h3 class="text-sm [overflow-wrap:anywhere]">
        <strong class="font-bold">{label}</strong>{#if title}: {title}{/if}
      </h3>

      {#if subtitle}
        <p class="text-xs text-muted-foreground [overflow-wrap:anywhere]">{subtitle}</p>
      {/if}
    </div>
  </div>
{/snippet}

{#snippet detail(label: string, value: string)}
  <div class="flex min-w-0 items-baseline justify-between gap-3">
    <dt class="min-w-0 text-muted-foreground">{label}</dt>
    <dd class="min-w-0 text-right tabular-nums [overflow-wrap:anywhere]">{value}</dd>
  </div>
{/snippet}

<div in:fade={{ duration: 150 }} class="flex min-h-[calc(100dvh-4rem)] flex-col">
  <div class="fixed top-4.5 right-4 z-50 md:hidden">
    <ActionIcon icon={X} tooltip={fTranslate('text_7d9eb7acb13e')} onclick={fHandleClose} />
  </div>

  <div class="sticky top-0 z-10 mt-4 mb-2 flex items-center gap-2 px-4 md:p-4 md:px-8">
    <Activity class="h-5 w-5 md:h-6 md:w-6" />

    <h1 class="text-lg font-semibold md:text-2xl">{fTranslate('text_4c2e1df4570b')}</h1>
  </div>

  <div class="mx-auto flex w-full min-w-0 max-w-6xl flex-col gap-8 {vClassName}">
    {#if vError}
      <p
        class="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive [overflow-wrap:anywhere]"
        role="alert"
      >
        {fTranslate('text_2da0a129e146', { p0: vError })}
      </p>
    {/if}

    {#if !vReport}
      {#if !vError}
        <div class="grid gap-3 md:grid-cols-2">
          <Skeleton class="h-40 rounded-xl" />
          <Skeleton class="h-40 rounded-xl" />
          <Skeleton class="h-48 rounded-xl md:col-span-2" />
        </div>
      {/if}
    {:else}
      <section class="flex min-w-0 flex-col gap-3">
        <h2 class="text-base font-semibold">{fTranslate('text_8d59829c1e15')}</h2>

        <div
          class="grid min-w-0 gap-3"
          style="grid-template-columns: repeat(auto-fill, minmax(min(20rem, 100%), 1fr));"
        >
          {#each vReport.gpus as vGpu, vIndex (vGpu.id)}
            <article class="flex min-w-0 flex-col gap-3 rounded-xl border bg-card p-4">
              {@render cardHeader(
                Gpu,
                `gpu-${vIndex + 1}`,
                vGpu.name,
                vGpu.backends.length > 0
                  ? fTranslate('text_251ae8dfb35f', { p0: vGpu.backends.join(', ') })
                  : vGpu.id
              )}

              <MonitorMeter
                label={fTranslate('text_8d59829c1e15')}
                value={vGpu.usage}
                size="md"
                class="text-sm"
              />

              {#if vGpu.memory_usage !== null}
                <MonitorMeter
                  label={fTranslate('text_46a7d46b7d30')}
                  value={vGpu.memory_usage}
                  size="md"
                  class="text-sm"
                />
              {/if}

              <dl class="flex min-w-0 flex-col gap-1 text-xs">
                {#if vGpu.total_bytes > 0}
                  {@render detail(
                    fTranslate('text_de956947095b'),
                    fTranslate('text_eee4ca253785', {
                      p0: fFormatMonitorBytes(vGpu.used_bytes),
                      p1: fFormatMonitorBytes(vGpu.total_bytes)
                    })
                  )}
                {/if}

                {#if vGpu.kernel}
                  {@render detail(
                    fTranslate('text_faa542a861aa'),
                    fTranslate('text_c52d07ce5178', {
                      p0: fFormatMonitorBytes(vGpu.kernel.vram_bytes),
                      p1: fFormatMonitorBytes(vGpu.kernel.gtt_bytes)
                    })
                  )}
                {/if}

                {#if vGpu.temperature_c !== null}
                  {@render detail(
                    fTranslate('text_b958ce8b871a'),
                    `${fFormatMonitorNumber(vGpu.temperature_c)} °C`
                  )}
                {/if}

                {#if vGpu.power_w !== null}
                  {@render detail(
                    fTranslate('text_848e96567d6b'),
                    vGpu.power_limit_w !== null
                      ? `${fFormatMonitorNumber(vGpu.power_w)} / ${fFormatMonitorNumber(vGpu.power_limit_w)} W`
                      : `${fFormatMonitorNumber(vGpu.power_w)} W`
                  )}
                {/if}
              </dl>
            </article>
          {:else}
            <article
              class="flex min-w-0 items-center gap-2 rounded-xl border border-dashed p-4 text-sm text-muted-foreground"
            >
              <Gpu class="size-4 shrink-0" />
              {fTranslate('text_e60a0ce3c3fc')}
            </article>
          {/each}

          {#each vLProcessors as vProcessor, vIndex (vProcessor.id)}
            <article class="flex min-w-0 flex-col gap-3 rounded-xl border bg-card p-4">
              {@render cardHeader(Cpu, `cpu-${vIndex + 1}`, vProcessor.model || undefined)}

              <MonitorMeter
                label={fTranslate('text_8d59829c1e15')}
                value={vProcessor.usage}
                size="md"
                class="text-sm"
              />

              <p class="text-xs text-muted-foreground">
                {fTranslate('text_131b2a6ed189', { p0: vProcessor.logical_cpus })}
              </p>

              <dl class="flex min-w-0 flex-col gap-1 text-xs">
                {#if vLoadAverage}
                  {@render detail(fTranslate('text_1d77f4555f3f'), vLoadAverage)}
                {/if}
              </dl>

              {#if vProcessor.cores.length > 0}
                <div class="flex min-w-0 flex-col gap-1">
                  <p class="text-xs text-muted-foreground">{fTranslate('text_994489c640f8')}</p>

                  <div
                    class="grid min-w-0 gap-0.5"
                    style="grid-template-columns: repeat(auto-fit, minmax(0.5rem, 1fr));"
                  >
                    {#each vProcessor.cores as vCore (vCore.id)}
                      {@const vCoreLabel = fTranslate('text_13b6d047ac72', {
                        p0: vCore.id,
                        p1: fFormatMonitorPercent(vCore.usage)
                      })}
                      <span
                        class="relative h-6 min-w-0 overflow-hidden rounded-sm"
                        style="background-color: var(--monitor-meter-free)"
                        role="img"
                        aria-label={vCoreLabel}
                        title={vCoreLabel}
                      >
                        <span
                          class="absolute inset-x-0 bottom-0 transition-[height] duration-500 ease-out"
                          style="height: {Math.min(
                            100,
                            Math.max(0, vCore.usage)
                          )}%; background-color: var(--monitor-meter-used)"
                        ></span>
                      </span>
                    {/each}
                  </div>
                </div>
              {/if}
            </article>
          {/each}
        </div>
      </section>

      <section class="flex min-w-0 flex-col gap-3">
        <div class="min-w-0">
          <h2 class="text-base font-semibold">{fTranslate('text_c3963aedaac6')}</h2>

          <p class="text-xs text-muted-foreground">{fTranslate('text_9e8127fbcb1a')}</p>
        </div>

        <div
          class="grid min-w-0 gap-3"
          style="grid-template-columns: repeat(auto-fill, minmax(min(28rem, 100%), 1fr));"
        >
          {#each vLMemoryGpus as vGpu (vGpu.id)}
            {@const vOutside = fMonitorOutsideVram(vGpu)}
            <div class="flex min-w-0 flex-col gap-2">
              {#if vOutside > 0}
                <p
                  class="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive [overflow-wrap:anywhere]"
                  role="alert"
                >
                  {fTranslate('text_d10c6eda5a5c', { p0: fFormatMonitorBytes(vOutside) })}
                </p>
              {/if}

              <MonitorMemoryBar
                title={fTranslate('text_de956947095b')}
                subtitle={vGpu.name}
                pool={vGpu}
              />
            </div>
          {/each}

          <MonitorMemoryBar
            title={fTranslate('text_bf4b0c03c646')}
            subtitle={vReport.ram.swap_total_bytes > 0
              ? fTranslate('text_78c3e7b6ab43', {
                  p0: fFormatMonitorBytes(vReport.ram.swap_used_bytes),
                  p1: fFormatMonitorBytes(vReport.ram.swap_total_bytes)
                })
              : undefined}
            pool={vReport.ram}
          />
        </div>
      </section>
    {/if}
  </div>
</div>
