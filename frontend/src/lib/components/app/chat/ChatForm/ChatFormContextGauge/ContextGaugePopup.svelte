<script lang="ts">
  import { fTranslate } from '../../../../../i18n';

  import { fFormatParameters } from '$lib/utils/formatters';
  import { fUseContextGauge } from '$lib/hooks/use-context-gauge.svelte';
  import ContextGaugeDetails from './ContextGaugeDetails.svelte';
  import ContextGaugeLoadModel from './ContextGaugeLoadModel.svelte';
  import { fColorLevelBgClass, fColorLevelTextClass } from './context-gauge';
  import {
    cGaugePopup,
    fGaugeCardEnter,
    fGaugeCardLeave,
    fGaugePopupClose
  } from '$lib/stores/context-gauge-popup.svelte';

  const cGauge = fUseContextGauge();

  // The gauge hook wraps a processing state instance that only follows the
  // live stream while its own monitoring flag is set, so the card instance
  // starts monitoring like the dial does.
  $effect(() => {
    cGauge.startMonitoring();
  });

  let vCardEl = $state<HTMLElement | null>(null);

  // Any press outside the card and outside the dial closes the card.
  // Presses on the dial are excluded because the dial handles its own
  // toggle; the listener only exists while the card is open.
  $effect(() => {
    if (!cGaugePopup.open) return;

    const fOnPointerDown = (pEvent: PointerEvent) => {
      const cTarget = pEvent.target;
      if (!(cTarget instanceof Node)) return;
      if (vCardEl?.contains(cTarget)) return;
      if (cTarget instanceof Element && cTarget.closest('[data-context-gauge-trigger]')) return;
      fGaugePopupClose();
    };

    document.addEventListener('pointerdown', fOnPointerDown, true);
    return () => document.removeEventListener('pointerdown', fOnPointerDown, true);
  });

  const cShowProgressBar = $derived(
    cGauge.contextTotal !== null &&
      cGauge.contextTotal > 0 &&
      (cGauge.activeModelId !== null || cGauge.isActiveModelLoaded)
  );
</script>

{#if cGaugePopup.open}
  <div
    role="status"
    bind:this={vCardEl}
    class="absolute z-50 w-64 -translate-x-1/2 rounded-lg border border-border/50 bg-popover p-3 text-sm text-popover-foreground shadow-lg ring-1 ring-foreground/10"
    style="left: {cGaugePopup.centerX}px; bottom: {cGaugePopup.bottom}px"
    onpointerenter={fGaugeCardEnter}
    onpointerleave={fGaugeCardLeave}
  >
    <div class="flex flex-col gap-2">
      <div class="flex items-center gap-2">
        <span class="font-medium">{fTranslate('text_a6e600a10fed')}</span>
        <span class="text-muted-foreground">·</span>
        <span class="font-mono text-muted-foreground">
          {fFormatParameters(cGauge.contextUsed)}
          / {cGauge.contextTotal !== null ? fFormatParameters(cGauge.contextTotal) : '-'}
        </span>
      </div>

      {#if cGauge.activeModelId !== null && !cGauge.isActiveModelLoaded}
        <ContextGaugeLoadModel
          modelId={cGauge.activeModelId}
          isLoading={cGauge.isActiveModelLoading}
          onLoad={cGauge.loadModel}
        />
      {:else if cShowProgressBar}
        <div class="h-1.5 w-full overflow-hidden rounded-full bg-muted">
          <div
            class="h-full rounded-full transition-all duration-300 {fColorLevelBgClass(
              cGauge.colorLevel
            )}"
            style="width: {cGauge.contextPercent}%"
          ></div>
        </div>

        <div class="flex justify-between text-xs text-muted-foreground">
          <span>
            <span class={fColorLevelTextClass(cGauge.colorLevel)}>{cGauge.contextPercent}%</span>
            {fTranslate('text_f83916135509')}
          </span>
          <span>
            {fFormatParameters((cGauge.contextTotal ?? 0) - cGauge.contextUsed)}
            {fTranslate('text_43e3ece73ff8')}
          </span>
        </div>
      {:else}
        <div class="text-xs text-muted-foreground">{fTranslate('text_2d1c5067fcc7')}</div>
      {/if}

      {#if cGauge.hasAnyUsage}
        <ContextGaugeDetails
          currentRead={cGauge.currentRead}
          currentFresh={cGauge.currentFresh}
          currentCache={cGauge.currentCache}
          currentOutput={cGauge.currentOutput}
          kvTotal={cGauge.kvTotal}
          cumulativeRead={cGauge.cumulativeRead}
          cumulativeOutput={cGauge.cumulativeOutput}
          cumulativeCacheTotal={cGauge.cumulativeCacheTotal}
          averageTokensPerSecond={cGauge.averageTokensPerSecond}
          transientDetails={cGauge.transientDetails}
        />
      {/if}
    </div>
  </div>
{/if}
