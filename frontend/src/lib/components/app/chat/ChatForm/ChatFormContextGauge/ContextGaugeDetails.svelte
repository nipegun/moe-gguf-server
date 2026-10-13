<script lang="ts">
  import { fTranslate } from '../../../../../i18n';

  import { ChevronDown } from '@lucide/svelte';
  import * as Collapsible from '$lib/components/ui/collapsible';
  import { cStatsUnits } from '$lib/constants';
  import ContextGaugeDetailRow from './ContextGaugeDetailRow.svelte';

  interface Props {
    currentRead: number;
    currentFresh: number;
    currentCache: number;
    currentOutput: number;
    kvTotal: number;
    cumulativeRead: number;
    cumulativeOutput: number;
    cumulativeCacheTotal: number;
    averageTokensPerSecond: number | null;
    transientDetails: string[];
  }

  let {
    currentRead: vCurrentRead,
    currentFresh: vCurrentFresh,
    currentCache: vCurrentCache,
    currentOutput: vCurrentOutput,
    kvTotal: vKvTotal,
    cumulativeRead: vCumulativeRead,
    cumulativeOutput: vCumulativeOutput,
    cumulativeCacheTotal: vCumulativeCacheTotal,
    averageTokensPerSecond: vAverageTokensPerSecond,
    transientDetails: vTransientDetails
  }: Props = $props();

  let vOpen = $state(false);

  const cHasCumulative = $derived(vCumulativeRead > 0 || vCumulativeOutput > 0);
  const cHasCurrent = $derived(vCurrentRead > 0 || vCurrentOutput > 0);
</script>

<Collapsible.Root bind:open={vOpen} class="mt-3 border-t border-border/50 pt-4">
  <Collapsible.Trigger
    class="flex w-full cursor-pointer items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
  >
    <span>{fTranslate('text_f59cbe0da913')}</span>

    <ChevronDown class={'ml-auto h-3 w-3 transition-transform' + (vOpen ? ' rotate-180' : '')} />
  </Collapsible.Trigger>

  <Collapsible.Content class="flex flex-col gap-4 text-xs pt-4">
    {#if cHasCumulative}
      <div>
        <h3 class="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/70 mb-2">
          {fTranslate('text_a1b285c2e617')}
        </h3>

        <div class="flex flex-col gap-2">
          {#if vCumulativeRead > 0}
            <ContextGaugeDetailRow
              label={fTranslate('text_5c5444cd1b95')}
              value={`${vCumulativeRead.toLocaleString()} tok`}
              subtitle={vCumulativeCacheTotal > 0
                ? `${vCumulativeCacheTotal.toLocaleString()} reused from KV cache`
                : undefined}
            />
          {/if}
          {#if vCumulativeOutput > 0}
            <ContextGaugeDetailRow
              label={fTranslate('text_2f3bda40333c')}
              value={`${vCumulativeOutput.toLocaleString()} tok`}
            />
          {/if}
        </div>
      </div>
    {/if}

    {#if cHasCurrent}
      <div>
        <h3 class="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/70 mb-2">
          {fTranslate('text_e896ec9b0612')}
        </h3>

        <div class="flex flex-col gap-2">
          {#if vCurrentRead > 0}
            <ContextGaugeDetailRow
              label={fTranslate('text_5c39123805ff')}
              value={`${vCurrentRead.toLocaleString()} tok`}
              subtitle={vCurrentCache > 0
                ? `${vCurrentFresh.toLocaleString()} fresh + ${vCurrentCache.toLocaleString()} cached`
                : undefined}
            />
          {/if}

          {#if vCurrentOutput > 0}
            <ContextGaugeDetailRow
              label={fTranslate('text_827ec8d9f99d')}
              value={`${vCurrentOutput.toLocaleString()} tok`}
            />
          {/if}

          <div class="pt-1 mt-0.5 border-t border-border/30">
            <div class="flex justify-between">
              <span class="text-muted-foreground">{fTranslate('text_4e453eb018ba')}</span>
              <span class="font-mono font-medium"
                >{vKvTotal.toLocaleString()} {fTranslate('text_1a7674eb4ee7')}</span
              >
            </div>
          </div>
        </div>
      </div>
    {/if}

    {#if vAverageTokensPerSecond !== null}
      <div class="pt-1.5 mt-1 border-t border-border/30">
        <ContextGaugeDetailRow
          label={fTranslate('text_607e4c5f4272')}
          value={`${vAverageTokensPerSecond.toFixed(1)}${cStatsUnits.TOKENS_PER_SECOND}`}
        />
      </div>
    {/if}

    {#each vTransientDetails as detail (detail)}
      <div class="font-mono text-muted-foreground">{detail}</div>
    {/each}
  </Collapsible.Content>
</Collapsible.Root>
