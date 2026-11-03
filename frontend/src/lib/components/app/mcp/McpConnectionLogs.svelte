<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { ChevronDown, ChevronRight } from '@lucide/svelte';
  import * as Collapsible from '$lib/components/ui/collapsible';
  import type { MCPConnectionLog } from '$lib/types';
  import { fFormatTime, fGetMcpLogLevelIcon, fGetMcpLogLevelClass } from '$lib/utils';

  interface Props {
    logs: MCPConnectionLog[];
    connectionTimeMs?: number;
    defaultExpanded?: boolean;
    class?: string;
  }

  let {
    logs: vLogs,
    connectionTimeMs: vConnectionTimeMs,
    defaultExpanded: vDefaultExpanded = false,
    class: vClassName
  }: Props = $props();

  let vIsExpanded = $derived(vDefaultExpanded);

  function fFormatLogDetails(pDetails: unknown): string {
    if (pDetails == null) {
      return '';
    }

    try {
      return JSON.stringify(pDetails, null, 2);
    } catch {
      return String(pDetails);
    }
  }
</script>

{#if vLogs.length > 0}
  <Collapsible.Root bind:open={vIsExpanded} class={vClassName}>
    <div class="space-y-2">
      <Collapsible.Trigger
        class="flex w-full items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
      >
        {#if vIsExpanded}
          <ChevronDown class="h-3.5 w-3.5" />
        {:else}
          <ChevronRight class="h-3.5 w-3.5" />
        {/if}

        <span>{fTranslate('text_021503c5ec57')}{vLogs.length})</span>

        {#if vConnectionTimeMs !== undefined}
          <span class="ml-1"
            >{fTranslate('text_a20a64e98af1')}
            {vConnectionTimeMs}{fTranslate('text_f785c3ce1d58')}</span
          >
        {/if}
      </Collapsible.Trigger>
    </div>

    <Collapsible.Content class="mt-2">
      <div
        class="max-h-64 space-y-0.5 overflow-y-auto rounded bg-muted/50 p-2 font-mono text-[10px]"
      >
        {#each vLogs as vLog (vLog.timestamp.getTime() + vLog.message)}
          {@const IconComponent = fGetMcpLogLevelIcon(vLog.level)}

          <div class={['flex items-start gap-1.5', fGetMcpLogLevelClass(vLog.level)]}>
            <span class="shrink-0 text-muted-foreground">
              {fFormatTime(vLog.timestamp)}
            </span>

            <IconComponent class="mt-0.5 h-3 w-3 shrink-0" />

            <span class="break-all">{vLog.message}</span>
          </div>

          {#if vLog.details !== undefined}
            <details class="ml-11">
              <summary class="cursor-pointer text-[10px] text-muted-foreground">
                {fTranslate('text_41ffd8b76afc')}
              </summary>

              <pre
                class="mt-1 overflow-x-auto rounded bg-background/70 p-2 text-[10px] break-all whitespace-pre-wrap text-foreground/80">
{fFormatLogDetails(vLog.details)}</pre>
            </details>
          {/if}
        {/each}
      </div>
    </Collapsible.Content>
  </Collapsible.Root>
{/if}
