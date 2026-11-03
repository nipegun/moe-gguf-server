<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { XCircle } from '@lucide/svelte';
  import { type AgenticSection } from '$lib/utils';
  import { fParseGrepSearchMeta } from './parsers/grep-search';
  import ToolCallBlock from './ToolCallBlock.svelte';

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    onToggle?: () => void;
  }

  let {
    section: vSection,
    open: vOpen,
    isStreaming: vIsStreaming,
    onToggle: vOnToggle
  }: Props = $props();

  const cGrepMeta = $derived(fParseGrepSearchMeta(vSection));
</script>

<ToolCallBlock
  section={vSection}
  open={vOpen}
  isStreaming={vIsStreaming}
  meta={cGrepMeta}
  onToggle={vOnToggle}
>
  {#snippet titleSnippet()}
    {#if cGrepMeta}
      <span class="text-muted-foreground">{fTranslate('text_6ea8f7433c6c')} </span>
      <span class="font-mono">{cGrepMeta.pattern}</span>
      <span class="text-muted-foreground"> {fTranslate('text_582967534d0f')} </span>
      <span class="font-mono">{cGrepMeta.path}</span>
    {/if}
  {/snippet}

  {#snippet children(pMeta, pCtx)}
    {#if pCtx.isPending}
      <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
        {fTranslate('text_78c9d9f6ace0')}
      </div>
    {:else if pMeta?.errorMessage}
      <div
        class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
      >
        <XCircle class="mt-0.5 h-3 w-3 shrink-0" />
        <span>{pMeta.errorMessage}</span>
      </div>
    {:else if pMeta && pMeta.matches.length > 0}
      <div class="max-h-96 overflow-auto">
        {#each pMeta.matches as vMatch, vMi (vMi)}
          <div class="font-mono text-[11px] leading-relaxed">
            <span class="text-muted-foreground/70">{vMatch.file}</span>
            {#if pMeta.showLineNumbers && vMatch.line != null}
              <span class="text-muted-foreground/70">:{vMatch.line}</span>
            {/if}
            <span class="text-muted-foreground/70">:</span>
            <span>{vMatch.content}</span>
          </div>
        {/each}
      </div>
      <div class="mt-1.5 text-xs text-muted-foreground/70 italic">
        {fTranslate('text_9f2dda0efa37')}
        <span class="font-mono">{pMeta.totalMatches ?? pMeta.matches.length}</span>
        {#if pMeta.showLineNumbers}
          &nbsp;<span class="italic">{fTranslate('text_f1edb1862f8d')}</span>
        {/if}
      </div>
    {:else}
      <div class="text-xs text-muted-foreground/70 italic">{fTranslate('text_2df01a03ff43')}</div>
      <div class="mt-1.5 text-xs text-muted-foreground/70 italic">
        {fTranslate('text_9f2dda0efa37')} <span class="font-mono">{pMeta?.totalMatches ?? 0}</span>
      </div>
    {/if}
  {/snippet}
</ToolCallBlock>
