<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { XCircle } from '@lucide/svelte';
  import { type AgenticSection } from '$lib/utils';
  import { fParseFileGlobSearchMeta } from './parsers/file-glob-search';
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

  const cFileGlobMeta = $derived(fParseFileGlobSearchMeta(vSection));
</script>

<ToolCallBlock
  section={vSection}
  open={vOpen}
  isStreaming={vIsStreaming}
  meta={cFileGlobMeta}
  onToggle={vOnToggle}
>
  {#snippet titleSnippet()}
    {#if cFileGlobMeta}
      <span class="text-muted-foreground"
        >{cFileGlobMeta.include === '**'
          ? fTranslate('text_e700817c9d5e')
          : fTranslate('text_179fed85ec50')}&nbsp;</span
      >
      {#if cFileGlobMeta.include !== '**'}
        <span class="font-mono">{cFileGlobMeta.include}</span>
      {/if}
      <span class="text-muted-foreground"> {fTranslate('text_582967534d0f')} </span>
      <span class="font-mono">{cFileGlobMeta.path}</span>
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
        {#each pMeta.matches as vMatch, vI (vI)}
          <div class="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">{vMatch}</div>
        {/each}
      </div>
      <div class="mt-1.5 text-xs text-muted-foreground/70 italic">
        {fTranslate('text_9f2dda0efa37')}
        <span class="font-mono">{pMeta.totalMatches ?? pMeta.matches.length}</span>
      </div>
    {:else}
      <div class="text-xs text-muted-foreground/70 italic">{fTranslate('text_2df01a03ff43')}</div>
      <div class="mt-1.5 text-xs text-muted-foreground/70 italic">
        {fTranslate('text_9f2dda0efa37')} <span class="font-mono">{pMeta?.totalMatches ?? 0}</span>
      </div>
    {/if}
  {/snippet}
</ToolCallBlock>
