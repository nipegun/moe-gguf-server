<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { SyntaxHighlightedCode } from '$lib/components/app';
  import { cDefaultLanguage, cMaxHeightCodeBlock } from '$lib/constants';
  import { type AgenticSection } from '$lib/utils';
  import { fParseReadFileMeta } from './parsers/read-file';
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

  const cReadFileMeta = $derived(fParseReadFileMeta(vSection));
</script>

<ToolCallBlock
  section={vSection}
  open={vOpen}
  isStreaming={vIsStreaming}
  meta={cReadFileMeta}
  onToggle={vOnToggle}
>
  {#snippet titleSnippet()}
    <span class="text-muted-foreground">{fTranslate('text_47659c3dccc8')} </span>
    <span class="font-mono">{cReadFileMeta?.fileName}</span>
    {#if cReadFileMeta?.lineRange}
      <span class="text-muted-foreground">
        {fTranslate('text_83f47275bb38')}
        {cReadFileMeta.lineRange.start}-{cReadFileMeta.lineRange.end})</span
      >
    {/if}
  {/snippet}

  {#snippet children(_meta, _ctx)}
    {#if vSection.toolResult}
      <SyntaxHighlightedCode
        code={vSection.toolResult}
        language={cReadFileMeta?.language ?? cDefaultLanguage}
        maxHeight={cMaxHeightCodeBlock}
      />
    {:else}
      <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
        {fTranslate('text_510899fbc0a1')}
      </div>
    {/if}
  {/snippet}
</ToolCallBlock>
