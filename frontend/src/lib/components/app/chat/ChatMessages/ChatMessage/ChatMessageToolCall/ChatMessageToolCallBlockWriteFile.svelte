<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { XCircle } from '@lucide/svelte';
  import { SyntaxHighlightedCode } from '$lib/components/app';
  import { cMaxHeightCodeBlock, cResultStatSeparator } from '$lib/constants';
  import { type AgenticSection } from '$lib/utils';
  import { fParseWriteFileMeta } from './parsers/write-file';
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

  const cWriteFileMeta = $derived(fParseWriteFileMeta(vSection));
</script>

<ToolCallBlock
  section={vSection}
  open={vOpen}
  isStreaming={vIsStreaming}
  meta={cWriteFileMeta}
  onToggle={vOnToggle}
>
  {#snippet titleSnippet()}
    <span class="text-muted-foreground">{fTranslate('text_997357233881')} </span>
    <span class="font-mono">{cWriteFileMeta?.filePath}</span>
    {#if cWriteFileMeta?.errorMessage}
      <span class="ml-1 text-xs italic text-muted-foreground/70"
        >{fTranslate('text_a1bf75497647')}</span
      >
    {/if}
  {/snippet}

  {#snippet children(meta, ctx)}
    {#if meta?.errorMessage}
      <div
        class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
      >
        <XCircle class="mt-0.5 h-3 w-3 shrink-0" />
        <span>{meta.errorMessage}</span>
      </div>
    {:else if meta}
      <SyntaxHighlightedCode
        code={meta.content}
        language={meta.language}
        maxHeight={cMaxHeightCodeBlock}
        streaming={ctx.isCodeStreaming}
      />
      <div class="mt-1.5 text-xs text-muted-foreground/70 italic">
        {#if meta.resultMessage}
          {meta.resultMessage}{meta.bytesWritten != null ? cResultStatSeparator : ''}{/if}
        {#if meta.bytesWritten != null}
          <span class="font-mono">{meta.bytesWritten}</span> {fTranslate('text_277089d91c0b')}
        {/if}
      </div>
    {/if}
  {/snippet}
</ToolCallBlock>
