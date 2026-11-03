<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  // Fall-through renderer for tool calls without a dedicated block.
  // Renders section.toolArgs / section.toolResult directly using the
  // shared chrome shell.

  import { Loader2 } from '@lucide/svelte';
  import { MarkdownContent, SyntaxHighlightedCode } from '$lib/components/app';
  import { FileTypeText, ToolResultKind } from '$lib/enums';
  import { cMaxHeightCodeBlock } from '$lib/constants';
  import {
    fClassifyToolResult,
    fFormatJsonPretty,
    fParseToolResultWithImages,
    type AgenticSection
  } from '$lib/utils';
  import { fGetBuiltinToolUi } from '$lib/constants/built-in-tools';
  import type { DatabaseMessageExtra } from '$lib/types';
  import ToolCallBlock from './ToolCallBlock.svelte';

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    attachments?: DatabaseMessageExtra[];
    onToggle?: () => void;
  }

  let {
    section: vSection,
    open: vOpen,
    isStreaming: vIsStreaming,
    attachments: vAttachments,
    onToggle: vOnToggle
  }: Props = $props();

  const cTitle = $derived(fGetBuiltinToolUi(vSection.toolName)?.label ?? vSection.toolName ?? '');
  const cOutputKind = $derived(fClassifyToolResult(vSection.toolResult));
  const cParsedLines = $derived(
    vSection.toolResult ? fParseToolResultWithImages(vSection.toolResult, vAttachments) : []
  );
</script>

<ToolCallBlock
  section={vSection}
  open={vOpen}
  isStreaming={vIsStreaming}
  meta={null}
  title={cTitle}
  onToggle={vOnToggle}
>
  {#snippet children(_meta, pCtx)}
    {#if pCtx.isStreamingCall}
      <div class="mb-2 flex items-center gap-2 text-xs text-muted-foreground/70">
        <span>{fTranslate('text_36ecb4f86691')}</span>
        {#if pCtx.isStreaming}
          <Loader2 class="h-3 w-3 animate-spin" />
        {/if}
      </div>
      {#if vSection.toolArgs}
        <SyntaxHighlightedCode
          code={fFormatJsonPretty(vSection.toolArgs)}
          language={FileTypeText.JSON}
          maxHeight={cMaxHeightCodeBlock}
          streaming={pCtx.isCodeStreaming}
        />
      {:else if pCtx.isStreaming}
        <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
          {fTranslate('text_75a15599785c')}
        </div>
      {:else}
        <div
          class="rounded bg-yellow-500/10 p-2 text-xs text-yellow-600 italic dark:text-yellow-400"
        >
          {fTranslate('text_4494ff030139')}
        </div>
      {/if}
    {:else}
      {@const cShowInput = Boolean(vSection.toolArgs)}
      {#if cShowInput}
        <div class="mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70">
          <span>{fTranslate('text_36ecb4f86691')}</span>
        </div>
        <SyntaxHighlightedCode
          code={fFormatJsonPretty(vSection.toolArgs ?? '')}
          language={FileTypeText.JSON}
          maxHeight={cMaxHeightCodeBlock}
          streaming={pCtx.isCodeStreaming}
        />
      {/if}
      <div
        class={cShowInput
          ? 'mt-4 mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70'
          : 'mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70'}
      >
        <span>{fTranslate('text_b2439bcb8dee')}</span>
        {#if pCtx.isPending}
          <Loader2 class="h-3 w-3 animate-spin" />
        {/if}
      </div>
      {#if pCtx.isPending}
        <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
          {fTranslate('text_3baf0a61c565')}
        </div>
      {:else if vSection.toolResult}
        {#if cOutputKind === ToolResultKind.JSON}
          <SyntaxHighlightedCode
            code={fFormatJsonPretty(vSection.toolResult)}
            language={FileTypeText.JSON}
            maxHeight={cMaxHeightCodeBlock}
          />
        {:else if cOutputKind === ToolResultKind.MARKDOWN}
          <MarkdownContent content={vSection.toolResult} attachments={vAttachments} />
        {:else}
          <div class="overflow-auto">
            {#each cParsedLines as vLine, vI (vI)}
              <div class="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
                {vLine.text}
              </div>
              {#if vLine.image}
                <img
                  src={vLine.image.base64Url}
                  alt={vLine.image.name}
                  class="mt-2 mb-2 h-auto max-w-full rounded-lg"
                  loading="lazy"
                />
              {/if}
            {/each}
          </div>
        {/if}
      {:else}
        <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
          {fTranslate('text_f7e31759b202')}
        </div>
      {/if}
    {/if}
  {/snippet}
</ToolCallBlock>
