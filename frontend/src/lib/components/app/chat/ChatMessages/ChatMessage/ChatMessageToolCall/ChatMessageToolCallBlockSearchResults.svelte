<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { cIconClassDefault, cIconClassSpin } from '$lib/constants/css-classes';
  import { Globe, Loader2 } from '@lucide/svelte';
  import { CollapsibleContentBlock } from '$lib/components/app';
  import * as HoverCard from '$lib/components/ui/hover-card';
  import { AgenticSectionType } from '$lib/enums';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import {
    extractSearchResults,
    extractSearchQuery,
    faviconForUrl,
    sanitizeExternalUrl,
    type SearchResult,
    type AgenticSection
  } from '$lib/utils';

  interface Props {
    section: AgenticSection;
    open?: boolean;
    isStreaming?: boolean;
    onToggle?: () => void;
  }

  let {
    section: vSection,
    open: vOpen = $bindable(false),
    isStreaming: vIsStreaming = false,
    onToggle: vOnToggle
  }: Props = $props();

  const cIsPending = $derived(vSection.type === AgenticSectionType.TOOL_CALL_PENDING);
  const cIsStreamingCall = $derived(vSection.type === AgenticSectionType.TOOL_CALL_STREAMING);
  const cShowSpinner = $derived(cIsPending || (cIsStreamingCall && vIsStreaming));

  const cResults = $derived(extractSearchResults(vSection.toolResult));
  const cQuery = $derived(extractSearchQuery(vSection.toolArgs));

  // Same icon-resolution chain as ChatMessageToolCallBlockDefault so
  // MCP-server branding is consistent across both views. Spinner wins
  // while the call is in flight so the user sees execution status.
  const cIconUrl = $derived(
    cShowSpinner ? null : cMcpStore.mGetServerFaviconForTool(vSection.toolName)
  );
  const cIcon = $derived(cShowSpinner ? Loader2 : undefined);
  const cIconClass = $derived(cShowSpinner ? cIconClassSpin : cIconClassDefault);

  // Verb reflects state: "Searching" while the call is in flight, "Searched"
  // once results (or a definitive empty response) have arrived. Lets the
  // heading read as a live progress indicator rather than a completed
  // retrospective.
  const cTitle = $derived.by(() => {
    const cVerb = cShowSpinner ? fTranslate('text_03bd6fca9cf5') : fTranslate('text_9fc7f1166869');
    return cQuery ? `${cVerb} web for "${cQuery}"` : `${cVerb} web`;
  });

  function fHideBrokenIcon(pEvent: Event) {
    (pEvent.currentTarget as HTMLImageElement).style.display = 'none';
  }

  function fFormatPublishDate(pIso: string | undefined): string | null {
    if (!pIso) return null;
    try {
      const cDate = new Date(pIso);
      if (Number.isNaN(cDate.getTime())) return pIso;
      return cDate.toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });
    } catch {
      return pIso;
    }
  }

  function fHostFor(pUrl: string): string | null {
    try {
      return new URL(pUrl).host;
    } catch {
      return null;
    }
  }

  function fHasDetails(pResult: SearchResult): boolean {
    return Boolean(pResult.highlights || pResult.published || pResult.author);
  }
</script>

{#snippet pill(result: SearchResult)}
  {@const faviconUrl = faviconForUrl(result.url)}
  {@const safeUrl = sanitizeExternalUrl(result.url)}
  {@const showHoverCard = safeUrl !== null && fHasDetails(result)}
  {#if safeUrl}
    <HoverCard.Root openDelay={150} closeDelay={100}>
      <HoverCard.Trigger
        href={safeUrl}
        target="_blank"
        rel="noopener noreferrer"
        class="hover:bg-muted/80 focus-visible:ring-ring inline-flex max-w-full items-center gap-1.5 rounded-full border bg-muted px-2.5 py-1 text-xs transition-colors outline-none focus-visible:ring-2"
      >
        {#if faviconUrl}
          <img
            src={faviconUrl}
            alt=""
            class="h-3 w-3 shrink-0 rounded-sm"
            onerror={fHideBrokenIcon}
          />
        {:else}
          <Globe class="text-muted-foreground/70 h-3 w-3 shrink-0" />
        {/if}
        <span class="truncate font-medium text-foreground/80">{result.title}</span>
      </HoverCard.Trigger>
      {#if showHoverCard}
        {@const publishDate = fFormatPublishDate(result.published)}
        {@const host = fHostFor(safeUrl)}
        <HoverCard.Content
          side="top"
          align="start"
          sideOffset={6}
          class="bg-popover text-popover-foreground z-50 w-80 max-w-[90vw] rounded-lg border p-0 shadow-lg"
        >
          <div class="flex flex-col gap-2 p-3">
            <a
              href={safeUrl}
              target="_blank"
              rel="noopener noreferrer"
              class="line-clamp-3 text-sm font-medium leading-snug hover:underline"
              >{result.title}</a
            >
            {#if publishDate || result.author}
              <div class="text-muted-foreground flex items-center gap-1.5 text-[11px]">
                {#if publishDate}
                  <span>{publishDate}</span>
                {/if}
                {#if publishDate && result.author}
                  <span class="opacity-50">&middot;</span>
                {/if}
                {#if result.author}
                  <span class="truncate">{result.author}</span>
                {/if}
              </div>
            {/if}
            {#if result.highlights}
              <p
                class="text-popover-foreground/85 line-clamp-5 text-xs leading-relaxed whitespace-pre-line"
              >
                {result.highlights}
              </p>
            {/if}
            {#if host}
              <div class="text-muted-foreground/80 truncate text-[11px]">{host}</div>
            {/if}
          </div>
        </HoverCard.Content>
      {/if}
    </HoverCard.Root>
  {/if}
{/snippet}

<CollapsibleContentBlock
  open={vOpen}
  class="my-2"
  icon={cIcon}
  iconClass={cIconClass}
  iconUrl={cIconUrl}
  title={cTitle}
  onToggle={vOnToggle}
>
  {#if cResults.length > 0}
    <div class="flex flex-wrap items-center gap-2 pb-1">
      {#each cResults as result (result.url)}
        {@render pill(result)}
      {/each}
    </div>
  {:else if cShowSpinner}
    <div class="text-muted-foreground/70 flex items-center gap-2 py-1 text-xs italic">
      <Loader2 class="h-3 w-3 animate-spin" />
      <span>{fTranslate('text_78c9d9f6ace0')}</span>
    </div>
  {:else}
    <div class="text-muted-foreground/70 py-1 text-xs italic">
      {fTranslate('text_a43619f32117')}
    </div>
  {/if}
</CollapsibleContentBlock>
