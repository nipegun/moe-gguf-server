<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { Clock, Loader2 } from '@lucide/svelte';
  import { AgenticSectionType } from '$lib/enums';
  import type { AgenticSection } from '$lib/utils';

  interface Props {
    section: AgenticSection;
    isStreaming?: boolean;
  }

  let { section: vSection, isStreaming: vIsStreaming = false }: Props = $props();

  const cIsPending = $derived(vSection.type === AgenticSectionType.TOOL_CALL_PENDING);
  const cIsStreamingCall = $derived(vSection.type === AgenticSectionType.TOOL_CALL_STREAMING);
  const cShowSpinner = $derived(cIsPending || (cIsStreamingCall && vIsStreaming));

  type GetDatetimeMeta = {
    dateString?: string;
    errorMessage?: string;
  };

  function fParseGetDatetimeMeta(pToolResultString: string | undefined): GetDatetimeMeta {
    if (!pToolResultString) return {};

    try {
      const cParsed: unknown = JSON.parse(pToolResultString);
      if (cParsed && typeof cParsed === 'object' && !Array.isArray(cParsed)) {
        const cObj = cParsed as Record<string, unknown>;
        if (typeof cObj.error === 'string') return { errorMessage: cObj.error };
        if (typeof cObj.result === 'string') return { dateString: cObj.result.trim() };
      }
    } catch {
      return { dateString: pToolResultString.trim() };
    }

    return {};
  }

  const cDateMeta = $derived(fParseGetDatetimeMeta(vSection.toolResult));
</script>

<div class="text-muted-foreground flex items-center gap-2 py-1.5">
  <Clock class="text-muted-foreground/60 h-3.5 w-3.5 shrink-0" />
  {#if cShowSpinner}
    <span class="text-foreground/80 text-sm font-medium">{fTranslate('text_06dafda7802b')}</span>
    <Loader2 class="text-muted-foreground/70 h-3 w-3 animate-spin" />
  {:else if cDateMeta.errorMessage}
    <span class="text-foreground/80 text-sm font-medium">{fTranslate('text_06dafda7802b')} </span>
    <span class="text-red-600 text-xs italic dark:text-red-400"
      >-&nbsp;{cDateMeta.errorMessage}</span
    >
  {:else if cDateMeta.dateString}
    <span class="text-foreground/80 text-sm font-medium">{fTranslate('text_61655615b3bd')} </span>
    <span class="font-mono text-foreground/90 text-sm">{cDateMeta.dateString}</span>
  {:else}
    <span class="text-foreground/80 text-sm font-medium">{fTranslate('text_06dafda7802b')}</span>
  {/if}
</div>
