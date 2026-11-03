<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { ChevronLeft, ChevronRight } from '@lucide/svelte';
  import { ActionIcon } from '$lib/components/app';

  interface Props {
    class?: string;
    siblingInfo: ChatMessageSiblingInfo | null;
    onNavigateToSibling?: (pSiblingId: string) => void;
  }

  let {
    class: vClassName = '',
    siblingInfo: vSiblingInfo,
    onNavigateToSibling: vOnNavigateToSibling
  }: Props = $props();

  let vHasPrevious = $derived(vSiblingInfo && vSiblingInfo.currentIndex > 0);
  let vHasNext = $derived(
    vSiblingInfo && vSiblingInfo.currentIndex < vSiblingInfo.totalSiblings - 1
  );
  let vNextSiblingId = $derived(
    vHasNext ? vSiblingInfo!.siblingIds[vSiblingInfo!.currentIndex + 1] : null
  );
  let vPreviousSiblingId = $derived(
    vHasPrevious ? vSiblingInfo!.siblingIds[vSiblingInfo!.currentIndex - 1] : null
  );
</script>

{#if vSiblingInfo && vSiblingInfo.totalSiblings > 1}
  <div
    aria-label={fTranslate('text_5ed1f74a0caa', {
      p0: vSiblingInfo.currentIndex + 1,
      p1: vSiblingInfo.totalSiblings
    })}
    class="flex items-center gap-1 text-xs text-muted-foreground {vClassName}"
    role="navigation"
  >
    <ActionIcon
      icon={ChevronLeft}
      tooltip={fTranslate('text_21d886b74bc1')}
      disabled={!vHasPrevious}
      class="h-5 w-5 p-0 {!vHasPrevious ? '!cursor-not-allowed opacity-30' : ''}"
      onclick={() => vOnNavigateToSibling?.(vPreviousSiblingId!)}
    />

    <span class="px-1 font-mono text-xs">
      {vSiblingInfo.currentIndex + 1}/{vSiblingInfo.totalSiblings}
    </span>

    <ActionIcon
      icon={ChevronRight}
      tooltip={fTranslate('text_0671058c0f46')}
      disabled={!vHasNext}
      class="h-5 w-5 p-0 {!vHasNext ? 'opacity-30' : ''}"
      onclick={() => vOnNavigateToSibling?.(vNextSiblingId!)}
    />
  </div>
{/if}
