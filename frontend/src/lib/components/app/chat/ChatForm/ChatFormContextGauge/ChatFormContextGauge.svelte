<script lang="ts">
  import { fTranslate } from '../../../../../i18n';

  import { untrack } from 'svelte';
  import { fActiveConversation, fActiveMessages } from '$lib/stores/conversations.svelte';
  import { cChatStore, fIsChatStreaming, fIsLoading } from '$lib/stores/chat.svelte';
  import { fUseContextGauge } from '$lib/hooks/use-context-gauge.svelte';
  import ContextGaugeDial from './ContextGaugeDial.svelte';
  import {
    fGaugeTriggerClick,
    fGaugeTriggerEnter,
    fGaugeTriggerKeydown,
    fGaugeTriggerLeave,
    fGaugeTriggerPointerDown
  } from '$lib/stores/context-gauge-popup.svelte';

  const cGauge = fUseContextGauge();

  $effect(() => {
    const cConv = fActiveConversation();
    untrack(() => cChatStore.mSetActiveProcessingConversation(cConv?.id ?? null));
  });

  $effect(() => {
    const cConv = fActiveConversation();
    const cMessages = fActiveMessages() as DatabaseMessage[];
    if (!cConv) return;
    if (fIsLoading() || fIsChatStreaming()) return;

    if (cMessages.length === 0) {
      untrack(() => cChatStore.mClearProcessingState(cConv.id));
      return;
    }

    untrack(() => cChatStore.mRestoreProcessingStateFromMessages(cMessages, cConv.id));
  });

  $effect(() => {
    cGauge.startMonitoring();
  });
</script>

<div
  role="button"
  tabindex="0"
  aria-label={fTranslate('text_36416faf4b8a')}
  data-context-gauge-trigger
  class="flex h-5 w-5 cursor-default items-center justify-center"
  onclick={fGaugeTriggerClick}
  onkeydown={fGaugeTriggerKeydown}
  onpointerdown={fGaugeTriggerPointerDown}
  onpointerenter={fGaugeTriggerEnter}
  onpointerleave={fGaugeTriggerLeave}
>
  <ContextGaugeDial percent={cGauge.contextPercent} level={cGauge.colorLevel} />
</div>
