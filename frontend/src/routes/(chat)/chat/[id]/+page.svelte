<script lang="ts">
  import { goto, replaceState } from '$app/navigation';
  import { page } from '$app/state';
  import { afterNavigate } from '$app/navigation';
  import { DialogModelNotAvailable } from '$lib/components/app';
  import { cAppName, cRoutes } from '$lib/constants';
  import { cChatStore } from '$lib/stores/chat.svelte';
  import { cConversationsStore, fActiveConversation } from '$lib/stores/conversations.svelte';
  import { cModelsStore, fModelOptions } from '$lib/stores/models.svelte';

  let vChatId = $derived(page.params.id);
  let vCurrentChatId: string | undefined = undefined;

  // URL parameters for prompt and model selection
  let vQParam = $derived(page.url.searchParams.get('q'));
  let vModelParam = $derived(page.url.searchParams.get('model'));

  // Dialog state for model not available error
  let vShowModelNotAvailable = $state(false);
  let vRequestedModelName = $state('');
  let vAvailableModelNames = $derived(fModelOptions().map((pM) => pM.model));

  // Track if URL params have been processed for this chat
  let vUrlParamsProcessed = $state(false);

  /**
   * Clear URL params after message is sent to prevent re-sending on refresh
   */
  function fClearUrlParams() {
    const cUrl = new URL(page.url);
    cUrl.searchParams.delete('q');
    cUrl.searchParams.delete('model');
    replaceState(cUrl.toString(), {});
  }

  async function fHandleUrlParams() {
    // Ensure models are loaded first
    await cModelsStore.mFetch();

    // Handle model parameter - select model if provided
    if (vModelParam) {
      const cModel = cModelsStore.mFindModelByName(vModelParam);
      if (cModel) {
        try {
          await cModelsStore.mSelectModelById(cModel.id);
        } catch (vError) {
          console.error('Failed to select model:', vError);
          vRequestedModelName = vModelParam;
          vShowModelNotAvailable = true;
          return;
        }
      } else {
        // Model not found - show error dialog
        vRequestedModelName = vModelParam;
        vShowModelNotAvailable = true;
        return;
      }
    }

    // Handle ?q= parameter - send message in current conversation
    if (vQParam !== null) {
      await cChatStore.mSendMessage(vQParam);
      // Clear URL params after message is sent
      fClearUrlParams();
    } else if (vModelParam) {
      // Clear params even if no message was sent (just model selection)
      fClearUrlParams();
    }

    vUrlParamsProcessed = true;
  }

  afterNavigate(() => {
    setTimeout(() => {
      void cModelsStore.mSelectModelFromLastAssistantResponse();
    }, 100);
  });

  $effect(() => {
    if (vChatId && vChatId !== vCurrentChatId) {
      vCurrentChatId = vChatId;
      vUrlParamsProcessed = false; // Reset for new chat

      // Skip loading if this conversation is already active (e.g., just created)
      if (fActiveConversation()?.id === vChatId) {
        void cChatStore.mDiscoverActiveStream(vChatId);
        if ((vQParam !== null || vModelParam !== null) && !vUrlParamsProcessed) {
          fHandleUrlParams();
        }
        return;
      }

      (async () => {
        const cSuccess = await cConversationsStore.mLoadConversation(vChatId);
        if (!cSuccess) {
          await goto(cRoutes.START);
          return;
        }
        cChatStore.mSyncLoadingStateForChat(vChatId);
        // server probe (with localStorage fallback) and attach
        await cChatStore.mDiscoverActiveStream(vChatId);

        if ((vQParam !== null || vModelParam !== null) && !vUrlParamsProcessed) {
          await fHandleUrlParams();
        }
      })();
    }
  });

  $effect(() => {
    if (typeof window === 'undefined' || typeof document === 'undefined') return;

    // when the tab comes back to the foreground, re-run discovery to catch any race
    // where the initial mount probe missed an active session
    const fOnVisibility = () => {
      if (document.visibilityState !== 'visible') return;
      if (!vChatId) return;
      void cChatStore.mDiscoverActiveStream(vChatId);
    };
    document.addEventListener('visibilitychange', fOnVisibility);
    return () => document.removeEventListener('visibilitychange', fOnVisibility);
  });
</script>

<svelte:head>
  <title>{fActiveConversation()?.name || 'Chat'} - {cAppName}</title>
</svelte:head>

<DialogModelNotAvailable
  bind:open={vShowModelNotAvailable}
  modelName={vRequestedModelName}
  availableModels={vAvailableModelNames}
/>
