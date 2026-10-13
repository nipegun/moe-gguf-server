<script lang="ts">
  import { DialogModelNotAvailable } from '$lib/components/app';
  import { cChatStore } from '$lib/stores/chat.svelte';
  import {
    cConversationsStore,
    fIsConversationsInitialized
  } from '$lib/stores/conversations.svelte';
  import { cModelsStore, fModelOptions } from '$lib/stores/models.svelte';
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { replaceState } from '$app/navigation';
  import { cAppName, cNewChatParam } from '$lib/constants';

  let vQParam = $derived(page.url.searchParams.get('q'));
  let vModelParam = $derived(page.url.searchParams.get('model'));
  let vNewChatParam = $derived(page.url.searchParams.get(cNewChatParam));

  // Dialog state for model not available error
  let vShowModelNotAvailable = $state(false);
  let vRequestedModelName = $state('');
  let vAvailableModelNames = $derived(fModelOptions().map((pM) => pM.model));

  /**
   * Clear URL params after message is sent to prevent re-sending on refresh
   */
  function fClearUrlParams() {
    const cUrl = new URL(page.url);

    cUrl.searchParams.delete('q');
    cUrl.searchParams.delete('model');
    cUrl.searchParams.delete(cNewChatParam);

    replaceState(cUrl.toString(), {});
  }

  async function fHandleUrlParams() {
    await cModelsStore.mFetch();

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
        vRequestedModelName = vModelParam;
        vShowModelNotAvailable = true;

        return;
      }
    }

    // Handle ?q= parameter - create new conversation and send message
    if (vQParam !== null) {
      await cConversationsStore.mCreateConversation();
      fClearUrlParams();
    } else if (vModelParam || vNewChatParam === 'true') {
      fClearUrlParams();
    }
  }

  onMount(async () => {
    if (!fIsConversationsInitialized()) {
      await cConversationsStore.mInit();
    }

    cConversationsStore.mClearActiveConversation();
    cChatStore.mClearUIState();

    await cModelsStore.mFetch();

    if (vQParam !== null || vModelParam !== null || vNewChatParam === 'true') {
      await fHandleUrlParams();
    }

    await cModelsStore.mEnsureFirstModelSelected();
  });
</script>

<svelte:head>
  <title>{cAppName}</title>
</svelte:head>

<DialogModelNotAvailable
  bind:open={vShowModelNotAvailable}
  modelName={vRequestedModelName}
  availableModels={vAvailableModelNames}
/>
