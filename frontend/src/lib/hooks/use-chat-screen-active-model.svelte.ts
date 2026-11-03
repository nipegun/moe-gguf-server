/**
 * Active model resolution and capability detection for the ChatScreen.
 *
 * Picks the model that should be used for the current view
 * (router: user-selected or conversation fallback; non-router: first
 * available option), and reactively tracks which modalities (vision /
 * audio / video) it supports — fetching model props from the server on
 * demand if they aren't cached yet.
 */

import { cModelsStore, fModelOptions, fSelectedModelId } from '$lib/stores/models.svelte';
import { fIsRouterMode } from '$lib/stores/server.svelte';
import { cChatStore } from '$lib/stores/chat.svelte';
import { fActiveMessages } from '$lib/stores/conversations.svelte';

export function fUseChatScreenActiveModel() {
  const cIsRouter = $derived(fIsRouterMode());
  const cConversationModel = $derived(
    cChatStore.mGetConversationModel(fActiveMessages() as DatabaseMessage[])
  );

  const cActiveModelId = $derived.by(() => {
    const cOptions = fModelOptions();

    if (!cIsRouter) {
      return cOptions.length > 0 ? cOptions[0].model : null;
    }

    const cSelectedId = fSelectedModelId();
    if (cSelectedId) {
      const cModel = cOptions.find((pM) => pM.id === cSelectedId);
      if (cModel) return cModel.model;
    }

    if (cConversationModel) {
      const cModel = cOptions.find((pM) => pM.model === cConversationModel);
      if (cModel) return cModel.model;
    }

    return null;
  });

  let vModelPropsVersion = $state(0);

  $effect(() => {
    if (cActiveModelId) {
      const cCached = cModelsStore.mGetModelProps(cActiveModelId);
      if (!cCached) {
        cModelsStore.mFetchModelProps(cActiveModelId).then(() => {
          vModelPropsVersion++;
        });
      }
    }
  });

  const cHasAudioModality = $derived.by(() => {
    if (cActiveModelId) {
      void vModelPropsVersion;
      return cModelsStore.mModelSupportsAudio(cActiveModelId);
    }
    return false;
  });

  const cHasVideoModality = $derived.by(() => {
    if (cActiveModelId) {
      void vModelPropsVersion;
      return cModelsStore.mModelSupportsVideo(cActiveModelId);
    }
    return false;
  });

  const cHasVisionModality = $derived.by(() => {
    if (cActiveModelId) {
      void vModelPropsVersion;
      return cModelsStore.mModelSupportsVision(cActiveModelId);
    }
    return false;
  });

  return {
    get isRouter() {
      return cIsRouter;
    },
    get conversationModel() {
      return cConversationModel;
    },
    get activeModelId() {
      return cActiveModelId;
    },
    get hasAudioModality() {
      return cHasAudioModality;
    },
    get hasVideoModality() {
      return cHasVideoModality;
    },
    get hasVisionModality() {
      return cHasVisionModality;
    }
  };
}
