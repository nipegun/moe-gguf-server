<script lang="ts">
  import { fTranslate } from '../../../../../i18n';

  import { cChatStore } from '$lib/stores/chat.svelte';
  import {
    cModelsStore,
    fModelOptions,
    fSelectedModelId,
    fSelectedModelName
  } from '$lib/stores/models.svelte';
  import { fIsRouterMode, fServerError } from '$lib/stores/server.svelte';
  import { ModelsSelectorDropdown, ModelsSelectorSheet } from '$lib/components/app';
  import { cIsMobile } from '$lib/stores/viewport.svelte';
  import { fActiveMessages } from '$lib/stores/conversations.svelte';

  interface Props {
    disabled?: boolean;
    forceForegroundText?: boolean;
    hasAudioModality?: boolean;
    hasVideoModality?: boolean;
    hasVisionModality?: boolean;
    hasModelSelected?: boolean;
    isSelectedModelInCache?: boolean;
    submitTooltip?: string;
    useGlobalSelection?: boolean;
  }

  let {
    disabled: vDisabled = false,
    forceForegroundText: vForceForegroundText = false,
    hasAudioModality: vHasAudioModality = $bindable(false),
    hasVideoModality: vHasVideoModality = $bindable(false),
    hasVisionModality: vHasVisionModality = $bindable(false),
    hasModelSelected: vHasModelSelected = $bindable(false),
    isSelectedModelInCache: vIsSelectedModelInCache = $bindable(true),
    submitTooltip: vSubmitTooltip = $bindable(''),
    useGlobalSelection: vUseGlobalSelection = false
  }: Props = $props();

  let vIsRouter = $derived(fIsRouterMode());
  let vIsOffline = $derived(!!fServerError());

  let vConversationModel = $derived(
    cChatStore.mGetConversationModel(fActiveMessages() as DatabaseMessage[])
  );

  let vLastSyncedConversationModel: string | null = null;

  let vSelectorModel = $derived.by(() => {
    const cStoreModel = fSelectedModelName();
    if (cStoreModel && cStoreModel !== vConversationModel) {
      return cStoreModel;
    }

    if (vConversationModel) {
      return vConversationModel;
    }

    return null;
  });

  $effect(() => {
    if (vConversationModel && vConversationModel !== vLastSyncedConversationModel) {
      if (fModelOptions().some((pM) => pM.model === vConversationModel)) {
        cModelsStore.vSelectedModelName = vConversationModel;
        cModelsStore.mSelectModelByName(vConversationModel);
      } else {
        cModelsStore.vSelectedModelName = null;
        cModelsStore.mClearSelection();
      }
      vLastSyncedConversationModel = vConversationModel;
    } else if (
      vIsRouter &&
      !cModelsStore.vSelectedModelId &&
      cModelsStore.mLoadedModelIds.length > 0 &&
      fActiveMessages().length > 0 &&
      !vConversationModel
    ) {
      vLastSyncedConversationModel = null;
      const cFirst = fModelOptions().find((pM) => cModelsStore.mLoadedModelIds.includes(pM.model));
      if (cFirst) cModelsStore.mSelectModelById(cFirst.id);
    }
  });

  let vActiveModelId = $derived.by(() => {
    const cOptions = fModelOptions();

    if (!vIsRouter) {
      return cOptions.length > 0 ? cOptions[0].model : null;
    }

    const cSelectedId = fSelectedModelId();

    if (cSelectedId) {
      const cModel = cOptions.find((pM) => pM.id === cSelectedId);

      if (cModel) return cModel.model;
    }

    if (vConversationModel) {
      const cModel = cOptions.find((pM) => pM.model === vConversationModel);

      if (cModel) return cModel.model;
    }

    return null;
  });

  let vModelPropsVersion = $state(0); // Used to trigger reactivity after fetch

  $effect(() => {
    if (vActiveModelId) {
      const cCached = cModelsStore.mGetModelProps(vActiveModelId);

      if (!cCached) {
        cModelsStore.mFetchModelProps(vActiveModelId).then(() => {
          vModelPropsVersion++;
        });
      }
    }
  });

  $effect(() => {
    void vModelPropsVersion;

    vHasAudioModality = vActiveModelId ? cModelsStore.mModelSupportsAudio(vActiveModelId) : false;
  });

  $effect(() => {
    void vModelPropsVersion;

    vHasVideoModality = vActiveModelId ? cModelsStore.mModelSupportsVideo(vActiveModelId) : false;
  });

  $effect(() => {
    void vModelPropsVersion;

    vHasVisionModality = vActiveModelId ? cModelsStore.mModelSupportsVision(vActiveModelId) : false;
  });

  $effect(() => {
    vHasModelSelected = !vIsRouter || !!vConversationModel || !!fSelectedModelId();
  });

  $effect(() => {
    if (!vIsRouter) {
      vIsSelectedModelInCache = true;
    } else if (vConversationModel) {
      vIsSelectedModelInCache = fModelOptions().some(
        (pOption) => pOption.model === vConversationModel
      );
    } else {
      const cCurrentModelId = fSelectedModelId();

      if (!cCurrentModelId) {
        vIsSelectedModelInCache = false;
      } else {
        vIsSelectedModelInCache = fModelOptions().some((pOption) => pOption.id === cCurrentModelId);
      }
    }
  });

  $effect(() => {
    if (!vHasModelSelected) {
      vSubmitTooltip = fTranslate('text_ee1a6ecc0ae0');
    } else if (!vIsSelectedModelInCache) {
      vSubmitTooltip = fTranslate('text_c8fd1ce5bc14');
    } else {
      vSubmitTooltip = '';
    }
  });

  let vSelectorModelRef: ModelsSelectorDropdown | ModelsSelectorSheet | undefined =
    $state(undefined);

  export function fOpen() {
    vSelectorModelRef?.fOpen();
  }
</script>

{#if cIsMobile.current}
  <ModelsSelectorSheet
    disabled={vDisabled || vIsOffline}
    bind:this={vSelectorModelRef}
    currentModel={vSelectorModel}
    forceForegroundText={vForceForegroundText}
    useGlobalSelection={vUseGlobalSelection}
  />
{:else}
  <ModelsSelectorDropdown
    disabled={vDisabled || vIsOffline}
    bind:this={vSelectorModelRef}
    currentModel={vSelectorModel}
    forceForegroundText={vForceForegroundText}
    useGlobalSelection={vUseGlobalSelection}
  />
{/if}
