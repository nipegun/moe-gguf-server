<script lang="ts">
  import { ModelBadge, ModelsSelectorDropdown } from '$lib/components/app';
  import { copyToClipboard } from '$lib/utils';
  import { cModelsStore } from '$lib/stores/models.svelte';
  import { ServerModelStatus } from '$lib/enums';

  interface Props {
    displayedModel: string | null;
    isRouter: boolean;
    isLoading: boolean;
    onRegenerate: (pModelOverride?: string) => void;
  }

  let {
    displayedModel: vDisplayedModel,
    isRouter: vIsRouter,
    isLoading: vIsLoading,
    onRegenerate: vOnRegenerate
  }: Props = $props();

  let vPendingModel = $state<string | null>(null);

  function fHandleCopyModel() {
    void copyToClipboard(vDisplayedModel ?? '');
  }
</script>

{#if vIsRouter}
  <ModelsSelectorDropdown
    currentModel={vPendingModel ?? vDisplayedModel}
    disabled={vIsLoading}
    onModelChange={async (modelId: string, modelName: string) => {
      const status = cModelsStore.mGetModelStatus(modelId);

      if (status !== ServerModelStatus.LOADED) {
        vPendingModel = modelId;

        try {
          await cModelsStore.mLoadModel(modelId);
        } finally {
          vPendingModel = null;
        }
      }

      vOnRegenerate(modelName);
      return true;
    }}
  />
{:else}
  <ModelBadge model={vDisplayedModel || undefined} onclick={fHandleCopyModel} />
{/if}
