<script lang="ts">
  import { ModelBadge, ModelsSelectorDropdown } from '$lib/components/app';
  import { fCopyToClipboard } from '$lib/utils';
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
    void fCopyToClipboard(vDisplayedModel ?? '');
  }
</script>

{#if vIsRouter}
  <ModelsSelectorDropdown
    currentModel={vPendingModel ?? vDisplayedModel}
    disabled={vIsLoading}
    onModelChange={async (pModelId: string, pModelName: string) => {
      const cStatus = cModelsStore.mGetModelStatus(pModelId);

      if (cStatus !== ServerModelStatus.LOADED) {
        vPendingModel = pModelId;

        try {
          await cModelsStore.mLoadModel(pModelId);
        } finally {
          vPendingModel = null;
        }
      }

      vOnRegenerate(pModelName);
      return true;
    }}
  />
{:else}
  <ModelBadge model={vDisplayedModel || undefined} onclick={fHandleCopyModel} />
{/if}
