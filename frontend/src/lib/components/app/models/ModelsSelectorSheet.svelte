<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { ChevronDown, Loader2, Package } from '@lucide/svelte';
  import * as Sheet from '$lib/components/ui/sheet';
  import { fUseModelsSelector } from '$lib/hooks/use-models-selector.svelte';
  import {
    DialogModelInformation,
    ModelId,
    ModelsSelectorList,
    SearchInput
  } from '$lib/components/app';
  import ModelLoadHighlight from './ModelLoadHighlight.svelte';
  import { ServerModelStatus } from '$lib/enums';
  import { cModelsStore, fRouterModels } from '$lib/stores/models.svelte';
  import { modelLoadFraction } from '$lib/utils';

  interface Props {
    class?: string;
    currentModel?: string | null;
    /** Callback when model changes. Return false to keep menu open (e.g., for validation failures) */
    onModelChange?: (pModelId: string, pModelName: string) => Promise<boolean> | boolean | void;
    disabled?: boolean;
    forceForegroundText?: boolean;
    /** When true, user's global selection takes priority over currentModel (for form selector) */
    useGlobalSelection?: boolean;
  }

  let {
    class: vClassName = '',
    currentModel: vCurrentModel = null,
    onModelChange: vOnModelChange,
    disabled: vDisabled = false,
    forceForegroundText: vForceForegroundText = false,
    useGlobalSelection: vUseGlobalSelection = false
  }: Props = $props();

  let vSheetOpen = $state(false);

  const cMs = fUseModelsSelector({
    currentModel: () => vCurrentModel,
    useGlobalSelection: () => vUseGlobalSelection,
    onModelChange: () => vOnModelChange,
    onOpenChange: (pOpen) => {
      vSheetOpen = pOpen;
    }
  });

  export function fOpen() {
    cMs.handleOpenChange(true);
  }

  function fHandleSheetOpenChange(pOpen: boolean) {
    if (!pOpen) {
      cMs.handleOpenChange(false);
    }
  }
</script>

<div class={['relative inline-flex flex-col items-end gap-1', vClassName]}>
  {#if cMs.loading && cMs.options.length === 0 && cMs.isRouter}
    <div class="flex items-center gap-2 text-xs text-muted-foreground">
      <Loader2 class="h-3.5 w-3.5 animate-spin" />
      {fTranslate('text_cc8b4624f765')}
    </div>
  {:else if cMs.options.length === 0 && cMs.isRouter}
    <p class="text-xs text-muted-foreground">{fTranslate('text_58064bf43a25')}</p>
  {:else}
    {@const selectedOption = cMs.getDisplayOption()}
    {@const triggerModel = selectedOption?.model}
    {@const triggerStatus = triggerModel
      ? fRouterModels().find((m) => m.id === triggerModel)?.status?.value
      : undefined}
    {@const triggerLoading =
      !!triggerModel &&
      (triggerStatus === ServerModelStatus.LOADING ||
        cModelsStore.mIsModelOperationInProgress(triggerModel))}
    {@const triggerLoadPercent = triggerLoading
      ? Math.round(modelLoadFraction(cModelsStore.mGetLoadProgress(triggerModel)) * 100)
      : 0}

    {#if cMs.isRouter}
      <button
        type="button"
        class={[
          `relative inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 max-sm:px-3 max-sm:py-2 max-sm:text-sm dark:bg-muted-foreground/15 dark:text-secondary-foreground`,
          !cMs.isCurrentModelInCache
            ? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
            : vForceForegroundText
              ? 'text-foreground'
              : cMs.isHighlightedCurrentModelActive
                ? 'text-foreground'
                : 'text-foreground',
          vSheetOpen && 'text-foreground'
        ]}
        style="max-width: min(calc(100cqw - 9rem), 20rem)"
        disabled={vDisabled || cMs.updating}
        onclick={() => cMs.handleOpenChange(true)}
      >
        <Package class="h-3.5 w-3.5 shrink-0" />

        {#if !selectedOption}
          <span class="min-w-0 font-medium">{fTranslate('text_1d446005add3')}</span>
        {:else}
          <ModelId
            class="text-xs"
            modelId={selectedOption?.model || ''}
            hideQuantization
            hideTags
            hideOrgName
          />
        {/if}

        {#if cMs.updating || cMs.isLoadingModel}
          <Loader2 class="h-3 w-3.5 shrink-0 animate-spin" />
        {:else}
          <ChevronDown class="h-3 w-3.5 shrink-0" />
        {/if}

        {#if triggerLoading}
          <ModelLoadHighlight percent={triggerLoadPercent} />
        {/if}
      </button>

      <Sheet.Root bind:open={vSheetOpen} onOpenChange={fHandleSheetOpenChange}>
        <Sheet.Content side="bottom" class="max-h-[85vh] gap-1">
          <Sheet.Header>
            <Sheet.Title>{fTranslate('text_e4813fd58ccf')}</Sheet.Title>

            <Sheet.Description class="sr-only">{fTranslate('text_0a527b8493e8')}</Sheet.Description>
          </Sheet.Header>

          <div class="flex flex-col gap-1 pb-4">
            <div class="mb-3 px-4">
              <SearchInput
                placeholder={fTranslate('text_37b90680b842')}
                value={cMs.searchTerm}
                onInput={(v) => cMs.setSearchTerm(v)}
              />
            </div>

            <div class="max-h-[60vh] overflow-y-auto px-2">
              {#if !cMs.isCurrentModelInCache && vCurrentModel}
                <button
                  type="button"
                  class="flex w-full cursor-not-allowed items-center rounded-md bg-red-400/10 px-3 py-2.5 text-left text-sm text-red-400"
                  disabled
                >
                  <span class="min-w-0 flex-1 truncate">
                    {selectedOption?.name || vCurrentModel}
                  </span>
                  <span class="ml-2 text-xs whitespace-nowrap opacity-70"
                    >{fTranslate('text_1c96e2456748')}</span
                  >
                </button>
                <div class="my-1 h-px bg-border"></div>
              {/if}

              {#if cMs.filteredOptions.length === 0}
                <p class="px-3 py-3 text-center text-sm text-muted-foreground">
                  {fTranslate('text_339e5fcda064')}
                </p>
              {/if}

              <ModelsSelectorList
                groups={cMs.groupedFilteredOptions}
                currentModel={vCurrentModel}
                activeId={cMs.activeId}
                sectionHeaderClass="px-2 py-2 text-xs font-semibold text-muted-foreground/60 select-none"
                orgHeaderClass="px-2 py-2 text-xs font-semibold text-muted-foreground/60 select-none [&:not(:first-child)]:mt-2"
                onSelect={cMs.handleSelect}
                onInfoClick={cMs.handleInfoClick}
              />
            </div>
          </div>
        </Sheet.Content>
      </Sheet.Root>
    {:else}
      <button
        class={[
          `inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground`,
          !cMs.isCurrentModelInCache
            ? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
            : vForceForegroundText
              ? 'text-foreground'
              : cMs.isHighlightedCurrentModelActive
                ? 'text-foreground'
                : 'text-foreground'
        ]}
        style="max-width: min(calc(100cqw - 6.5rem), 32rem)"
        onclick={() => cMs.handleOpenChange(true)}
        disabled={vDisabled || cMs.updating}
      >
        <Package class="h-3.5 w-3.5 shrink-0" />

        <ModelId modelId={selectedOption?.model || ''} class="font-medium" hideQuantization />

        {#if cMs.updating}
          <Loader2 class="h-3 w-3.5 shrink-0 animate-spin" />
        {/if}
      </button>
    {/if}
  {/if}
</div>

{#if cMs.showModelDialog}
  <DialogModelInformation
    open={cMs.showModelDialog}
    onOpenChange={(v) => cMs.setShowModelDialog(v)}
    modelId={cMs.infoModelId}
  />
{/if}
