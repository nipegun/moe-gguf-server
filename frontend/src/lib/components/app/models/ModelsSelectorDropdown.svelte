<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { ChevronDown, Loader2, Package } from '@lucide/svelte';
  import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { KeyboardKey, ServerModelStatus } from '$lib/enums';
  import { fUseModelsSelector } from '$lib/hooks/use-models-selector.svelte';
  import { cModelsStore, fRouterModels } from '$lib/stores/models.svelte';
  import { modelLoadFraction } from '$lib/utils';
  import {
    DialogModelInformation,
    DropdownMenuSearchable,
    ModelId,
    ModelsSelectorList,
    ModelsSelectorOption
  } from '$lib/components/app';
  import ModelLoadHighlight from './ModelLoadHighlight.svelte';
  import type { ModelItem } from './utils';

  interface Props {
    class?: string;
    currentModel?: string | null;
    disabled?: boolean;
    forceForegroundText?: boolean;
    onModelChange?: (pModelId: string, pModelName: string) => Promise<boolean> | boolean | void;
    useGlobalSelection?: boolean;
  }

  let {
    class: vClassName = '',
    currentModel: vCurrentModel = null,
    disabled: vDisabled = false,
    forceForegroundText: vForceForegroundText = false,
    onModelChange: vOnModelChange,
    useGlobalSelection: vUseGlobalSelection = false
  }: Props = $props();

  let vIsOpen = $state(false);
  let vHighlightedIndex = $state<number>(-1);

  const cMs = fUseModelsSelector({
    currentModel: () => vCurrentModel,
    useGlobalSelection: () => vUseGlobalSelection,
    onModelChange: () => vOnModelChange,
    onOpenChange: (pOpen) => {
      vIsOpen = pOpen;
      vHighlightedIndex = -1;
    }
  });

  $effect(() => {
    void cMs.searchTerm;
    vHighlightedIndex = -1;
  });

  export function fOpen() {
    cMs.handleOpenChange(true);
  }

  function fHandleSearchKeyDown(pEvent: KeyboardEvent) {
    if (pEvent.isComposing) return;

    if (pEvent.key === KeyboardKey.ARROW_DOWN) {
      pEvent.preventDefault();

      if (cMs.filteredOptions.length === 0) return;

      if (vHighlightedIndex === -1 || vHighlightedIndex === cMs.filteredOptions.length - 1) {
        vHighlightedIndex = 0;
      } else {
        vHighlightedIndex += 1;
      }
    } else if (pEvent.key === KeyboardKey.ARROW_UP) {
      pEvent.preventDefault();

      if (cMs.filteredOptions.length === 0) return;

      if (vHighlightedIndex === -1 || vHighlightedIndex === 0) {
        vHighlightedIndex = cMs.filteredOptions.length - 1;
      } else {
        vHighlightedIndex -= 1;
      }
    } else if (pEvent.key === KeyboardKey.ENTER) {
      pEvent.preventDefault();

      if (vHighlightedIndex >= 0 && vHighlightedIndex < cMs.filteredOptions.length) {
        const cOption = cMs.filteredOptions[vHighlightedIndex];

        cMs.handleSelect(cOption.id);
      } else if (cMs.filteredOptions.length > 0) {
        vHighlightedIndex = 0;
      }
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
    {#if vCurrentModel}
      <span
        class={[
          'inline-flex items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs text-muted-foreground',
          vClassName
        ]}
        style="max-width: min(calc(100cqw - 10rem), 20rem)"
      >
        <Package class="h-3.5 w-3.5 shrink-0" />
      </span>
    {:else}
      <p class="text-xs text-muted-foreground">{fTranslate('text_58064bf43a25')}</p>
    {/if}
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
      <DropdownMenu.Root bind:open={vIsOpen} onOpenChange={cMs.handleOpenChange}>
        <Tooltip.Root>
          <Tooltip.Trigger>
            <!-- prevent another nested button element -->
            {#snippet child({ props })}
              <DropdownMenu.Trigger
                {...props}
                class={[
                  `relative inline-grid cursor-pointer grid-cols-[1fr_auto_1fr] items-center gap-1.5 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground`,
                  !cMs.isCurrentModelInCache
                    ? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
                    : vForceForegroundText
                      ? 'text-foreground'
                      : cMs.isHighlightedCurrentModelActive
                        ? 'text-foreground'
                        : 'text-foreground',
                  vIsOpen && 'text-foreground',
                  'max-w-[min(calc(100vw-4rem) md:max-w-[min(calc(100cqw-9rem),25rem)]'
                ]}
                disabled={vDisabled || cMs.updating}
              >
                <Package class="h-3.5 w-3.5 shrink-0" />

                {#if selectedOption}
                  <ModelId
                    modelId={selectedOption.model}
                    class="min-w-0 overflow-hidden"
                    hideOrgName={false}
                    hideQuantization
                  />
                {:else}
                  <span class="min-w-0 font-medium">{fTranslate('text_1d446005add3')}</span>
                {/if}

                {#if cMs.updating || cMs.isLoadingModel}
                  <Loader2 class="h-3 w-3.5 shrink-0 animate-spin" />
                {:else}
                  <ChevronDown class="h-3 w-3.5 shrink-0" />
                {/if}

                {#if triggerLoading}
                  <ModelLoadHighlight percent={triggerLoadPercent} />
                {/if}
              </DropdownMenu.Trigger>
            {/snippet}
          </Tooltip.Trigger>

          {#if selectedOption}
            <Tooltip.Content>
              <p class="font-mono">{selectedOption.model}</p>
            </Tooltip.Content>
          {/if}
        </Tooltip.Root>

        <DropdownMenu.Content
          align="end"
          class="w-full max-w-[100vw] pt-0 sm:w-max sm:max-w-[calc(100vw-2rem)]"
        >
          <DropdownMenuSearchable
            searchValue={cMs.searchTerm}
            onSearchChange={(v) => cMs.setSearchTerm(v)}
            placeholder={fTranslate('text_37b90680b842')}
            onSearchKeyDown={fHandleSearchKeyDown}
            emptyMessage={fTranslate('text_339e5fcda064')}
            isEmpty={cMs.filteredOptions.length === 0 && cMs.isCurrentModelInCache}
          >
            <div class="models-list">
              {#if !cMs.isCurrentModelInCache && vCurrentModel}
                <!-- Show unavailable model as first option (disabled) -->
                <button
                  type="button"
                  class="flex w-full cursor-not-allowed items-center bg-red-400/10 p-2 text-left text-sm text-red-400"
                  role="option"
                  aria-selected="true"
                  aria-disabled="true"
                  disabled
                >
                  <ModelId modelId={vCurrentModel} class="flex-1" hideQuantization />

                  <span class="ml-2 text-xs whitespace-nowrap opacity-70"
                    >{fTranslate('text_1c96e2456748')}</span
                  >
                </button>
              {/if}

              {#if cMs.filteredOptions.length === 0}
                <p class="px-4 py-3 text-sm text-muted-foreground">
                  {fTranslate('text_339e5fcda064')}
                </p>
              {/if}

              {#snippet modelOption(item: ModelItem, hideOrgName: boolean)}
                {@const { option, flatIndex } = item}
                {@const isSelected = vCurrentModel === option.model || cMs.activeId === option.id}
                {@const isHighlighted = flatIndex === vHighlightedIndex}
                {@const isFav = cMs.isFavorite(option.model)}

                <ModelsSelectorOption
                  {option}
                  {isSelected}
                  {isHighlighted}
                  {isFav}
                  {hideOrgName}
                  onSelect={cMs.handleSelect}
                  onInfoClick={cMs.handleInfoClick}
                  onMouseEnter={() => (vHighlightedIndex = flatIndex)}
                  onKeyDown={(event) => {
                    if (event.key === KeyboardKey.ENTER || event.key === KeyboardKey.SPACE) {
                      event.preventDefault();
                      cMs.handleSelect(option.id);
                    }
                  }}
                />
              {/snippet}

              <ModelsSelectorList
                groups={cMs.groupedFilteredOptions}
                currentModel={vCurrentModel}
                activeId={cMs.activeId}
                sectionHeaderClass="my-1.5 px-2 py-2 text-[13px] font-semibold text-muted-foreground/70 select-none"
                onSelect={cMs.handleSelect}
                onInfoClick={cMs.handleInfoClick}
                renderOption={modelOption}
              />
            </div>
          </DropdownMenuSearchable>
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    {:else}
      <Tooltip.Root>
        <Tooltip.Trigger>
          <!-- prevent another nested button element -->
          {#snippet child({ props })}
            <button
              {...props}
              class={[
                `inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground`,
                !cMs.isCurrentModelInCache
                  ? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
                  : vForceForegroundText
                    ? 'text-foreground'
                    : cMs.isHighlightedCurrentModelActive
                      ? 'text-foreground'
                      : 'text-foreground',
                vIsOpen && 'text-foreground'
              ]}
              style="max-width: min(calc(100cqw - 6.5rem), 32rem)"
              onclick={() => cMs.handleOpenChange(true)}
              disabled={vDisabled || cMs.updating}
            >
              <Package class="h-3.5 w-3.5 shrink-0" />

              {#if selectedOption}
                <ModelId
                  modelId={selectedOption.model}
                  class="min-w-0 overflow-hidden"
                  hideOrgName={false}
                  hideQuantization
                />
              {/if}

              {#if cMs.updating}
                <Loader2 class="h-3 w-3.5 shrink-0 animate-spin" />
              {/if}
            </button>
          {/snippet}
        </Tooltip.Trigger>

        {#if selectedOption}
          <Tooltip.Content>
            <p class="font-mono">{selectedOption.model}</p>
          </Tooltip.Content>
        {/if}
      </Tooltip.Root>
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
