<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import {
    CircleAlert,
    Heart,
    HeartOff,
    Info,
    Loader2,
    Power,
    PowerOff,
    RotateCw
  } from '@lucide/svelte';
  import { ActionIcon, ModelId } from '$lib/components/app';
  import ModelLoadHighlight from './ModelLoadHighlight.svelte';
  import type { ModelOption } from '$lib/types/models';
  import { ServerModelStatus } from '$lib/enums';
  import { cModelsStore, fRouterModels } from '$lib/stores/models.svelte';
  import { modelLoadFraction, modelLoadProgressText } from '$lib/utils';

  interface Props {
    option: ModelOption;
    isSelected: boolean;
    isHighlighted: boolean;
    isFav: boolean;
    hideOrgName?: boolean;
    onSelect: (pModelId: string) => void;
    onMouseEnter: () => void;
    onKeyDown: (pE: KeyboardEvent) => void;
    onInfoClick?: (pModelName: string) => void;
  }

  let {
    option: vOption,
    isSelected: vIsSelected,
    isHighlighted: vIsHighlighted,
    isFav: vIsFav,
    hideOrgName: vHideOrgName = false,
    onSelect: vOnSelect,
    onMouseEnter: vOnMouseEnter,
    onKeyDown: vOnKeyDown,
    onInfoClick: vOnInfoClick
  }: Props = $props();

  let vCurrentRouterModels = $derived(fRouterModels());
  let vServerStatus = $derived.by(() => {
    const cModel = vCurrentRouterModels.find((pM) => pM.id === vOption.model);
    return (cModel?.status?.value as ServerModelStatus) ?? null;
  });
  let vIsOperationInProgress = $derived(cModelsStore.mIsModelOperationInProgress(vOption.model));
  let vIsFailed = $derived(vServerStatus === ServerModelStatus.FAILED);
  let vIsSleeping = $derived(vServerStatus === ServerModelStatus.SLEEPING);
  let vIsLoaded = $derived(
    (vServerStatus === ServerModelStatus.LOADED || vIsSleeping) && !vIsOperationInProgress
  );
  let vIsLoading = $derived(vServerStatus === ServerModelStatus.LOADING || vIsOperationInProgress);

  let vLoadProgress = $derived(vIsLoading ? cModelsStore.mGetLoadProgress(vOption.model) : null);
  let vLoadPercent = $derived(Math.round(modelLoadFraction(vLoadProgress) * 100));
  let vLoadTitle = $derived(modelLoadProgressText(vLoadProgress));
</script>

<div
  class={[
    'group relative flex w-full items-center gap-2 rounded-sm p-2 text-left text-sm transition focus:outline-none',
    'cursor-pointer hover:bg-muted focus:bg-muted',
    (vIsSelected || vIsHighlighted) && 'bg-accent text-accent-foreground',
    !(vIsSelected || vIsHighlighted) && 'hover:bg-accent hover:text-accent-foreground',
    vIsLoaded ? 'text-popover-foreground' : 'text-muted-foreground'
  ]}
  role="option"
  aria-selected={vIsSelected || vIsHighlighted}
  title={vLoadTitle}
  tabindex="0"
  onclick={() => vOnSelect(vOption.id)}
  onmouseenter={vOnMouseEnter}
  onkeydown={vOnKeyDown}
>
  <ModelId
    modelId={vOption.model}
    hideOrgName={vHideOrgName}
    aliases={vOption.aliases}
    tags={vOption.tags}
    class="flex-1"
  />

  <div class="flex shrink-0 items-center gap-1">
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <div
      class="pointer-events-none flex items-center justify-center gap-0.75 pl-2 opacity-0 group-hover:pointer-events-auto group-hover:opacity-100 [@media(pointer:coarse)]:pointer-events-auto [@media(pointer:coarse)]:opacity-100"
      onclick={(e) => e.stopPropagation()}
    >
      {#if vIsFav}
        <ActionIcon
          iconSize="h-2.5 w-2.5"
          icon={HeartOff}
          tooltip={fTranslate('text_18720329ca30')}
          class="h-3 w-3 hover:text-foreground"
          onclick={() => cModelsStore.mToggleFavorite(vOption.model)}
        />
      {:else}
        <ActionIcon
          iconSize="h-2.5 w-2.5"
          icon={Heart}
          tooltip={fTranslate('text_7f3c0782afd7')}
          class="h-3 w-3 hover:text-foreground"
          onclick={() => cModelsStore.mToggleFavorite(vOption.model)}
        />
      {/if}

      <!-- info button: only shown when model is loaded and callback is provided -->
      {#if vIsLoaded && vOnInfoClick}
        <ActionIcon
          iconSize="h-2.5 w-2.5"
          icon={Info}
          tooltip={fTranslate('text_3a584c762603')}
          class="h-3 w-3 hover:text-foreground"
          onclick={() => vOnInfoClick(vOption.model)}
        />
      {/if}
    </div>

    {#if vIsLoading}
      <div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-5">
        <Loader2 class="{cIconClassDefault} animate-spin text-muted-foreground" />
      </div>
    {:else if vIsFailed}
      <div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-auto">
        <CircleAlert
          class="h-3.5 w-3.5 text-red-500 group-hover:hidden [@media(pointer:coarse)]:hidden"
        />

        <div class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
          <ActionIcon
            iconSize="h-2.5 w-2.5"
            icon={RotateCw}
            tooltip={fTranslate('text_3890e4decfe9')}
            class="h-3 w-3 text-red-500 hover:text-foreground"
            onclick={() => cModelsStore.mLoadModel(vOption.model)}
            stopPropagationOnClick
          />
        </div>
      </div>
    {:else if vIsSleeping}
      <div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-auto">
        <span
          class="h-2 w-2 rounded-full bg-orange-400 group-hover:hidden [@media(pointer:coarse)]:hidden"
        ></span>

        <div class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
          <ActionIcon
            iconSize="h-2.5 w-2.5"
            icon={PowerOff}
            tooltip={fTranslate('text_c2ed103ced3e')}
            class="h-3 w-3 text-red-500 hover:text-red-600 [@media(pointer:coarse)]:text-amber-500 [@media(pointer:coarse)]:hover:text-amber-600"
            onclick={(e) => {
              e?.stopPropagation();
              cModelsStore.mUnloadModel(vOption.model);
            }}
          />
        </div>
      </div>
    {:else if vIsLoaded}
      <div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-auto">
        <span
          class="h-2 w-2 rounded-full bg-green-500 group-hover:hidden [@media(pointer:coarse)]:hidden"
        ></span>

        <div class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
          <ActionIcon
            iconSize="h-2.5 w-2.5"
            icon={PowerOff}
            tooltip={fTranslate('text_c2ed103ced3e')}
            class="h-3 w-3 text-red-500 hover:text-red-600 [@media(pointer:coarse)]:text-green-500 [@media(pointer:coarse)]:hover:text-green-600"
            onclick={() => cModelsStore.mUnloadModel(vOption.model)}
            stopPropagationOnClick
          />
        </div>
      </div>
    {:else}
      <div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-auto">
        <span
          class="h-2 w-2 rounded-full bg-muted-foreground/50 group-hover:hidden [@media(pointer:coarse)]:hidden"
        ></span>

        <div class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
          <ActionIcon
            iconSize="h-2.5 w-2.5"
            icon={Power}
            tooltip={fTranslate('text_889fad74ece5')}
            class="h-3 w-3 [@media(pointer:coarse)]:text-muted-foreground"
            onclick={() => cModelsStore.mLoadModel(vOption.model)}
            stopPropagationOnClick
          />
        </div>
      </div>
    {/if}
  </div>

  {#if vIsLoading}
    <ModelLoadHighlight percent={vLoadPercent} />
  {/if}
</div>
