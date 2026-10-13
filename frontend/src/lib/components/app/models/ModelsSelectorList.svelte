<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { cModelsStore } from '$lib/stores/models.svelte';
  import { ModelsSelectorOption } from '$lib/components/app';
  import type { GroupedModelOptions, ModelItem } from './utils';

  interface Props {
    groups: GroupedModelOptions;
    currentModel: string | null;
    activeId: string | null;
    sectionHeaderClass?: string;
    orgHeaderClass?: string;
    onSelect: (pModelId: string) => void;
    onInfoClick: (pModelName: string) => void;
    renderOption?: import('svelte').Snippet<[ModelItem, boolean]>;
  }

  let {
    groups: vGroups,
    currentModel: vCurrentModel,
    activeId: vActiveId,
    sectionHeaderClass:
      vSectionHeaderClass = 'my-1 px-2 py-2 text-[13px] font-semibold text-muted-foreground/70 select-none',
    orgHeaderClass:
      vOrgHeaderClass = 'px-2 py-2 text-[11px] font-semibold text-muted-foreground/50 select-none [&:not(:first-child)]:mt-1',
    onSelect: vOnSelect,
    onInfoClick: vOnInfoClick,
    renderOption: vRenderOption
  }: Props = $props();
  let vRender = $derived(vRenderOption ?? defaultOption);
</script>

{#snippet defaultOption(item: ModelItem, hideOrgName: boolean)}
  {@const { option } = item}
  {@const isSelected = vCurrentModel === option.model || vActiveId === option.id}
  {@const isFav = cModelsStore.vFavoriteModelIds.has(option.model)}

  <ModelsSelectorOption
    {option}
    {isSelected}
    isHighlighted={false}
    {isFav}
    {hideOrgName}
    onSelect={vOnSelect}
    onInfoClick={vOnInfoClick}
    onMouseEnter={() => {}}
    onKeyDown={() => {}}
  />
{/snippet}

{#if vGroups.loaded.length > 0}
  <p class={vSectionHeaderClass}>{fTranslate('text_fb65b87d1b5e')}</p>
  {#each vGroups.loaded as item (`loaded-${item.option.id}`)}
    {@render vRender(item, false)}
  {/each}
{/if}

{#if vGroups.favorites.length > 0}
  <p class={vSectionHeaderClass}>{fTranslate('text_7715ea73a9bf')}</p>
  {#each vGroups.favorites as item (`fav-${item.option.id}`)}
    {@render vRender(item, true)}
  {/each}
{/if}

{#if vGroups.available.length > 0}
  <p class={vSectionHeaderClass}>{fTranslate('text_9dbfd0478bd5')}</p>
  {#each vGroups.available as group (group.orgName)}
    {#if group.orgName}
      <p class={vOrgHeaderClass}>{group.orgName}</p>
    {/if}
    {#each group.items as item (item.option.id)}
      {@render vRender(item, true)}
    {/each}
  {/each}
{/if}
