<script lang="ts">
  import { ModelsService } from '$lib/services/models.service';
  import { fConfig } from '$lib/stores/settings.svelte';
  import { TruncatedText } from '$lib/components/app';

  interface Props {
    modelId: string;
    hideOrgName?: boolean;
    showRaw?: boolean;
    hideQuantization?: boolean;
    hideTags?: boolean;
    aliases?: string[];
    tags?: string[];
    class?: string;
  }

  let {
    modelId: vModelId,
    hideOrgName: vHideOrgName = false,
    showRaw: vShowRaw = undefined,
    hideQuantization: vHideQuantization,
    hideTags: vHideTags,
    aliases: vAliases,
    tags: vTags,
    class: vClassName = '',
    ...vRest
  }: Props = $props();

  const cBadgeClass =
    'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono bg-foreground/15 dark:bg-foreground/10 text-foreground [a&]:hover:bg-foreground/25';
  const cTagBadgeClass =
    'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono text-foreground [a&]:hover:bg-accent [a&]:hover:text-accent-foreground';

  let vParsed = $derived(ModelsService.mParseModelId(vModelId));
  let vResolvedShowRaw = $derived(vShowRaw ?? (fConfig().showRawModelNames as boolean) ?? false);
  let vResolvedHideQuantization = $derived(vHideQuantization ?? !fConfig().showModelQuantization);
  let vResolvedHideTags = $derived(vHideTags ?? !fConfig().showModelTags);

  let vUniqueAliases = $derived([...new Set(vAliases ?? [])]);
  let vUniqueTags = $derived([...new Set([...(vParsed.tags ?? []), ...(vTags ?? [])])]);

  let vPrimaryAlias = $derived(vUniqueAliases.length === 1 ? vUniqueAliases[0] : null);
  let vDisplayName = $derived(vPrimaryAlias ?? vParsed.modelName ?? vModelId);
</script>

{#if vResolvedShowRaw}
  <TruncatedText class="font-medium {vClassName}" showTooltip={false} text={vModelId} {...vRest} />
{:else}
  <span class="flex min-w-0 flex-wrap items-center gap-1 {vClassName}" {...vRest}>
    <span class="min-w-0 truncate font-medium">
      {#if !vHideOrgName && vParsed.orgName}{vParsed.orgName}/{/if}{vDisplayName}
    </span>

    {#if vParsed.params}
      <span class={cBadgeClass}>
        {vParsed.params}{vParsed.activatedParams ? `-${vParsed.activatedParams}` : ''}
      </span>
    {/if}

    {#if vParsed.quantization && !vResolvedHideQuantization}
      <span class={cBadgeClass}>
        {vParsed.quantization}
      </span>
    {/if}

    {#if vPrimaryAlias}
      {#if vPrimaryAlias !== vParsed.modelName}
        <span class={cBadgeClass}>{vParsed.modelName ?? vModelId}</span>
      {/if}
    {:else if vUniqueAliases.length > 1}
      {#each vUniqueAliases as alias (alias)}
        <span class={cBadgeClass}>{alias}</span>
      {/each}
    {/if}

    {#if vUniqueTags.length > 0 && !vResolvedHideTags}
      {#each vUniqueTags as tag (tag)}
        <span class={cTagBadgeClass}>{tag}</span>
      {/each}
    {/if}
  </span>
{/if}
