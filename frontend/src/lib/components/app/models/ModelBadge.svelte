<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { Package } from '@lucide/svelte';
  import { BadgeInfo, ActionIconCopyToClipboard } from '$lib/components/app';
  import ModelId from './ModelId.svelte';
  import { cModelsStore } from '$lib/stores/models.svelte';
  import { cServerStore } from '$lib/stores/server.svelte';
  import * as Tooltip from '$lib/components/ui/tooltip';

  interface Props {
    class?: string;
    model?: string;
    onclick?: () => void;
    showCopyIcon?: boolean;
    showTooltip?: boolean;
  }

  let {
    class: vClassName = '',
    model: vModelProp,
    onclick: vOnclick,
    showCopyIcon: vShowCopyIcon = false,
    showTooltip: vShowTooltip = false
  }: Props = $props();

  let vModel = $derived(vModelProp || cModelsStore.mSingleModelName);
  let vIsModelMode = $derived(cServerStore.mIsModelMode);
  let vShouldShow = $derived(vModel && (vModelProp !== undefined || vIsModelMode));
</script>

{#snippet badgeContent(triggerProps?: Record<string, unknown>)}
  <BadgeInfo {...triggerProps ?? {}} class={vClassName} onclick={vOnclick}>
    {#snippet icon()}
      <Package class="h-3 w-3" />
    {/snippet}

    {#if vModel}
      <ModelId modelId={vModel} />
    {/if}

    {#if vShowCopyIcon}
      <ActionIconCopyToClipboard text={vModel || ''} ariaLabel="Copy model name" />
    {/if}
  </BadgeInfo>
{/snippet}

{#if vShouldShow}
  {#if vShowTooltip}
    <Tooltip.Root>
      <Tooltip.Trigger>
        <!-- prevent another nested button element -->
        {#snippet child({ props })}
          {@render badgeContent(props)}
        {/snippet}
      </Tooltip.Trigger>

      <Tooltip.Content>
        {vOnclick ? fTranslate('text_4d44206fe9e5') : vModel}
      </Tooltip.Content>
    </Tooltip.Root>
  {:else}
    {@render badgeContent()}
  {/if}
{/if}
