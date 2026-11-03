<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { ActionIcon } from '$lib/components/app';
  import { X } from '@lucide/svelte';

  interface Props {
    class?: string;
    height?: string;
    id: string;
    imageClass?: string;
    onclick?: (pEvent?: MouseEvent) => void;
    onRemove?: (pId: string) => void;
    name: string;
    preview: string;
    readonly?: boolean;
    width?: string;
  }

  let {
    class: vClassName = '',
    height: vHeight = 'h-16',
    id: vId,
    imageClass: vImageClass = '',
    onclick: vOnclick,
    onRemove: vOnRemove,
    name: vName,
    preview: vPreview,
    readonly: vReadonly = false,
    width: vWidth = 'w-auto'
  }: Props = $props();
</script>

{#snippet image()}
  <img
    src={vPreview}
    alt={vName}
    class="{vHeight} {vWidth} cursor-pointer object-cover {vImageClass}"
  />
{/snippet}

<div
  class="group relative overflow-hidden rounded-lg bg-muted shadow-lg dark:border dark:border-muted {vClassName}"
>
  {#if vOnclick}
    <button
      aria-label={fTranslate('text_e100b47318c5', { p0: vName })}
      class="block h-full w-full rounded-lg focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:outline-none"
      onclick={vOnclick}
      type="button"
    >
      {@render image()}
    </button>
  {:else}
    {@render image()}
  {/if}

  {#if !vReadonly}
    <div
      class="absolute top-1 right-1 flex items-center justify-center opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100"
    >
      <ActionIcon
        class="text-white"
        icon={X}
        onclick={() => vOnRemove?.(vId)}
        stopPropagationOnClick
        tooltip={fTranslate('text_c3812fc4acb8')}
      />
    </div>
  {/if}
</div>
