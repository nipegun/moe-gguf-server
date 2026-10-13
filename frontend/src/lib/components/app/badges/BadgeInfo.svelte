<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { HTMLButtonAttributes } from 'svelte/elements';

  interface Props extends HTMLButtonAttributes {
    children: Snippet;
    class?: string;
    icon?: Snippet;
  }

  let { children: vChildren, class: vClassName = '', icon: vIcon, ...vRest }: Props = $props();
</script>

<button
  {...vRest}
  class={[
    'inline-flex cursor-pointer items-center gap-1 rounded-sm bg-muted-foreground/15 px-1.5 py-0.75',
    vClassName
  ]}
>
  {#if vIcon}
    {@render vIcon()}
  {/if}

  {@render vChildren()}
</button>
