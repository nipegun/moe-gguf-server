<script lang="ts">
  import { ArrowBigUp } from '@lucide/svelte';

  interface Props {
    keys: string[];
    variant?: 'default' | 'destructive';
    class?: string;
  }

  let { keys: vKeys, variant: vVariant = 'default', class: vClassName = '' }: Props = $props();

  let vBaseClasses =
    'px-1 pointer-events-none inline-flex select-none items-center gap-0.5 font-sans text-md font-medium opacity-0 transition-opacity -my-1';
  let vVariantClasses = $derived(
    vVariant === 'destructive' ? 'text-destructive' : 'text-muted-foreground'
  );
</script>

<kbd class="{vBaseClasses} {vVariantClasses} {vClassName}">
  {#each vKeys as vKey, vIndex (vIndex)}
    {#if vKey === 'shift'}
      <ArrowBigUp class="h-1 w-1 {vVariant === 'destructive' ? 'text-destructive' : ''} -mr-1" />
    {:else if vKey === 'cmd'}
      <span class={vVariant === 'destructive' ? 'text-destructive' : ''}>⌘</span>
    {:else}
      {vKey.toUpperCase()}
    {/if}

    {#if vIndex < vKeys.length - 1}
      <span> </span>
    {/if}
  {/each}
</kbd>
