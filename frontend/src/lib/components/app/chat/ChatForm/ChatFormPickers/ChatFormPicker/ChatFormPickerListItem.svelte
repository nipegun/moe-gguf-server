<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    isSelected?: boolean;
    onclick: () => void;
    dataIndex?: number;
    children: Snippet;
  }

  let {
    isSelected: vIsSelected = false,
    onclick: vOnclick,
    dataIndex: vDataIndex,
    children: vChildren
  }: Props = $props();
</script>

<button
  type="button"
  data-picker-index={vDataIndex}
  onclick={vOnclick}
  class="flex w-full cursor-pointer items-start gap-3 rounded-lg px-3 py-2 text-left hover:bg-accent/50 {vIsSelected
    ? 'bg-accent/50'
    : ''}"
>
  {@render vChildren()}
</button>
