<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    children: Snippet;
  }

  let { title: vTitle, children: vChildren }: Props = $props();
</script>

<div>
  <h3 class="mb-6 text-base font-semibold">{vTitle}</h3>

  <div class="space-y-8">
    {@render vChildren()}
  </div>
</div>
