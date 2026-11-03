<script lang="ts">
  import { cIconClassDefault } from '$lib/constants/css-classes';
  import type { Snippet, Component } from 'svelte';

  interface Props {
    icon: Component<{ class?: string }>;
    message: Snippet;
    actions: Snippet;
  }

  let { icon: IconComponent, message: vMessage, actions: vActions }: Props = $props();
</script>

<div class="my-2 rounded-lg border border-border bg-card p-3">
  <div class="mb-3 flex items-center gap-2 text-sm">
    <IconComponent class="{cIconClassDefault} shrink-0 text-muted-foreground" />
    <span>
      {@render vMessage()}
    </span>
  </div>
  <div class="flex flex-wrap items-center gap-2">
    {@render vActions()}
  </div>
</div>
