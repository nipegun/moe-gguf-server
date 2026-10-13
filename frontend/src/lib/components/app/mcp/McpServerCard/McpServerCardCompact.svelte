<script lang="ts">
  import * as Card from '$lib/components/ui/card';
  import { mode } from 'mode-watcher';
  import type { RecommendedMCPServer } from '$lib/types';

  interface Props {
    server: RecommendedMCPServer;
    onClick?: () => void;
    selected?: boolean;
    dimmed?: boolean;
  }

  let {
    server: vServer,
    onClick: vOnClick,
    selected: vSelected = false,
    dimmed: vDimmed = false
  }: Props = $props();

  let vActiveIconUrl = $derived.by(() => {
    const cIsDark = mode.current === 'dark';

    if (cIsDark && vServer.iconUrlDark) return vServer.iconUrlDark;
    if (!cIsDark && vServer.iconUrlLight) return vServer.iconUrlLight;

    return vServer.iconUrl;
  });
</script>

<Card.Root
  class={`relative gap-3! select-none bg-muted/30 p-4 transition-all ${vOnClick ? 'cursor-pointer hover:bg-muted/50 hover:opacity-100' : ''} ${vSelected ? 'bg-muted/30 ring-1 ring-primary/40' : ''} ${vDimmed ? 'opacity-50' : ''}`}
  onclick={vOnClick}
>
  <div class="flex min-w-0 items-center gap-2">
    {#if vActiveIconUrl}
      <img
        src={vActiveIconUrl}
        alt=""
        class="h-5 w-5 shrink-0 rounded"
        loading="lazy"
        decoding="async"
      />
    {/if}

    <h4 class="min-w-0 flex-1 truncate font-medium">{vServer.name}</h4>
  </div>

  <p class="text-xs text-muted-foreground">{vServer.description}</p>
</Card.Root>
