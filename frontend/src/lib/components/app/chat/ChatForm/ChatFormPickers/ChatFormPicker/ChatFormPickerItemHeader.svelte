<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { MCPServerSettingsEntry } from '$lib/types';
  import { cMcpStore } from '$lib/stores/mcp.svelte';

  interface Props {
    server: MCPServerSettingsEntry | undefined;
    serverLabel: string;
    title: string;
    description?: string;
    titleExtra?: Snippet;
    subtitle?: Snippet;
  }

  let {
    server: vServer,
    serverLabel: vServerLabel,
    title: vTitle,
    description: vDescription,
    titleExtra: vTitleExtra,
    subtitle: vSubtitle
  }: Props = $props();

  let vFaviconUrl = $derived(vServer ? cMcpStore.mGetServerFavicon(vServer.id) : null);
</script>

<div class="min-w-0 flex-1">
  <div class="mb-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
    {#if vFaviconUrl}
      <img
        src={vFaviconUrl}
        alt=""
        class="h-3 w-3 shrink-0 rounded-sm"
        onerror={(e) => {
          (e.currentTarget as HTMLImageElement).style.display = 'none';
        }}
      />
    {/if}

    <span>{vServerLabel}</span>
  </div>

  <div class="flex items-center gap-2">
    <span class="font-medium">
      {vTitle}
    </span>

    {#if vTitleExtra}
      {@render vTitleExtra()}
    {/if}
  </div>

  {#if vDescription}
    <p class="mt-0.5 truncate text-sm text-muted-foreground">
      {vDescription}
    </p>
  {/if}

  {#if vSubtitle}
    {@render vSubtitle()}
  {/if}
</div>
