<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { Loader2, AlertCircle } from '@lucide/svelte';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import type { MCPResourceAttachment } from '$lib/types';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { ActionIcon } from '$lib/components/app';
  import { X } from '@lucide/svelte';
  import { getResourceIcon, getResourceDisplayName } from '$lib/utils';

  interface Props {
    attachment: MCPResourceAttachment;
    class?: string;
    onclick?: () => void;
    onRemove?: (pAttachmentId: string) => void;
  }

  let {
    attachment: vAttachment,
    class: vClassName,
    onclick: vOnclick,
    onRemove: vOnRemove
  }: Props = $props();

  const ResourceIcon = $derived(
    getResourceIcon(vAttachment.resource.mimeType, vAttachment.resource.uri)
  );
  const cServerName = $derived(cMcpStore.mGetServerDisplayName(vAttachment.resource.serverName));
  const cFavicon = $derived(cMcpStore.mGetServerFavicon(vAttachment.resource.serverName));

  function fGetStatusClass(pAttachment: MCPResourceAttachment): string {
    if (pAttachment.error) return 'border-red-500/50 bg-red-500/10';
    if (pAttachment.loading) return 'border-border/50 bg-muted/30';

    return 'border-border/50 bg-muted/30';
  }
</script>

<Tooltip.Root>
  <Tooltip.Trigger>
    <button
      class={[
        'flex flex-shrink-0 items-center gap-1.5 rounded-md border px-2 py-0.75 text-sm transition-colors',
        fGetStatusClass(vAttachment),
        vOnclick && 'cursor-pointer hover:bg-muted/50',
        vClassName
      ]}
      disabled={!vOnclick}
      onclick={vOnclick}
      type="button"
    >
      {#if vAttachment.loading}
        <Loader2 class="h-3 w-3 animate-spin text-muted-foreground" />
      {:else if vAttachment.error}
        <AlertCircle class="h-3 w-3 text-red-500" />
      {:else}
        <ResourceIcon class="h-3 w-3 text-muted-foreground" />
      {/if}

      <span class="max-w-[150px] truncate text-xs">
        {getResourceDisplayName(vAttachment.resource)}
      </span>

      {#if vOnRemove}
        <ActionIcon
          class="-my-2 -mr-1.5 bg-transparent"
          icon={X}
          iconSize="h-2 w-2"
          onclick={() => vOnRemove?.(vAttachment.id)}
          stopPropagationOnClick
          tooltip={fTranslate('text_c3812fc4acb8')}
        />
      {/if}
    </button>
  </Tooltip.Trigger>

  <Tooltip.Content>
    <div class="flex items-center gap-1 text-xs">
      {#if cFavicon}
        <img
          alt={vAttachment.resource.serverName}
          class="h-3 w-3 shrink-0 rounded-sm"
          onerror={(e) => {
            (e.currentTarget as HTMLImageElement).style.display = 'none';
          }}
          src={cFavicon}
        />
      {/if}

      <span class="truncate">
        {cServerName}
      </span>
    </div>
  </Tooltip.Content>
</Tooltip.Root>
