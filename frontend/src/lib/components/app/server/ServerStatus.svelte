<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { AlertTriangle, Server } from '@lucide/svelte';
  import { Badge } from '$lib/components/ui/badge';
  import { Button } from '$lib/components/ui/button';
  import { fServerProps, fServerLoading, fServerError } from '$lib/stores/server.svelte';
  import { fSingleModelName } from '$lib/stores/models.svelte';

  interface Props {
    class?: string;
    showActions?: boolean;
  }

  let { class: vClassName = '', showActions: vShowActions = false }: Props = $props();

  let vError = $derived(fServerError());
  let vLoading = $derived(fServerLoading());
  let vModel = $derived(fSingleModelName());
  let vServerData = $derived(fServerProps());

  function fGetStatusColor() {
    if (vLoading) return 'bg-yellow-500';
    if (vError) return 'bg-red-500';
    if (vServerData) return 'bg-green-500';

    return 'bg-gray-500';
  }

  function fGetStatusText() {
    if (vLoading) return 'Connecting...';
    if (vError) return fTranslate('text_ddb621f87c5c');
    if (vServerData) return 'Connected';

    return 'Unknown';
  }
</script>

<div class="flex items-center space-x-3 {vClassName}">
  <div class="flex items-center space-x-2">
    <div class="h-2 w-2 rounded-full {fGetStatusColor()}"></div>

    <span class="text-sm text-muted-foreground">{fGetStatusText()}</span>
  </div>

  {#if vServerData && !vError}
    <Badge variant="outline" class="text-xs">
      <Server class="mr-1 h-3 w-3" />

      {vModel || fTranslate('text_3abcd81da434')}
    </Badge>

    {#if vServerData?.default_generation_settings?.n_ctx}
      <Badge variant="secondary" class="text-xs">
        {fTranslate('text_051765386b25')}
        {vServerData.default_generation_settings.n_ctx.toLocaleString()}
      </Badge>
    {/if}
  {/if}

  {#if vShowActions && vError}
    <Button variant="outline" size="sm" class="text-destructive">
      <AlertTriangle class={cIconClassDefault} />

      {vError}
    </Button>
  {/if}
</div>
