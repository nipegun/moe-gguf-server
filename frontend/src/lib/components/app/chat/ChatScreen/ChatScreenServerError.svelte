<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { AlertTriangle, Loader2, RefreshCw } from '@lucide/svelte';
  import * as Alert from '$lib/components/ui/alert';
  import {
    fServerError,
    fServerLoading,
    fServerStatus,
    cServerStore
  } from '$lib/stores/server.svelte';

  let vHasError = $derived(!!fServerError());
  let vIsLoadingModel = $derived(fServerStatus() === 503);
</script>

{#if vHasError}
  <div class="pointer-events-auto mx-auto mb-4 max-w-[48rem] px-1">
    <Alert.Root variant={vIsLoadingModel ? 'default' : 'destructive'}>
      {#if vIsLoadingModel}
        <Loader2 class="{cIconClassDefault} animate-spin" />
      {:else}
        <AlertTriangle class={cIconClassDefault} />
      {/if}

      <Alert.Title class="flex items-center justify-between">
        <span
          >{vIsLoadingModel
            ? fTranslate('text_64590e4b6bc5')
            : fTranslate('text_b0f398a57ea3')}</span
        >

        {#if !vIsLoadingModel}
          <button
            onclick={() => cServerStore.mFetch()}
            disabled={fServerLoading()}
            class="flex items-center gap-1.5 rounded-lg bg-destructive/20 px-2 py-1 text-xs font-medium hover:bg-destructive/30 disabled:opacity-50"
          >
            <RefreshCw class="h-3 w-3 {fServerLoading() ? 'animate-spin' : ''}" />
            {fServerLoading() ? fTranslate('text_84a657bcf3d9') : fTranslate('text_942087cc2d41')}
          </button>
        {/if}
      </Alert.Title>

      {#if !vIsLoadingModel}
        <Alert.Description>{fServerError()}</Alert.Description>
      {/if}
    </Alert.Root>
  </div>
{/if}
