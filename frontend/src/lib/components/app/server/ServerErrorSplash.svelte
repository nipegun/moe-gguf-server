<script lang="ts">
  import { fApiUrl } from '$lib/api-url';

  import { fTranslate } from '../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { base } from '$app/paths';
  import { AlertTriangle, RefreshCw, Key, CheckCircle, XCircle } from '@lucide/svelte';
  import { goto } from '$app/navigation';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import Label from '$lib/components/ui/label/label.svelte';
  import { cServerStore, fServerLoading } from '$lib/stores/server.svelte';
  import { fConfig, cSettingsStore } from '$lib/stores/settings.svelte';
  import { cAuthorizationHeader, cBearerPrefix, cSettingsKeys } from '$lib/constants';
  import { cRoutes } from '$lib/constants/routes';
  import { fade, fly, scale } from 'svelte/transition';
  import { KeyboardKey } from '$lib/enums';

  interface Props {
    class?: string;
    error: string;
    onRetry?: () => void;
    showRetry?: boolean;
    showTroubleshooting?: boolean;
  }

  let {
    class: vClassName = '',
    error: vError,
    onRetry: vOnRetry,
    showRetry: vShowRetry = true,
    showTroubleshooting: vShowTroubleshooting = false
  }: Props = $props();

  let vIsServerLoading = $derived(fServerLoading());
  let vIsAccessDeniedError = $derived(
    vError.toLowerCase().includes('access denied') ||
      vError.toLowerCase().includes('invalid api key') ||
      vError.toLowerCase().includes('unauthorized') ||
      vError.toLowerCase().includes('401') ||
      vError.toLowerCase().includes('403')
  );

  let vApiKeyInput = $state('');
  let vShowApiKeyInput = $state(false);
  let vApiKeyState = $state<'idle' | 'validating' | 'success' | 'error'>('idle');
  let vApiKeyError = $state('');

  function fHandleRetryConnection() {
    if (vOnRetry) {
      vOnRetry();
    } else {
      cServerStore.mFetch();
    }
  }

  function fHandleShowApiKeyInput() {
    vShowApiKeyInput = true;
    // Pre-fill with current API key if it exists
    const cCurrentConfig = fConfig();
    vApiKeyInput = cCurrentConfig.apiKey?.toString() || '';
  }

  async function fHandleSaveApiKey() {
    if (!vApiKeyInput.trim()) return;

    vApiKeyState = 'validating';
    vApiKeyError = '';

    try {
      // Update the API key in settings first
      cSettingsStore.mUpdateConfig(cSettingsKeys.API_KEY, vApiKeyInput.trim());

      // Test the API key by making a real request to the server
      const cResponse = await fetch(fApiUrl('/props'), {
        headers: {
          'Content-Type': 'application/json',
          [cAuthorizationHeader]: `${cBearerPrefix}${vApiKeyInput.trim()}`
        }
      });

      if (cResponse.ok) {
        // API key is valid - User Story B
        vApiKeyState = 'success';

        // Show success state briefly, then navigate to home
        setTimeout(() => {
          goto(cRoutes.START);
        }, 1000);
      } else {
        // API key is invalid - User Story A
        vApiKeyState = 'error';

        if (cResponse.status === 401 || cResponse.status === 403) {
          vApiKeyError = fTranslate('text_1fa2d2208266');
        } else {
          vApiKeyError = fTranslate('text_af83e449bf6a', { p0: cResponse.status });
        }

        // Reset to idle state after showing error (don't reload UI)
        setTimeout(() => {
          vApiKeyState = 'idle';
        }, 3000);
      }
    } catch (vError) {
      // Network or other errors - User Story A
      vApiKeyState = 'error';

      if (vError instanceof Error) {
        if (vError.message.includes('fetch')) {
          vApiKeyError = fTranslate('text_0cbdcfd3ffe0');
        } else {
          vApiKeyError = vError.message;
        }
      } else {
        vApiKeyError = fTranslate('text_d965734118e6');
      }

      // Reset to idle state after showing error (don't reload UI)
      setTimeout(() => {
        vApiKeyState = 'idle';
      }, 3000);
    }
  }

  function fHandleApiKeyKeydown(pEvent: KeyboardEvent) {
    if (pEvent.key === KeyboardKey.ENTER) {
      fHandleSaveApiKey();
    }
  }
</script>

<div class="flex h-full items-center justify-center {vClassName}">
  <div class="w-full max-w-md px-4 text-center">
    <div class="mb-6" in:fade={{ duration: 300 }}>
      <div
        class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-destructive/10"
      >
        <AlertTriangle class="h-8 w-8 text-destructive" />
      </div>

      <h2 class="mb-2 text-xl font-semibold">{fTranslate('text_8d9809100ff0')}</h2>

      <p class="mb-4 text-sm text-muted-foreground">
        {vError}
      </p>
    </div>

    {#if vIsAccessDeniedError && !vShowApiKeyInput}
      <div in:fly={{ y: 10, duration: 300, delay: 200 }} class="mb-4">
        <Button onclick={fHandleShowApiKeyInput} variant="outline" class="w-full">
          <Key class={cIconClassDefault} />
          {fTranslate('text_895f77e86e47')}
        </Button>
      </div>
    {/if}

    {#if vShowApiKeyInput}
      <div in:fly={{ y: 10, duration: 300, delay: 200 }} class="mb-4 space-y-3 text-left">
        <div class="space-y-2">
          <Label for="api-key-input" class="text-sm font-medium"
            >{fTranslate('text_23189d55f697')}</Label
          >

          <div class="relative">
            <Input
              id="api-key-input"
              placeholder={fTranslate('text_cfc419f196c7')}
              bind:value={vApiKeyInput}
              onkeydown={fHandleApiKeyKeydown}
              class="w-full pr-10 {vApiKeyState === 'error'
                ? 'border-destructive'
                : vApiKeyState === 'success'
                  ? 'border-green-500'
                  : ''}"
              disabled={vApiKeyState === 'validating'}
            />
            {#if vApiKeyState === 'validating'}
              <div class="absolute top-1/2 right-3 -translate-y-1/2">
                <RefreshCw class="{cIconClassDefault} animate-spin text-muted-foreground" />
              </div>
            {:else if vApiKeyState === 'success'}
              <div
                class="absolute top-1/2 right-3 -translate-y-1/2"
                in:scale={{ duration: 200, start: 0.8 }}
              >
                <CheckCircle class="{cIconClassDefault} text-green-500" />
              </div>
            {:else if vApiKeyState === 'error'}
              <div
                class="absolute top-1/2 right-3 -translate-y-1/2"
                in:scale={{ duration: 200, start: 0.8 }}
              >
                <XCircle class="{cIconClassDefault} text-destructive" />
              </div>
            {/if}
          </div>
          {#if vApiKeyError}
            <p class="text-sm text-destructive" in:fly={{ y: -10, duration: 200 }}>
              {vApiKeyError}
            </p>
          {/if}
          {#if vApiKeyState === 'success'}
            <p class="text-sm text-green-600" in:fly={{ y: -10, duration: 200 }}>
              {fTranslate('text_e40ab11d04df')}
            </p>
          {/if}
        </div>
        <div class="flex gap-2">
          <Button
            onclick={fHandleSaveApiKey}
            disabled={!vApiKeyInput.trim() ||
              vApiKeyState === 'validating' ||
              vApiKeyState === 'success'}
            class="flex-1"
          >
            {#if vApiKeyState === 'validating'}
              <RefreshCw class="{cIconClassDefault} animate-spin" />
              {fTranslate('text_b28b48d59a08')}
            {:else if vApiKeyState === 'success'}
              {fTranslate('text_662efaf46c61')}
            {:else}
              {fTranslate('text_49673e62edaf')}
            {/if}
          </Button>
          <Button
            onclick={() => {
              vShowApiKeyInput = false;
              vApiKeyState = 'idle';
              vApiKeyError = '';
            }}
            variant="outline"
            class="flex-1"
            disabled={vApiKeyState === 'validating'}
          >
            {fTranslate('text_19766ed6ccb2')}
          </Button>
        </div>
      </div>
    {/if}

    {#if vShowRetry}
      <div in:fly={{ y: 10, duration: 300, delay: 200 }}>
        <Button onclick={fHandleRetryConnection} disabled={vIsServerLoading} class="w-full">
          {#if vIsServerLoading}
            <RefreshCw class="{cIconClassDefault} animate-spin" />
            {fTranslate('text_5f04ae9ed6a8')}
          {:else}
            <RefreshCw class={cIconClassDefault} /> {fTranslate('text_35f09f1991fb')}
          {/if}
        </Button>
      </div>
    {/if}

    {#if vShowTroubleshooting}
      <div class="mt-4 text-left" in:fly={{ y: 10, duration: 300, delay: 400 }}>
        <details class="text-sm">
          <summary class="cursor-pointer text-muted-foreground hover:text-foreground">
            {fTranslate('text_c3af076f92c5')}
          </summary>

          <div class="mt-2 space-y-3 text-xs text-muted-foreground">
            <div class="space-y-2">
              <p class="mb-4 font-medium">{fTranslate('text_d5420ce33622')}</p>

              <div class="rounded bg-muted/50 px-2 py-1 font-mono text-xs">
                <p>{fTranslate('text_177038b4d0bd')}</p>
              </div>

              <p>{fTranslate('text_7175517a370b')}</p>

              <div class="rounded bg-muted/50 px-2 py-1 font-mono text-xs">
                <p class="mt-1">{fTranslate('text_6b398631299a')}</p>
              </div>
            </div>
            <ul class="list-disc space-y-1 pl-4">
              <li>{fTranslate('text_4d326da4f4c4')}</li>

              <li>{fTranslate('text_f5cb1a5f2421')}</li>

              <li>{fTranslate('text_7b3d3f648257')}</li>
            </ul>
          </div>
        </details>
      </div>
    {/if}
  </div>
</div>
