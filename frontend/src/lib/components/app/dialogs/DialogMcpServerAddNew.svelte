<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { Button } from '$lib/components/ui/button';
  import * as Dialog from '$lib/components/ui/dialog';
  import { McpServerCardCompact, McpServerForm } from '$lib/components/app/mcp';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { cConversationsStore } from '$lib/stores/conversations.svelte';
  import { cUiSettingsStore } from '$lib/stores/ui-settings.svelte';
  import { parseHeadersToArray, uuid, canonicalizeServerUrl } from '$lib/utils';
  import {
    cBearerPrefix,
    cMcpServerIdPrefix,
    aRecommendedMcpServers,
    sRedactedHeaders
  } from '$lib/constants';
  import { HealthCheckStatus } from '$lib/enums';

  interface Props {
    open: boolean;
    onOpenChange?: (pOpen: boolean) => void;
  }

  let { open: vOpen = $bindable(), onOpenChange: vOnOpenChange }: Props = $props();

  let vNewServerUrl = $state('');
  let vNewServerName = $state('');
  let vNameAutoFilled = $state('');
  let vNameTouched = $state(false);

  let vPreviewRun = 0;

  function fHandleNameChange(pValue: string) {
    vNewServerName = pValue;
    vNameTouched = true;
  }
  let vNewServerHeaders = $state('');
  let vNewServerUseProxy = $state(false);

  let vNewServerWantsAuthorization = $state(false);

  let vSelectedRecommendationId = $derived.by(() => {
    const cUrl = vNewServerUrl.trim();
    if (!cUrl) return null;
    const cTargetCanonical = canonicalizeServerUrl(cUrl);
    return (
      aRecommendedMcpServers.find((pRec) => canonicalizeServerUrl(pRec.url) === cTargetCanonical)
        ?.id ?? null
    );
  });
  let vSelectedRecommendation = $derived(
    vSelectedRecommendationId
      ? (aRecommendedMcpServers.find((pRec) => pRec.id === vSelectedRecommendationId) ?? null)
      : null
  );
  let vAuthRequired = $derived(vSelectedRecommendation?.needsAuthorization ?? false);

  let vBearerTokenFilled = $derived.by(() => {
    const cPairs = parseHeadersToArray(vNewServerHeaders);
    const cNormalizedBearerPrefix = cBearerPrefix.toLowerCase();
    const cBearer = cPairs.find(
      (pP) =>
        sRedactedHeaders.has(pP.key.trim().toLowerCase()) &&
        pP.value.trim().toLowerCase().startsWith(cNormalizedBearerPrefix)
    );

    if (!cBearer) return false;

    return cBearer.value.trim().slice(cNormalizedBearerPrefix.length).trim().length > 0;
  });

  let vNewServerUrlError = $derived.by(() => {
    if (!vNewServerUrl.trim()) return 'URL is required';
    try {
      new URL(vNewServerUrl);

      return null;
    } catch {
      return fTranslate('text_fb0564517178');
    }
  });
  let vNewServerHeaderPairsValid = $derived(
    parseHeadersToArray(vNewServerHeaders).every((pP) => pP.key.trim() && pP.value.trim())
  );
  let vCanSave = $derived(
    !vNewServerUrlError && vNewServerHeaderPairsValid && (!vAuthRequired || vBearerTokenFilled)
  );

  // kept in the server's settings document (section "dismissedRecommendedMcpServers")
  function fReadRecommendationsDismissed(): boolean {
    return cUiSettingsStore.mGet('dismissedRecommendedMcpServers') === true;
  }

  function fWriteRecommendationsDismissed(pDismissed: boolean) {
    vRecommendationsDismissed = pDismissed;
    cUiSettingsStore.mSet('dismissedRecommendedMcpServers', pDismissed);
  }

  let vRecommendationsDismissed = $state<boolean>(fReadRecommendationsDismissed());

  // Read-only once a recommendation is picked: switch is disabled, so we keep
  // the Authorization field in sync with the requirement.
  $effect(() => {
    if (vAuthRequired) {
      vNewServerWantsAuthorization = true;
    }
  });

  // Debounced preview handshake: once the URL is valid and stable, fetch the
  // server-reported name to prefill the display name field. A manual edit
  // freezes the autofill for good, and failures stay silent.
  $effect(() => {
    const cUrl = vNewServerUrl.trim();
    const cHeaders = vNewServerHeaders.trim();
    const cUseProxy = vNewServerUseProxy;

    if (!vOpen || vNewServerUrlError || !cUrl) return;

    const cRun = ++vPreviewRun;
    // One throwaway id per run: concurrent previews (URL typed, then the
    // bearer token pasted) would poison each other's shared health state.
    const cPreviewId = `${cMcpServerIdPrefix}-preview-${cRun}`;
    const cTimer = setTimeout(async () => {
      await cMcpStore.mRunHealthCheck({
        id: cPreviewId,
        enabled: false,
        url: cUrl,
        headers: cHeaders || undefined,
        useProxy: cUseProxy
      });

      const cState = cMcpStore.mGetHealthCheckState(cPreviewId);

      cMcpStore.mClearHealthCheck(cPreviewId);

      if (cRun !== vPreviewRun) return;

      if (cState.status !== HealthCheckStatus.SUCCESS) return;

      const cAutoName = cState.serverInfo?.title || cState.serverInfo?.name || '';

      if (cAutoName && !vNameTouched) {
        vNewServerName = cAutoName;
        vNameAutoFilled = cAutoName;
      }
    }, 600);

    return () => clearTimeout(cTimer);
  });

  let vHasSelection = $derived(vSelectedRecommendationId !== null);

  let vUnconfiguredRecommendations = $derived.by(() => {
    const sConfiguredCanonicals = new Set(
      cMcpStore.mGetServers().map((pS) => canonicalizeServerUrl(pS.url))
    );

    return aRecommendedMcpServers.filter(
      (pRec) => !sConfiguredCanonicals.has(canonicalizeServerUrl(pRec.url))
    );
  });

  let vRecommendationsToShow = $derived(
    vRecommendationsDismissed ? [] : vUnconfiguredRecommendations
  );

  function fHandleRecommendationClick(pRecommendedId: string) {
    const cRecommendation = aRecommendedMcpServers.find((pRec) => pRec.id === pRecommendedId);

    if (!cRecommendation) return;

    vNewServerUrl = cRecommendation.url;
    vNewServerHeaders = '';
    vNewServerWantsAuthorization = cRecommendation.needsAuthorization ?? false;
  }

  function fHandleDismissAll() {
    fWriteRecommendationsDismissed(true);
  }

  function fHandleOpenChange(pValue: boolean) {
    if (!pValue) {
      vNewServerUrl = '';
      vNewServerName = '';
      vNameAutoFilled = '';
      vNameTouched = false;
      vPreviewRun++;
      vNewServerHeaders = '';
      vNewServerUseProxy = false;
      vNewServerWantsAuthorization = false;
    }
    vOpen = pValue;
    vOnOpenChange?.(pValue);
  }

  function fSaveNewServer() {
    if (!vCanSave) return;

    const cNewServerId = uuid() ?? `${cMcpServerIdPrefix}-${Date.now()}`;

    cMcpStore.mAddServer({
      id: cNewServerId,
      enabled: true,
      url: vNewServerUrl.trim(),
      // A name equal to the autofilled server-reported one is not a
      // customization: keep following the automatic label.
      displayName:
        vNewServerName.trim() && vNewServerName.trim() !== vNameAutoFilled.trim()
          ? vNewServerName.trim()
          : undefined,
      headers: vNewServerHeaders.trim() || undefined,
      useProxy: vNewServerUseProxy
    });

    cConversationsStore.mSetMcpServerOverride(cNewServerId, true);

    fHandleOpenChange(false);
  }

  function fHandleSubmit(pEvent: SubmitEvent) {
    pEvent.preventDefault();
    fSaveNewServer();
  }
</script>

<Dialog.Root open={vOpen} onOpenChange={fHandleOpenChange}>
  <Dialog.Content class="sm:max-w-2xl">
    <Dialog.Header>
      <Dialog.Title class="select-none">{fTranslate('text_143882ba828c')}</Dialog.Title>
    </Dialog.Header>

    {#if vRecommendationsToShow.length > 0}
      <div class="space-y-3 pt-2">
        <div class="flex items-center justify-between gap-3">
          <h3 class="text-sm font-medium">{fTranslate('text_17bdcbc59576')}</h3>
          <Button
            class="text-muted-foreground"
            variant="ghost"
            size="sm"
            onclick={fHandleDismissAll}>{fTranslate('text_48845bff334a')}</Button
          >
        </div>

        <div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {#each vRecommendationsToShow as recommendation (recommendation.id)}
            <McpServerCardCompact
              server={recommendation}
              onClick={() => fHandleRecommendationClick(recommendation.id)}
              selected={vSelectedRecommendationId === recommendation.id}
              dimmed={vHasSelection && vSelectedRecommendationId !== recommendation.id}
            />
          {/each}
        </div>
      </div>
    {/if}

    <form onsubmit={fHandleSubmit} class="contents">
      <div class="space-y-4 py-4">
        <McpServerForm
          url={vNewServerUrl}
          name={vNewServerName}
          onNameChange={fHandleNameChange}
          headers={vNewServerHeaders}
          useProxy={vNewServerUseProxy}
          onUrlChange={(v) => (vNewServerUrl = v)}
          onHeadersChange={(v) => (vNewServerHeaders = v)}
          onUseProxyChange={(v) => (vNewServerUseProxy = v)}
          urlError={vNewServerUrl ? vNewServerUrlError : null}
          id="new-server"
          bind:wantsAuthorization={vNewServerWantsAuthorization}
          required={vAuthRequired}
        />
      </div>

      <Dialog.Footer>
        <Button variant="secondary" size="sm" onclick={() => fHandleOpenChange(false)}>
          {fTranslate('text_19766ed6ccb2')}
        </Button>

        <Button
          variant="default"
          size="sm"
          type="submit"
          disabled={!vCanSave}
          aria-label={fTranslate('text_1509f561f241')}
        >
          {fTranslate('text_9fd728c66c9a')}
        </Button>
      </Dialog.Footer>
    </form>
  </Dialog.Content>
</Dialog.Root>
