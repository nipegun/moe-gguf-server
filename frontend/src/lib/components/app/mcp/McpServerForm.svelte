<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { Input } from '$lib/components/ui/input';
  import { Switch } from '$lib/components/ui/switch';
  import { KeyValuePairs } from '$lib/components/app';
  import type { KeyValuePair } from '$lib/types';
  import { parseHeadersToArray, serializeHeaders } from '$lib/utils';
  import { UrlProtocol } from '$lib/enums';
  import {
    cAuthorizationHeader,
    cBearerPrefix,
    cCliFlags,
    cMcpServerUrlPlaceholder,
    sRedactedHeaders
  } from '$lib/constants';
  import { cMcpStore } from '$lib/stores/mcp.svelte';

  interface Props {
    url: string;
    headers: string;
    name?: string;
    onNameChange?: (pName: string) => void;
    /** Shown in the empty display name field, e.g. the current automatic label. */
    namePlaceholder?: string;
    useProxy?: boolean;
    onUrlChange: (pUrl: string) => void;
    onHeadersChange: (pHeaders: string) => void;
    onUseProxyChange?: (pUseProxy: boolean) => void;
    urlError?: string | null;
    id?: string;
    /**
     * "Wants Authorization" is the user's *intent* to add a Bearer token
     * (separate from `hasAuthorization` which reflects what's already in
     * the headers). Bindable so a parent - e.g. the recommendation cards
     * on the "Add New Server" dialog - can flip the switch on when the
     * picked server ships a `needsAuthorization: true` flag.
     */
    wantsAuthorization?: boolean;
    /**
     * Marks the "Authorization" field as required. Locks the toggle so the
     * user can't dismiss it, and visually marks the field with a red
     * asterisk. The parent is expected to gate its submit affordance on
     * the bearer token actually being filled. Used by the "Add New Server"
     * dialog for recommendations whose `needsAuthorization` flag is true.
     */
    required?: boolean;
  }

  let {
    url: vUrl,
    headers: vHeaders,
    name: vName = '',
    onNameChange: vOnNameChange,
    namePlaceholder: vNamePlaceholder = fTranslate('text_680326dbf89f'),
    useProxy: vUseProxy = false,
    onUrlChange: vOnUrlChange,
    onHeadersChange: vOnHeadersChange,
    onUseProxyChange: vOnUseProxyChange,
    urlError: vUrlError = null,
    id: vId = 'server',
    wantsAuthorization: vWantsAuthorization = $bindable(false),
    required: vRequired = false
  }: Props = $props();

  let vIsWebSocket = $derived(
    vUrl.toLowerCase().startsWith(UrlProtocol.WEBSOCKET) ||
      vUrl.toLowerCase().startsWith(UrlProtocol.WEBSOCKET_SECURE)
  );

  let vHeaderPairs = $derived<KeyValuePair[]>(parseHeadersToArray(vHeaders));

  // Heuristic: this dedicated UI only owns Authorization headers that already
  // carry a Bearer scheme. Anything else (e.g. Basic, raw tokens) stays in the
  // KV section so the user can still edit those values verbatim.
  const fMatchesAuthorizationKey = (pKey: string): boolean =>
    sRedactedHeaders.has(pKey.trim().toLowerCase());

  const fIsBearerScheme = (pValue: string): boolean =>
    pValue.trim().toLowerCase().startsWith(cBearerPrefix.toLowerCase());

  const fOwnedByBearerUi = (pP: KeyValuePair): boolean =>
    fMatchesAuthorizationKey(pP.key) && fIsBearerScheme(pP.value);

  let vHasAuthorization = $derived(vHeaderPairs.some(fOwnedByBearerUi));

  let vShowAuthorization = $derived(vHasAuthorization || vWantsAuthorization);

  let vUrlInput: HTMLInputElement | null = $state(null);
  let vBearerInput: HTMLInputElement | null = $state(null);

  $effect(() => {
    vUrlInput?.focus();
  });

  $effect(() => {
    if (vWantsAuthorization && vBearerInput) {
      vBearerInput.focus();
    }
  });

  let vBearerToken = $derived.by(() => {
    const cAuth = vHeaderPairs.find(fOwnedByBearerUi);
    if (!cAuth) return '';
    return cAuth.value.trim().slice(cBearerPrefix.length).trim();
  });

  $effect(() => {
    if (!vHeaders.trim()) {
      vWantsAuthorization = false;
    }
  });

  function fUpdateHeaderPairs(pNewPairs: KeyValuePair[]) {
    vHeaderPairs = pNewPairs;
    vOnHeadersChange(serializeHeaders(pNewPairs));
  }

  // The dedicated UI owns the Authorization slot end-to-end when the user
  // engages it: any prior Authorization row (Bearer or otherwise) is replaced
  // by exactly one { Authorization: "Bearer <token>" } entry. JSON's last-key
  // behavior would otherwise pick one arbitrarily, so we strip first.
  function fUpdateBearerToken(pToken: string) {
    const cFiltered = vHeaderPairs.filter((pP) => !fMatchesAuthorizationKey(pP.key));

    const cTrimmed = pToken.trim();

    if (cTrimmed) {
      cFiltered.push({ key: cAuthorizationHeader, value: `${cBearerPrefix}${cTrimmed}` });
    }

    fUpdateHeaderPairs(cFiltered);
  }

  function fSetUseAuthorization(pChecked: boolean) {
    vWantsAuthorization = pChecked;

    if (!pChecked) {
      // Only drop the entry this UI owns; a non-Bearer Authorization row
      // authored in the KV section must survive a toggle off untouched.
      const cFiltered = vHeaderPairs.filter((pP) => !fOwnedByBearerUi(pP));
      fUpdateHeaderPairs(cFiltered);
    }
  }
</script>

<div class="grid gap-2">
  <div class="mb-4">
    <label for="server-url-{vId}" class="mb-2 block text-xs font-medium select-none">
      {fTranslate('text_22f5ebc3ad3f')} <span class="text-destructive">*</span>
    </label>

    <Input
      id="server-url-{vId}"
      type="url"
      placeholder={cMcpServerUrlPlaceholder}
      value={vUrl}
      oninput={(e) => vOnUrlChange(e.currentTarget.value)}
      class={vUrlError ? 'border-destructive' : ''}
      bind:ref={vUrlInput}
    />

    {#if vUrlError}
      <p class="mt-1.5 text-xs text-destructive">{vUrlError}</p>
    {/if}
  </div>

  <div class="mb-4">
    <label for="server-name-{vId}" class="mb-2 block text-xs font-medium select-none">
      {fTranslate('text_2b7f6a84de91')}
    </label>

    <Input
      id="server-name-{vId}"
      type="text"
      placeholder={vNamePlaceholder}
      value={vName}
      oninput={(e) => vOnNameChange?.(e.currentTarget.value)}
    />
  </div>

  <label class="flex items-center gap-2 cursor-pointer select-none">
    <Switch
      id="use-authorization-{vId}"
      checked={vShowAuthorization}
      onCheckedChange={fSetUseAuthorization}
      disabled={vRequired}
    />

    <span class="text-xs text-muted-foreground">
      {fTranslate('text_ca5839e38a15')}{#if vRequired}
        <span class="text-destructive">*</span>{/if}
    </span>
  </label>

  {#if vShowAuthorization}
    <div class="relative mt-2">
      <Input
        id="bearer-token-{vId}"
        type="password"
        autocomplete="off"
        placeholder={fTranslate('text_4a1c9cb993d2')}
        value={vBearerToken}
        oninput={(e) => fUpdateBearerToken(e.currentTarget.value)}
        class="pl-16"
        bind:ref={vBearerInput}
      />

      <span
        class="pointer-events-none absolute inset-y-0 left-3 flex items-center text-sm font-medium text-foreground"
      >
        {fTranslate('text_710e0dbdd422')}
      </span>
    </div>
  {/if}

  <KeyValuePairs
    class="mt-3"
    pairs={vHeaderPairs.filter((p) => !fOwnedByBearerUi(p))}
    onPairsChange={(pairs) => {
      const auth = vHeaderPairs.find(fOwnedByBearerUi);
      fUpdateHeaderPairs(auth ? [...pairs, auth] : pairs);
    }}
    keyPlaceholder="Header name"
    valuePlaceholder="Value"
    addButtonLabel="Add"
    emptyMessage={fTranslate('text_ce538952aa31')}
    sectionLabel="Custom Headers"
    sectionLabelOptional
  />

  {#if !vIsWebSocket && vOnUseProxyChange}
    <label
      class={[
        'mt-3 flex items-start gap-2',
        cMcpStore.mIsProxyAvailable && 'cursor-pointer',
        !cMcpStore.mIsProxyAvailable && 'opacity-80'
      ]}
    >
      <Switch
        class="mt-1"
        id="use-proxy-{vId}"
        checked={vUseProxy}
        disabled={!cMcpStore.mIsProxyAvailable}
        onCheckedChange={(checked) => vOnUseProxyChange?.(checked)}
      />

      <span>
        <span class="text-xs text-muted-foreground">{fTranslate('text_a3831acb8899')}</span>

        <br />

        {#if !cMcpStore.mIsProxyAvailable}
          <span class="inline-flex gap-0.75 text-xs text-muted-foreground/60"
            >{fTranslate('text_2f7f3e22ea61')}
            <pre>mgs</pre>
            {fTranslate('text_0695b563acde')}
            <pre>{cCliFlags.MCP_PROXY}</pre>
            {fTranslate('text_f9921b14d171')}</span
          >
        {/if}
      </span>
    </label>
  {/if}
</div>
