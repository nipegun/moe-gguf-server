<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { Button } from '$lib/components/ui/button';
  import { McpServerForm } from '$lib/components/app/mcp';
  import { parseHeadersToArray } from '$lib/utils';

  interface Props {
    serverId: string;
    serverUrl: string;
    serverUseProxy?: boolean;
    /** Current automatic label, prefilled so the user can customize it. */
    serverLabel?: string;
    onSave: (pUrl: string, pHeaders: string, pUseProxy: boolean, pName?: string) => void;
    onCancel: () => void;
  }

  let {
    serverId: vServerId,
    serverUrl: vServerUrl,
    serverUseProxy: vServerUseProxy = false,
    serverLabel: vServerLabel = '',
    onSave: vOnSave,
    onCancel: vOnCancel
  }: Props = $props();

  let vEditUrl = $derived(vServerUrl);
  let vEditName = $derived(vServerLabel);
  let vEditHeaders = $state('');
  let vEditUseProxy = $derived(vServerUseProxy);

  let vUrlError = $derived.by(() => {
    if (!vEditUrl.trim()) return 'URL is required';
    try {
      new URL(vEditUrl);
      return null;
    } catch {
      return fTranslate('text_fb0564517178');
    }
  });

  let vHeaderPairsValid = $derived(
    parseHeadersToArray(vEditHeaders).every((pP) => pP.key.trim() && pP.value.trim())
  );
  let vCanSave = $derived(!vUrlError && vHeaderPairsValid);

  function fHandleSave() {
    if (!vCanSave) return;

    // An unchanged prefill keeps following the automatic label; only an
    // actual edit becomes a persisted custom display name.
    const cName = vEditName.trim() !== vServerLabel.trim() ? vEditName.trim() : undefined;

    vOnSave(vEditUrl.trim(), vEditHeaders.trim(), vEditUseProxy, cName);
  }

  function fHandleSubmit(pEvent: SubmitEvent) {
    pEvent.preventDefault();
    fHandleSave();
  }

  export function fSetInitialValues(
    pUrl: string,
    pHeaders: string,
    pUseProxy: boolean,
    pName = ''
  ) {
    vEditUrl = pUrl;
    vEditHeaders = pHeaders;
    vEditUseProxy = pUseProxy;
    vEditName = pName;
  }
</script>

<form onsubmit={fHandleSubmit} class="contents">
  <div class="space-y-4">
    <p class="font-medium">{fTranslate('text_b86b2fbec250')}</p>

    <McpServerForm
      url={vEditUrl}
      name={vEditName}
      onNameChange={(v) => (vEditName = v)}
      headers={vEditHeaders}
      useProxy={vEditUseProxy}
      onUrlChange={(v) => (vEditUrl = v)}
      onHeadersChange={(v) => (vEditHeaders = v)}
      onUseProxyChange={(v) => (vEditUseProxy = v)}
      urlError={vEditUrl ? vUrlError : null}
      id={vServerId}
    />

    <div class="flex items-center justify-end gap-2">
      <Button variant="secondary" size="sm" onclick={vOnCancel}
        >{fTranslate('text_19766ed6ccb2')}</Button
      >

      <Button size="sm" type="submit" disabled={!vCanSave}>
        {vServerUrl.trim() ? fTranslate('text_c1c1009d3f37') : fTranslate('text_9fd728c66c9a')}
      </Button>
    </div>
  </div>
</form>
