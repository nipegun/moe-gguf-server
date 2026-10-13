<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { tick } from 'svelte';
  import * as Card from '$lib/components/ui/card';
  import { Skeleton } from '$lib/components/ui/skeleton';
  import type { MCPServerSettingsEntry, HealthCheckState } from '$lib/types';
  import { HealthCheckStatus } from '$lib/enums';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import {
    McpServerCardActions,
    McpServerCardDeleteDialog,
    McpServerCardEditForm,
    McpServerCardHeader,
    McpServerCardToolsList,
    McpConnectionLogs,
    McpServerInfo
  } from '$lib/components/app/mcp';

  interface Props {
    server: MCPServerSettingsEntry;
    enabled?: boolean;
    onToggle: (pEnabled: boolean) => void;
    onUpdate: (pUpdates: Partial<MCPServerSettingsEntry>) => void;
    onDelete: () => void;
  }

  let {
    server: vServer,
    enabled: vEnabled,
    onToggle: vOnToggle,
    onUpdate: vOnUpdate,
    onDelete: vOnDelete
  }: Props = $props();

  let vHealthState = $derived<HealthCheckState>(cMcpStore.mGetHealthCheckState(vServer.id));
  let vDisplayName = $derived(cMcpStore.mGetServerLabel(vServer));
  let vFaviconUrl = $derived(cMcpStore.mGetServerFavicon(vServer.id));
  let vIsIdle = $derived(vHealthState.status === HealthCheckStatus.IDLE);
  let vIsHealthChecking = $derived(vHealthState.status === HealthCheckStatus.CONNECTING);
  let vIsConnected = $derived(vHealthState.status === HealthCheckStatus.SUCCESS);
  let vIsError = $derived(vHealthState.status === HealthCheckStatus.ERROR);
  // Disabled servers stay IDLE (no startup health check), so the body
  // skeleton only applies while a check is running or expected to run.
  let vShowSkeleton = $derived(vIsHealthChecking || (vIsIdle && vServer.enabled));
  let vErrorMessage = $derived(
    vHealthState.status === HealthCheckStatus.ERROR ? vHealthState.message : undefined
  );
  let vTools = $derived(
    vHealthState.status === HealthCheckStatus.SUCCESS ? vHealthState.tools : []
  );

  let vConnectionLogs = $derived(
    vHealthState.status === HealthCheckStatus.CONNECTING ||
      vHealthState.status === HealthCheckStatus.SUCCESS ||
      vHealthState.status === HealthCheckStatus.ERROR
      ? vHealthState.logs
      : []
  );

  let vSuccessState = $derived(
    vHealthState.status === HealthCheckStatus.SUCCESS ? vHealthState : null
  );
  let vServerInfo = $derived(vSuccessState?.serverInfo);
  let vCapabilities = $derived(vSuccessState?.capabilities);
  let vTransportType = $derived(vSuccessState?.transportType);
  let vProtocolVersion = $derived(vSuccessState?.protocolVersion);
  let vConnectionTimeMs = $derived(vSuccessState?.connectionTimeMs);
  let vInstructions = $derived(vSuccessState?.instructions);

  let vIsEditing = $derived(!vServer.url.trim());
  let vShowDeleteDialog = $state(false);
  let vEditFormRef: McpServerCardEditForm | null = $state(null);

  function fHandleHealthCheck() {
    cMcpStore.mRunHealthCheck(vServer);
  }

  async function fStartEditing() {
    vIsEditing = true;
    await tick();
    vEditFormRef?.fSetInitialValues(
      vServer.url,
      vServer.headers || '',
      vServer.useProxy || false,
      vDisplayName
    );
  }

  function fCancelEditing() {
    if (vServer.url.trim()) {
      vIsEditing = false;
    } else {
      vOnDelete();
    }
  }

  function fSaveEditing(pUrl: string, pHeaders: string, pUseProxy: boolean, pName?: string) {
    vOnUpdate({
      url: pUrl,
      // undefined = prefill untouched, keep any existing custom name;
      // empty string = field cleared, back to the automatic label
      displayName: pName === undefined ? vServer.displayName : pName.trim() || undefined,
      headers: pHeaders || undefined,
      useProxy: pUseProxy
    });
    vIsEditing = false;

    if (vServer.enabled && pUrl) {
      setTimeout(
        () => cMcpStore.mRunHealthCheck({ ...vServer, url: pUrl, useProxy: pUseProxy }),
        100
      );
    }
  }

  function fHandleDeleteClick() {
    vShowDeleteDialog = true;
  }
</script>

<Card.Root class="!gap-3 bg-muted/30 p-4">
  {#if vIsEditing}
    <McpServerCardEditForm
      bind:this={vEditFormRef}
      serverId={vServer.id}
      serverUrl={vServer.url}
      serverUseProxy={vServer.useProxy}
      serverLabel={vDisplayName}
      onSave={fSaveEditing}
      onCancel={fCancelEditing}
    />
  {:else}
    <McpServerCardHeader
      displayName={vDisplayName}
      faviconUrl={vFaviconUrl}
      enabled={vEnabled ?? vServer.enabled}
      disabled={vIsError}
      onToggle={vOnToggle}
      serverInfo={vServerInfo}
      capabilities={vCapabilities}
      transportType={vTransportType}
    />

    {#if vIsError && vErrorMessage}
      <p class="text-xs text-destructive">{vErrorMessage}</p>
    {/if}

    {#if vIsConnected && vServerInfo?.description}
      <p class="line-clamp-2 text-xs text-muted-foreground">
        {vServerInfo.description}
      </p>
    {/if}

    <div class="grid gap-3">
      {#if vShowSkeleton}
        <div class="space-y-2">
          <div class="flex items-center gap-2">
            <Skeleton class="{cIconClassDefault} rounded" />
            <Skeleton class="h-3 w-24" />
          </div>
          <div class="flex flex-wrap gap-1.5">
            <Skeleton class="h-5 w-16 rounded-full" />
            <Skeleton class="h-5 w-20 rounded-full" />
            <Skeleton class="h-5 w-14 rounded-full" />
          </div>
        </div>

        <div class="space-y-1.5">
          <div class="flex items-center gap-2">
            <Skeleton class="{cIconClassDefault} rounded" />
            <Skeleton class="h-3 w-32" />
          </div>
        </div>
      {:else}
        {#if vIsConnected && vInstructions}
          <McpServerInfo instructions={vInstructions} />
        {/if}

        {#if vTools.length > 0}
          <McpServerCardToolsList tools={vTools} />
        {/if}

        {#if vConnectionLogs.length > 0}
          <McpConnectionLogs logs={vConnectionLogs} connectionTimeMs={vConnectionTimeMs} />
        {/if}
      {/if}
    </div>

    <div class="mt-auto flex justify-between gap-4">
      {#if vShowSkeleton}
        <Skeleton class="h-3 w-28" />
      {:else if vProtocolVersion}
        <div class="flex flex-wrap items-center gap-1">
          <span class="text-[10px] text-muted-foreground">
            {fTranslate('text_cd6999408a95')}
            {vProtocolVersion}
          </span>
        </div>
      {/if}

      <McpServerCardActions
        isHealthChecking={vIsHealthChecking}
        onEdit={fStartEditing}
        onRefresh={fHandleHealthCheck}
        onDelete={fHandleDeleteClick}
      />
    </div>
  {/if}
</Card.Root>

<McpServerCardDeleteDialog
  bind:open={vShowDeleteDialog}
  displayName={vDisplayName}
  onOpenChange={(open) => (vShowDeleteDialog = open)}
  onConfirm={vOnDelete}
/>
