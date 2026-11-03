<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { Plus, SquareTerminal } from '@lucide/svelte';
  import { toast } from 'svelte-sonner';
  import { Switch } from '$lib/components/ui/switch';
  import { PropsService } from '$lib/services/props.service';
  import { cServerStore } from '$lib/stores/server.svelte';
  import { cUiSettingsStore } from '$lib/stores/ui-settings.svelte';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { cConversationsStore } from '$lib/stores/conversations.svelte';
  import { cToolsStore } from '$lib/stores/tools.svelte';
  import { Button } from '$lib/components/ui/button';
  import * as Empty from '$lib/components/ui/empty';
  import { McpServerCard, McpServerCardSkeleton } from '$lib/components/app';
  import { DialogMcpServerAddNew } from '$lib/components/app/dialogs';
  import { HealthCheckStatus } from '$lib/enums';
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { replaceState } from '$app/navigation';

  interface Props {
    class?: string;
  }

  let { class: vClassName }: Props = $props();

  let vServers = $derived(cMcpStore.mGetServers());

  let vIsAddingServer = $state(false);

  // the local shell MCP server of mgs: turned on with --local-shell-mcp-server or here
  let vShellOn = $derived(Boolean(cServerStore.vProps?.shell_mcp_server));
  let vShellForced = $derived(Boolean(cServerStore.vProps?.shell_mcp_server_forced));
  let vShellToggle = $derived(Boolean(cServerStore.vProps?.shell_mcp_server_toggle));
  let vShellBusy = $state(false);

  async function fToggleShellServer(pEnabled: boolean) {
    vShellBusy = true;
    try {
      const cState = await PropsService.mSetShellMcpServer(pEnabled);
      // the server has already saved it; the copy of the document must hold the same value, or the
      // next save of the settings would bring the previous state back
      cUiSettingsStore.mSet('localShellMcpServer', cState.shell_mcp_server ? true : undefined);
      // the new /props adds the server to the list, or removes it (mSyncShellServer)
      await cServerStore.mFetch({ background: true });
    } catch (vError) {
      toast.error(vError instanceof Error ? vError.message : String(vError));
    } finally {
      vShellBusy = false;
    }
  }

  // ?add (the MCP menu of the chat when there is no server yet) opens the dialog to add one
  onMount(() => {
    if (page.url.searchParams.has('add')) {
      vIsAddingServer = true;

      const cNewUrl = new URL(page.url);
      cNewUrl.searchParams.delete('add');

      replaceState(cNewUrl, {});
    }
  });

  // Each card decides for itself whether to render based on its own
  // health-check state, so adding a server only flashes the new card
  // (not every other already-loaded card) until its health check resolves.
  // Disabled servers never receive a startup health check, so IDLE only
  // counts as pending when the server is enabled; otherwise the real card
  // renders and keeps the enable toggle reachable.
  function fIsServerPending(pServerId: string, pEnabled: boolean): boolean {
    const cStatus = cMcpStore.mGetHealthCheckState(pServerId).status;
    return (
      cStatus === HealthCheckStatus.CONNECTING || (cStatus === HealthCheckStatus.IDLE && pEnabled)
    );
  }
</script>

<!-- Settings > MCP servers: the section shows the title, so this is only its content -->
<div class="flex flex-col gap-6">
  <DialogMcpServerAddNew bind:open={vIsAddingServer} />

  <!-- the global search highlights this switch by its id -->
  <section
    id="settings-local-shell-mcp-server"
    class="flex items-start gap-3 rounded-xl border border-border/50 p-4"
  >
    <SquareTerminal class="mt-0.5 h-5 w-5 shrink-0 text-muted-foreground" />

    <div class="min-w-0 flex-1 space-y-1">
      <label for="local-shell-mcp-server" class="text-sm font-medium"
        >{fTranslate('text_a65fc168a110')}</label
      >

      <p class="text-xs text-muted-foreground">{fTranslate('text_42f4a2baa426')}</p>

      {#if vShellForced}
        <p class="text-xs text-muted-foreground">{fTranslate('text_df64dfdaf5d6')}</p>
      {:else if !vShellToggle}
        <p class="text-xs text-muted-foreground">{fTranslate('text_9297a038237b')}</p>
      {/if}
    </div>

    <Switch
      id="local-shell-mcp-server"
      checked={vShellOn}
      disabled={vShellBusy || vShellForced || !vShellToggle}
      onCheckedChange={fToggleShellServer}
    />
  </section>

  {#if vServers.length === 0}
    <div class="flex flex-1 items-center justify-center py-16">
      <Empty.Root class="max-w-md">
        <Empty.Header>
          <Empty.Media variant="icon">
            <Plus />
          </Empty.Media>

          <Empty.Title>{fTranslate('text_b2b642a46b33')}</Empty.Title>

          <Empty.Description>{fTranslate('text_341e549ff92d')}</Empty.Description>
        </Empty.Header>

        <Empty.Content>
          <Button size="sm" onclick={() => (vIsAddingServer = true)}>
            <Plus />
            {fTranslate('text_554d104442c2')}
          </Button>
        </Empty.Content>
      </Empty.Root>
    </div>
  {:else}
    <div class="grid gap-3 {vClassName}">
      {#each vServers as vServer (vServer.id)}
        {#if fIsServerPending(vServer.id, vServer.enabled)}
          <McpServerCardSkeleton />
        {:else}
          <!-- the global search highlights the server by this id -->
          <div id="mcp-server-{vServer.id}" class="rounded-xl">
            <McpServerCard
              server={vServer}
              enabled={cConversationsStore.mIsMcpServerEnabledForChat(vServer.id)}
              onToggle={async () => {
                const cWasEnabled = cConversationsStore.mIsMcpServerEnabledForChat(vServer.id);
                await cConversationsStore.mToggleMcpServerForChat(vServer.id);
                if (!cWasEnabled) {
                  // Promote the connection so tools/prompts/resources become
                  // available right away instead of waiting for the next chat-init.
                  await cMcpStore.mRunHealthCheck(vServer, true);
                  cToolsStore.mEnableAllToolsForServer(vServer.id);
                }
              }}
              onUpdate={(pUpdates) => cMcpStore.mUpdateServer(vServer.id, pUpdates)}
              onDelete={() => cMcpStore.mRemoveServer(vServer.id)}
            />
          </div>
        {/if}
      {/each}

      {#if !vIsAddingServer}
        <Empty.Root class="border">
          <Empty.Header>
            <Empty.Media variant="icon">
              <Plus />
            </Empty.Media>

            <Empty.Title>{fTranslate('text_857b19e08d14')}</Empty.Title>

            <Empty.Description>{fTranslate('text_341e549ff92d')}</Empty.Description>
          </Empty.Header>

          <Empty.Content>
            <Button size="sm" onclick={() => (vIsAddingServer = true)}>
              <Plus />
              {fTranslate('text_554d104442c2')}
            </Button>
          </Empty.Content>
        </Empty.Root>
      {/if}
    </div>
  {/if}
</div>
