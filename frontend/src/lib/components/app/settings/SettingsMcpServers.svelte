<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { X, Plus } from '@lucide/svelte';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { cConversationsStore } from '$lib/stores/conversations.svelte';
  import { cToolsStore } from '$lib/stores/tools.svelte';
  import { Button } from '$lib/components/ui/button';
  import * as Empty from '$lib/components/ui/empty';
  import { ActionIcon, McpServerCard, McpServerCardSkeleton } from '$lib/components/app';
  import { DialogMcpServerAddNew } from '$lib/components/app/dialogs';
  import { HealthCheckStatus } from '$lib/enums';
  import { cRoutes } from '$lib/constants';
  import { fade } from 'svelte/transition';
  import { onMount } from 'svelte';
  import McpLogo from '../mcp/McpLogo.svelte';
  import { browser } from '$app/environment';
  import { page } from '$app/state';
  import { goto, replaceState } from '$app/navigation';

  interface Props {
    class?: string;
  }

  let { class: vClassName }: Props = $props();

  let vServers = $derived(cMcpStore.mGetServers());

  let vIsAddingServer = $state(false);

  let vPreviousRouteId = $state<string | null>(null);

  $effect(() => {
    const cCurrentId = page.route.id;
    return () => {
      vPreviousRouteId = cCurrentId;
    };
  });

  function fHandleClose() {
    const cPrevIsMcpServers = vPreviousRouteId === '/mcp-servers';
    if (browser && window.history.length > 1 && !cPrevIsMcpServers) {
      history.back();
    } else {
      goto(cRoutes.START);
    }
  }

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

<div in:fade={{ duration: 150 }} class="flex min-h-[calc(100dvh-4rem)] flex-col">
  <div class="fixed top-4.5 right-4 z-50 md:hidden">
    <ActionIcon icon={X} tooltip={fTranslate('text_7d9eb7acb13e')} onclick={fHandleClose} />
  </div>

  <div
    class="sticky top-0 z-10 mt-4 mb-2 flex items-start gap-4 md:p-4 p-0 px-4 md:justify-between md:px-8"
  >
    <div class="flex items-center gap-2">
      <McpLogo class="h-5 w-5 md:h-6 md:w-6" />

      <h1 class="text-lg font-semibold md:text-2xl">{fTranslate('text_aab669a72882')}</h1>
    </div>
  </div>

  <DialogMcpServerAddNew bind:open={vIsAddingServer} />

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
    <div
      class="grid gap-3 {vClassName}"
      style="grid-template-columns: repeat(auto-fill, minmax(min(32rem, calc(100dvw - 2rem)), 1fr));"
    >
      {#each vServers as server (server.id)}
        {#if fIsServerPending(server.id, server.enabled)}
          <McpServerCardSkeleton />
        {:else}
          <McpServerCard
            {server}
            enabled={cConversationsStore.mIsMcpServerEnabledForChat(server.id)}
            onToggle={async () => {
              const wasEnabled = cConversationsStore.mIsMcpServerEnabledForChat(server.id);
              await cConversationsStore.mToggleMcpServerForChat(server.id);
              if (!wasEnabled) {
                // Promote the connection so tools/prompts/resources become
                // available right away instead of waiting for the next chat-init.
                await cMcpStore.mRunHealthCheck(server, true);
                cToolsStore.mEnableAllToolsForServer(server.id);
              }
            }}
            onUpdate={(updates) => cMcpStore.mUpdateServer(server.id, updates)}
            onDelete={() => cMcpStore.mRemoveServer(server.id)}
          />
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
