<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { Settings, Plus } from '@lucide/svelte';
  import { Switch } from '$lib/components/ui/switch';
  import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
  import { McpLogo, DropdownMenuSearchable, McpServerIdentity } from '$lib/components/app';
  import { cConversationsStore } from '$lib/stores/conversations.svelte';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { HealthCheckStatus } from '$lib/enums';
  import type { MCPServerSettingsEntry } from '$lib/types';
  import { goto } from '$app/navigation';
  import { cRoutes } from '$lib/constants/routes';

  interface Props {
    onMcpSettingsClick?: () => void;
  }

  let { onMcpSettingsClick: vOnMcpSettingsClick }: Props = $props();

  let vMcpSearchQuery = $state('');
  // Every configured server is listed; `enabled` is an on/off state,
  // not a visibility filter, so a disabled server stays toggleable.
  let vMcpServers = $derived(cMcpStore.mGetServers());
  let vHasMcpServers = $derived(vMcpServers.length > 0);
  let vFilteredMcpServers = $derived.by(() => {
    const cQuery = vMcpSearchQuery.toLowerCase().trim();
    if (!cQuery) return vMcpServers;
    return vMcpServers.filter((pS) => {
      const cName = fGetServerLabel(pS).toLowerCase();
      const cUrl = pS.url.toLowerCase();
      return cName.includes(cQuery) || cUrl.includes(cQuery);
    });
  });

  function fGetServerLabel(pServer: MCPServerSettingsEntry): string {
    return cMcpStore.mGetServerLabel(pServer);
  }

  function fIsServerEnabledForChat(pServerId: string): boolean {
    return cConversationsStore.mIsMcpServerEnabledForChat(pServerId);
  }

  async function fToggleServerForChat(pServerId: string) {
    await cConversationsStore.mToggleMcpServerForChat(pServerId);
  }

  function fHandleMcpSubMenuOpen(pOpen: boolean) {
    if (pOpen) {
      vMcpSearchQuery = '';
      cMcpStore.mRunHealthChecksForServers(vMcpServers);
    }
  }

  function fHandleMcpSettingsClick() {
    vOnMcpSettingsClick?.();

    goto(`${vHasMcpServers ? '' : '?add'}${cRoutes.MCP_SERVERS}`);
  }
</script>

<DropdownMenu.Root>
  <DropdownMenu.Sub onOpenChange={fHandleMcpSubMenuOpen}>
    <DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
      <McpLogo class={cIconClassDefault} />

      <span>{fTranslate('text_aab669a72882')}</span>
    </DropdownMenu.SubTrigger>

    <DropdownMenu.SubContent class="w-72 pt-0">
      {#if vHasMcpServers}
        <DropdownMenuSearchable
          placeholder={fTranslate('text_10ae62d666e9')}
          bind:searchValue={vMcpSearchQuery}
          emptyMessage={fTranslate('text_71e873a7b235')}
          isEmpty={vFilteredMcpServers.length === 0}
        >
          <div class="max-h-64 overflow-y-auto">
            {#each vFilteredMcpServers as server (server.id)}
              {@const healthState = cMcpStore.mGetHealthCheckState(server.id)}
              {@const hasError = healthState.status === HealthCheckStatus.ERROR}
              {@const isEnabledForChat = fIsServerEnabledForChat(server.id)}
              {@const displayName = fGetServerLabel(server)}
              {@const faviconUrl = cMcpStore.mGetServerFavicon(server.id)}

              <button
                type="button"
                class="flex w-full items-center justify-between gap-2 rounded-sm px-2 py-2 text-left transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
                onclick={() => !hasError && fToggleServerForChat(server.id)}
                disabled={hasError}
              >
                <div class="flex min-w-0 flex-1 items-center gap-2">
                  <div class="min-w-0 flex-1">
                    <McpServerIdentity
                      {displayName}
                      {faviconUrl}
                      iconClass={cIconClassDefault}
                      iconRounded="rounded-sm"
                      showVersion={false}
                      nameClass="text-sm"
                    />
                  </div>

                  {#if hasError}
                    <span
                      class="shrink-0 rounded bg-destructive/15 px-1.5 py-0.5 text-xs text-destructive"
                    >
                      {fTranslate('text_54a0e8c17ebb')}
                    </span>
                  {/if}
                </div>

                <Switch
                  checked={isEnabledForChat}
                  disabled={hasError}
                  onclick={(e) => e.stopPropagation()}
                  onCheckedChange={() => fToggleServerForChat(server.id)}
                />
              </button>
            {/each}
          </div>

          {#snippet footer()}
            <DropdownMenu.Item
              class="flex cursor-pointer items-center gap-2"
              onclick={fHandleMcpSettingsClick}
            >
              <Settings class={cIconClassDefault} />

              <span>{fTranslate('text_1a7cd357096b')}</span>
            </DropdownMenu.Item>
          {/snippet}
        </DropdownMenuSearchable>
      {:else}
        <div class="px-2 py-3 text-center text-sm text-muted-foreground">
          {fTranslate('text_013e6b9db9d2')}
        </div>

        <DropdownMenu.Separator />

        <DropdownMenu.Item
          class="flex cursor-pointer items-center gap-2"
          onclick={fHandleMcpSettingsClick}
        >
          <Plus class={cIconClassDefault} />

          <span>{fTranslate('text_f33f83f4e0ed')}</span>
        </DropdownMenu.Item>
      {/if}
    </DropdownMenu.SubContent>
  </DropdownMenu.Sub>
</DropdownMenu.Root>
