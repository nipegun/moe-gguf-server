<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cConversationsStore } from '$lib/stores/conversations.svelte';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { HealthCheckStatus } from '$lib/enums';
  import { cMaxDisplayedMcpAvatars } from '$lib/constants';
  import McpLogo from './McpLogo.svelte';

  interface Props {
    class?: string;
    onclick?: () => void;
  }

  let { class: vClassName = '', onclick: vOnclick }: Props = $props();

  let vMcpServers = $derived(cMcpStore.mGetServers().filter((pS) => pS.enabled));
  let vEnabledMcpServersForChat = $derived(
    vMcpServers.filter(
      (pS) => cConversationsStore.mIsMcpServerEnabledForChat(pS.id) && pS.url.trim()
    )
  );
  let vHealthyEnabledMcpServers = $derived(
    vEnabledMcpServersForChat.filter((pS) => {
      const cHealthState = cMcpStore.mGetHealthCheckState(pS.id);
      return cHealthState.status !== HealthCheckStatus.ERROR;
    })
  );
  let vHasEnabledMcpServers = $derived(vEnabledMcpServersForChat.length > 0);
  let vExtraServersCount = $derived(
    Math.max(0, vHealthyEnabledMcpServers.length - cMaxDisplayedMcpAvatars)
  );
  let vMcpFavicons = $derived(
    vHealthyEnabledMcpServers
      .slice(0, cMaxDisplayedMcpAvatars)
      .map((pS) => ({
        id: pS.id,
        name: cMcpStore.mGetServerDisplayName(pS.id),
        url: cMcpStore.mGetServerFavicon(pS.id)
      }))
      .filter((pF) => pF.url !== null)
  );
</script>

{#if !vHasEnabledMcpServers}
  <button
    class={[
      'inline-flex cursor-pointer items-center gap-0.75 opacity-70 transition-opacity hover:opacity-100',
      vClassName,
      'opacity-50 hover:opacity-100'
    ]}
    onclick={vOnclick}
  >
    <Tooltip.Root>
      <Tooltip.Trigger>
        <McpLogo class={cIconClassDefault} />
      </Tooltip.Trigger>

      <Tooltip.Content>
        <p>{fTranslate('text_aab669a72882')}</p>
      </Tooltip.Content>
    </Tooltip.Root>
  </button>
{:else if vMcpFavicons.length > 0}
  <button class={['inline-flex items-center gap-0.75', vClassName]} onclick={vOnclick}>
    <div class="flex -space-x-1">
      {#each vMcpFavicons as favicon (favicon.id)}
        <Tooltip.Root>
          <Tooltip.Trigger>
            <div class="box-shadow-lg overflow-hidden rounded-full bg-muted ring-1 ring-muted">
              <img
                src={favicon.url}
                alt=""
                class={cIconClassDefault}
                onerror={(e) => {
                  (e.currentTarget as HTMLImageElement).style.display = 'none';
                }}
              />
            </div>
          </Tooltip.Trigger>
          <Tooltip.Content>
            <p>{favicon.name}</p>
          </Tooltip.Content>
        </Tooltip.Root>
      {/each}
    </div>

    {#if vExtraServersCount > 0}
      <span class="text-xs text-muted-foreground">+{vExtraServersCount}</span>
    {/if}
  </button>
{/if}
