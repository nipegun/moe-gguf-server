<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { ChevronDown, ChevronRight } from '@lucide/svelte';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import * as Collapsible from '$lib/components/ui/collapsible';
  import { TruncatedText, McpServerIdentity } from '$lib/components/app';
  import { cToolsStore } from '$lib/stores/tools.svelte';
  import { cPermissionsStore } from '$lib/stores/permissions.svelte';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { fGetBuiltinToolUi } from '$lib/constants/built-in-tools';
  import { ToolSource } from '$lib/enums/tools.enums';
  import { SvelteSet } from 'svelte/reactivity';

  let vExpandedGroups = new SvelteSet<string>();
  let vGroups = $derived(cToolsStore.mToolGroups);

  function fToggleExpanded(pKey: string) {
    if (vExpandedGroups.has(pKey)) {
      vExpandedGroups.delete(pKey);
    } else {
      vExpandedGroups.add(pKey);
    }
  }
</script>

{#if vGroups.length === 0}
  <div class="py-8 text-center text-sm text-muted-foreground">
    {fTranslate('text_0ec9813e343b')}
  </div>
{:else}
  <div class="space-y-2">
    {#each vGroups as group (group.key)}
      {@const isExpanded = vExpandedGroups.has(group.key)}
      <Collapsible.Root open={isExpanded} onOpenChange={() => fToggleExpanded(group.key)}>
        <Collapsible.Trigger
          class="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-muted/50"
        >
          {#if isExpanded}
            <ChevronDown class="h-3.5 w-3.5 shrink-0" />
          {:else}
            <ChevronRight class="h-3.5 w-3.5 shrink-0" />
          {/if}

          {@const faviconUrl = group.serverId ? cMcpStore.mGetServerFavicon(group.serverId) : null}

          <span class="inline-flex min-w-0 items-center gap-1.5 font-medium">
            {#if group.source === 'mcp'}
              <McpServerIdentity
                iconClass={cIconClassDefault}
                iconRounded="rounded-sm"
                showVersion={false}
                displayName={group.label}
                {faviconUrl}
              />
            {:else}
              <TruncatedText text={group.label} class="font-medium" />
            {/if}
          </span>

          <span class="ml-auto shrink-0 text-xs text-muted-foreground">
            {group.tools.length}
            {fTranslate('text_7c9bbe5ec9b3')}{group.tools.length !== 1 ? 's' : ''}
          </span>
        </Collapsible.Trigger>

        <Collapsible.Content>
          <div class="ml-4 border-l border-border/50 pl-2">
            <!-- Header row -->
            <div class="flex items-center gap-2 px-2 py-1 text-xs text-muted-foreground">
              <span class="min-w-0 flex-1">{fTranslate('text_2e53bdcd0740')}</span>
              <span class="w-16 shrink-0 text-center">{fTranslate('text_92c1cdfdf4cb')}</span>
              <span class="w-20 shrink-0 text-center">{fTranslate('text_977618bd8bc7')}</span>
            </div>

            {#each group.tools as entry (entry.key)}
              {@const toolName = entry.definition.function.name}
              {@const builtinUi =
                entry.source === ToolSource.BUILTIN || entry.source === ToolSource.FRONTEND
                  ? fGetBuiltinToolUi(toolName)
                  : null}
              {@const displayLabel = builtinUi?.label ?? toolName}
              {@const IconComponent = builtinUi?.icon ?? null}
              {@const isEnabled = cToolsStore.mIsToolEnabled(entry.key)}
              {@const permissionKey = entry.key}
              {@const isAlwaysAllowed = cPermissionsStore.mHasTool(permissionKey)}

              <div class="flex items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-muted/50">
                <span class="flex min-w-0 flex-1 items-center gap-1.5">
                  {#if IconComponent}
                    <IconComponent class={cIconClassDefault} />
                  {/if}
                  <TruncatedText text={displayLabel} class="min-w-0" showTooltip={true} />
                </span>

                <div class="flex w-16 shrink-0 justify-center">
                  <Checkbox
                    checked={isEnabled}
                    onCheckedChange={() => cToolsStore.mToggleTool(entry.key)}
                    class={cIconClassDefault}
                  />
                </div>

                <div class="flex w-20 shrink-0 justify-center">
                  <Checkbox
                    checked={isAlwaysAllowed}
                    onCheckedChange={() => {
                      if (isAlwaysAllowed) {
                        cPermissionsStore.mRevokeTool(permissionKey);
                      } else {
                        cPermissionsStore.mAllowTool(permissionKey);
                      }
                    }}
                    class={cIconClassDefault}
                  />
                </div>
              </div>
            {/each}
          </div>
        </Collapsible.Content>
      </Collapsible.Root>
    {/each}
  </div>
{/if}
