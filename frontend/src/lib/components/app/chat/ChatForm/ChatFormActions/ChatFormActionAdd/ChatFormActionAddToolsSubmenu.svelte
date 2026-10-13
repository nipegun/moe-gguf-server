<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { PencilRuler, ChevronDown, ChevronRight, Loader2, Info, Check } from '@lucide/svelte';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import * as Collapsible from '$lib/components/ui/collapsible';
  import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cToolsStore } from '$lib/stores/tools.svelte';
  import { cCliFlags } from '$lib/constants';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { fUseToolsPanel } from '$lib/hooks/use-tools-panel.svelte';

  const cToolsPanel = fUseToolsPanel();
  const cHasMcpServersAvailable = $derived(cMcpStore.mGetServers().length > 0);
</script>

<DropdownMenu.Sub onOpenChange={(open) => open && cToolsPanel.handleOpen()}>
  <DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
    <PencilRuler class={cIconClassDefault} />

    <span>{fTranslate('text_ea93d6a262ec')}</span>
  </DropdownMenu.SubTrigger>

  <DropdownMenu.SubContent class="w-72 p-0">
    {#if cToolsPanel.totalToolCount === 0}
      {#if cToolsStore.mLoading}
        <div class="px-3 py-4 text-center text-sm text-muted-foreground">
          <Loader2 class="mx-auto mb-1 {cIconClassDefault} animate-spin" />
          {fTranslate('text_efc190cd4ce7')}
        </div>
      {:else if cToolsStore.mIsToolsEndpointUnreachable}
        <div class="grid gap-2.5 px-3 py-4 text-sm text-muted-foreground">
          <span class="flex gap-2">
            <Info class="mt-0.5 {cIconClassDefault} shrink-0" />

            <span>
              {fTranslate('text_9b5e82028f67')} <code>{cCliFlags.TOOLS}</code>
              {fTranslate('text_534e6d6e87c7')} <strong>{fTranslate('text_92a726148df7')}</strong>.
            </span>
          </span>

          <span class="flex gap-2">
            <Info class="mt-0.5 {cIconClassDefault} shrink-0" />

            <span>
              {cHasMcpServersAvailable
                ? fTranslate('text_5342e09f2729')
                : fTranslate('text_9fd728c66c9a')}
              {fTranslate('text_35302548c7fd')} <strong>{fTranslate('text_ecd806d3c25e')}</strong>.
            </span>
          </span>
        </div>
      {:else if cToolsStore.mError}
        <div class="px-3 py-4 text-center text-sm text-muted-foreground">
          {fTranslate('text_5ceb12c7da97')}
        </div>
      {:else if cToolsPanel.noToolsInfoMessage}
        <div class="flex gap-2 px-3 py-4 text-sm text-muted-foreground">
          <Info class="mt-0.5 {cIconClassDefault} shrink-0" />

          <span>{cToolsPanel.noToolsInfoMessage}</span>
        </div>
      {:else}
        <div class="px-3 py-4 text-center text-sm text-muted-foreground">
          {fTranslate('text_0ec9813e343b')}
        </div>
      {/if}
    {:else}
      <div class="max-h-80 overflow-y-auto p-2 pr-1">
        {#each cToolsPanel.activeGroups as group (group.key)}
          {@const isExpanded = cToolsPanel.expandedGroups.has(group.key)}
          {@const checked = cToolsPanel.isGroupChecked(group)}
          {@const favicon = cToolsPanel.getFavicon(group)}

          <Collapsible.Root
            open={isExpanded}
            onOpenChange={() => cToolsPanel.toggleGroupExpanded(group.key)}
          >
            <div class="flex items-center gap-1">
              <Collapsible.Trigger
                class="flex min-w-0 flex-1 items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-muted/50"
              >
                {#if isExpanded}
                  <ChevronDown class="h-3.5 w-3.5 shrink-0" />
                {:else}
                  <ChevronRight class="h-3.5 w-3.5 shrink-0" />
                {/if}

                <span class="inline-flex min-w-0 items-center gap-1.5 font-medium">
                  {#if favicon}
                    <img
                      src={favicon}
                      alt=""
                      class="{cIconClassDefault} shrink-0 rounded-sm"
                      onerror={(e) => {
                        (e.currentTarget as HTMLImageElement).style.display = 'none';
                      }}
                    />
                  {/if}

                  <span class="truncate">{group.label}</span>
                </span>

                <span class="ml-auto shrink-0 text-xs text-muted-foreground">
                  {cToolsPanel.getEnabledToolCount(group)}/{group.tools.length}
                </span>
              </Collapsible.Trigger>

              <Tooltip.Root>
                <Tooltip.Trigger>
                  {#snippet child({ props })}
                    <Checkbox
                      {...props}
                      {checked}
                      onCheckedChange={() => cToolsPanel.toggleGroupByKey(group.key)}
                      class="mr-2 {cIconClassDefault} shrink-0"
                    />
                  {/snippet}
                </Tooltip.Trigger>

                <Tooltip.Content side="right">
                  <p>
                    {checked ? fTranslate('text_b7e3e4aa4257') : fTranslate('text_5342e09f2729')}
                    {group.tools.length}
                    {fTranslate('text_7c9bbe5ec9b3')}{group.tools.length !== 1 ? 's' : ''}
                  </p>
                </Tooltip.Content>
              </Tooltip.Root>
            </div>

            <Collapsible.Content>
              <div class="ml-4 flex flex-col gap-0.5 border-l border-border/50 pl-2">
                {#each group.tools as entry (entry.key)}
                  {@const enabled = cToolsStore.mIsToolEnabled(entry.key)}
                  <button
                    type="button"
                    class="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm transition-colors hover:bg-muted/50"
                    onclick={() => cToolsStore.mToggleTool(entry.key)}
                  >
                    <span
                      data-slot="checkbox"
                      data-state={enabled ? 'checked' : 'unchecked'}
                      class="flex size-4 shrink-0 items-center justify-center rounded-[4px] border border-input data-[state=checked]:border-primary data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground"
                    >
                      {#if enabled}
                        <Check class="size-3.5" />
                      {/if}
                    </span>

                    <span class="min-w-0 flex-1 truncate font-mono text-[12px]">
                      {entry.definition.function.name}
                    </span>
                  </button>
                {/each}
              </div>
            </Collapsible.Content>
          </Collapsible.Root>
        {/each}
      </div>
    {/if}
  </DropdownMenu.SubContent>
</DropdownMenu.Sub>
