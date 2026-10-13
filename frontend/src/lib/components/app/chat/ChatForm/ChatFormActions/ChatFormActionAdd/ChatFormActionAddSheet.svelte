<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import type { Snippet } from 'svelte';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import * as Sheet from '$lib/components/ui/sheet';
  import * as Collapsible from '$lib/components/ui/collapsible';
  import { File, MessageSquare, Zap, FolderOpen } from '@lucide/svelte';
  import { Switch } from '$lib/components/ui/switch';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import { cTooltipDelayDuration } from '$lib/constants';
  import { aAttachmentFileItems } from '$lib/constants/attachment-menu';
  import { fUseAttachmentMenu } from '$lib/hooks/use-attachment-menu.svelte';
  import { fUseToolsPanel } from '$lib/hooks/use-tools-panel.svelte';
  import { fUseReasoningMenu } from '$lib/hooks/use-reasoning-menu.svelte';
  import { cConversationsStore } from '$lib/stores/conversations.svelte';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { McpLogo } from '$lib/components/app';
  import {
    PencilRuler,
    ChevronDown,
    ChevronRight,
    Lightbulb,
    LightbulbOff,
    Check
  } from '@lucide/svelte';
  import { HealthCheckStatus } from '$lib/enums';
  import { AttachmentAction } from '$lib/enums/attachment.enums';

  interface Props {
    class?: string;
    disabled?: boolean;
    hasAudioModality?: boolean;
    hasVideoModality?: boolean;
    hasVisionModality?: boolean;
    hasMcpPromptsSupport?: boolean;
    hasMcpResourcesSupport?: boolean;
    onFileUpload?: () => void;
    onSystemPromptClick?: () => void;
    onMcpPromptClick?: () => void;
    onMcpResourcesClick?: () => void;
    trigger: Snippet<[{ disabled: boolean; onclick?: () => void }]>;
  }

  let {
    class: vClassName = '',
    disabled: vDisabled = false,
    hasAudioModality: vHasAudioModality = false,
    hasVisionModality: vHasVisionModality = false,
    hasVideoModality: vHasVideoModality = false,
    hasMcpPromptsSupport: vHasMcpPromptsSupport = false,
    hasMcpResourcesSupport: vHasMcpResourcesSupport = false,
    onFileUpload: vOnFileUpload,
    onSystemPromptClick: vOnSystemPromptClick,
    onMcpPromptClick: vOnMcpPromptClick,
    onMcpResourcesClick: vOnMcpResourcesClick,
    trigger: vTrigger
  }: Props = $props();

  let vSheetOpen = $state(false);
  let vReasoningExpanded = $state(false);
  let vFilesExpanded = $state(true);
  let vToolsExpanded = $state(false);
  let vMcpExpanded = $state(false);

  const cAttachmentMenu = fUseAttachmentMenu(
    () => ({
      hasVisionModality: vHasVisionModality,
      hasAudioModality: vHasAudioModality,
      hasVideoModality: vHasVideoModality,
      hasMcpPromptsSupport: vHasMcpPromptsSupport,
      hasMcpResourcesSupport: vHasMcpResourcesSupport
    }),
    () => ({
      onFileUpload: vOnFileUpload,
      onSystemPromptClick: vOnSystemPromptClick,
      onMcpPromptClick: vOnMcpPromptClick,
      onMcpResourcesClick: vOnMcpResourcesClick
    }),
    () => {
      vSheetOpen = false;
    }
  );

  const cToolsPanel = fUseToolsPanel();
  const cReasoning = fUseReasoningMenu();

  const cSheetItemClass =
    'flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left text-sm transition-colors hover:bg-accent active:bg-accent disabled:cursor-not-allowed disabled:opacity-50';

  const cSheetItemRowClass =
    'flex w-full items-center justify-between gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent';

  let vMcpServers = $derived(cMcpStore.mGetServers());
</script>

<div class="flex items-center gap-1 {vClassName}">
  <Sheet.Root bind:open={vSheetOpen}>
    {@render vTrigger({ disabled: vDisabled, onclick: () => (vSheetOpen = true) })}

    <Sheet.Content side="bottom" class="max-h-[85vh] gap-0 overflow-y-auto">
      <Sheet.Header>
        <Sheet.Title>{fTranslate('text_b33b61718ca0')}</Sheet.Title>

        <Sheet.Description class="sr-only">{fTranslate('text_320f701a326b')}</Sheet.Description>
      </Sheet.Header>

      <div class="flex flex-col gap-1 px-1.5 pb-2">
        {#if cReasoning.modelSupportsThinking}
          <Collapsible.Root
            open={vReasoningExpanded}
            onOpenChange={(open) => (vReasoningExpanded = open)}
          >
            <Collapsible.Trigger class={cSheetItemClass}>
              {#if vReasoningExpanded}
                <ChevronDown class="{cIconClassDefault} shrink-0" />
              {:else}
                <ChevronRight class="{cIconClassDefault} shrink-0" />
              {/if}

              {#if cReasoning.thinkingEnabled}
                <Lightbulb class="{cIconClassDefault} shrink-0 text-amber-400" />
              {:else if cReasoning.isOff}
                <LightbulbOff class="{cIconClassDefault} shrink-0 text-muted-foreground" />
              {:else}
                <Lightbulb class="{cIconClassDefault} shrink-0 text-muted-foreground" />
              {/if}

              <span class="flex-1 whitespace-nowrap">{fTranslate('text_d8211e24e83d')}</span>

              <span class="text-xs whitespace-nowrap text-muted-foreground">
                {cReasoning.currentLabel}
              </span>
            </Collapsible.Trigger>

            <Collapsible.Content>
              <div class="flex flex-col gap-0.5 pl-4">
                {#each cReasoning.levels as level (level.value)}
                  {@const tokenLabel = cReasoning.tokenLabel(level)}
                  <button
                    type="button"
                    class={cSheetItemRowClass}
                    class:bg-accent={cReasoning.isSelected(level)}
                    onclick={() => cReasoning.select(level)}
                  >
                    <div class="flex shrink-0 items-center gap-3">
                      {#if cReasoning.isSelected(level)}
                        <Check class="{cIconClassDefault} shrink-0 text-foreground" />
                      {:else}
                        <div class="{cIconClassDefault} shrink-0"></div>
                      {/if}

                      <span class="text-sm whitespace-nowrap">{level.label}</span>
                    </div>

                    {#if tokenLabel}
                      <!-- the hint wraps when space is short; the level name never does -->
                      <span class="min-w-0 text-right text-[11px] text-muted-foreground opacity-60">
                        {tokenLabel}
                      </span>
                    {/if}
                  </button>
                {/each}
              </div>
            </Collapsible.Content>
          </Collapsible.Root>
        {/if}

        <Collapsible.Root open={vFilesExpanded} onOpenChange={(open) => (vFilesExpanded = open)}>
          <Collapsible.Trigger class={cSheetItemClass}>
            {#if vFilesExpanded}
              <ChevronDown class="{cIconClassDefault} shrink-0" />
            {:else}
              <ChevronRight class="{cIconClassDefault} shrink-0" />
            {/if}

            <File class="{cIconClassDefault} shrink-0" />

            <span class="flex-1">{fTranslate('text_01792a33c228')}</span>
          </Collapsible.Trigger>

          <Collapsible.Content>
            <div class="flex flex-col gap-0.5 pl-4">
              {#each aAttachmentFileItems as item (item.id)}
                {@const enabled = cAttachmentMenu.isItemEnabled(item.enabledWhen)}
                {#if enabled}
                  <button
                    type="button"
                    class={cSheetItemClass}
                    onclick={() => cAttachmentMenu.callbacks[item.action]()}
                  >
                    <item.icon class="{cIconClassDefault} shrink-0" />

                    <span>{item.label}</span>
                  </button>
                {:else if item.disabledTooltip}
                  <Tooltip.Root delayDuration={cTooltipDelayDuration}>
                    <Tooltip.Trigger>
                      <button type="button" class={cSheetItemClass} disabled>
                        <item.icon class="{cIconClassDefault} shrink-0" />

                        <span>{item.label}</span>
                      </button>
                    </Tooltip.Trigger>

                    <Tooltip.Content side="right">
                      <p>{item.disabledTooltip}</p>
                    </Tooltip.Content>
                  </Tooltip.Root>
                {/if}
              {/each}
            </div>
          </Collapsible.Content>
        </Collapsible.Root>

        <Collapsible.Root open={vMcpExpanded} onOpenChange={(open) => (vMcpExpanded = open)}>
          <Collapsible.Trigger class={cSheetItemClass}>
            {#if vMcpExpanded}
              <ChevronDown class="{cIconClassDefault} shrink-0" />
            {:else}
              <ChevronRight class="{cIconClassDefault} shrink-0" />
            {/if}

            <McpLogo class="inline {cIconClassDefault} shrink-0" />

            <span class="flex-1">{fTranslate('text_aab669a72882')}</span>

            <span class="text-xs text-muted-foreground">
              {vMcpServers.length}
              {fTranslate('text_b3eacd33433b')}{vMcpServers.length !== 1 ? 's' : ''}
            </span>
          </Collapsible.Trigger>

          <Collapsible.Content>
            <div class="flex flex-col gap-0.5 pl-4">
              {#each vMcpServers as server (server.id)}
                {@const healthState = cMcpStore.mGetHealthCheckState(server.id)}
                {@const hasError = healthState.status === HealthCheckStatus.ERROR}
                {@const displayName = cMcpStore.mGetServerLabel(server)}
                {@const faviconUrl = cMcpStore.mGetServerFavicon(server.id)}
                {@const isEnabled = cConversationsStore.mIsMcpServerEnabledForChat(server.id)}

                <button
                  type="button"
                  class={cSheetItemRowClass}
                  onclick={() =>
                    !hasError && cConversationsStore.mToggleMcpServerForChat(server.id)}
                  disabled={hasError}
                >
                  <div class="flex min-w-0 flex-1 items-center gap-2">
                    {#if faviconUrl}
                      <img
                        src={faviconUrl}
                        alt=""
                        class="{cIconClassDefault} shrink-0 rounded-sm"
                        onerror={(e) => {
                          (e.currentTarget as HTMLImageElement).style.display = 'none';
                        }}
                      />
                    {/if}

                    <span class="min-w-0 truncate text-sm">{displayName}</span>
                  </div>

                  {#if hasError}
                    <span
                      class="shrink-0 rounded bg-destructive/15 px-1.5 py-0.5 text-xs text-destructive"
                    >
                      {fTranslate('text_54a0e8c17ebb')}
                    </span>
                  {:else}
                    <Switch
                      checked={isEnabled}
                      onCheckedChange={() => cConversationsStore.mToggleMcpServerForChat(server.id)}
                    />
                  {/if}
                </button>
              {/each}

              {#if vMcpServers.length === 0}
                <div class="px-3 py-2 text-center text-sm text-muted-foreground">
                  {fTranslate('text_013e6b9db9d2')}
                </div>
              {/if}
            </div>
          </Collapsible.Content>
        </Collapsible.Root>

        {#if cToolsPanel.totalToolCount > 0}
          <Collapsible.Root open={vToolsExpanded} onOpenChange={(open) => (vToolsExpanded = open)}>
            <Collapsible.Trigger class={cSheetItemClass}>
              {#if vToolsExpanded}
                <ChevronDown class="{cIconClassDefault} shrink-0" />
              {:else}
                <ChevronRight class="{cIconClassDefault} shrink-0" />
              {/if}

              <PencilRuler class="inline {cIconClassDefault} shrink-0" />

              <span class="flex-1">{fTranslate('text_ea93d6a262ec')}</span>

              <span class="text-xs text-muted-foreground">
                {cToolsPanel.totalToolCount}
                {fTranslate('text_7c9bbe5ec9b3')}{cToolsPanel.totalToolCount !== 1 ? 's' : ''}
              </span>
            </Collapsible.Trigger>

            <Collapsible.Content>
              <div class="flex flex-col gap-0.5 pl-4">
                {#each cToolsPanel.activeGroups as group (group.key)}
                  {@const checked = cToolsPanel.isGroupChecked(group)}
                  {@const enabledCount = cToolsPanel.getEnabledToolCount(group)}
                  {@const favicon = cToolsPanel.getFavicon(group)}

                  <button
                    type="button"
                    class={cSheetItemRowClass}
                    onclick={() => cToolsPanel.toggleGroupByKey(group.key)}
                  >
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

                    <span class="min-w-0 flex-1 truncate text-sm font-medium">{group.label}</span>

                    <span class="shrink-0 text-xs text-muted-foreground">
                      {enabledCount}/{group.tools.length}
                    </span>

                    <Checkbox
                      {checked}
                      class="{cIconClassDefault} shrink-0"
                      onclick={(e) => e.stopPropagation()}
                      onCheckedChange={() => cToolsPanel.toggleGroupByKey(group.key)}
                    />
                  </button>
                {/each}
              </div>
            </Collapsible.Content>
          </Collapsible.Root>
        {/if}

        <button
          type="button"
          class={cSheetItemClass}
          onclick={() => cAttachmentMenu.callbacks[AttachmentAction.SYSTEM_PROMPT_CLICK]()}
        >
          <MessageSquare class="{cIconClassDefault} shrink-0" />

          <span>{fTranslate('text_8e5a3143184d')}</span>
        </button>

        {#if vHasMcpPromptsSupport}
          <button
            type="button"
            class={cSheetItemClass}
            onclick={() => cAttachmentMenu.callbacks[AttachmentAction.MCP_PROMPT_CLICK]()}
          >
            <Zap class="{cIconClassDefault} shrink-0" />

            <span>{fTranslate('text_6829fa6ca0ec')}</span>
          </button>
        {/if}

        {#if vHasMcpResourcesSupport}
          <button
            type="button"
            class={cSheetItemClass}
            onclick={() => cAttachmentMenu.callbacks[AttachmentAction.MCP_RESOURCES_CLICK]()}
          >
            <FolderOpen class="{cIconClassDefault} shrink-0" />

            <span>{fTranslate('text_77a814db2847')}</span>
          </button>
        {/if}
      </div>
    </Sheet.Content>
  </Sheet.Root>
</div>
