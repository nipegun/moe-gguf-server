<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { goto } from '$app/navigation';
  import type { Component } from 'svelte';
  import { Activity, Box, Boxes, MessageSquare, PencilRuler, Settings, SquarePen } from '@lucide/svelte';
  import McpLogo from '../../mcp/McpLogo.svelte';
  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { aSettingsChatSections, cRoutes, cSettingsSectionSlugs, cSettingsSectionTitles } from '$lib/constants';
  import { fGetBuiltinToolUi } from '$lib/constants/built-in-tools';
  import { ToolSource } from '$lib/enums/tools.enums';
  import { DatabaseService } from '$lib/services/database.service';
  import { RouterService } from '$lib/services/router.service';
  import { fConversations } from '$lib/stores/conversations.svelte';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { fRouterModels } from '$lib/stores/models.svelte';
  import { fIsRouterMode } from '$lib/stores/server.svelte';
  import { cToolsStore } from '$lib/stores/tools.svelte';
  import {
    fRevealSearchTarget,
    fSearchScore,
    fSearchSnippet,
    fSearchTokens
  } from '$lib/utils';
  import SidebarNavigationConversationItem from './SidebarNavigationConversationItem.svelte';

  interface Props {
    query: string;
    currentChatId: string | undefined;
    /** after a result was opened, so the sidebar can clear the query and collapse on mobile */
    onNavigated: () => void;
    onEdit: (pId: string) => void;
    onDelete: (pId: string) => void;
    onStop: (pId: string) => void;
  }

  let {
    query: vQuery,
    currentChatId: vCurrentChatId,
    onNavigated: vOnNavigated,
    onEdit: vOnEdit,
    onDelete: vOnDelete,
    onStop: vOnStop
  }: Props = $props();

  // Something of the web UI that the search can find: a page or section, a setting, a model, a
  // tool, an MCP server, a chat or a message. targets are the ids of the element to highlight on
  // the page, the first one that the page renders (fRevealSearchTarget)
  interface SearchEntry {
    key: string;
    title: string;
    subtitle?: string;
    details?: string;
    href: string;
    targets?: string[];
    icon: Component;
    /** a chat found by its title, shown with its actions */
    conversation?: DatabaseConversation;
    /** the chat of a message found by its content */
    chatId?: string;
  }

  interface SearchGroup {
    key: string;
    title: string;
    entries: SearchEntry[];
  }

  const cMaxPerGroup = 30;
  const cMaxMessageChats = 20;

  let vTokens = $derived(fSearchTokens(vQuery));

  function fSettingsHref(pSlug: string): string {
    return RouterService.mSettings(pSlug);
  }

  // pages, their parts and the sections of the settings
  function fSectionEntries(): SearchEntry[] {
    const cModelTitle = fTranslate('text_5e2c614c23f0');
    const cMonitorTitle = fTranslate('text_4c2e1df4570b');
    const cSettingsTitle = fTranslate('text_74a883a037bc');
    const aEntries: SearchEntry[] = [
      {
        key: 'page:new-chat',
        title: fTranslate('text_db18382a249e'),
        href: cRoutes.NEW_CHAT,
        icon: SquarePen
      },
      {
        key: 'page:monitor',
        title: cMonitorTitle,
        details: 'CPU GPU RAM VRAM',
        href: cRoutes.MONITOR,
        icon: Activity
      },
      {
        key: 'page:monitor:usage',
        title: fTranslate('text_8d59829c1e15'),
        subtitle: cMonitorTitle,
        details: 'CPU GPU',
        href: cRoutes.MONITOR,
        targets: ['monitor-usage'],
        icon: Activity
      },
      {
        key: 'page:monitor:memory',
        title: fTranslate('text_c3963aedaac6'),
        subtitle: cMonitorTitle,
        details: 'RAM VRAM',
        href: cRoutes.MONITOR,
        targets: ['monitor-memory'],
        icon: Activity
      },
      { key: 'page:model', title: cModelTitle, href: cRoutes.MODEL, icon: Box },
      {
        key: 'page:model:file',
        title: fTranslate('text_3264391adcc7'),
        subtitle: cModelTitle,
        details: 'GGUF',
        href: cRoutes.MODEL,
        targets: ['model-file'],
        icon: Box
      },
      {
        key: 'page:model:load',
        title: fTranslate('text_0832cad691aa'),
        subtitle: cModelTitle,
        details: 'MTP flash attention --fit',
        href: cRoutes.MODEL,
        targets: ['model-load-parameters'],
        icon: Box
      },
      {
        key: 'page:model:command-line',
        title: fTranslate('text_c925a11891b8'),
        subtitle: cModelTitle,
        href: cRoutes.MODEL,
        targets: ['model-command-line'],
        icon: Box
      }
    ];

    for (const cSection of aSettingsChatSections) {
      aEntries.push({
        key: `section:${cSection.slug}`,
        title: cSection.title,
        subtitle: cSettingsTitle,
        href: fSettingsHref(cSection.slug),
        icon: cSection.icon
      });
    }

    return aEntries;
  }

  // every setting of the sections with fields, plus the parts of the other sections
  function fSettingEntries(): SearchEntry[] {
    const aEntries: SearchEntry[] = [];

    for (const cSection of aSettingsChatSections) {
      for (const cField of cSection.fields ?? []) {
        const aDetails = [
          cField.help ?? '',
          ...(cField.options ?? []).map((pOption) => pOption.label),
          ...(cField.radioOptions ?? []).map((pOption) => pOption.label)
        ];
        aEntries.push({
          key: `setting:${cField.key}`,
          title: cField.label,
          subtitle: cSection.title,
          details: aDetails.join(' '),
          href: fSettingsHref(cSection.slug),
          targets: [
            `setting-${cField.key}`,
            ...(cField.dependsOn ? [`setting-${cField.dependsOn}`] : [])
          ],
          icon: cSection.icon
        });
      }
    }

    aEntries.push({
      key: 'setting:language',
      title: fTranslate('language'),
      subtitle: cSettingsSectionTitles.GENERAL,
      details: fTranslate('languageReload'),
      href: fSettingsHref(cSettingsSectionSlugs.GENERAL),
      targets: ['setting-language'],
      icon: Settings
    });

    if (fIsRouterMode()) {
      aEntries.push(
        {
          key: 'setting:models-folder',
          title: fTranslate('text_9460a43cccd9'),
          subtitle: cSettingsSectionTitles.MODELS,
          details: 'GGUF',
          href: fSettingsHref(cSettingsSectionSlugs.MODELS),
          targets: ['settings-models-folder'],
          icon: Boxes
        },
        {
          key: 'setting:models-available',
          title: fTranslate('text_9dbfd0478bd5'),
          subtitle: cSettingsSectionTitles.MODELS,
          details: 'GGUF',
          href: fSettingsHref(cSettingsSectionSlugs.MODELS),
          targets: ['settings-models-available'],
          icon: Boxes
        }
      );
    }

    // Import/Export: the actions of its two groups, conversations and settings
    const cImportExport = aSettingsChatSections.find(
      (pSection) => pSection.slug === cSettingsSectionSlugs.IMPORT_EXPORT
    );
    const aActions: [string, string, string, string][] = [
      ['conversations-export', 'text_1d432f58690c', 'text_3664895579f0', 'text_e1c29b8d7188'],
      ['conversations-import', 'text_1d432f58690c', 'text_2cff9baabf56', 'text_a659626215c6'],
      ['conversations-delete', 'text_1d432f58690c', 'text_ef85afa1b92c', 'text_21814f9e74f4'],
      ['settings-export', 'text_74a883a037bc', 'text_3664895579f0', 'text_72df65773b40'],
      ['settings-import', 'text_74a883a037bc', 'text_2cff9baabf56', 'text_6235faca5e63']
    ];
    for (const [cId, cGroupKey, cTitleKey, cDescriptionKey] of aActions) {
      aEntries.push({
        key: `setting:import-export:${cId}`,
        title: `${fTranslate(cTitleKey)}: ${fTranslate(cGroupKey)}`,
        subtitle: cSettingsSectionTitles.IMPORT_EXPORT,
        details: fTranslate(cDescriptionKey),
        href: fSettingsHref(cSettingsSectionSlugs.IMPORT_EXPORT),
        targets: [`settings-import-export-${cId}`],
        icon: cImportExport?.icon ?? Settings
      });
    }

    return aEntries;
  }

  // models of the router, in the list of Settings > Models
  function fModelEntries(): SearchEntry[] {
    if (!fIsRouterMode()) return [];

    return fRouterModels().map((pModel) => ({
      key: `model:${pModel.id}`,
      title: pModel.id,
      subtitle: cSettingsSectionTitles.MODELS,
      details: pModel.path ?? '',
      href: fSettingsHref(cSettingsSectionSlugs.MODELS),
      targets: [`settings-model-${pModel.id}`],
      icon: Boxes
    }));
  }

  function fToolEntries(): SearchEntry[] {
    const aEntries: SearchEntry[] = [];

    for (const cGroup of cToolsStore.mToolGroups) {
      for (const cTool of cGroup.tools) {
        const cName = cTool.definition.function.name;
        const cBuiltin =
          cTool.source === ToolSource.BUILTIN || cTool.source === ToolSource.FRONTEND
            ? fGetBuiltinToolUi(cName)
            : null;
        aEntries.push({
          key: `tool:${cTool.key}`,
          title: cBuiltin?.label ?? cName,
          subtitle: `${cSettingsSectionTitles.TOOLS} · ${cGroup.label}`,
          details: `${cName} ${cTool.definition.function.description ?? ''}`,
          href: fSettingsHref(cSettingsSectionSlugs.TOOLS),
          targets: [`settings-tool-group-${cGroup.key}`],
          icon: PencilRuler
        });
      }
    }

    return aEntries;
  }

  function fMcpServerEntries(): SearchEntry[] {
    const cShellServer: SearchEntry = {
      key: 'mcp:local-shell-switch',
      title: fTranslate('text_a65fc168a110'),
      subtitle: cSettingsSectionTitles.MCP_SERVERS,
      details: `${fTranslate('text_42f4a2baa426')} bash zsh sh PowerShell`,
      href: fSettingsHref(cSettingsSectionSlugs.MCP_SERVERS),
      targets: ['settings-local-shell-mcp-server'],
      icon: McpLogo
    };

    return [cShellServer, ...cMcpStore.mGetServers().map((pServer) => ({
      key: `mcp:${pServer.id}`,
      title: cMcpStore.mGetServerLabel(pServer),
      subtitle: cSettingsSectionTitles.MCP_SERVERS,
      details: `${pServer.url} ${pServer.name ?? ''}`,
      href: fSettingsHref(cSettingsSectionSlugs.MCP_SERVERS),
      targets: [`mcp-server-${pServer.id}`],
      icon: McpLogo
    }))];
  }

  // the entries of pEntries that hold every word of the query, best first
  function fMatch(pEntries: SearchEntry[]): SearchEntry[] {
    return pEntries
      .map((pEntry) => ({
        vEntry: pEntry,
        vScore: fSearchScore(vTokens, pEntry.title, `${pEntry.subtitle ?? ''} ${pEntry.details ?? ''}`)
      }))
      .filter((pScored) => pScored.vScore > 0)
      .sort((pA, pB) => pB.vScore - pA.vScore)
      .slice(0, cMaxPerGroup)
      .map((pScored) => pScored.vEntry);
  }

  let vChatEntries = $derived(
    fMatch(
      fConversations().map((pConversation) => ({
        key: `chat:${pConversation.id}`,
        title: pConversation.name,
        href: RouterService.mChat(pConversation.id),
        icon: MessageSquare,
        conversation: pConversation
      }))
    )
  );

  // messages are read from IndexedDB, a moment after the last key press
  let vMessageEntries = $state<SearchEntry[]>([]);
  let vMessagesPending = $state(false);
  let vMessageRequest = 0;

  $effect(() => {
    const aTokens = vTokens;
    const cQuery = vQuery;
    const cRequest = ++vMessageRequest;

    if (aTokens.length === 0) {
      vMessageEntries = [];
      vMessagesPending = false;
      return;
    }

    vMessagesPending = true;
    const cTimer = setTimeout(async () => {
      try {
        // the server compares the words without regard to case or accents
        const aMessages = await DatabaseService.mSearchMessages(cQuery, cMaxMessageChats);
        if (cRequest !== vMessageRequest) return;

        const dConversations = new Map(fConversations().map((pC) => [pC.id, pC]));
        vMessageEntries = aMessages.flatMap((pMessage) => {
          const cConversation = dConversations.get(pMessage.convId);
          if (!cConversation) return [];

          return [
            {
              key: `message:${pMessage.id}`,
              title: cConversation.name,
              subtitle: fSearchSnippet(pMessage.content, aTokens),
              href: RouterService.mChat(cConversation.id),
              icon: MessageSquare,
              chatId: cConversation.id
            }
          ];
        });
      } catch (pError) {
        console.error('Failed to search the messages:', pError);
        if (cRequest === vMessageRequest) vMessageEntries = [];
      } finally {
        if (cRequest === vMessageRequest) vMessagesPending = false;
      }
    }, 200);

    return () => clearTimeout(cTimer);
  });

  let vGroups = $derived.by((): SearchGroup[] => {
    if (vTokens.length === 0) return [];

    const sChatIds = new Set(vChatEntries.map((pEntry) => pEntry.conversation?.id));
    const aGroups: SearchGroup[] = [
      { key: 'sections', title: fTranslate('text_9bae918add9a'), entries: fMatch(fSectionEntries()) },
      {
        key: 'settings',
        title: fTranslate('text_74a883a037bc'),
        entries: fMatch(fSettingEntries())
      },
      { key: 'models', title: cSettingsSectionTitles.MODELS, entries: fMatch(fModelEntries()) },
      { key: 'tools', title: cSettingsSectionTitles.TOOLS, entries: fMatch(fToolEntries()) },
      {
        key: 'mcp',
        title: cSettingsSectionTitles.MCP_SERVERS,
        entries: fMatch(fMcpServerEntries())
      },
      { key: 'chats', title: fTranslate('text_ef5b4049bb36'), entries: vChatEntries },
      {
        key: 'messages',
        title: fTranslate('text_8d847ba21aee'),
        // a chat whose title already matched is not repeated
        entries: vMessageEntries.filter((pEntry) => !sChatIds.has(pEntry.chatId))
      }
    ];

    return aGroups.filter((pGroup) => pGroup.entries.length > 0);
  });

  async function fOpen(pEntry: SearchEntry) {
    await goto(pEntry.href);
    if (pEntry.targets) fRevealSearchTarget(pEntry.targets);
    vOnNavigated();
  }

  /** Opens the first result, for Enter in the search field. */
  export function mOpenFirst(): void {
    const cFirst = vGroups[0]?.entries[0];
    if (cFirst) void fOpen(cFirst);
  }
</script>

<div class="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-2 pb-2">
  {#each vGroups as vGroup (vGroup.key)}
    <section class="flex min-w-0 flex-col gap-0.5">
      <h3
        class="text-muted-foreground flex h-7 shrink-0 items-center px-2 text-xs font-medium whitespace-nowrap"
      >
        {vGroup.title}
      </h3>

      <ul class="flex w-full min-w-0 flex-col gap-1">
        {#each vGroup.entries as vEntry (vEntry.key)}
          <li class="group/item relative min-w-0 p-0">
            {#if vEntry.conversation}
              <SidebarNavigationConversationItem
                conversation={vEntry.conversation}
                isActive={vCurrentChatId === vEntry.conversation.id}
                onSelect={() => fOpen(vEntry)}
                onEdit={vOnEdit}
                onDelete={vOnDelete}
                onStop={vOnStop}
              />
            {:else}
              <button
                type="button"
                class="flex w-full min-w-0 items-start gap-2 rounded-lg px-2 py-1.5 text-start text-sm hover:bg-accent hover:text-accent-foreground"
                onclick={() => fOpen(vEntry)}
              >
                <vEntry.icon class="{cIconClassDefault} mt-0.5 shrink-0" />

                <span class="flex min-w-0 flex-1 flex-col">
                  <span class="truncate">{vEntry.title}</span>

                  {#if vEntry.subtitle}
                    <span class="truncate text-xs text-muted-foreground">{vEntry.subtitle}</span>
                  {/if}
                </span>
              </button>
            {/if}
          </li>
        {/each}
      </ul>
    </section>
  {/each}

  {#if vGroups.length === 0 && !vMessagesPending && vTokens.length > 0}
    <p class="p-4 text-center text-sm text-muted-foreground">{fTranslate('text_7d7e36058609')}</p>
  {/if}
</div>
