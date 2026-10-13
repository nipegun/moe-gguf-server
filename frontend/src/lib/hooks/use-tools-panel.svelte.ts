import { fTranslate } from '../i18n';
import { cCliFlags } from '$lib/constants';
import { SvelteSet } from 'svelte/reactivity';
import { ToolSource } from '$lib/enums';
import { cConversationsStore } from '$lib/stores/conversations.svelte';
import { cMcpStore } from '$lib/stores/mcp.svelte';
import { cToolsStore } from '$lib/stores/tools.svelte';
import type { ToolGroup } from '$lib/types';

export interface UseToolsPanelReturn {
  readonly expandedGroups: SvelteSet<string>;
  readonly groups: ToolGroup[];
  readonly activeGroups: ToolGroup[];
  readonly totalToolCount: number;
  readonly noToolsInfoMessage: string | null;
  isGroupChecked(pGroup: ToolGroup): boolean;
  getEnabledToolCount(pGroup: ToolGroup): number;
  getFavicon(pGroup: ToolGroup): string | null;
  isGroupDisabled(pGroup: ToolGroup): boolean;
  toggleGroupExpanded(pKey: string): void;
  /** Toggle all tools in a group by its stable key (avoids stale group object references). */
  toggleGroupByKey(pKey: string): void;
  handleOpen(): void;
}

/**
 * Shared reactive state and helpers for the tools panel UI.
 *
 * Used by both the desktop dropdown (`ChatFormActionAddToolsSubmenu`)
 * and the mobile sheet (`ChatFormActionAddSheet`) to avoid
 * duplicating group filtering, checked-state derivation, and favicon logic.
 */
export function fUseToolsPanel(): UseToolsPanelReturn {
  const cExpandedGroups = new SvelteSet<string>();

  const cGroups = $derived(cToolsStore.mToolGroups);
  const cActiveGroups = $derived(
    cGroups.filter(
      (pG) =>
        pG.source !== ToolSource.MCP ||
        !pG.serverId ||
        cConversationsStore.mIsMcpServerEnabledForChat(pG.serverId)
    )
  );
  const cTotalToolCount = $derived(cActiveGroups.reduce((pN, pG) => pN + pG.tools.length, 0));
  const cNoToolsInfoMessage = $derived.by(() => {
    if (cToolsStore.mLoading) return null;
    if (cToolsStore.mToolGroups.length > 0) return null;
    // A 403 means the server was started without tools.
    if (cToolsStore.mIsToolsEndpointUnreachable) {
      return fTranslate('text_9a67bb312cb8', { p0: cCliFlags.TOOLS, p1: cCliFlags.TOOLS });
    }
    // Other errors — return null so UI shows "Failed to load tools"
    if (cToolsStore.mError) return null;
    return fTranslate('text_9a67bb312cb8', { p0: cCliFlags.TOOLS, p1: cCliFlags.TOOLS });
  });

  function fIsGroupChecked(pGroup: ToolGroup): boolean {
    return cToolsStore.mIsGroupFullyEnabled(pGroup);
  }

  function fGetEnabledToolCount(pGroup: ToolGroup): number {
    return pGroup.tools.filter((pTool) => cToolsStore.mIsToolEnabled(pTool.key)).length;
  }

  function fGetFavicon(pGroup: ToolGroup): string | null {
    if (pGroup.source !== ToolSource.MCP || !pGroup.serverId) return null;

    return cMcpStore.mGetServerFavicon(pGroup.serverId);
  }

  function fIsGroupDisabled(pGroup: ToolGroup): boolean {
    return (
      pGroup.source === ToolSource.MCP &&
      !!pGroup.serverId &&
      !cConversationsStore.mIsMcpServerEnabledForChat(pGroup.serverId)
    );
  }

  function fToggleGroupExpanded(pKey: string): void {
    if (cExpandedGroups.has(pKey)) {
      cExpandedGroups.delete(pKey);
    } else {
      cExpandedGroups.add(pKey);
    }
  }

  function fToggleGroupByKey(pKey: string): void {
    // Find current group by key to get up-to-date tool references
    const cGroup = cActiveGroups.find((pG) => pG.key === pKey);
    if (!cGroup) return;
    cToolsStore.mToggleGroup(cGroup);
  }

  function fHandleOpen(): void {
    if (cToolsStore.mBuiltinTools.length === 0 && !cToolsStore.mLoading) {
      cToolsStore.mFetchBuiltinTools();
    }
    cMcpStore.mRunHealthChecksForServers(cMcpStore.mGetServers().filter((pS) => pS.enabled));
  }

  return {
    expandedGroups: cExpandedGroups,
    get groups() {
      return cGroups;
    },
    get activeGroups() {
      return cActiveGroups;
    },
    get totalToolCount() {
      return cTotalToolCount;
    },
    get noToolsInfoMessage() {
      return cNoToolsInfoMessage;
    },
    isGroupChecked: fIsGroupChecked,
    getEnabledToolCount: fGetEnabledToolCount,
    getFavicon: fGetFavicon,
    isGroupDisabled: fIsGroupDisabled,
    toggleGroupExpanded: fToggleGroupExpanded,
    toggleGroupByKey: fToggleGroupByKey,
    handleOpen: fHandleOpen
  };
}
