import { fTranslate } from '../i18n';
/**
 * conversationsStore - Reactive State Store for Conversations
 *
 * Manages conversation lifecycle, persistence, navigation, and MCP server overrides.
 *
 * **Architecture & Relationships:**
 * - **DatabaseService**: Stateless IndexedDB layer
 * - **conversationsStore** (this): Reactive state + business logic
 * - **chatStore**: Chat-specific state (streaming, loading)
 *
 * **Key Responsibilities:**
 * - Conversation CRUD (create, load, delete)
 * - Message management and tree navigation
 * - MCP server per-chat overrides
 * - Import/Export functionality
 * - Title management with confirmation
 *
 * @see DatabaseService in services/database.ts for IndexedDB operations
 */

import { goto } from '$app/navigation';
import { browser } from '$app/environment';
import { toast } from 'svelte-sonner';
import { DatabaseService } from '$lib/services/database.service';
import { fConfig } from '$lib/stores/settings.svelte';
import { cMcpStore } from '$lib/stores/mcp.svelte';
import { cUiSettingsStore } from '$lib/stores/ui-settings.svelte';
import { filterByLeafNodeId, findLeafNode, generateConversationTitle } from '$lib/utils';
import type { McpServerOverride } from '$lib/types/database';
import { zipSync, unzipSync, strToU8, strFromU8 } from 'fflate';
import {
  AttachmentType,
  MessageRole,
  FileExtensionText,
  MimeTypeText,
  MimeTypeApplication,
  ReasoningEffort,
  SessionRecordType
} from '$lib/enums';
import {
  cIsoDateTimeSeparator,
  cIsoDateTimeSeparatorReplacement,
  cIsoTimestampSliceLength,
  cExportConvIdTrimLength,
  cExportConvNonalnumReplacement,
  cExportConvNameSuffixMaxLength,
  cIsoTimeSeparator,
  cIsoTimeSeparatorReplacement,
  cNonAlphanumericRegex,
  cMultipleUnderscoreRegex,
  cNewline,
  cSessionHarness,
  aZipMagic,
  cBytesPerMegabyte,
  cImportFileMaxBytes,
  cImportJsonlMaxBytes,
  cImportZipMaxEntries,
  cImportZipMaxExtractedBytes
} from '$lib/constants';
import { fIsDataImageUrl } from '$lib/utils/safe-url';

import { cRoutes } from '$lib/constants/routes';
import { RouterService } from '$lib/services/router.service';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';

export interface ConversationTreeItem {
  conversation: DatabaseConversation;
  depth: number;
}

class ConversationsStore {
  /**
   *
   *
   * State
   *
   *
   */

  /** List of all conversations */
  vConversations = $state<DatabaseConversation[]>([]);

  /** Currently active conversation */
  vActiveConversation = $state<DatabaseConversation | null>(null);

  /** Messages in the active conversation (filtered by currNode path) */
  vActiveMessages = $state<DatabaseMessage[]>([]);

  /** Whether the store has been initialized */
  vIsInitialized = $state(false);

  /** Global (non-conversation-specific) reasoning effort default */
  vPendingReasoningEffort = $state<ReasoningEffort>(ConversationsStore.mLoadReasoningEffortDefault());

  /**
   * Reasoning effort default kept in the server's settings document (section
   * "reasoningEffortDefault"), DEFAULT defers to the server
   */
  private static mLoadReasoningEffortDefault(): ReasoningEffort {
    const cSaved = cUiSettingsStore.mGet('reasoningEffortDefault');
    return (cSaved as ReasoningEffort) || ReasoningEffort.DEFAULT;
  }

  /** Save the reasoning effort default on the server */
  private mSaveReasoningEffortDefaults(): void {
    cUiSettingsStore.mSet('reasoningEffortDefault', this.vPendingReasoningEffort);
  }

  constructor() {
    // the saved default arrives with the settings document, after this store is created
    cUiSettingsStore.mRegister('reasoningEffortDefault', () => {
      if (!this.vActiveConversation) {
        this.vPendingReasoningEffort = ConversationsStore.mLoadReasoningEffortDefault();
      }
    });
  }

  /**
   * Callback for updating message content in chatStore.
   * Registered by chatStore to enable cross-store updates without circular dependency.
   */
  private vMessageUpdateCallback:
    | ((pMessageId: string, pUpdates: Partial<DatabaseMessage>) => void)
    | null = null;

  /** In-flight init run; shared by concurrent callers, reset on failure to allow retry */
  private vInitPromise: Promise<void> | null = null;

  /**
   *
   *
   * Lifecycle
   *
   *
   */

  /**
   * Initialize the store by loading conversations from database.
   * Safe to call multiple times: concurrent callers share a single run,
   * and a failed run can be retried by calling again.
   */
  mInit(): Promise<void> {
    if (!browser) return Promise.resolve();
    if (this.vInitPromise) return this.vInitPromise;

    this.vInitPromise = (async () => {
      try {
        await this.mLoadConversations();
        this.vIsInitialized = true;
      } catch (vError) {
        console.error('Failed to initialize conversations:', vError);
        this.vInitPromise = null;
      }
    })();

    return this.vInitPromise;
  }

  /**
   * Register a callback for message updates from other stores.
   * Called by chatStore during initialization.
   */
  mRegisterMessageUpdateCallback(
    pCallback: (pMessageId: string, pUpdates: Partial<DatabaseMessage>) => void
  ): void {
    this.vMessageUpdateCallback = pCallback;
  }

  /**
   *
   *
   * Message Array Operations
   *
   *
   */

  /**
   * Adds a message to the active messages array
   */
  mAddMessageToActive(pMessage: DatabaseMessage): void {
    this.vActiveMessages.push(pMessage);
  }

  /**
   * Updates a message at a specific index in active messages
   */
  mUpdateMessageAtIndex(pIndex: number, pUpdates: Partial<DatabaseMessage>): void {
    const cMessage = pIndex === -1 ? undefined : this.vActiveMessages[pIndex];

    if (!cMessage) return;

    // Assign field by field rather than replacing the object. Replacing it
    // changes the array slot, which invalidates every consumer that merely
    // walks the list - notably ChatMessages.displayMessages, which rebuilds
    // entries for every message in the conversation. Deep $state proxies make
    // per-field writes fine-grained, so only readers of the changed field wake.
    const cTarget = cMessage as unknown as Record<string, unknown>;

    for (const [cKey, cValue] of Object.entries(pUpdates)) {
      if (cTarget[cKey] !== cValue) {
        cTarget[cKey] = cValue;
      }
    }
  }

  /**
   * Finds the index of a message in active messages
   */
  mFindMessageIndex(pMessageId: string): number {
    return this.vActiveMessages.findIndex((pM) => pM.id === pMessageId);
  }

  /**
   * Removes messages from active messages starting at an index
   */
  mSliceActiveMessages(pStartIndex: number): void {
    this.vActiveMessages = this.vActiveMessages.slice(0, pStartIndex);
  }

  /**
   * Removes a message from active messages by index
   */
  mRemoveMessageAtIndex(pIndex: number): DatabaseMessage | undefined {
    if (pIndex !== -1) {
      return this.vActiveMessages.splice(pIndex, 1)[0];
    }
    return undefined;
  }

  /**
   *
   *
   * Conversation CRUD
   *
   *
   */

  /**
   * Loads all conversations from the database
   */
  async mLoadConversations(): Promise<void> {
    const cConversations = await DatabaseService.mGetAllConversations();
    this.vConversations = cConversations;
  }

  /**
   * Creates a new conversation and navigates to it
   * @param pName - Optional name for the conversation
   * @returns The ID of the created conversation
   */
  async mCreateConversation(pName?: string): Promise<string> {
    const cConversationName =
      pName || fTranslate('text_1744850b6bba', { p0: new Date().toLocaleString() });

    // No MCP override list is seeded: getAllMcpServerOverrides resolves
    // servers without a per-conversation override to `mcpServers[i].enabled`,
    // and only explicit toggles are stored on the conversation.
    const cConversation = await DatabaseService.mCreateConversation(cConversationName, {
      reasoningEffort: this.vPendingReasoningEffort
    });

    this.vConversations = [cConversation, ...this.vConversations];
    this.vActiveConversation = cConversation;
    this.vActiveMessages = [];

    await goto(RouterService.mChat(cConversation.id));

    return cConversation.id;
  }

  /**
   * Loads a specific conversation and its messages
   * @param pConvId - The conversation ID to load
   * @returns True if conversation was loaded successfully
   */
  async mLoadConversation(pConvId: string): Promise<boolean> {
    try {
      const cConversation = await DatabaseService.mGetConversation(pConvId);

      if (!cConversation) {
        return false;
      }

      this.vActiveConversation = cConversation;

      if (cConversation.currNode) {
        const cAllMessages = await DatabaseService.mGetConversationMessages(pConvId);
        const cFilteredMessages = filterByLeafNodeId(
          cAllMessages,
          cConversation.currNode,
          false
        ) as DatabaseMessage[];
        this.vActiveMessages = cFilteredMessages;
      } else {
        const cMessages = await DatabaseService.mGetConversationMessages(pConvId);
        this.vActiveMessages = cMessages;
      }

      return true;
    } catch (vError) {
      console.error('Failed to load conversation:', vError);
      return false;
    }
  }

  /**
   * Clears the active conversation and messages.
   */
  mClearActiveConversation(): void {
    this.vActiveConversation = null;
    this.vActiveMessages = [];
    // reload defaults so new chats inherit persisted state
    this.vPendingReasoningEffort = ConversationsStore.mLoadReasoningEffortDefault();
  }

  /**
   * Deletes a conversation and all its messages
   * @param pConvId - The conversation ID to delete
   */
  async mDeleteConversation(
    pConvId: string,
    pOptions?: { deleteWithForks?: boolean }
  ): Promise<void> {
    try {
      await DatabaseService.mDeleteConversation(pConvId, pOptions);

      if (pOptions?.deleteWithForks) {
        // Collect all descendants recursively
        const cIdsToRemove = new SvelteSet([pConvId]);
        const aQueue = [pConvId];
        while (aQueue.length > 0) {
          const cParentId = aQueue.pop()!;
          for (const cC of this.vConversations) {
            if (cC.forkedFromConversationId === cParentId && !cIdsToRemove.has(cC.id)) {
              cIdsToRemove.add(cC.id);
              aQueue.push(cC.id);
            }
          }
        }
        this.vConversations = this.vConversations.filter((pC) => !cIdsToRemove.has(pC.id));

        if (this.vActiveConversation && cIdsToRemove.has(this.vActiveConversation.id)) {
          this.mClearActiveConversation();
          await goto(cRoutes.NEW_CHAT);
        }
      } else {
        // Reparent direct children to deleted conv's parent (or promote to top-level)
        const cDeletedConv = this.vConversations.find((pC) => pC.id === pConvId);
        const cNewParent = cDeletedConv?.forkedFromConversationId;
        this.vConversations = this.vConversations
          .filter((pC) => pC.id !== pConvId)
          .map((pC) =>
            pC.forkedFromConversationId === pConvId
              ? { ...pC, forkedFromConversationId: cNewParent }
              : pC
          );

        if (this.vActiveConversation?.id === pConvId) {
          this.mClearActiveConversation();
          await goto(cRoutes.NEW_CHAT);
        }
      }
    } catch (vError) {
      console.error('Failed to delete conversation:', vError);
    }
  }

  /**
   * Deletes all conversations and their messages
   */
  async mDeleteAll(): Promise<void> {
    try {
      const cAllConversations = await DatabaseService.mGetAllConversations();
      await DatabaseService.mBulkDeleteConversations(cAllConversations.map((pC) => pC.id));

      this.mClearActiveConversation();
      this.vConversations = [];

      toast.success(fTranslate('text_2f3c5f7a351a'));

      await goto(cRoutes.NEW_CHAT);
    } catch (vError) {
      console.error('Failed to delete all conversations:', vError);
      toast.error(fTranslate('text_4c0d52d097ba'));
    }
  }

  /**
   * Deletes multiple conversations in sequence.
   * Mirrors deleteConversation() per-id; navigates to NEW_CHAT only if the
   * currently-open chat was among the deleted ones.
   * @param pConvIds - Conversation IDs to delete
   */
  async mBulkDeleteConversations(pConvIds: string[]): Promise<void> {
    if (pConvIds.length === 0) return;

    try {
      const cIdsToRemove = new SvelteSet(pConvIds);
      // Collect all descendants recursively so the local cache stays consistent
      // even when deleteWithForks is omitted.
      const aQueue = [...pConvIds];
      while (aQueue.length > 0) {
        const cParentId = aQueue.pop()!;
        for (const cC of this.vConversations) {
          if (cC.forkedFromConversationId === cParentId && !cIdsToRemove.has(cC.id)) {
            cIdsToRemove.add(cC.id);
            aQueue.push(cC.id);
          }
        }
      }

      const cActiveWasDeleted =
        this.vActiveConversation !== null && cIdsToRemove.has(this.vActiveConversation.id);

      await DatabaseService.mBulkDeleteConversations([...cIdsToRemove]);

      this.vConversations = this.vConversations.filter((pC) => !cIdsToRemove.has(pC.id));

      if (cActiveWasDeleted) {
        this.mClearActiveConversation();
        await goto(cRoutes.NEW_CHAT);
      }

      toast.success(
        cIdsToRemove.size === 1
          ? fTranslate('text_09d6990a101a')
          : `${cIdsToRemove.size} conversations deleted`
      );
    } catch (vError) {
      console.error('Failed to bulk delete conversations:', vError);
      toast.error(fTranslate('text_4c0d52d097ba'));
    }
  }

  /**
   * Toggles the pinned state of each conversation individually.
   * Mixed-pin selections are intentionally not normalised here; the bulk
   * action UI surfaces them as a disabled mixed-state instead.
   * @param pConvIds - Conversation IDs to toggle
   */
  async mBulkToggleConversationPin(pConvIds: string[]): Promise<void> {
    if (pConvIds.length === 0) return;

    try {
      const cUpdates = await DatabaseService.mBulkToggleConversationPins(pConvIds);

      const cActiveId = this.vActiveConversation?.id;
      if (cActiveId && cUpdates.has(cActiveId)) {
        this.vActiveConversation = {
          ...this.vActiveConversation!,
          pinned: cUpdates.get(cActiveId)!
        };
      }
      for (let vI = 0; vI < this.vConversations.length; vI++) {
        const cNewPinned = cUpdates.get(this.vConversations[vI].id);
        if (cNewPinned !== undefined) this.vConversations[vI].pinned = cNewPinned;
      }

      toast.success(
        pConvIds.length === 1
          ? fTranslate('text_ebd14ca8d807')
          : fTranslate('text_4f3ea7bbbf7c', { p0: pConvIds.length })
      );
    } catch (vError) {
      console.error('Failed to bulk toggle pin:', vError);
      toast.error(fTranslate('text_cbc9a6188b1f'));
    }
  }

  /**
   * Bundles the given conversations into a single zip archive and triggers a
   * browser download (one JSONL file per conversation).
   * @param pConvIds - Conversation IDs to export
   */
  async mBulkExportConversations(pConvIds: string[]): Promise<void> {
    if (pConvIds.length === 0) return;

    try {
      const cExported = await this.mGetConversationsForExport(pConvIds);
      if (cExported.length === 0) {
        toast.error(fTranslate('text_bf0779e06dd8'));
        return;
      }

      this.mDownloadConversationsArchive(cExported);

      toast.success(
        cExported.length === 1
          ? fTranslate('text_ad1e5f4b1983')
          : `${cExported.length} conversations exported`
      );
    } catch (vError) {
      console.error('Failed to bulk export conversations:', vError);
      toast.error(fTranslate('text_96f775dbb3fc'));
    }
  }

  /**
   *
   *
   * Message Management
   *
   *
   */

  /**
   * Refreshes active messages based on currNode after branch navigation.
   */
  async mRefreshActiveMessages(): Promise<void> {
    if (!this.vActiveConversation) return;

    const cAllMessages = await DatabaseService.mGetConversationMessages(
      this.vActiveConversation.id
    );

    if (cAllMessages.length === 0) {
      this.vActiveMessages = [];
      return;
    }

    const cLeafNodeId =
      this.vActiveConversation.currNode ||
      cAllMessages.reduce((pLatest, pMsg) => (pMsg.timestamp > pLatest.timestamp ? pMsg : pLatest))
        .id;

    const cCurrentPath = filterByLeafNodeId(cAllMessages, cLeafNodeId, false) as DatabaseMessage[];

    this.vActiveMessages = cCurrentPath;
  }

  /**
   * Gets all messages for a specific conversation
   * @param pConvId - The conversation ID
   * @returns Array of messages
   */
  async mGetConversationMessages(pConvId: string): Promise<DatabaseMessage[]> {
    return await DatabaseService.mGetConversationMessages(pConvId);
  }

  /** Read persisted metadata and the full message tree together, preserving selection order. */
  async mGetConversationsForExport(pConversationIds: string[]): Promise<ExportedConversation[]> {
    const dConversations = await DatabaseService.mGetConversationsWithMessages(pConversationIds);
    return pConversationIds
      .map((pId) => dConversations.get(pId))
      .filter((pEntry): pEntry is ExportedConversation => pEntry !== undefined);
  }

  /**
   *
   *
   * Title Management
   *
   *
   */

  /**
   * Updates the name of a conversation.
   * @param pConvId - The conversation ID to update
   * @param pName - The new name for the conversation
   */
  async mUpdateConversationName(pConvId: string, pName: string): Promise<void> {
    try {
      await DatabaseService.mUpdateConversation(pConvId, { name: pName });

      const cConvIndex = this.vConversations.findIndex((pC) => pC.id === pConvId);

      if (cConvIndex !== -1) {
        this.vConversations[cConvIndex].name = pName;
      }

      if (this.vActiveConversation?.id === pConvId) {
        this.vActiveConversation = { ...this.vActiveConversation, name: pName };
      }
    } catch (vError) {
      console.error('Failed to update conversation name:', vError);
    }
  }

  /**
   * Toggles the pinned status of a conversation.
   * @param pConvId - The conversation ID to toggle
   * @returns The new pinned status
   */
  async mToggleConversationPin(pConvId: string): Promise<boolean> {
    try {
      const cNewPinnedState = await DatabaseService.mToggleConversationPin(pConvId);

      const cConvIndex = this.vConversations.findIndex((pC) => pC.id === pConvId);

      if (cConvIndex !== -1) {
        this.vConversations[cConvIndex].pinned = cNewPinnedState;
      }

      if (this.vActiveConversation?.id === pConvId) {
        this.vActiveConversation = { ...this.vActiveConversation, pinned: cNewPinnedState };
      }

      return cNewPinnedState;
    } catch (vError) {
      console.error('Failed to toggle conversation pin:', vError);
      return false;
    }
  }

  /**
   * Marks a conversation as recently active: stamps lastModified (persisted)
   * and moves it to the top of the list. Only message-activity flows call
   * this; metadata updates (rename, pin, settings) do not.
   *
   * @param pConvId - Conversation that produced the activity, defaults to the active one
   */
  mUpdateConversationTimestamp(pConvId?: string): void {
    const cTargetId = pConvId ?? this.vActiveConversation?.id;
    if (!cTargetId) return;

    const cNow = Date.now();

    const cChatIndex = this.vConversations.findIndex((pC) => pC.id === cTargetId);

    if (cChatIndex !== -1) {
      this.vConversations[cChatIndex].lastModified = cNow;
      const cUpdatedConv = this.vConversations.splice(cChatIndex, 1)[0];
      this.vConversations = [cUpdatedConv, ...this.vConversations];
    }

    if (this.vActiveConversation?.id === cTargetId) {
      this.vActiveConversation = { ...this.vActiveConversation, lastModified: cNow };
    }

    DatabaseService.mUpdateConversation(cTargetId, { lastModified: cNow }).catch((pError) =>
      console.error('Failed to update conversation timestamp:', pError)
    );
  }

  /**
   * Updates the current node of the active conversation
   * @param pNodeId - The new current node ID
   */
  async mUpdateCurrentNode(pNodeId: string): Promise<void> {
    if (!this.vActiveConversation) return;

    await DatabaseService.mUpdateCurrentNode(this.vActiveConversation.id, pNodeId);
    this.vActiveConversation = { ...this.vActiveConversation, currNode: pNodeId };
  }

  /**
   *
   *
   * Branch Navigation
   *
   *
   */

  /**
   * Navigates to a specific sibling branch by updating currNode and refreshing messages.
   * @param pSiblingId - The sibling message ID to navigate to
   */
  async mNavigateToSibling(pSiblingId: string): Promise<void> {
    if (!this.vActiveConversation) return;

    const cAllMessages = await DatabaseService.mGetConversationMessages(
      this.vActiveConversation.id
    );
    const cRootMessage = cAllMessages.find((pM) => pM.type === 'root' && pM.parent === null);
    const cCurrentFirstUserMessage = this.vActiveMessages.find(
      (pM) => pM.role === MessageRole.USER && pM.parent === cRootMessage?.id
    );

    const cCurrentLeafNodeId = findLeafNode(cAllMessages, pSiblingId);

    await DatabaseService.mUpdateCurrentNode(this.vActiveConversation.id, cCurrentLeafNodeId);
    this.vActiveConversation = { ...this.vActiveConversation, currNode: cCurrentLeafNodeId };
    await this.mRefreshActiveMessages();

    if (cRootMessage && this.vActiveMessages.length > 0) {
      const cNewFirstUserMessage = this.vActiveMessages.find(
        (pM) => pM.role === MessageRole.USER && pM.parent === cRootMessage.id
      );

      if (
        cNewFirstUserMessage &&
        cNewFirstUserMessage.content.trim() &&
        (!cCurrentFirstUserMessage ||
          cNewFirstUserMessage.id !== cCurrentFirstUserMessage.id ||
          cNewFirstUserMessage.content.trim() !== cCurrentFirstUserMessage.content.trim())
      ) {
        await this.mUpdateConversationName(
          this.vActiveConversation.id,
          generateConversationTitle(
            cNewFirstUserMessage.content,
            Boolean(fConfig().titleGenerationUseFirstLine)
          )
        );
      }
    }
  }

  /**
   *
   *
   * MCP Server Overrides
   *
   *
   */

  /**
   * Resolve the default enabled value for a server: its own `enabled`
   * flag in `mcpServers`, so the global on/off state lives in one place.
   */
  #getDefaultOverride(pServerId: string): McpServerOverride | undefined {
    const cServer = cMcpStore.mGetServers().find((pS) => pS.id === pServerId);
    if (!cServer) return undefined;
    return { serverId: pServerId, enabled: cServer.enabled };
  }

  /**
   * Gets the effective MCP server override for a specific server.
   * A per-conversation override wins when present; a server without one
   * resolves to its `mcpServers[i].enabled` default.
   * @param pServerId - The server ID to check
   * @returns The effective override, undefined if no matching server
   */
  mGetMcpServerOverride(pServerId: string): McpServerOverride | undefined {
    const cOverride = this.vActiveConversation?.mcpServerOverrides?.find(
      (pO: McpServerOverride) => pO.serverId === pServerId
    );
    if (cOverride) return cOverride;
    return this.#getDefaultOverride(pServerId);
  }

  /**
   * Gets the effective override list for the current conversation:
   * one entry per configured server, resolved per server. The stored
   * per-conversation list is sparse and only holds explicit toggles.
   */
  mGetAllMcpServerOverrides(): McpServerOverride[] {
    const cOverrides = this.vActiveConversation?.mcpServerOverrides;
    return cMcpStore.mGetServers().map((pS) => {
      const cOverride = cOverrides?.find((pO: McpServerOverride) => pO.serverId === pS.id);
      return { serverId: pS.id, enabled: cOverride?.enabled ?? pS.enabled };
    });
  }

  /**
   * Checks if an MCP server is enabled for the active conversation.
   * @param pServerId - The server ID to check
   * @returns True if server is enabled for this conversation
   */
  mIsMcpServerEnabledForChat(pServerId: string): boolean {
    const cOverride = this.mGetMcpServerOverride(pServerId);
    return cOverride?.enabled ?? false;
  }

  /**
   * Sets or removes MCP server override for the active conversation.
   * If no conversation exists, persists `enabled` onto `mcpServers[i].enabled`
   * (the single source of truth for new-chat defaults).
   * @param pServerId - The server ID to override
   * @param pEnabled - The enabled state, or undefined to remove per-conversation override
   */
  async mSetMcpServerOverride(pServerId: string, pEnabled: boolean | undefined): Promise<void> {
    if (!this.vActiveConversation) {
      if (pEnabled !== undefined) {
        cMcpStore.mUpdateServer(pServerId, { enabled: pEnabled });
      }
      return;
    }

    // Clone to plain objects to avoid Proxy serialization issues with IndexedDB
    const cCurrentOverrides = (this.vActiveConversation.mcpServerOverrides || []).map(
      (pO: McpServerOverride) => ({
        serverId: pO.serverId,
        enabled: pO.enabled
      })
    );
    let aNewOverrides: McpServerOverride[];

    if (pEnabled === undefined) {
      aNewOverrides = cCurrentOverrides.filter(
        (pO: McpServerOverride) => pO.serverId !== pServerId
      );
    } else {
      const cExistingIndex = cCurrentOverrides.findIndex(
        (pO: McpServerOverride) => pO.serverId === pServerId
      );
      if (cExistingIndex >= 0) {
        aNewOverrides = [...cCurrentOverrides];
        aNewOverrides[cExistingIndex] = { serverId: pServerId, enabled: pEnabled };
      } else {
        aNewOverrides = [...cCurrentOverrides, { serverId: pServerId, enabled: pEnabled }];
      }
    }

    await DatabaseService.mUpdateConversation(this.vActiveConversation.id, {
      mcpServerOverrides: aNewOverrides.length > 0 ? aNewOverrides : undefined
    });

    this.vActiveConversation = {
      ...this.vActiveConversation,
      mcpServerOverrides: aNewOverrides.length > 0 ? aNewOverrides : undefined
    };

    const cConvIndex = this.vConversations.findIndex(
      (pC) => pC.id === this.vActiveConversation!.id
    );
    if (cConvIndex !== -1) {
      this.vConversations[cConvIndex].mcpServerOverrides =
        aNewOverrides.length > 0 ? aNewOverrides : undefined;
    }
  }

  /**
   * Toggles MCP server enabled state for the active conversation.
   * @param pServerId - The server ID to toggle
   */
  async mToggleMcpServerForChat(pServerId: string): Promise<void> {
    const cCurrentEnabled = this.mIsMcpServerEnabledForChat(pServerId);
    await this.mSetMcpServerOverride(pServerId, !cCurrentEnabled);
  }

  /**
   * Removes MCP server override for the active conversation.
   * @param pServerId - The server ID to remove override for
   */
  async mRemoveMcpServerOverride(pServerId: string): Promise<void> {
    await this.mSetMcpServerOverride(pServerId, undefined);
  }

  /**
   * Gets the effective reasoning effort for the active conversation.
   * Returns the conversation override if set, otherwise the global default.
   * DEFAULT means no override is sent and the server decides.
   */
  mGetReasoningEffort(): ReasoningEffort {
    if (this.vActiveConversation) {
      if (this.vActiveConversation.reasoningEffort !== undefined) {
        return this.vActiveConversation.reasoningEffort;
      }
      // conversations created before the tri-state store an explicit
      // opt-out only as thinkingEnabled = false
      if (this.vActiveConversation.thinkingEnabled === false) {
        return ReasoningEffort.OFF;
      }
    }
    return this.vPendingReasoningEffort;
  }

  /**
   * Sets the reasoning effort for the active conversation.
   * If no conversation exists, stores the global default.
   * @param pEffort - The effort level ('default' | 'off' | 'low' | 'medium' | 'high' | 'max')
   */
  async mSetReasoningEffort(pEffort: ReasoningEffort): Promise<void> {
    if (!this.vActiveConversation) {
      this.vPendingReasoningEffort = pEffort;
      this.mSaveReasoningEffortDefaults();
      return;
    }

    this.vActiveConversation = {
      ...this.vActiveConversation,
      reasoningEffort: pEffort
    };

    await DatabaseService.mUpdateConversation(this.vActiveConversation.id, {
      reasoningEffort: pEffort
    });

    const cConvIndex = this.vConversations.findIndex(
      (pC) => pC.id === this.vActiveConversation!.id
    );
    if (cConvIndex !== -1) {
      this.vConversations[cConvIndex].reasoningEffort = pEffort;
    }
  }

  /**
   * Forks a conversation at a specific message, creating a new conversation
   * containing messages from root up to the target message, then navigates to it.
   *
   * @param pMessageId - The message ID to fork at
   * @param pOptions - Fork options (name and whether to include attachments)
   * @returns The new conversation ID, or null if fork failed
   */
  async mForkConversation(
    pMessageId: string,
    pOptions: { name: string; includeAttachments: boolean }
  ): Promise<string | null> {
    if (!this.vActiveConversation) return null;

    try {
      const cNewConv = await DatabaseService.mForkConversation(
        this.vActiveConversation.id,
        pMessageId,
        pOptions
      );

      this.vConversations = [cNewConv, ...this.vConversations];

      await goto(RouterService.mChat(cNewConv.id));

      toast.success(fTranslate('text_95de71e25038'));

      return cNewConv.id;
    } catch (vError) {
      console.error('Failed to fork conversation:', vError);
      toast.error(fTranslate('text_32ad3abe4479'));

      return null;
    }
  }

  /**
   *
   *
   * Import & Export
   *
   *
   */

  /**
   * Generates a sanitized filename for a conversation export
   * @param pConversation - The conversation metadata
   * @param pMsgs - Optional array of messages belonging to the conversation
   * @returns The generated filename string
   */
  mGenerateConversationFilename(
    pConversation: { id?: string; name?: string },
    pMsgs?: DatabaseMessage[]
  ): string {
    const cConversationName = (pConversation.name ?? '').trim().toLowerCase();

    const cSanitizedName = cConversationName
      .replace(cNonAlphanumericRegex, cExportConvNonalnumReplacement)
      .replace(cMultipleUnderscoreRegex, '_')
      .substring(0, cExportConvNameSuffixMaxLength);

    // If we have messages, use the timestamp of the newest message
    const cReferenceDate = pMsgs?.length
      ? new Date(Math.max(...pMsgs.map((pM) => pM.timestamp)))
      : new Date();

    const cIso = cReferenceDate.toISOString().slice(0, cIsoTimestampSliceLength);
    const cFormattedDate = cIso
      .replace(cIsoDateTimeSeparator, cIsoDateTimeSeparatorReplacement)
      .replaceAll(cIsoTimeSeparator, cIsoTimeSeparatorReplacement);
    const cTrimmedConvId = pConversation.id?.slice(0, cExportConvIdTrimLength) ?? '';
    return `${cFormattedDate}_conv_${cTrimmedConvId}_${cSanitizedName}${FileExtensionText.JSONL}`;
  }

  /**
   * Serializes a session (a conversation with its messages) as JSONL.
   * The first line is the session header (a `SessionRecordType.SESSION` record
   * carrying the conversation properties); each subsequent line is a single message.
   * @param pData - The exported conversation payload
   * @returns The JSONL string (one record per line)
   */
  mSerializeSessionToJsonl(pData: ExportedConversation): string {
    const { conv: cConv, messages: cMessages } = pData;

    const cSessionLine = JSON.stringify({
      type: SessionRecordType.SESSION,
      harness: cSessionHarness,
      ...cConv
    });
    const cMessageLines = cMessages.map((pMessage: DatabaseMessage) => {
      // `toolCalls` is stored as a JSON string; drop it when empty, otherwise parse it.
      const { toolCalls: cToolCalls, ...cRest } = pMessage;
      const cNormalized = cToolCalls ? { ...cRest, toolCalls: JSON.parse(cToolCalls) } : cRest;

      return JSON.stringify({ type: SessionRecordType.MESSAGE, message: cNormalized });
    });

    return [cSessionLine, ...cMessageLines].join(cNewline);
  }

  /**
   * Parses the JSONL session format produced by {@link mSerializeSessionToJsonl}.
   * A `SessionRecordType.SESSION` line starts a new session; following
   * `SessionRecordType.MESSAGE` lines are appended to it. Supports multiple
   * sessions in a single file.
   * @param pText - The JSONL file contents
   * @returns The parsed conversations with their messages
   */
  mParseSessionsJsonl(pText: string): ExportedConversation[] {
    const aSessions: ExportedConversation[] = [];
    let vCurrent: ExportedConversation | null = null;

    for (const cLine of pText.split(cNewline)) {
      const cTrimmed = cLine.trim();
      if (!cTrimmed) continue;

      const cRecord = JSON.parse(cTrimmed);
      if (!fIsPlainObject(cRecord)) continue;

      if (cRecord.type === SessionRecordType.SESSION) {
        // Drop the discriminator and harness marker; the rest is the conversation.
        const dConv = { ...cRecord };
        delete dConv.type;
        delete dConv.harness;
        if (!fIsImportedConversation(dConv)) {
          throw new Error(fTranslate('text_c863959c59d0'));
        }
        vCurrent = { conv: dConv, messages: [] };
        aSessions.push(vCurrent);
      } else if (cRecord.type === SessionRecordType.MESSAGE) {
        if (!vCurrent) {
          throw new Error(fTranslate('text_8b7bddc8d9c3'));
        }

        if (!fIsPlainObject(cRecord.message)) {
          throw new Error(fTranslate('text_c34a118bb0e0'));
        }
        const dMessage = { ...cRecord.message };
        // `toolCalls` is parsed to an array on export; the DB stores it as a string.
        if (dMessage.toolCalls !== undefined && typeof dMessage.toolCalls !== 'string') {
          dMessage.toolCalls = JSON.stringify(dMessage.toolCalls);
        }
        if (!fIsImportedMessage(dMessage)) {
          throw new Error(fTranslate('text_c34a118bb0e0'));
        }
        // Each message must belong to the session that contains it, never to another conversation
        if (dMessage.convId !== vCurrent.conv.id) {
          throw new Error(fTranslate('text_8442c2fbba9d'));
        }
        fDropUnsafeImageAttachments(dMessage);
        vCurrent.messages.push(dMessage);
      }
      // Ignore unknown record types for forward compatibility.
    }

    return aSessions;
  }

  /**
   * Reports whether the text is the JSONL session format, whose first non-empty
   * line is a `SessionRecordType.SESSION` record.
   * @param pText - The file contents
   */
  private mIsSessionsJsonl(pText: string): boolean {
    const cTrimmed = pText.trimStart();
    const cLineEnd = cTrimmed.indexOf(cNewline);
    const cFirstLine = cLineEnd === -1 ? cTrimmed : cTrimmed.slice(0, cLineEnd);

    try {
      return JSON.parse(cFirstLine).type === SessionRecordType.SESSION;
    } catch {
      // Not a standalone JSON record, so not the JSONL format.
      return false;
    }
  }

  /**
   * Parses an import file into conversations, accepting the JSONL and ZIP
   * formats. The format comes from the contents, so an import works whatever
   * the file is named.
   * @param pFile - The user-selected file
   * @returns The parsed conversations with their messages
   */
  async mParseImportFile(pFile: File): Promise<ExportedConversation[]> {
    // Bounded before the file is read: the archive is expanded in memory on the UI thread
    if (pFile.size > cImportFileMaxBytes) {
      throw new Error(
        fTranslate('text_5f79d6ab101f', { p0: cImportFileMaxBytes / cBytesPerMegabyte })
      );
    }

    const cBytes = new Uint8Array(await pFile.arrayBuffer());

    if (aZipMagic.every((pByte, pIndex) => cBytes[pIndex] === pByte)) {
      // The filter sees each entry's declared sizes before it is decompressed; fflate never
      // writes more than the declared size, so these checks bound the memory it allocates
      let vEntryCount = 0;
      let vExtractedBytes = 0;
      const cEntries = unzipSync(cBytes, {
        filter: (pInfo) => {
          vEntryCount += 1;
          if (vEntryCount > cImportZipMaxEntries) {
            throw new Error(fTranslate('text_926b21c2ce0d', { p0: cImportZipMaxEntries }));
          }
          if (!pInfo.name.toLowerCase().endsWith(FileExtensionText.JSONL)) return false;
          if (pInfo.originalSize > cImportJsonlMaxBytes) {
            throw new Error(
              fTranslate('text_5e1ca8cb6c63', { p0: cImportJsonlMaxBytes / cBytesPerMegabyte })
            );
          }
          vExtractedBytes += pInfo.originalSize;
          if (vExtractedBytes > cImportZipMaxExtractedBytes) {
            throw new Error(
              fTranslate('text_e5f389d16037', {
                p0: cImportZipMaxExtractedBytes / cBytesPerMegabyte
              })
            );
          }
          return true;
        }
      });
      const aSessions: ExportedConversation[] = [];
      for (const cEntryBytes of Object.values(cEntries)) {
        aSessions.push(...this.mParseSessionsJsonl(strFromU8(cEntryBytes)));
      }
      return aSessions;
    }

    if (cBytes.length > cImportJsonlMaxBytes) {
      throw new Error(
        fTranslate('text_5f79d6ab101f', { p0: cImportJsonlMaxBytes / cBytesPerMegabyte })
      );
    }

    const cText = strFromU8(cBytes);

    if (this.mIsSessionsJsonl(cText)) {
      return this.mParseSessionsJsonl(cText);
    }

    throw new Error(fTranslate('text_f91f79f34955'));
  }

  /**
   * Triggers a browser download of the provided exported conversation data
   * @param pData - The exported conversation payload (a single conversation with its messages)
   * @param pFilename - Filename; if omitted, a deterministic name is generated
   */
  mDownloadConversationFile(pData: ExportedConversation, pFilename?: string): void {
    const { conv: cConversation, messages: cMsgs } = pData;

    if (!cConversation) {
      console.error('Invalid data: missing conversation');
      return;
    }

    const cDownloadFilename = pFilename ?? this.mGenerateConversationFilename(cConversation, cMsgs);

    const cJsonl = this.mSerializeSessionToJsonl(pData);
    const cBlob = new Blob([cJsonl], { type: MimeTypeText.JSONL });
    this.mTriggerDownload(cBlob, cDownloadFilename);
  }

  /**
   * Triggers a browser download of multiple conversations as a `.zip`, one
   * `.jsonl` file per conversation.
   * @param pData - The conversations to export
   */
  mDownloadConversationsArchive(pData: ExportedConversation[]): void {
    if (pData.length === 0) {
      console.error('Invalid data: no conversations to export');
      return;
    }

    const cUsedNames = new SvelteSet<string>();
    const dFiles: Record<string, Uint8Array> = {};

    for (const cSession of pData) {
      const cBaseName = this.mGenerateConversationFilename(cSession.conv, cSession.messages);

      // Disambiguate any duplicate filenames within the archive.
      let vEntryName = cBaseName;
      let vSuffix = 1;
      while (cUsedNames.has(vEntryName)) {
        vEntryName = cBaseName.replace(
          new RegExp(`${FileExtensionText.JSONL}$`),
          `_${vSuffix++}${FileExtensionText.JSONL}`
        );
      }
      cUsedNames.add(vEntryName);

      dFiles[vEntryName] = strToU8(this.mSerializeSessionToJsonl(cSession));
    }

    const cArchiveName = `${new Date().toISOString().split(cIsoDateTimeSeparator)[0]}_conversations${FileExtensionText.ZIP}`;

    const cZipped = zipSync(dFiles);
    const cBlob = new Blob([cZipped], { type: MimeTypeApplication.ZIP });
    this.mTriggerDownload(cBlob, cArchiveName);
  }

  /**
   * Triggers a browser download of a blob under the given filename.
   */
  private mTriggerDownload(pBlob: Blob, pFilename: string): void {
    const cUrl = URL.createObjectURL(pBlob);
    const cA = document.createElement('a');
    cA.href = cUrl;
    cA.download = pFilename;
    document.body.appendChild(cA);
    cA.click();
    document.body.removeChild(cA);
    URL.revokeObjectURL(cUrl);
  }

  /**
   * Downloads a single conversation as a JSONL file, serializing the full message tree.
   * @param pConvId - The conversation ID to download
   */
  async mDownloadConversation(pConvId: string): Promise<void> {
    const [vExportedConversation] = await this.mGetConversationsForExport([pConvId]);
    if (!vExportedConversation) return;
    this.mDownloadConversationFile(vExportedConversation);
  }

  /**
   * Imports conversations from provided data (without file picker)
   * @param pData - Array of conversation data with messages
   * @returns The conversations written to the database and the ones skipped
   */
  async mImportConversationsData(
    pData: ExportedConversation[]
  ): Promise<{ imported: DatabaseConversation[]; skipped: DatabaseConversation[] }> {
    const cResult = await DatabaseService.mImportConversations(pData);
    await this.mLoadConversations();
    return cResult;
  }
}

/**
 * Structural checks for imported records. An import file comes from outside the
 * app, so its records are validated instead of being trusted as typed values.
 */
function fIsPlainObject(pValue: unknown): pValue is Record<string, unknown> {
  return typeof pValue === 'object' && pValue !== null && !Array.isArray(pValue);
}

function fIsOptionalString(pValue: unknown): boolean {
  return pValue === undefined || typeof pValue === 'string';
}

function fIsImportedConversation(
  pValue: Record<string, unknown>
): pValue is Record<string, unknown> & DatabaseConversation {
  return (
    typeof pValue.id === 'string' &&
    pValue.id.length > 0 &&
    typeof pValue.name === 'string' &&
    typeof pValue.lastModified === 'number' &&
    Number.isFinite(pValue.lastModified) &&
    (pValue.currNode === null || typeof pValue.currNode === 'string') &&
    (pValue.mcpServerOverrides === undefined || Array.isArray(pValue.mcpServerOverrides)) &&
    fIsOptionalString(pValue.forkedFromConversationId)
  );
}

function fIsImportedMessage(
  pValue: Record<string, unknown>
): pValue is Record<string, unknown> & DatabaseMessage {
  return (
    typeof pValue.id === 'string' &&
    pValue.id.length > 0 &&
    typeof pValue.convId === 'string' &&
    typeof pValue.type === 'string' &&
    typeof pValue.role === 'string' &&
    typeof pValue.content === 'string' &&
    typeof pValue.timestamp === 'number' &&
    Number.isFinite(pValue.timestamp) &&
    (pValue.parent === null || typeof pValue.parent === 'string') &&
    Array.isArray(pValue.children) &&
    pValue.children.every((pChild: unknown) => typeof pChild === 'string') &&
    (pValue.extra === undefined ||
      (Array.isArray(pValue.extra) &&
        pValue.extra.every(
          (pExtra: unknown) => fIsPlainObject(pExtra) && typeof pExtra.type === 'string'
        ))) &&
    fIsOptionalString(pValue.reasoningContent) &&
    fIsOptionalString(pValue.toolCalls) &&
    fIsOptionalString(pValue.completionId) &&
    fIsOptionalString(pValue.toolCallId) &&
    fIsOptionalString(pValue.model)
  );
}

/**
 * Keeps only image attachments whose source is a base64 image data URL, the
 * only form the app stores; anything else (javascript:, other data: types,
 * remote URLs) would otherwise reach image sources and links when rendered.
 */
function fDropUnsafeImageAttachments(pMessage: DatabaseMessage): void {
  if (!pMessage.extra) return;
  pMessage.extra = pMessage.extra.filter(
    (pExtra: DatabaseMessageExtra) =>
      pExtra.type !== AttachmentType.IMAGE || fIsDataImageUrl(pExtra.base64Url)
  );
}

export const cConversationsStore = new ConversationsStore();

// Auto-initialize in browser
if (browser) {
  cConversationsStore.mInit();
}

export const fConversations = () => cConversationsStore.vConversations;
export const fActiveConversation = () => cConversationsStore.vActiveConversation;
export const fActiveMessages = () => cConversationsStore.vActiveMessages;
export const fIsConversationsInitialized = () => cConversationsStore.vIsInitialized;

/**
 * Builds a flat tree of conversations with depth levels for nested forks.
 * Accepts a pre-filtered list so search filtering stays in the component.
 *
 * Output order matches the sidebar render exactly: pinned first, then
 * unpinned by lastModified desc, with forks interleaved under their parents.
 * Range-select / marquee in the sidebar rely on this alignment.
 */

// Pinned conversations first, then by lastModified descending
const fComparePinnedThenRecent = (pA: DatabaseConversation, pB: DatabaseConversation) => {
  if (pA.pinned && !pB.pinned) return -1;
  if (!pA.pinned && pB.pinned) return 1;
  return pB.lastModified - pA.lastModified;
};

export function fBuildConversationTree(pConvs: DatabaseConversation[]): ConversationTreeItem[] {
  const cChildrenByParent = new SvelteMap<string, DatabaseConversation[]>();
  const cForkIds = new SvelteSet<string>();

  for (const cConv of pConvs) {
    if (cConv.forkedFromConversationId) {
      cForkIds.add(cConv.id);

      const cSiblings = cChildrenByParent.get(cConv.forkedFromConversationId) || [];

      cSiblings.push(cConv);
      cChildrenByParent.set(cConv.forkedFromConversationId, cSiblings);
    }
  }

  const aResult: ConversationTreeItem[] = [];
  const cVisited = new SvelteSet<string>();

  function fWalk(pConv: DatabaseConversation, pDepth: number) {
    cVisited.add(pConv.id);
    aResult.push({ conversation: pConv, depth: pDepth });

    const cChildren = cChildrenByParent.get(pConv.id);
    if (cChildren) {
      cChildren.sort(fComparePinnedThenRecent);

      for (const cChild of cChildren) {
        fWalk(cChild, pDepth + 1);
      }
    }
  }

  const cRoots = pConvs.filter((pC) => !cForkIds.has(pC.id)).sort(fComparePinnedThenRecent);
  for (const cRoot of cRoots) {
    fWalk(cRoot, 0);
  }

  for (const cConv of pConvs) {
    if (!cVisited.has(cConv.id)) {
      fWalk(cConv, 1);
    }
  }

  return aResult;
}
