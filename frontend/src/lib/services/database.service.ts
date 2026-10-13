import { fTranslate } from '../i18n';
import Dexie, { type EntityTable } from 'dexie';
import { findDescendantMessages, uuid, filterByLeafNodeId } from '$lib/utils';
import { cIdxdbTables, cIdxdbStores, cStorageAppName } from '$lib/constants';
import { MessageRole } from '$lib/enums';
import type { McpServerOverride } from '$lib/types/database';
import type { ExportedConversation } from '$lib/types/database';

class MgsUiDatabase extends Dexie {
  [cIdxdbTables.conversations]!: EntityTable<DatabaseConversation, string>;
  [cIdxdbTables.messages]!: EntityTable<DatabaseMessage, string>;

  constructor() {
    super(cStorageAppName);

    this.version(1).stores(cIdxdbStores);
  }
}

const cDb = new MgsUiDatabase();

export class DatabaseService {
  /**
   *
   *
   * Conversations
   *
   *
   */

  /**
   * Creates a new conversation.
   *
   * @param pName - Name of the conversation
   * @param pFields - Optional extra fields (e.g. reasoningEffort)
   * @returns The created conversation
   */
  static async mCreateConversation(
    pName: string,
    pFields?: Partial<Omit<DatabaseConversation, 'id' | 'name' | 'lastModified'>>
  ): Promise<DatabaseConversation> {
    const dConversation: DatabaseConversation = {
      id: uuid(),
      name: pName,
      lastModified: Date.now(),
      currNode: '',
      ...pFields
    };

    await cDb[cIdxdbTables.conversations].add(dConversation);
    return dConversation;
  }

  /**
   *
   *
   * Messages
   *
   *
   */

  /**
   * Creates a new message branch by adding a message and updating parent/child relationships.
   * Also updates the conversation's currNode to point to the new message.
   *
   * @param pMessage - Message to add (without id)
   * @param pParentId - Parent message ID to attach to
   * @returns The created message
   */
  static async mCreateMessageBranch(
    pMessage: Omit<DatabaseMessage, 'id'>,
    pParentId: string | null
  ): Promise<DatabaseMessage> {
    return await cDb.transaction(
      'rw',
      [cDb[cIdxdbTables.conversations], cDb[cIdxdbTables.messages]],
      async () => {
        // Handle null parent (root message case)
        if (pParentId !== null) {
          const cParentMessage = await cDb[cIdxdbTables.messages].get(pParentId);
          if (!cParentMessage) {
            throw new Error(fTranslate('text_cb1c1c4c66b5', { p0: pParentId }));
          }
        }

        const dNewMessage: DatabaseMessage = {
          ...pMessage,
          id: uuid(),
          parent: pParentId,
          toolCalls: pMessage.toolCalls ?? '',
          children: []
        };

        await cDb[cIdxdbTables.messages].add(dNewMessage);

        // Update parent's children array if parent exists
        if (pParentId !== null) {
          const cParentMessage = await cDb[cIdxdbTables.messages].get(pParentId);
          if (cParentMessage) {
            await cDb[cIdxdbTables.messages].update(pParentId, {
              children: [...cParentMessage.children, dNewMessage.id]
            });
          }
        }

        await this.mUpdateConversation(pMessage.convId, {
          currNode: dNewMessage.id
        });

        return dNewMessage;
      }
    );
  }

  /**
   * Creates a root message for a new conversation.
   * Root messages are not displayed but serve as the tree root for branching.
   *
   * @param pConvId - Conversation ID
   * @returns The created root message
   */
  static async mCreateRootMessage(pConvId: string): Promise<string> {
    const dRootMessage: DatabaseMessage = {
      id: uuid(),
      convId: pConvId,
      type: 'root',
      timestamp: Date.now(),
      role: MessageRole.SYSTEM,
      content: '',
      parent: null,
      toolCalls: '',
      children: []
    };

    await cDb[cIdxdbTables.messages].add(dRootMessage);
    return dRootMessage.id;
  }

  /**
   * Creates a system prompt message for a conversation.
   *
   * @param pConvId - Conversation ID
   * @param pSystemPrompt - The system prompt content (must be non-empty)
   * @param pParentId - Parent message ID (typically the root message)
   * @returns The created system message
   * @throws Error if systemPrompt is empty or the parent message does not exist
   */
  static async mCreateSystemMessage(
    pConvId: string,
    pSystemPrompt: string,
    pParentId: string
  ): Promise<DatabaseMessage> {
    const cTrimmedPrompt = pSystemPrompt.trim();
    if (!cTrimmedPrompt) {
      throw new Error(fTranslate('text_3e4406904217'));
    }

    return await cDb.transaction('rw', cDb[cIdxdbTables.messages], async () => {
      const cParentMessage = await cDb[cIdxdbTables.messages].get(pParentId);
      if (!cParentMessage) {
        throw new Error(fTranslate('text_cb1c1c4c66b5', { p0: pParentId }));
      }

      const dSystemMessage: DatabaseMessage = {
        id: uuid(),
        convId: pConvId,
        type: MessageRole.SYSTEM,
        timestamp: Date.now(),
        role: MessageRole.SYSTEM,
        content: cTrimmedPrompt,
        parent: pParentId,
        children: []
      };

      await cDb[cIdxdbTables.messages].add(dSystemMessage);
      await cDb[cIdxdbTables.messages].update(pParentId, {
        children: [...cParentMessage.children, dSystemMessage.id]
      });

      return dSystemMessage;
    });
  }

  /**
   * Deletes a conversation and all its messages.
   *
   * @param pId - Conversation ID
   */
  static async mDeleteConversation(
    pId: string,
    pOptions?: { deleteWithForks?: boolean }
  ): Promise<void> {
    await cDb.transaction(
      'rw',
      [cDb[cIdxdbTables.conversations], cDb[cIdxdbTables.messages]],
      async () => {
        if (pOptions?.deleteWithForks) {
          // Recursively collect all descendant IDs
          const aIdsToDelete: string[] = [];
          const aQueue = [pId];

          while (aQueue.length > 0) {
            const cParentId = aQueue.pop()!;
            const cChildren = await cDb[cIdxdbTables.conversations]
              .filter((pC) => pC.forkedFromConversationId === cParentId)
              .toArray();

            for (const cChild of cChildren) {
              aIdsToDelete.push(cChild.id);
              aQueue.push(cChild.id);
            }
          }

          for (const cForkId of aIdsToDelete) {
            await cDb[cIdxdbTables.conversations].delete(cForkId);
            await cDb[cIdxdbTables.messages].where('convId').equals(cForkId).delete();
          }
        } else {
          await this.mReparentDirectChildren(pId);
        }

        await cDb[cIdxdbTables.conversations].delete(pId);
        await cDb[cIdxdbTables.messages].where('convId').equals(pId).delete();
      }
    );
  }

  /**
   * Reparents direct children of `parentId` to the nearest surviving
   * ancestor (or promotes them to top-level when the immediate parent was
   * top-level). Walking skips any ancestor listed in `excludeIds`, since
   * those will be deleted in the same batch — leaving a grandchild pointing
   * at an `excludeIds` entry would orphan it. Children whose own id is in
   * `excludeIds` are dropped from the updates (the bulk-delete pass will
   * remove them). `prefetched` may carry a pre-fetched ancestor map to
   * avoid repeat reads inside a bulk transaction.
   */
  private static async mReparentDirectChildren(
    pParentId: string,
    pExcludeIds: ReadonlySet<string> = new Set(),
    pPrefetched?: ReadonlyMap<string, DatabaseConversation>
  ): Promise<void> {
    const cConv =
      pPrefetched?.get(pParentId) ?? (await cDb[cIdxdbTables.conversations].get(pParentId));
    if (!cConv) return;

    let vNewParent = cConv.forkedFromConversationId;
    const sVisited = new Set<string>([pParentId]);
    while (vNewParent && pExcludeIds.has(vNewParent)) {
      if (sVisited.has(vNewParent)) {
        vNewParent = undefined;
        break;
      }
      sVisited.add(vNewParent);
      const cNext =
        pPrefetched?.get(vNewParent) ?? (await cDb[cIdxdbTables.conversations].get(vNewParent));
      if (!cNext) {
        vNewParent = undefined;
        break;
      }
      vNewParent = cNext.forkedFromConversationId;
    }

    const cDirectChildren = await cDb[cIdxdbTables.conversations]
      .filter((pC) => pC.forkedFromConversationId === pParentId)
      .toArray();

    const aUpdates: DatabaseConversation[] = [];
    for (const cChild of cDirectChildren) {
      if (pExcludeIds.has(cChild.id)) continue;
      aUpdates.push({ ...cChild, forkedFromConversationId: vNewParent });
    }
    if (aUpdates.length === 0) return;
    await cDb[cIdxdbTables.conversations].bulkPut(aUpdates);
  }

  /**
   * Deletes multiple conversations in a single transaction. Each deleted
   * conversation has its direct children reparented to the nearest surviving
   * ancestor (or promoted to top-level). Children also in `ids` are dropped
   * entirely rather than reparented.
   *
   * @param pIds - Conversation IDs to delete
   */
  static async mBulkDeleteConversations(pIds: string[]): Promise<void> {
    const cCleanIds = pIds.filter(
      (pId): pId is string => typeof pId === 'string' && pId.length > 0
    );
    if (cCleanIds.length === 0) return;
    const sIdSet = new Set(cCleanIds);

    await cDb.transaction(
      'rw',
      [cDb[cIdxdbTables.conversations], cDb[cIdxdbTables.messages]],
      async () => {
        // Pre-load each to-delete conversation so the per-id reparent
        // walk-up doesn't ping-pong the same ancestry chain.
        const dPrefetched = new Map<string, DatabaseConversation>();
        let aFrontier = [...cCleanIds];
        const sRequested = new Set<string>(aFrontier);
        while (aFrontier.length > 0) {
          const cFetched = await cDb[cIdxdbTables.conversations].bulkGet(aFrontier);
          aFrontier = [];
          for (let vI = 0; vI < cFetched.length; vI++) {
            const cConv = cFetched[vI];
            if (!cConv || !cConv.id) continue;
            dPrefetched.set(cConv.id, cConv);
            const cAncestor = cConv.forkedFromConversationId;
            if (cAncestor && !dPrefetched.has(cAncestor) && !sRequested.has(cAncestor)) {
              aFrontier.push(cAncestor);
              sRequested.add(cAncestor);
            }
          }
        }

        for (const cId of cCleanIds) {
          await this.mReparentDirectChildren(cId, sIdSet, dPrefetched);
        }

        await cDb[cIdxdbTables.conversations].bulkDelete(cCleanIds);
        await cDb[cIdxdbTables.messages].where('convId').anyOf(cCleanIds).delete();
      }
    );
  }

  /**
   * Deletes a message and removes it from its parent's children array.
   *
   * @param pMessageId - ID of the message to delete
   */
  static async mDeleteMessage(pMessageId: string): Promise<void> {
    await cDb.transaction('rw', cDb[cIdxdbTables.messages], async () => {
      const cMessage = await cDb[cIdxdbTables.messages].get(pMessageId);
      if (!cMessage) return;

      // Remove this message from its parent's children array
      if (cMessage.parent) {
        const cParent = await cDb[cIdxdbTables.messages].get(cMessage.parent);
        if (cParent) {
          cParent.children = cParent.children.filter((pChildId: string) => pChildId !== pMessageId);
          await cDb[cIdxdbTables.messages].put(cParent);
        }
      }

      // Delete the message
      await cDb[cIdxdbTables.messages].delete(pMessageId);
    });
  }

  /**
   * Deletes a message and all its descendant messages (cascading deletion).
   * This removes the entire branch starting from the specified message.
   *
   * @param pConversationId - ID of the conversation containing the message
   * @param pMessageId - ID of the root message to delete (along with all descendants)
   * @returns Array of all deleted message IDs
   */
  static async mDeleteMessageCascading(
    pConversationId: string,
    pMessageId: string
  ): Promise<string[]> {
    return await cDb.transaction('rw', cDb[cIdxdbTables.messages], async () => {
      // Get all messages in the conversation to find descendants
      const cAllMessages = await cDb[cIdxdbTables.messages]
        .where('convId')
        .equals(pConversationId)
        .toArray();

      // Find all descendant messages
      const cDescendants = findDescendantMessages(cAllMessages, pMessageId);
      const aAllToDelete = [pMessageId, ...cDescendants];

      // Get the message to delete for parent cleanup
      const cMessage = await cDb[cIdxdbTables.messages].get(pMessageId);
      if (cMessage && cMessage.parent) {
        const cParent = await cDb[cIdxdbTables.messages].get(cMessage.parent);
        if (cParent) {
          cParent.children = cParent.children.filter((pChildId: string) => pChildId !== pMessageId);
          await cDb[cIdxdbTables.messages].put(cParent);
        }
      }

      // Delete all messages in the branch
      await cDb[cIdxdbTables.messages].bulkDelete(aAllToDelete);

      return aAllToDelete;
    });
  }

  /**
   * Gets all conversations, sorted by last modified time (newest first).
   *
   * @returns Array of conversations
   */
  static async mGetAllConversations(): Promise<DatabaseConversation[]> {
    return await cDb[cIdxdbTables.conversations].orderBy('lastModified').reverse().toArray();
  }

  /**
   * Gets a conversation by ID.
   *
   * @param pId - Conversation ID
   * @returns The conversation if found, otherwise undefined
   */
  static async mGetConversation(pId: string): Promise<DatabaseConversation | undefined> {
    return await cDb[cIdxdbTables.conversations].get(pId);
  }

  /**
   * Gets all messages in a conversation, sorted by timestamp (oldest first).
   *
   * @param pConvId - Conversation ID
   * @returns Array of messages in the conversation
   */
  static async mGetConversationMessages(pConvId: string): Promise<DatabaseMessage[]> {
    return await cDb[cIdxdbTables.messages].where('convId').equals(pConvId).sortBy('timestamp');
  }

  /**
   * Loads multiple conversations with all of their messages in two bulk
   * reads. Missing conversations are silently omitted from the result.
   *
   * @param pConvIds - Conversation IDs to load
   * @returns Map of id -> { conv, messages }. Messages are sorted ascending by timestamp.
   */
  static async mGetConversationsWithMessages(
    pConvIds: string[]
  ): Promise<Map<string, ExportedConversation>> {
    const dResult = new Map<string, ExportedConversation>();
    const cCleanIds = pConvIds.filter(
      (pId): pId is string => typeof pId === 'string' && pId.length > 0
    );
    if (cCleanIds.length === 0) return dResult;

    const [cConvs, cAllMessages] = await Promise.all([
      cDb[cIdxdbTables.conversations].bulkGet(cCleanIds),
      cDb[cIdxdbTables.messages].where('convId').anyOf(cCleanIds).toArray()
    ]);

    const dMessagesByConv = new Map<string, DatabaseMessage[]>();
    for (const cMsg of cAllMessages) {
      const cBucket = dMessagesByConv.get(cMsg.convId);
      if (cBucket) cBucket.push(cMsg);
      else dMessagesByConv.set(cMsg.convId, [cMsg]);
    }

    for (let vI = 0; vI < cCleanIds.length; vI++) {
      const cConv = cConvs[vI];
      if (!cConv) continue;
      const cMessages = (dMessagesByConv.get(cConv.id) ?? []).sort(
        (pA, pB) => pA.timestamp - pB.timestamp
      );
      dResult.set(cConv.id, { conv: cConv, messages: cMessages });
    }
    return dResult;
  }

  /**
   * Updates a conversation. `lastModified` is never stamped implicitly;
   * pass it in `updates` to bump the conversation in recency ordering.
   *
   * @param pId - Conversation ID
   * @param pUpdates - Partial updates to apply
   * @returns Promise that resolves when the conversation is updated
   */
  static async mUpdateConversation(
    pId: string,
    pUpdates: Partial<Omit<DatabaseConversation, 'id'>>
  ): Promise<void> {
    await cDb[cIdxdbTables.conversations].update(pId, pUpdates);
  }

  /**
   *
   *
   * Navigation
   *
   *
   */

  /**
   * Toggles the pinned status of a conversation.
   *
   * @param pId - Conversation ID
   * @returns The new pinned status
   */
  static async mToggleConversationPin(pId: string): Promise<boolean> {
    const cConversation = await cDb[cIdxdbTables.conversations].get(pId);
    if (!cConversation) {
      throw new Error(fTranslate('text_4d5c9425ff50', { p0: pId }));
    }
    const cNewPinnedState = !cConversation.pinned;
    await this.mUpdateConversation(pId, { pinned: cNewPinnedState });
    return cNewPinnedState;
  }

  /**
   * Toggles the pinned status of each conversation in `ids` inside a single
   * transaction. Treats `pinned === undefined` as `false`, matching the
   * semantics of {@link mToggleConversationPin} where `!undefined` evaluates
   * to `true`. Returns the resulting pinned state for every id that was
   * updated; missing ids are omitted from the map.
   *
   * @param pIds - Conversation IDs to toggle
   * @returns Map of id -> new pinned state
   */
  static async mBulkToggleConversationPins(pIds: string[]): Promise<Map<string, boolean>> {
    const cCleanIds = pIds.filter(
      (pId): pId is string => typeof pId === 'string' && pId.length > 0
    );
    const dResult = new Map<string, boolean>();
    if (cCleanIds.length === 0) return dResult;

    await cDb.transaction('rw', cDb[cIdxdbTables.conversations], async () => {
      const cConvs = await cDb[cIdxdbTables.conversations].bulkGet(cCleanIds);
      const aUpdates: DatabaseConversation[] = [];
      for (let vI = 0; vI < cCleanIds.length; vI++) {
        const cConv = cConvs[vI];
        if (!cConv) continue;
        const cNewPinned = !cConv.pinned;
        aUpdates.push({ ...cConv, pinned: cNewPinned });
        dResult.set(cCleanIds[vI], cNewPinned);
      }
      if (aUpdates.length === 0) return;
      await cDb[cIdxdbTables.conversations].bulkPut(aUpdates);
    });
    return dResult;
  }

  /**
   * Updates the conversation's current node (active branch).
   * This determines which conversation path is currently being viewed.
   *
   * @param pConvId - Conversation ID
   * @param pNodeId - Message ID to set as current node
   */
  static async mUpdateCurrentNode(pConvId: string, pNodeId: string): Promise<void> {
    await this.mUpdateConversation(pConvId, {
      currNode: pNodeId
    });
  }

  /**
   * Updates a message.
   *
   * @param pId - Message ID
   * @param pUpdates - Partial updates to apply
   * @returns Promise that resolves when the message is updated
   */
  static async mUpdateMessage(
    pId: string,
    pUpdates: Partial<Omit<DatabaseMessage, 'id'>>
  ): Promise<void> {
    await cDb[cIdxdbTables.messages].update(pId, pUpdates);
  }

  /**
   *
   *
   * Import
   *
   *
   */

  /**
   * Imports multiple conversations and their messages.
   * Skips conversations that already exist, and conversations whose messages do not
   * all belong to them or whose message ids repeat or are already in the database,
   * so an import never overwrites or reassigns an existing record.
   *
   * @param pData - Array of { conv, messages } objects
   * @returns The conversations written to the database and the ones skipped
   */
  static async mImportConversations(
    pData: { conv: DatabaseConversation; messages: DatabaseMessage[] }[]
  ): Promise<{ imported: DatabaseConversation[]; skipped: DatabaseConversation[] }> {
    const aImported: DatabaseConversation[] = [];
    const aSkipped: DatabaseConversation[] = [];
    // Message ids written by this import, so two imported conversations can not share one
    const sImportedMessageIds = new Set<string>();

    return await cDb.transaction(
      'rw',
      [cDb[cIdxdbTables.conversations], cDb[cIdxdbTables.messages]],
      async () => {
        for (const cItem of pData) {
          const { conv: cConv, messages: cMessages } = cItem;

          const cExisting = await cDb[cIdxdbTables.conversations].get(cConv.id);
          if (cExisting) {
            aSkipped.push(cConv);
            continue;
          }

          const aMessageIds = cMessages.map((pMsg) => pMsg.id);
          const sMessageIds = new Set(aMessageIds);
          const cExistingMessages = await cDb[cIdxdbTables.messages].bulkGet(aMessageIds);
          if (
            sMessageIds.size !== aMessageIds.length ||
            aMessageIds.some((pId) => sImportedMessageIds.has(pId)) ||
            cExistingMessages.some((pMsg) => pMsg !== undefined) ||
            cMessages.some((pMsg) => pMsg.convId !== cConv.id)
          ) {
            aSkipped.push(cConv);
            continue;
          }

          await cDb[cIdxdbTables.conversations].add(cConv);
          await cDb[cIdxdbTables.messages].bulkAdd(cMessages);
          for (const cId of aMessageIds) sImportedMessageIds.add(cId);

          aImported.push(cConv);
        }

        return { imported: aImported, skipped: aSkipped };
      }
    );
  }

  /**
   *
   *
   * Forking
   *
   *
   */

  /**
   * Forks a conversation at a specific message, creating a new conversation
   * containing all messages from the root up to (and including) the target message.
   *
   * @param pSourceConvId - The source conversation ID
   * @param pAtMessageId - The message ID to fork at (the new conversation ends here)
   * @param pOptions - Fork options (name and whether to include attachments)
   * @returns The newly created conversation
   */
  static async mForkConversation(
    pSourceConvId: string,
    pAtMessageId: string,
    pOptions: { name: string; includeAttachments: boolean }
  ): Promise<DatabaseConversation> {
    return await cDb.transaction(
      'rw',
      [cDb[cIdxdbTables.conversations], cDb[cIdxdbTables.messages]],
      async () => {
        const cSourceConv = await cDb[cIdxdbTables.conversations].get(pSourceConvId);
        if (!cSourceConv) {
          throw new Error(fTranslate('text_5f6b68cb4211', { p0: pSourceConvId }));
        }

        const cAllMessages = await cDb[cIdxdbTables.messages]
          .where('convId')
          .equals(pSourceConvId)
          .toArray();

        const cPathMessages = filterByLeafNodeId(
          cAllMessages,
          pAtMessageId,
          true
        ) as DatabaseMessage[];
        if (cPathMessages.length === 0) {
          throw new Error(fTranslate('text_e154ca68fed4', { p0: pAtMessageId }));
        }

        const dIdMap = new Map<string, string>();

        for (const cMsg of cPathMessages) {
          dIdMap.set(cMsg.id, uuid());
        }

        const cNewConvId = uuid();
        const aClonedMessages: DatabaseMessage[] = cPathMessages.map((pMsg) => {
          const cNewId = dIdMap.get(pMsg.id)!;
          const cNewParent = pMsg.parent ? (dIdMap.get(pMsg.parent) ?? null) : null;
          const cNewChildren = pMsg.children
            .filter((pChildId: string) => dIdMap.has(pChildId))
            .map((pChildId: string) => dIdMap.get(pChildId)!);

          return {
            ...pMsg,
            id: cNewId,
            convId: cNewConvId,
            parent: cNewParent,
            children: cNewChildren,
            extra: pOptions.includeAttachments ? pMsg.extra : undefined
          };
        });

        const cLastClonedMessage = aClonedMessages[aClonedMessages.length - 1];
        const dNewConv: DatabaseConversation = {
          id: cNewConvId,
          name: pOptions.name,
          lastModified: Date.now(),
          currNode: cLastClonedMessage.id,
          forkedFromConversationId: pSourceConvId,
          mcpServerOverrides: cSourceConv.mcpServerOverrides
            ? cSourceConv.mcpServerOverrides.map((pO: McpServerOverride) => ({
                serverId: pO.serverId,
                enabled: pO.enabled
              }))
            : undefined
        };

        await cDb[cIdxdbTables.conversations].add(dNewConv);

        for (const cMsg of aClonedMessages) {
          await cDb[cIdxdbTables.messages].add(cMsg);
        }

        return dNewConv;
      }
    );
  }
}
