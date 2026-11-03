import { fTranslate } from '../i18n';
import {
  fApiFetch,
  fApiFetchWithParams,
  fApiPost,
  fFindDescendantMessages,
  fUuid,
  fFilterByLeafNodeId,
  fMessageGraphHasCycle
} from '$lib/utils';
import { cChatsEndpoint } from '$lib/constants/api-endpoints';
import { MessageRole } from '$lib/enums';
import type { McpServerOverride } from '$lib/types/database';
import type { ExportedConversation } from '$lib/types/database';

/**
 * The chats live on the server, in ~/.mgs/chats/ (one JSONL file per chat), so every browser sees
 * the same ones. This service keeps a copy in memory of what it has read and sends every change to
 * the server: the list of conversations, and the messages of every chat opened in this page.
 */

interface ApiChatList {
  version: number;
  conversations: DatabaseConversation[];
}

interface ApiChat {
  conversation: DatabaseConversation;
  messages: DatabaseMessage[];
}

interface ApiChatSaved {
  version: number;
  conversation: DatabaseConversation;
}

/** A change of one chat: the conversation, messages added or replaced, ids of messages removed */
interface ChatChange {
  conversation?: DatabaseConversation;
  messages?: DatabaseMessage[];
  deleted?: string[];
  /** messages is the whole list of the chat */
  replace?: boolean;
}

const dConversations = new Map<string, DatabaseConversation>();
const dMessages = new Map<string, Map<string, DatabaseMessage>>();
const dMessageChat = new Map<string, string>();
let vListLoaded = false;
// version of the chats after the last read or write of this page: another one means another browser wrote
let vKnownVersion = 0;
// the writes of a chat go one after the other, in the order they were made
const dWriteQueues = new Map<string, Promise<unknown>>();
// writes of every chat asked by this page and the last one the server answered: while they
// differ, the server does not have every change made here yet
const dWritesAsked = new Map<string, number>();
const dWritesAnswered = new Map<string, number>();

/** Numbers a new write of a chat; the server has it once dWritesAnswered reaches that number. */
function fAskWrite(pConvId: string): number {
  const cWriteNumber = (dWritesAsked.get(pConvId) ?? 0) + 1;
  dWritesAsked.set(pConvId, cWriteNumber);
  return cWriteNumber;
}

/**
 * Whether an answer of the server holds every change made here to a chat: only when every write
 * asked until now had been answered before the read started (pAnsweredAtStart). Otherwise the
 * answer is older than the copy in memory and must not replace it: a chat being created would
 * lose its root message, and the first message would fail because its parent is not found.
 */
function fServerHasEveryChange(pConvId: string, pAnsweredAtStart: number): boolean {
  return (dWritesAsked.get(pConvId) ?? 0) === pAnsweredAtStart;
}

function fSortByTimestamp(pMessages: DatabaseMessage[]): DatabaseMessage[] {
  return pMessages.map(fCopy).sort((pA, pB) => pA.timestamp - pB.timestamp);
}

function fChatUrl(pId: string): string {
  return `${cChatsEndpoint}/${encodeURIComponent(pId)}`;
}

function fCopy<T extends object>(pObject: T): T {
  return { ...pObject };
}

function fCacheMessages(pConvId: string, pMessages: DatabaseMessage[]): Map<string, DatabaseMessage> {
  const dChatMessages = new Map<string, DatabaseMessage>();
  for (const cMessage of pMessages) {
    dChatMessages.set(cMessage.id, cMessage);
    dMessageChat.set(cMessage.id, pConvId);
  }
  dMessages.set(pConvId, dChatMessages);
  return dChatMessages;
}

function fForgetChat(pConvId: string): void {
  dConversations.delete(pConvId);
  for (const cMessageId of dMessages.get(pConvId)?.keys() ?? []) {
    dMessageChat.delete(cMessageId);
  }
  dMessages.delete(pConvId);
}

/** Sends a change of a chat to the server, after the previous ones of the same chat. */
function fSaveChat(pConvId: string, pChange: ChatChange): Promise<DatabaseConversation> {
  const cWriteNumber = fAskWrite(pConvId);
  const cPrevious = dWriteQueues.get(pConvId) ?? Promise.resolve();
  const cWrite = cPrevious
    .catch(() => {})
    .then(async () => {
      try {
        const cSaved = await fApiPost<ApiChatSaved, ChatChange>(fChatUrl(pConvId), pChange);
        vKnownVersion = cSaved.version;
        // the server gives a new chat its number and its creation date
        const cCached = dConversations.get(pConvId);
        if (cCached) {
          cCached.number = cSaved.conversation.number;
          cCached.createdAt = cSaved.conversation.createdAt;
        } else {
          dConversations.set(pConvId, cSaved.conversation);
        }
        return cSaved.conversation;
      } finally {
        dWritesAnswered.set(pConvId, cWriteNumber);
      }
    });
  dWriteQueues.set(pConvId, cWrite);
  return cWrite;
}

async function fDeleteChat(pConvId: string): Promise<void> {
  await (dWriteQueues.get(pConvId) ?? Promise.resolve()).catch(() => {});
  const cWriteNumber = fAskWrite(pConvId);
  try {
    const cResult = await fApiFetch<{ version: number }>(fChatUrl(pConvId), { method: 'DELETE' });
    vKnownVersion = cResult.version;
  } finally {
    dWritesAnswered.set(pConvId, cWriteNumber);
  }
  dWriteQueues.delete(pConvId);
  fForgetChat(pConvId);
}

async function fEnsureList(): Promise<void> {
  if (!vListLoaded) {
    await DatabaseService.mGetAllConversations();
  }
}

/** The messages of a chat, read from the server the first time. */
async function fChatMessages(pConvId: string): Promise<Map<string, DatabaseMessage>> {
  const cCached = dMessages.get(pConvId);
  if (cCached) return cCached;

  await DatabaseService.mGetConversationMessages(pConvId);
  return dMessages.get(pConvId) ?? fCacheMessages(pConvId, []);
}

/** The conversation kept in memory (not a copy), reading the list from the server the first time. */
async function fCachedConversation(pId: string): Promise<DatabaseConversation | undefined> {
  if (!dConversations.has(pId)) {
    await fEnsureList();
  }
  return dConversations.get(pId);
}

export class DatabaseService {
  /**
   *
   *
   * Synchronization
   *
   *
   */

  /** Version of the chats on the server; it changes with every write, so another one means news. */
  static async mGetVersion(): Promise<number> {
    const cResult = await fApiFetch<{ version: number }>(`${cChatsEndpoint}/version`);
    return cResult.version;
  }

  /** Version that this page last read or wrote. */
  static mGetKnownVersion(): number {
    return vKnownVersion;
  }

  /**
   *
   *
   * Conversations
   *
   *
   */

  /**
   * Creates a new conversation. The server gives it its number and its creation date.
   *
   * @param pName - Name of the conversation
   * @param pFields - Optional extra fields (e.g. reasoningEffort)
   * @returns The created conversation
   */
  static async mCreateConversation(
    pName: string,
    pFields?: Partial<Omit<DatabaseConversation, 'id' | 'name' | 'lastModified'>>
  ): Promise<DatabaseConversation> {
    const cNow = Date.now();
    const dConversation: DatabaseConversation = {
      id: fUuid(),
      name: pName,
      lastModified: cNow,
      currNode: '',
      ...pFields,
      number: 0,
      createdAt: cNow
    };

    dConversations.set(dConversation.id, dConversation);
    fCacheMessages(dConversation.id, []);
    await fSaveChat(dConversation.id, { conversation: dConversation, messages: [], replace: true });
    return fCopy(dConversation);
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
    // every read comes before the changes: between a change and its write there must be no wait,
    // or a read of the server answered meanwhile would not know the change and would drop it
    await fCachedConversation(pMessage.convId);
    const dChatMessages = await fChatMessages(pMessage.convId);

    // Handle null parent (root message case)
    const cParentMessage = pParentId !== null ? dChatMessages.get(pParentId) : undefined;
    if (pParentId !== null && !cParentMessage) {
      throw new Error(fTranslate('text_cb1c1c4c66b5', { p0: pParentId }));
    }

    const dNewMessage: DatabaseMessage = {
      ...pMessage,
      id: fUuid(),
      parent: pParentId,
      toolCalls: pMessage.toolCalls ?? '',
      children: []
    };
    dChatMessages.set(dNewMessage.id, dNewMessage);
    dMessageChat.set(dNewMessage.id, pMessage.convId);

    // Update parent's children array if parent exists
    const aChanged: DatabaseMessage[] = [dNewMessage];
    if (cParentMessage) {
      cParentMessage.children = [...cParentMessage.children, dNewMessage.id];
      aChanged.push(cParentMessage);
    }

    const cConversation = dConversations.get(pMessage.convId);
    if (cConversation) {
      cConversation.currNode = dNewMessage.id;
    }

    await fSaveChat(pMessage.convId, { conversation: cConversation, messages: aChanged });
    return fCopy(dNewMessage);
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
      id: fUuid(),
      convId: pConvId,
      type: 'root',
      timestamp: Date.now(),
      role: MessageRole.SYSTEM,
      content: '',
      parent: null,
      toolCalls: '',
      children: []
    };

    const dChatMessages = await fChatMessages(pConvId);
    dChatMessages.set(dRootMessage.id, dRootMessage);
    dMessageChat.set(dRootMessage.id, pConvId);
    await fSaveChat(pConvId, { messages: [dRootMessage] });
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

    const dChatMessages = await fChatMessages(pConvId);
    const cParentMessage = dChatMessages.get(pParentId);
    if (!cParentMessage) {
      throw new Error(fTranslate('text_cb1c1c4c66b5', { p0: pParentId }));
    }

    const dSystemMessage: DatabaseMessage = {
      id: fUuid(),
      convId: pConvId,
      type: MessageRole.SYSTEM,
      timestamp: Date.now(),
      role: MessageRole.SYSTEM,
      content: cTrimmedPrompt,
      parent: pParentId,
      children: []
    };

    dChatMessages.set(dSystemMessage.id, dSystemMessage);
    dMessageChat.set(dSystemMessage.id, pConvId);
    cParentMessage.children = [...cParentMessage.children, dSystemMessage.id];
    await fSaveChat(pConvId, { messages: [dSystemMessage, cParentMessage] });

    return fCopy(dSystemMessage);
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
    await fEnsureList();

    if (pOptions?.deleteWithForks) {
      // Recursively collect all descendant IDs
      const aIdsToDelete: string[] = [];
      const aQueue = [pId];

      while (aQueue.length > 0) {
        const cParentId = aQueue.pop()!;
        for (const cChild of dConversations.values()) {
          if (cChild.forkedFromConversationId === cParentId) {
            aIdsToDelete.push(cChild.id);
            aQueue.push(cChild.id);
          }
        }
      }

      for (const cForkId of aIdsToDelete) {
        await fDeleteChat(cForkId);
      }
    } else {
      await this.mReparentDirectChildren(pId);
    }

    await fDeleteChat(pId);
  }

  /**
   * Reparents direct children of `parentId` to the nearest surviving
   * ancestor (or promotes them to top-level when the immediate parent was
   * top-level). Walking skips any ancestor listed in `excludeIds`, since
   * those will be deleted in the same batch — leaving a grandchild pointing
   * at an `excludeIds` entry would orphan it. Children whose own id is in
   * `excludeIds` are dropped from the updates (the bulk-delete pass will
   * remove them).
   */
  private static async mReparentDirectChildren(
    pParentId: string,
    pExcludeIds: ReadonlySet<string> = new Set()
  ): Promise<void> {
    const cConv = dConversations.get(pParentId);
    if (!cConv) return;

    let vNewParent = cConv.forkedFromConversationId;
    const sVisited = new Set<string>([pParentId]);
    while (vNewParent && pExcludeIds.has(vNewParent)) {
      if (sVisited.has(vNewParent)) {
        vNewParent = undefined;
        break;
      }
      sVisited.add(vNewParent);
      const cNext = dConversations.get(vNewParent);
      if (!cNext) {
        vNewParent = undefined;
        break;
      }
      vNewParent = cNext.forkedFromConversationId;
    }

    for (const cChild of [...dConversations.values()]) {
      if (cChild.forkedFromConversationId !== pParentId || pExcludeIds.has(cChild.id)) continue;
      cChild.forkedFromConversationId = vNewParent;
      await fSaveChat(cChild.id, { conversation: cChild });
    }
  }

  /**
   * Deletes multiple conversations. Each deleted conversation has its direct
   * children reparented to the nearest surviving ancestor (or promoted to
   * top-level). Children also in `ids` are dropped entirely rather than
   * reparented.
   *
   * @param pIds - Conversation IDs to delete
   */
  static async mBulkDeleteConversations(pIds: string[]): Promise<void> {
    const cCleanIds = pIds.filter(
      (pId): pId is string => typeof pId === 'string' && pId.length > 0
    );
    if (cCleanIds.length === 0) return;
    const sIdSet = new Set(cCleanIds);

    await fEnsureList();
    for (const cId of cCleanIds) {
      await this.mReparentDirectChildren(cId, sIdSet);
    }
    for (const cId of cCleanIds) {
      await fDeleteChat(cId);
    }
  }

  /**
   * Deletes a message and removes it from its parent's children array.
   *
   * @param pMessageId - ID of the message to delete
   */
  static async mDeleteMessage(pMessageId: string): Promise<void> {
    const cConvId = dMessageChat.get(pMessageId);
    if (!cConvId) return;
    const dChatMessages = await fChatMessages(cConvId);
    const cMessage = dChatMessages.get(pMessageId);
    if (!cMessage) return;

    // Remove this message from its parent's children array
    const aChanged: DatabaseMessage[] = [];
    const cParent = cMessage.parent ? dChatMessages.get(cMessage.parent) : undefined;
    if (cParent) {
      cParent.children = cParent.children.filter((pChildId: string) => pChildId !== pMessageId);
      aChanged.push(cParent);
    }

    // Delete the message
    dChatMessages.delete(pMessageId);
    dMessageChat.delete(pMessageId);
    await fSaveChat(cConvId, { messages: aChanged, deleted: [pMessageId] });
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
    const dChatMessages = await fChatMessages(pConversationId);

    // Find all descendant messages
    const cDescendants = fFindDescendantMessages([...dChatMessages.values()], pMessageId);
    const aAllToDelete = [pMessageId, ...cDescendants];

    // Remove the branch from the children of its parent
    const aChanged: DatabaseMessage[] = [];
    const cMessage = dChatMessages.get(pMessageId);
    const cParent = cMessage?.parent ? dChatMessages.get(cMessage.parent) : undefined;
    if (cParent) {
      cParent.children = cParent.children.filter((pChildId: string) => pChildId !== pMessageId);
      aChanged.push(cParent);
    }

    // Delete all messages in the branch
    for (const cId of aAllToDelete) {
      dChatMessages.delete(cId);
      dMessageChat.delete(cId);
    }
    await fSaveChat(pConversationId, { messages: aChanged, deleted: aAllToDelete });

    return aAllToDelete;
  }

  /**
   * Gets all conversations from the server, sorted by last modified time (newest first).
   *
   * @returns Array of conversations
   */
  static async mGetAllConversations(): Promise<DatabaseConversation[]> {
    const dAnsweredAtStart = new Map(dWritesAnswered);
    const cList = await fApiFetch<ApiChatList>(cChatsEndpoint);
    vKnownVersion = cList.version;

    const sIds = new Set(cList.conversations.map((pC) => pC.id));
    for (const cId of [...dConversations.keys()]) {
      // a chat whose creation the server had not saved yet is not missing
      if (!sIds.has(cId) && fServerHasEveryChange(cId, dAnsweredAtStart.get(cId) ?? 0)) {
        fForgetChat(cId);
      }
    }
    for (const cConversation of cList.conversations) {
      // a chat changed (or deleted) here during the read keeps the copy in memory
      if (!fServerHasEveryChange(cConversation.id, dAnsweredAtStart.get(cConversation.id) ?? 0)) {
        continue;
      }
      dConversations.set(cConversation.id, cConversation);
    }
    vListLoaded = true;

    return [...dConversations.values()]
      .map(fCopy)
      .sort((pA, pB) => pB.lastModified - pA.lastModified);
  }

  /**
   * Gets a conversation by ID.
   *
   * @param pId - Conversation ID
   * @returns The conversation if found, otherwise undefined
   */
  static async mGetConversation(pId: string): Promise<DatabaseConversation | undefined> {
    await fEnsureList();
    const cCached = dConversations.get(pId);
    return cCached ? fCopy(cCached) : undefined;
  }

  /**
   * Messages of the user and of the assistant whose content holds every word of pQuery, newest
   * first: at most one per conversation and pLimit conversations, for the global search of the
   * sidebar. The server compares them without regard to case or accents.
   */
  static async mSearchMessages(pQuery: string, pLimit: number): Promise<DatabaseMessage[]> {
    if (!pQuery.trim()) return [];

    const cResult = await fApiFetchWithParams<{ messages: DatabaseMessage[] }>(
      `${cChatsEndpoint}/search`,
      { q: pQuery, limit: String(pLimit) }
    );
    return cResult.messages;
  }

  /**
   * Gets all messages in a conversation from the server, sorted by timestamp (oldest first).
   *
   * @param pConvId - Conversation ID
   * @returns Array of messages in the conversation
   */
  static async mGetConversationMessages(pConvId: string): Promise<DatabaseMessage[]> {
    // a write still on its way would be missing from the answer
    await (dWriteQueues.get(pConvId) ?? Promise.resolve()).catch(() => {});

    const cAnsweredAtStart = dWritesAnswered.get(pConvId) ?? 0;
    let cChat: ApiChat;
    try {
      cChat = await fApiFetch<ApiChat>(fChatUrl(pConvId));
    } catch {
      // a chat that is not on the server (deleted from another browser) has no messages
      return [];
    }

    // the chat changed here during the read (a new chat gets its root message, for example):
    // the answer of the server is older than the copy in memory, which is kept
    if (!fServerHasEveryChange(pConvId, cAnsweredAtStart)) {
      const dCachedMessages = dMessages.get(pConvId);
      if (dCachedMessages) {
        return fSortByTimestamp([...dCachedMessages.values()]);
      }
      // deleted here during the read
      if (!dConversations.has(pConvId)) {
        return [];
      }
      // only the conversation changed: the messages of the server are up to date
      fCacheMessages(pConvId, cChat.messages);
      return fSortByTimestamp(cChat.messages);
    }

    dConversations.set(pConvId, cChat.conversation);
    fCacheMessages(pConvId, cChat.messages);

    return fSortByTimestamp(cChat.messages);
  }

  /**
   * Loads multiple conversations with all of their messages. Missing conversations are silently
   * omitted from the result.
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

    for (const cId of cCleanIds) {
      const cMessages = await this.mGetConversationMessages(cId);
      const cConv = dConversations.get(cId);
      if (!cConv) continue;
      dResult.set(cId, { conv: fCopy(cConv), messages: cMessages });
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
    const cConversation = await fCachedConversation(pId);
    if (!cConversation) return;

    Object.assign(cConversation, pUpdates);
    await fSaveChat(pId, { conversation: cConversation });
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
    const cConversation = await fCachedConversation(pId);
    if (!cConversation) {
      throw new Error(fTranslate('text_4d5c9425ff50', { p0: pId }));
    }
    const cNewPinnedState = !cConversation.pinned;
    await this.mUpdateConversation(pId, { pinned: cNewPinnedState });
    return cNewPinnedState;
  }

  /**
   * Toggles the pinned status of each conversation in `ids`. Treats
   * `pinned === undefined` as `false`, matching the semantics of
   * {@link mToggleConversationPin} where `!undefined` evaluates to `true`.
   * Returns the resulting pinned state for every id that was updated; missing
   * ids are omitted from the map.
   *
   * @param pIds - Conversation IDs to toggle
   * @returns Map of id -> new pinned state
   */
  static async mBulkToggleConversationPins(pIds: string[]): Promise<Map<string, boolean>> {
    const cCleanIds = pIds.filter(
      (pId): pId is string => typeof pId === 'string' && pId.length > 0
    );
    const dResult = new Map<string, boolean>();

    await fEnsureList();
    for (const cId of cCleanIds) {
      const cConv = dConversations.get(cId);
      if (!cConv) continue;
      const cNewPinned = !cConv.pinned;
      await this.mUpdateConversation(cId, { pinned: cNewPinned });
      dResult.set(cId, cNewPinned);
    }
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
    const cConvId = dMessageChat.get(pId);
    const cMessage = cConvId ? dMessages.get(cConvId)?.get(pId) : undefined;
    if (!cConvId || !cMessage) {
      console.warn(`[DatabaseService] message ${pId} is not loaded, its update is lost`);
      return;
    }

    Object.assign(cMessage, pUpdates);
    await fSaveChat(cConvId, { messages: [cMessage] });
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
   * all belong to them or whose message ids repeat, so an import never overwrites or
   * reassigns an existing record. Conversations whose parent or children links form a
   * cycle are skipped too, since a conversation is a tree and opening one with a cycle
   * would never finish. The server gives every imported chat a number of its own.
   *
   * @param pData - Array of { conv, messages } objects
   * @returns The conversations written to the server and the ones skipped
   */
  static async mImportConversations(
    pData: { conv: DatabaseConversation; messages: DatabaseMessage[] }[]
  ): Promise<{ imported: DatabaseConversation[]; skipped: DatabaseConversation[] }> {
    const aImported: DatabaseConversation[] = [];
    const aSkipped: DatabaseConversation[] = [];
    // Message ids written by this import, so two imported conversations can not share one
    const sImportedMessageIds = new Set<string>();

    await fEnsureList();
    for (const cItem of pData) {
      const { conv: cConv, messages: cMessages } = cItem;

      if (dConversations.has(cConv.id)) {
        aSkipped.push(cConv);
        continue;
      }

      const aMessageIds = cMessages.map((pMsg) => pMsg.id);
      const sMessageIds = new Set(aMessageIds);
      if (
        sMessageIds.size !== aMessageIds.length ||
        aMessageIds.some((pId) => sImportedMessageIds.has(pId) || dMessageChat.has(pId)) ||
        cMessages.some((pMsg) => pMsg.convId !== cConv.id) ||
        fMessageGraphHasCycle(cMessages)
      ) {
        aSkipped.push(cConv);
        continue;
      }

      // the numbers of another server mean nothing here: the chat takes the lowest free one
      const cCreatedAt =
        cConv.createdAt ??
        cMessages.reduce((pMin, pMsg) => Math.min(pMin, pMsg.timestamp), cConv.lastModified);
      const dConv: DatabaseConversation = { ...cConv, number: 0, createdAt: cCreatedAt };
      dConversations.set(dConv.id, dConv);
      fCacheMessages(dConv.id, cMessages);
      const cSaved = await fSaveChat(dConv.id, {
        conversation: dConv,
        messages: cMessages,
        replace: true
      });
      for (const cId of aMessageIds) sImportedMessageIds.add(cId);

      aImported.push({ ...dConv, number: cSaved.number });
    }

    return { imported: aImported, skipped: aSkipped };
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
    const cSourceConv = await fCachedConversation(pSourceConvId);
    if (!cSourceConv) {
      throw new Error(fTranslate('text_5f6b68cb4211', { p0: pSourceConvId }));
    }

    const cAllMessages = [...(await fChatMessages(pSourceConvId)).values()];

    const cPathMessages = fFilterByLeafNodeId(
      cAllMessages,
      pAtMessageId,
      true
    ) as DatabaseMessage[];
    if (cPathMessages.length === 0) {
      throw new Error(fTranslate('text_e154ca68fed4', { p0: pAtMessageId }));
    }

    const dIdMap = new Map<string, string>();

    for (const cMsg of cPathMessages) {
      dIdMap.set(cMsg.id, fUuid());
    }

    const cNewConvId = fUuid();
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
    const cNow = Date.now();
    const dNewConv: DatabaseConversation = {
      id: cNewConvId,
      name: pOptions.name,
      lastModified: cNow,
      number: 0,
      createdAt: cNow,
      currNode: cLastClonedMessage.id,
      forkedFromConversationId: pSourceConvId,
      mcpServerOverrides: cSourceConv.mcpServerOverrides
        ? cSourceConv.mcpServerOverrides.map((pO: McpServerOverride) => ({
            serverId: pO.serverId,
            enabled: pO.enabled
          }))
        : undefined
    };

    dConversations.set(cNewConvId, dNewConv);
    fCacheMessages(cNewConvId, aClonedMessages);
    await fSaveChat(cNewConvId, {
      conversation: dNewConv,
      messages: aClonedMessages,
      replace: true
    });

    return fCopy(dNewConv);
  }
}
