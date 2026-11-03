/**
 * Utility functions for conversation data manipulation
 */
import type { DatabaseMessage } from '$lib/types';
import { cSidebarRecentConversations } from '$lib/constants/ui';

/**
 * Creates a map of conversation IDs to their message counts from exported conversation data
 * @param pExportedData - Array of exported conversations with their messages
 * @returns Map of conversation ID to message count
 */
export function fCreateMessageCountMap(
  pExportedData: Array<{ conv: DatabaseConversation; messages: DatabaseMessage[] }>
): Map<string, number> {
  const dCountMap = new Map<string, number>();

  for (const cItem of pExportedData) {
    dCountMap.set(cItem.conv.id, cItem.messages.length);
  }

  return dCountMap;
}

/**
 * Gets the message count for a specific conversation from the count map
 * @param pConversationId - The ID of the conversation
 * @param pCountMap - Map of conversation IDs to message counts
 * @returns The message count, or 0 if not found
 */
export function fGetMessageCount(pConversationId: string, pCountMap: Map<string, number>): number {
  return pCountMap.get(pConversationId) ?? 0;
}

/** The sections of the conversations in the sidebar, in the order they are shown. */
export interface SidebarConversationGroups {
  pinned: DatabaseConversation[];
  /** the five unpinned chats the user worked with last */
  recent: DatabaseConversation[];
  /** the other unpinned chats, newest first by creation */
  others: DatabaseConversation[];
}

/**
 * Splits the conversations of the sidebar into its sections: pinned ones (last worked with first),
 * "Recent chats" (the unpinned ones worked with last) and "Chats" (the rest, newest first). A chat
 * that the user works with again moves up into "Recent chats".
 */
export function fGroupSidebarConversations(
  pConversations: DatabaseConversation[]
): SidebarConversationGroups {
  const aByActivity = [...pConversations].sort((pA, pB) => pB.lastModified - pA.lastModified);
  const aUnpinned = aByActivity.filter((pC) => !pC.pinned);

  return {
    pinned: aByActivity.filter((pC) => pC.pinned),
    recent: aUnpinned.slice(0, cSidebarRecentConversations),
    others: aUnpinned
      .slice(cSidebarRecentConversations)
      .sort((pA, pB) => (pB.createdAt ?? 0) - (pA.createdAt ?? 0))
  };
}
