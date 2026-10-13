/**
 * Utility functions for conversation data manipulation
 */
import type { DatabaseMessage } from '$lib/types';

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
