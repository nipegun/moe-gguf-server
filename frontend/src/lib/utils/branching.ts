/**
 * Message branching utilities for conversation tree navigation.
 *
 * Conversation branching allows users to edit messages and create alternate paths
 * while preserving the original conversation flow. Each message has parent/children
 * relationships forming a tree structure.
 *
 * Example tree:
 * root
 *  ├── message 1 (user)
 *  │      └── message 2 (assistant)
 *  │             ├── message 3 (user)
 *  │             └── message 6 (user) ← new branch
 *  └── message 4 (user)
 *        └── message 5 (assistant)
 */

import { MessageRole } from '$lib/enums';

/**
 * Finds a message by its ID in the given messages array.
 */
export function fFindMessageById(
  pMessages: readonly DatabaseMessage[],
  pId: string | null | undefined
): DatabaseMessage | undefined {
  if (!pId) return undefined;
  return pMessages.find((pM) => pM.id === pId);
}

/**
 * Filters messages to get the conversation path from root to a specific leaf node.
 * If the leafNodeId doesn't exist, returns the path with the latest timestamp.
 *
 * @param pMessages - All messages in the conversation
 * @param pLeafNodeId - The target leaf node ID to trace back from
 * @param pIncludeRoot - Whether to include root messages in the result
 * @returns Array of messages from root to leaf, sorted by timestamp
 */
export function fFilterByLeafNodeId(
  pMessages: readonly DatabaseMessage[],
  pLeafNodeId: string,
  pIncludeRoot: boolean = false
): readonly DatabaseMessage[] {
  const aResult: DatabaseMessage[] = [];
  const dNodeMap = new Map<string, DatabaseMessage>();

  // Build node map for quick lookups
  for (const cMsg of pMessages) {
    dNodeMap.set(cMsg.id, cMsg);
  }

  // Find the starting node (leaf node or latest if not found)
  let vStartNode: DatabaseMessage | undefined = dNodeMap.get(pLeafNodeId);
  if (!vStartNode) {
    // If leaf node not found, use the message with latest timestamp
    let vLatestTime = -1;
    for (const cMsg of pMessages) {
      if (cMsg.timestamp > vLatestTime) {
        vStartNode = cMsg;
        vLatestTime = cMsg.timestamp;
      }
    }
  }

  // Traverse from leaf to root, collecting messages
  let vCurrentNode: DatabaseMessage | undefined = vStartNode;
  while (vCurrentNode) {
    // Include message if it's not root, or if we want to include root
    if (vCurrentNode.type !== 'root' || pIncludeRoot) {
      aResult.push(vCurrentNode);
    }

    // Stop traversal if parent is null (reached root)
    if (vCurrentNode.parent === null) {
      break;
    }
    vCurrentNode = dNodeMap.get(vCurrentNode.parent);
  }

  // Sort: system messages first, then by timestamp
  aResult.sort((pA, pB) => {
    if (pA.role === MessageRole.SYSTEM && pB.role !== MessageRole.SYSTEM) return -1;
    if (pA.role !== MessageRole.SYSTEM && pB.role === MessageRole.SYSTEM) return 1;

    return pA.timestamp - pB.timestamp;
  });
  return aResult;
}

/**
 * Finds the leaf node (message with no children) for a given message branch.
 * Traverses down the tree following the last child until reaching a leaf.
 *
 * @param pNodeMap - Map of messages keyed by ID
 * @param pMessageId - Starting message ID to find leaf for
 * @returns The leaf node ID, or the original messageId if no children
 */
function fFindLeafNodeInMap(
  pNodeMap: ReadonlyMap<string, DatabaseMessage>,
  pMessageId: string
): string {
  let vCurrentNode: DatabaseMessage | undefined = pNodeMap.get(pMessageId);
  while (vCurrentNode && vCurrentNode.children.length > 0) {
    // Follow the last child (most recent branch)
    const cLastChildId = vCurrentNode.children[vCurrentNode.children.length - 1];
    vCurrentNode = pNodeMap.get(cLastChildId);
  }

  return vCurrentNode?.id ?? pMessageId;
}

/**
 * Convenience wrapper around {@link fFindLeafNodeInMap} for callers that have a flat message array.
 */
export function fFindLeafNode(pMessages: readonly DatabaseMessage[], pMessageId: string): string {
  const dNodeMap = new Map(pMessages.map((pMsg) => [pMsg.id, pMsg] as const));
  return fFindLeafNodeInMap(dNodeMap, pMessageId);
}

/**
 * Finds all descendant messages (children, grandchildren, etc.) of a given message.
 * This is used for cascading deletion to remove all messages in a branch.
 *
 * @param pMessages - All messages in the conversation
 * @param pMessageId - The root message ID to find descendants for
 * @returns Array of all descendant message IDs
 */
export function fFindDescendantMessages(
  pMessages: readonly DatabaseMessage[],
  pMessageId: string
): string[] {
  const dNodeMap = new Map<string, DatabaseMessage>();

  // Build node map for quick lookups
  for (const cMsg of pMessages) {
    dNodeMap.set(cMsg.id, cMsg);
  }

  const aDescendants: string[] = [];
  const aQueue: string[] = [pMessageId];

  while (aQueue.length > 0) {
    const cCurrentId = aQueue.shift()!;
    const cCurrentNode = dNodeMap.get(cCurrentId);

    if (cCurrentNode) {
      // Add all children to the queue and descendants list
      for (const cChildId of cCurrentNode.children) {
        aDescendants.push(cChildId);
        aQueue.push(cChildId);
      }
    }
  }

  return aDescendants;
}

/**
 * Gets sibling information for a message, including all sibling IDs and current position.
 * Siblings are messages that share the same parent.
 *
 * @param pNodeMap - Map of messages keyed by ID
 * @param pMessageId - The message to get sibling info for
 * @returns Sibling information including leaf node IDs for navigation
 */
export function fGetMessageSiblings(
  pNodeMap: ReadonlyMap<string, DatabaseMessage>,
  pMessageId: string
): ChatMessageSiblingInfo | null {
  const cMessage = pNodeMap.get(pMessageId);
  if (!cMessage) {
    return null;
  }

  // Handle null parent (root message) case
  if (cMessage.parent === null) {
    // No parent means this is likely a root node with no siblings
    return {
      message: cMessage,
      siblingIds: [pMessageId],
      currentIndex: 0,
      totalSiblings: 1
    };
  }

  const cParentNode = pNodeMap.get(cMessage.parent);
  if (!cParentNode) {
    // Parent not found - treat as single message
    return {
      message: cMessage,
      siblingIds: [pMessageId],
      currentIndex: 0,
      totalSiblings: 1
    };
  }

  // Get all sibling IDs (including self)
  const cSiblingIds = cParentNode.children;

  // Convert sibling message IDs to their corresponding leaf node IDs
  // This allows navigation between different conversation branches
  const cSiblingLeafIds = cSiblingIds.map((pSiblingId: string) =>
    fFindLeafNodeInMap(pNodeMap, pSiblingId)
  );

  // Find current message's position among siblings
  const cCurrentIndex = cSiblingIds.indexOf(pMessageId);

  return {
    message: cMessage,
    siblingIds: cSiblingLeafIds,
    currentIndex: cCurrentIndex,
    totalSiblings: cSiblingIds.length
  };
}

/**
 * Builds sibling information for every message in a conversation.
 *
 * @param pMessages - All messages in the conversation
 * @returns Map of message ID to its sibling information
 */
export function fBuildSiblingInfoMap(
  pMessages: readonly DatabaseMessage[]
): Map<string, ChatMessageSiblingInfo> {
  const dNodeMap = new Map(pMessages.map((pMsg) => [pMsg.id, pMsg] as const));
  const dSiblingMap = new Map<string, ChatMessageSiblingInfo>();
  for (const cMsg of pMessages) {
    const cInfo = fGetMessageSiblings(dNodeMap, cMsg.id);
    if (cInfo) {
      dSiblingMap.set(cMsg.id, cInfo);
    }
  }
  return dSiblingMap;
}
