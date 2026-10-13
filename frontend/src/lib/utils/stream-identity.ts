/**
 * Build the conversation identity used by the server side replay buffer.
 *
 * The server identifies a stream session by a conversation id sent in the
 * X-Conversation-Id header. When the user has explicitly picked a model the
 * client appends ::modelName, so a per model session stays distinct and the
 * router resolves the owning child through its conv_id -> model map.
 */
export function fStreamIdentity(pConversationId: string, pModel?: string | null): string {
  if (!pConversationId) return '';
  if (!pModel) return pConversationId;
  return `${pConversationId}::${pModel}`;
}
