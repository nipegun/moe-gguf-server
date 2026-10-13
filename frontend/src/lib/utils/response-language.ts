import { fGetLanguage, fTranslate } from '$lib/i18n';
import { ContentPartType, MessageRole } from '$lib/enums';
import type { ApiChatMessageData } from '$lib/types/api';

/** Add the current locale instruction to the outgoing copy, never to stored history. */
export function fApplyResponseLanguage(
  pMessages: ApiChatMessageData[],
  pEnabled: boolean
): ApiChatMessageData[] {
  if (!pEnabled) return pMessages;

  const cInstruction = fTranslate('responseLanguage.instruction', { p0: fGetLanguage() });
  const cFirstMessage = pMessages[0];

  // Keep a single leading system message for templates that require it.
  if (cFirstMessage?.role === MessageRole.SYSTEM) {
    const cContent =
      typeof cFirstMessage.content === 'string'
        ? [cFirstMessage.content, cInstruction].filter(Boolean).join('\n\n')
        : [...cFirstMessage.content, { type: ContentPartType.TEXT, text: cInstruction }];

    return [{ ...cFirstMessage, content: cContent }, ...pMessages.slice(1)];
  }

  return [{ role: MessageRole.SYSTEM, content: cInstruction }, ...pMessages];
}
