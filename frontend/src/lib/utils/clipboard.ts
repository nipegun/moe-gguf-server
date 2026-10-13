import { fTranslate } from '../i18n';
import { toast } from 'svelte-sonner';
import { AttachmentType } from '$lib/enums';
import type {
  DatabaseMessageExtra,
  DatabaseMessageExtraTextFile,
  DatabaseMessageExtraMcpPrompt,
  DatabaseMessageExtraMcpResource,
  ClipboardTextAttachment,
  ClipboardMcpPromptAttachment,
  ClipboardAttachment,
  ParsedClipboardContent
} from '$lib/types';

/**
 * Copy text to clipboard with toast notification
 * Uses modern clipboard API when available, falls back to legacy method for non-secure contexts
 * @param pText - Text to copy to clipboard
 * @param pSuccessMessage - Custom success message (optional)
 * @param pErrorMessage - Custom error message (optional)
 * @returns Promise<boolean> - True if successful, false otherwise
 */
export async function fCopyToClipboard(
  pText: string,
  pSuccessMessage = fTranslate('text_d37078fe6a1f'),
  pErrorMessage = fTranslate('text_978a1dc5401a')
): Promise<boolean> {
  try {
    // Try modern clipboard API first (secure contexts only)
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(pText);
      toast.success(pSuccessMessage);
      return true;
    }

    // Fallback for non-secure contexts
    const cTextArea = document.createElement('textarea');
    cTextArea.value = pText;
    cTextArea.style.position = 'fixed';
    cTextArea.style.left = '-999999px';
    cTextArea.style.top = '-999999px';
    document.body.appendChild(cTextArea);
    cTextArea.focus();
    cTextArea.select();

    const cSuccessful = document.execCommand('copy');
    document.body.removeChild(cTextArea);

    if (cSuccessful) {
      toast.success(pSuccessMessage);
      return true;
    } else {
      throw new Error(fTranslate('text_1b51fd2932aa'));
    }
  } catch (vError) {
    console.error('Failed to copy to clipboard:', vError);
    toast.error(pErrorMessage);
    return false;
  }
}

/**
 * Copy code with HTML entity decoding and toast notification
 * @param pRawCode - Raw code string that may contain HTML entities
 * @param pSuccessMessage - Custom success message (optional)
 * @param pErrorMessage - Custom error message (optional)
 * @returns Promise<boolean> - True if successful, false otherwise
 */
export async function fCopyCodeToClipboard(
  pRawCode: string,
  pSuccessMessage = fTranslate('text_d9c171407b14'),
  pErrorMessage = fTranslate('text_2c3e3b385b7a')
): Promise<boolean> {
  return fCopyToClipboard(pRawCode, pSuccessMessage, pErrorMessage);
}

/**
 * Formats a message with text attachments for clipboard copying.
 *
 * Default format (asPlainText = false):
 * ```
 * "Text message content"
 * [
 *   {"type":"TEXT","name":"filename.txt","content":"..."},
 *   {"type":"TEXT","name":"another.txt","content":"..."}
 * ]
 * ```
 *
 * Plain text format (asPlainText = true):
 * ```
 * Text message content
 *
 * file content here
 *
 * another file content
 * ```
 *
 * @param pContent - The message text content
 * @param pExtras - Optional array of message attachments
 * @param pAsPlainText - If true, format as plain text without JSON structure
 * @returns Formatted string for clipboard
 */
export function fFormatMessageForClipboard(
  pContent: string,
  pExtras?: DatabaseMessageExtra[],
  pAsPlainText: boolean = false
): string {
  // Filter text-like attachments (TEXT, MCP_PROMPT, and MCP_RESOURCE types)
  const cTextAttachments =
    pExtras?.filter(
      (
        pExtra
      ): pExtra is
        | DatabaseMessageExtraTextFile
        | DatabaseMessageExtraMcpPrompt
        | DatabaseMessageExtraMcpResource =>
        pExtra.type === AttachmentType.TEXT ||
        pExtra.type === AttachmentType.MCP_PROMPT ||
        pExtra.type === AttachmentType.MCP_RESOURCE
    ) ?? [];

  if (cTextAttachments.length === 0) {
    return pContent;
  }

  if (pAsPlainText) {
    const aParts = [pContent];
    for (const cAtt of cTextAttachments) {
      aParts.push(cAtt.content);
    }
    return aParts.join('\n\n');
  }

  const aClipboardAttachments: ClipboardAttachment[] = cTextAttachments.map((pAtt) => {
    if (pAtt.type === AttachmentType.MCP_PROMPT) {
      const cMcpAtt = pAtt as DatabaseMessageExtraMcpPrompt;
      return {
        type: AttachmentType.MCP_PROMPT,
        name: cMcpAtt.name,
        serverName: cMcpAtt.serverName,
        promptName: cMcpAtt.promptName,
        content: cMcpAtt.content,
        arguments: cMcpAtt.arguments
      } as ClipboardMcpPromptAttachment;
    }
    return {
      type: AttachmentType.TEXT,
      name: pAtt.name,
      content: pAtt.content
    } as ClipboardTextAttachment;
  });

  return `${JSON.stringify(pContent)}\n${JSON.stringify(aClipboardAttachments, null, 2)}`;
}

/**
 * Parses clipboard content to extract message and text attachments.
 * Supports both plain text and the special format with attachments.
 *
 * @param pClipboardText - Raw text from clipboard
 * @returns Parsed content with message and attachments
 */
export function fParseClipboardContent(pClipboardText: string): ParsedClipboardContent {
  const dDefaultResult: ParsedClipboardContent = {
    message: pClipboardText,
    textAttachments: [],
    mcpPromptAttachments: []
  };

  if (!pClipboardText.startsWith('"')) {
    return dDefaultResult;
  }

  try {
    let vStringEndIndex = -1;
    let vEscaped = false;

    for (let vI = 1; vI < pClipboardText.length; vI++) {
      const cChar = pClipboardText[vI];

      if (vEscaped) {
        vEscaped = false;
        continue;
      }

      if (cChar === '\\') {
        vEscaped = true;
        continue;
      }

      if (cChar === '"') {
        vStringEndIndex = vI;
        break;
      }
    }

    if (vStringEndIndex === -1) {
      return dDefaultResult;
    }

    const cJsonStringPart = pClipboardText.substring(0, vStringEndIndex + 1);
    const cRemainingPart = pClipboardText.substring(vStringEndIndex + 1).trim();

    const cMessage = JSON.parse(cJsonStringPart) as string;

    if (!cRemainingPart || !cRemainingPart.startsWith('[')) {
      return {
        message: cMessage,
        textAttachments: [],
        mcpPromptAttachments: []
      };
    }

    const cAttachments = JSON.parse(cRemainingPart) as unknown[];

    const aValidTextAttachments: ClipboardTextAttachment[] = [];
    const aValidMcpPromptAttachments: ClipboardMcpPromptAttachment[] = [];

    for (const cAtt of cAttachments) {
      if (fIsValidMcpPromptAttachment(cAtt)) {
        aValidMcpPromptAttachments.push({
          type: AttachmentType.MCP_PROMPT,
          name: cAtt.name,
          serverName: cAtt.serverName,
          promptName: cAtt.promptName,
          content: cAtt.content,
          arguments: cAtt.arguments
        });
      } else if (fIsValidTextAttachment(cAtt)) {
        aValidTextAttachments.push({
          type: AttachmentType.TEXT,
          name: cAtt.name,
          content: cAtt.content
        });
      }
    }

    return {
      message: cMessage,
      textAttachments: aValidTextAttachments,
      mcpPromptAttachments: aValidMcpPromptAttachments
    };
  } catch {
    return dDefaultResult;
  }
}

/**
 * Type guard to validate an MCP prompt attachment object
 * @param pObj The object to validate
 * @returns true if the object is a valid MCP prompt attachment
 */
function fIsValidMcpPromptAttachment(pObj: unknown): pObj is {
  type: string;
  name: string;
  serverName: string;
  promptName: string;
  content: string;
  arguments?: Record<string, string>;
} {
  if (typeof pObj !== 'object' || pObj === null) {
    return false;
  }

  const cRecord = pObj as Record<string, unknown>;

  return (
    (cRecord.type === AttachmentType.MCP_PROMPT || cRecord.type === 'MCP_PROMPT') &&
    typeof cRecord.name === 'string' &&
    typeof cRecord.serverName === 'string' &&
    typeof cRecord.promptName === 'string' &&
    typeof cRecord.content === 'string'
  );
}

/**
 * Type guard to validate a text attachment object
 * @param pObj The object to validate
 * @returns true if the object is a valid text attachment
 */
function fIsValidTextAttachment(
  pObj: unknown
): pObj is { type: string; name: string; content: string } {
  if (typeof pObj !== 'object' || pObj === null) {
    return false;
  }

  const cRecord = pObj as Record<string, unknown>;

  return (
    (cRecord.type === AttachmentType.TEXT || cRecord.type === 'TEXT') &&
    typeof cRecord.name === 'string' &&
    typeof cRecord.content === 'string'
  );
}

/**
 * Checks if clipboard content contains our special format with attachments
 * @param pClipboardText - Raw text from clipboard
 * @returns true if the clipboard content contains our special format with attachments
 */
export function fHasClipboardAttachments(pClipboardText: string): boolean {
  if (!pClipboardText.startsWith('"')) {
    return false;
  }

  const cParsed = fParseClipboardContent(pClipboardText);
  return cParsed.textAttachments.length > 0 || cParsed.mcpPromptAttachments.length > 0;
}
