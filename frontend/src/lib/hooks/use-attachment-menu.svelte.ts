import { fTranslate } from '../i18n';
import { page } from '$app/state';
import { AttachmentAction } from '$lib/enums';

export interface AttachmentModalityFlags {
  hasVisionModality: boolean;
  hasAudioModality: boolean;
  hasVideoModality: boolean;
  hasMcpPromptsSupport: boolean;
  hasMcpResourcesSupport: boolean;
}

export interface AttachmentActionCallbacks {
  onFileUpload?: () => void;
  onSystemPromptClick?: () => void;
  onMcpPromptClick?: () => void;
  onMcpResourcesClick?: () => void;
}

export interface UseAttachmentMenuReturn {
  readonly callbacks: Record<string, () => void>;
  isItemEnabled(pEnabledWhen: string | undefined): boolean;
  isItemVisible(pVisibleWhen: string | undefined): boolean;
  getSystemMessageTooltip(): string;
}

/**
 * useAttachmentMenu - Shared logic for attachment menu components.
 *
 * Encapsulates the modality-flag checks and callback wrapping that is
 * identical across the desktop dropdown (`ChatFormActionAddDropdown`)
 * and the mobile sheet (`ChatFormActionAddSheet`).
 *
 * @param pGetFlags   - Getter returning the current modality capability flags.
 * @param pGetCallbacks - Getter returning the raw action callbacks from props.
 * @param pClose      - Function that dismisses the hosting UI element (dropdown / sheet).
 */
export function fUseAttachmentMenu(
  pGetFlags: () => AttachmentModalityFlags,
  pGetCallbacks: () => AttachmentActionCallbacks,
  pClose: () => void
): UseAttachmentMenuReturn {
  const cModalityFlags = $derived(pGetFlags());

  const cCallbacks = $derived.by(() => {
    const cCbs = pGetCallbacks();
    const fWrap = (pFn?: () => void) => () => {
      pClose();
      pFn?.();
    };
    return {
      [AttachmentAction.FILE_UPLOAD]: fWrap(cCbs.onFileUpload),
      [AttachmentAction.SYSTEM_PROMPT_CLICK]: fWrap(cCbs.onSystemPromptClick),
      [AttachmentAction.MCP_PROMPT_CLICK]: fWrap(cCbs.onMcpPromptClick),
      [AttachmentAction.MCP_RESOURCES_CLICK]: fWrap(cCbs.onMcpResourcesClick)
    };
  });

  function fIsItemEnabled(pEnabledWhen: string | undefined): boolean {
    if (!pEnabledWhen || pEnabledWhen === 'always') return true;
    return !!cModalityFlags[pEnabledWhen as keyof AttachmentModalityFlags];
  }

  function fIsItemVisible(pVisibleWhen: string | undefined): boolean {
    if (!pVisibleWhen) return true;
    return !!cModalityFlags[pVisibleWhen as keyof AttachmentModalityFlags];
  }

  function fGetSystemMessageTooltip(): string {
    return !page.params.id ? fTranslate('text_6ac8d18c5928') : fTranslate('text_a945bdbb8e12');
  }

  return {
    get callbacks() {
      return cCallbacks;
    },
    isItemEnabled: fIsItemEnabled,
    isItemVisible: fIsItemVisible,
    getSystemMessageTooltip: fGetSystemMessageTooltip
  };
}
