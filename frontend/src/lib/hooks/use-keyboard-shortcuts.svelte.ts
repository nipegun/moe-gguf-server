import { goto } from '$app/navigation';
import { KeyboardKey } from '$lib/enums';
import { cRoutes } from '$lib/constants/routes';

interface KeyboardShortcutsCallbacks {
  activateSearchMode?: () => void;
  editActiveConversation?: () => void;
  onSearchActivated?: () => void;
  deleteActiveConversation?: () => void;
  navigateToPrevConversation?: () => void;
  navigateToNextConversation?: () => void;
  toggleSidebar?: () => void;
}

export function fUseKeyboardShortcuts(pCallbacks: KeyboardShortcutsCallbacks) {
  function fHandleKeydown(pEvent: KeyboardEvent) {
    const cIsCmdOrCtrl = pEvent.metaKey || pEvent.ctrlKey;

    if (cIsCmdOrCtrl && pEvent.key === KeyboardKey.K_LOWER) {
      pEvent.preventDefault();
      pCallbacks.activateSearchMode?.();
      pCallbacks.onSearchActivated?.();
    }

    if (cIsCmdOrCtrl && pEvent.key === KeyboardKey.B_LOWER) {
      pEvent.preventDefault();
      pCallbacks.toggleSidebar?.();
    }

    if (
      cIsCmdOrCtrl &&
      pEvent.shiftKey &&
      (pEvent.key === KeyboardKey.O_LOWER || pEvent.key === KeyboardKey.O_UPPER)
    ) {
      pEvent.preventDefault();

      goto(cRoutes.NEW_CHAT);
    }

    if (pEvent.shiftKey && cIsCmdOrCtrl && pEvent.key === KeyboardKey.E_UPPER) {
      pEvent.preventDefault();
      pCallbacks.editActiveConversation?.();
    }

    if (
      cIsCmdOrCtrl &&
      pEvent.shiftKey &&
      (pEvent.key === KeyboardKey.D_LOWER || pEvent.key === KeyboardKey.D_UPPER)
    ) {
      pEvent.preventDefault();
      pCallbacks.deleteActiveConversation?.();
    }

    if (cIsCmdOrCtrl && pEvent.shiftKey && pEvent.key === KeyboardKey.ARROW_UP) {
      pEvent.preventDefault();
      pCallbacks.navigateToPrevConversation?.();
    }

    if (cIsCmdOrCtrl && pEvent.shiftKey && pEvent.key === KeyboardKey.ARROW_DOWN) {
      pEvent.preventDefault();
      pCallbacks.navigateToNextConversation?.();
    }
  }

  return { handleKeydown: fHandleKeydown };
}
