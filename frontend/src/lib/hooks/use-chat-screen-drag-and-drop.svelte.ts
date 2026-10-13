/**
 * Drag-and-drop state machine for the ChatScreen.
 *
 * Tracks pointer enter/leave nesting so the overlay stays visible while the
 * cursor traverses child elements, then routes the dropped files either to
 * the active message-edit handler (if a message is being edited) or to the
 * caller's onDrop callback.
 */

import { fGetAddFilesHandler, fIsEditing } from '$lib/stores/chat.svelte';

interface UseChatScreenDragAndDropOptions {
  /** Called when the user drops files and no message is being edited. */
  onDrop: (pFiles: File[]) => void;
}

export function fUseChatScreenDragAndDrop(pOptions: UseChatScreenDragAndDropOptions) {
  let vDragCounter = $state(0);
  let vIsDragOver = $state(false);

  function fHandleDragEnter(pEvent: DragEvent) {
    pEvent.preventDefault();
    vDragCounter++;
    if (pEvent.dataTransfer?.types.includes('Files')) {
      vIsDragOver = true;
    }
  }

  function fHandleDragLeave(pEvent: DragEvent) {
    pEvent.preventDefault();
    vDragCounter--;
    if (vDragCounter === 0) {
      vIsDragOver = false;
    }
  }

  function fHandleDragOver(pEvent: DragEvent) {
    pEvent.preventDefault();
  }

  async function fHandleDrop(pEvent: DragEvent) {
    pEvent.preventDefault();
    vIsDragOver = false;
    vDragCounter = 0;

    if (!pEvent.dataTransfer?.files) return;

    const cFiles = Array.from(pEvent.dataTransfer.files);

    if (fIsEditing()) {
      const cHandler = fGetAddFilesHandler();
      if (cHandler) {
        cHandler(cFiles);
        return;
      }
    }

    pOptions.onDrop(cFiles);
  }

  return {
    get isDragOver() {
      return vIsDragOver;
    },
    dragHandlers: {
      dragenter: fHandleDragEnter,
      dragleave: fHandleDragLeave,
      dragover: fHandleDragOver,
      drop: fHandleDrop
    }
  };
}
