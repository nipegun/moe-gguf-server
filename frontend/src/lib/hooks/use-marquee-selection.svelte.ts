/**
 * Reusable selection state-machine: shift+click/shift+drag range select plus
 * rubber-band marquee drag, anchored on the last clicked row. Both the sidebar
 * conversation list and the dialog conversations table share this.
 *
 * The hook mutates the consumer's SvelteSet<string> directly; the consumer
 * owns the source of truth and reads it like any other $state. orderedIds
 * must reflect the current visual order of selectable rows so the range
 * matches what the user sees on screen.
 */

import { SvelteSet } from 'svelte/reactivity';

interface UseMarqueeSelectionOptions {
  /** Latest selected-IDs set. Re-read per selection event so consumer-side reassignment works. */
  selectedIds: () => SvelteSet<string>;
  /** IDs in the current rendered order; used to compute shift+click ranges and gate marquee visibility. */
  orderedIds: () => string[];
  /** Document listeners attach only while the getter returns true. */
  enabled: () => boolean;
  /** DOM attribute key (after the `data-` prefix) that marks selectable rows. */
  attributeName?: () => string;
  /** Minimum pixel distance before a press becomes a marquee drag. */
  dragThresholdPx?: number;
}

export function fUseMarqueeSelection(pOptions: UseMarqueeSelectionOptions) {
  const cDragThresholdPx = pOptions.dragThresholdPx ?? 5;

  let vDragAnchorId = $state<string | null>(null);
  let vIsMarqueeDragging = $state(false);
  let vMouseDownActive = false;
  let vDragStartX = 0;
  let vDragStartY = 0;
  let vMousedownRowId: string | null = null;
  let vDragMode: 'add' | 'remove' | null = null;
  let vSuppressNextClick = false;

  function fResolveAttributeName(): string {
    return pOptions.attributeName?.() ?? 'conversation-row';
  }

  /**
   * `dataset` keys are camelCased. `data-conversation-row` -> `conversationRow`.
   * We resolve the attribute name once per call and read via the camelCase key.
   */
  function fDatasetKey(pKey: string = fResolveAttributeName()): string {
    return pKey.replace(/-([a-z])/g, (pIgnored, pC) => pC.toUpperCase());
  }

  function fDecideDragMode(pStartingRowId: string | null, pCurrentlySelected: ReadonlySet<string>) {
    return pStartingRowId !== null && pCurrentlySelected.has(pStartingRowId) ? 'remove' : 'add';
  }

  /**
   * Range-select uses Finder-style toggle-by-target semantics: if the target
   * row is currently selected the range becomes deselected, otherwise it
   * becomes selected. Anchor moves to `toId` so chained shift+clicks keep
   * extending from the previous endpoint.
   */
  function fRangeSelect(pFromId: string, pToId: string) {
    const cSelected = pOptions.selectedIds();
    const cOrder = pOptions.orderedIds();
    const cFromIdx = cOrder.indexOf(pFromId);
    const cToIdx = cOrder.indexOf(pToId);
    if (cFromIdx === -1 || cToIdx === -1) return;
    const [cLo, cHi] = cFromIdx < cToIdx ? [cFromIdx, cToIdx] : [cToIdx, cFromIdx];
    const cShouldSelect = !cSelected.has(pToId);
    for (let vI = cLo; vI <= cHi; vI++) {
      const cId = cOrder[vI];
      if (cShouldSelect) cSelected.add(cId);
      else cSelected.delete(cId);
    }
  }

  function fFindRowAtPoint(pX: number, pY: number): string | null {
    const cAttr = fResolveAttributeName();
    const cSelector = `[data-${cAttr}]`;
    const cKey = fDatasetKey(cAttr);
    let vBestMatch: HTMLElement | null = null;
    let vBestCenterDistance = Infinity;

    for (const cRow of document.querySelectorAll<HTMLElement>(cSelector)) {
      const cRect = cRow.getBoundingClientRect();
      if (pY >= cRect.top && pY <= cRect.bottom && pX >= cRect.left && pX <= cRect.right) {
        return cRow.dataset[cKey] ?? null;
      }
      if (pX >= cRect.left && pX <= cRect.right) {
        const cCenterDistance = Math.abs(pY - (cRect.top + cRect.height / 2));
        if (cCenterDistance < vBestCenterDistance) {
          vBestCenterDistance = cCenterDistance;
          vBestMatch = cRow;
        }
      }
    }
    return vBestMatch ? (vBestMatch.dataset[cKey] ?? null) : null;
  }

  function fUpdateMarqueeRect(pCurrentX: number, pCurrentY: number) {
    const cAttr = fResolveAttributeName();
    const cSelector = `[data-${cAttr}]`;
    const cKey = fDatasetKey(cAttr);
    const cSelected = pOptions.selectedIds();
    const cLeft = Math.min(vDragStartX, pCurrentX);
    const cTop = Math.min(vDragStartY, pCurrentY);
    const cRight = Math.max(vDragStartX, pCurrentX);
    const cBottom = Math.max(vDragStartY, pCurrentY);
    const cVisibleIds = new SvelteSet(pOptions.orderedIds());

    for (const cRow of document.querySelectorAll<HTMLElement>(cSelector)) {
      const cId = cRow.dataset[cKey];
      if (!cId || !cVisibleIds.has(cId)) continue;

      const cRect = cRow.getBoundingClientRect();
      const cIntersects = !(
        cRect.right < cLeft ||
        cRect.left > cRight ||
        cRect.bottom < cTop ||
        cRect.top > cBottom
      );

      if (vDragMode === 'add') {
        if (cIntersects) cSelected.add(cId);
      } else if (vDragMode === 'remove') {
        if (cIntersects && cSelected.has(cId)) cSelected.delete(cId);
      }
    }
  }

  function fHandleDocumentMouseMove(pEvent: MouseEvent) {
    if (!vMouseDownActive) return;

    if (pEvent.shiftKey && vDragAnchorId !== null) {
      const cTarget = fFindRowAtPoint(pEvent.clientX, pEvent.clientY);
      if (cTarget && cTarget !== vMousedownRowId) fRangeSelect(vDragAnchorId, cTarget);
      return;
    }

    if (!vIsMarqueeDragging) {
      const cDx = pEvent.clientX - vDragStartX;
      const cDy = pEvent.clientY - vDragStartY;
      if (Math.hypot(cDx, cDy) < cDragThresholdPx) return;
      vIsMarqueeDragging = true;
      vDragMode = fDecideDragMode(vMousedownRowId, pOptions.selectedIds());
    }
    fUpdateMarqueeRect(pEvent.clientX, pEvent.clientY);
  }

  function fHandleDocumentMouseUp(pEvent: MouseEvent) {
    if (vIsMarqueeDragging) {
      vSuppressNextClick = true;
      const cTarget = fFindRowAtPoint(pEvent.clientX, pEvent.clientY);
      if (cTarget) vDragAnchorId = cTarget;
    }
    vIsMarqueeDragging = false;
    vMouseDownActive = false;
    vMousedownRowId = null;
    vDragMode = null;
    vDragStartX = 0;
    vDragStartY = 0;
  }

  function fHandleClickCapture(pEvent: MouseEvent) {
    if (vSuppressNextClick) {
      pEvent.stopPropagation();
      pEvent.preventDefault();
      vSuppressNextClick = false;
    }
  }

  $effect(() => {
    if (!pOptions.enabled()) {
      fReset();
      return;
    }
    document.addEventListener('mousemove', fHandleDocumentMouseMove);
    document.addEventListener('mouseup', fHandleDocumentMouseUp);
    document.addEventListener('click', fHandleClickCapture, { capture: true });
    return () => {
      document.removeEventListener('mousemove', fHandleDocumentMouseMove);
      document.removeEventListener('mouseup', fHandleDocumentMouseUp);
      document.removeEventListener('click', fHandleClickCapture, { capture: true });
    };
  });

  function fRowMouseDown(pId: string, pEvent: MouseEvent) {
    if (!pOptions.enabled()) return;
    if (pEvent.button !== 0) return;
    pEvent.preventDefault();
    vMouseDownActive = true;
    vMousedownRowId = pId;
    vDragStartX = pEvent.clientX;
    vDragStartY = pEvent.clientY;
    vIsMarqueeDragging = false;
    vDragMode = null;
  }

  function fRowClick(pId: string, pShiftKey: boolean) {
    if (!pOptions.enabled()) return;
    const cSelected = pOptions.selectedIds();

    if (pShiftKey) {
      const cAnchor = vDragAnchorId;
      if (cAnchor !== null && cAnchor !== pId) {
        fRangeSelect(cAnchor, pId);
      } else if (cSelected.has(pId)) {
        cSelected.delete(pId);
      } else {
        cSelected.add(pId);
      }
      vDragAnchorId = pId;
      return;
    }

    if (cSelected.has(pId)) cSelected.delete(pId);
    else cSelected.add(pId);
    vDragAnchorId = pId;
  }

  function fReset() {
    vDragAnchorId = null;
    vIsMarqueeDragging = false;
    vMouseDownActive = false;
    vSuppressNextClick = false;
    vMousedownRowId = null;
    vDragMode = null;
    vDragStartX = 0;
    vDragStartY = 0;
  }

  return {
    rowMouseDown: fRowMouseDown,
    rowClick: fRowClick,
    reset: fReset,
    get dragAnchorId() {
      return vDragAnchorId;
    }
  };
}
