// Shared state for the context gauge popup. The dial and the card live in
// different DOM subtrees, so open state and placement are coordinated here.
// centerX and bottom place the card just above the dial: both are measured
// once at open time, relative to the closest form ancestor; the dial and
// the card share that positioning frame, so the values stay exact for the
// whole time the card is open.
// Mouse pointers open on hover with a short grace delay to travel from
// dial to card; touch pointers toggle on tap.
import {
  cContextGaugeCardHalfWidthPx,
  cContextGaugeCloseGraceMs,
  cContextGaugeDialGapPx,
  cContextGaugeEdgeMarginPx
} from '$lib/constants';

let vCloseTimer: ReturnType<typeof setTimeout> | undefined;
let vLastPointerType = '';

export const cGaugePopup = $state({ open: false, centerX: 0, bottom: 0 });

function fOpenFrom(pTrigger: HTMLElement): void {
  clearTimeout(vCloseTimer);
  const cFrame = pTrigger.closest('form');
  if (cFrame) {
    const cFrameRect = cFrame.getBoundingClientRect();
    const cTriggerRect = pTrigger.getBoundingClientRect();
    const cCenterX = cTriggerRect.left + cTriggerRect.width / 2 - cFrameRect.left;
    const cMin = cContextGaugeCardHalfWidthPx + cContextGaugeEdgeMarginPx;
    const cMax = cFrameRect.width - cContextGaugeCardHalfWidthPx - cContextGaugeEdgeMarginPx;
    cGaugePopup.centerX = Math.min(Math.max(cCenterX, cMin), Math.max(cMin, cMax));
    cGaugePopup.bottom = cFrameRect.bottom - cTriggerRect.top + cContextGaugeDialGapPx;
  }
  cGaugePopup.open = true;
}

function fToggleFrom(pTrigger: HTMLElement): void {
  if (cGaugePopup.open) {
    clearTimeout(vCloseTimer);
    cGaugePopup.open = false;
  } else {
    fOpenFrom(pTrigger);
  }
}

export function fGaugePopupClose(): void {
  clearTimeout(vCloseTimer);
  cGaugePopup.open = false;
}

export function fGaugeTriggerPointerDown(pEvent: PointerEvent): void {
  vLastPointerType = pEvent.pointerType;
}

export function fGaugeTriggerClick(pEvent: MouseEvent): void {
  if (vLastPointerType !== 'touch') return;
  fToggleFrom(pEvent.currentTarget as HTMLElement);
}

export function fGaugeTriggerKeydown(pEvent: KeyboardEvent): void {
  if (pEvent.key !== 'Enter' && pEvent.key !== ' ') return;
  pEvent.preventDefault();
  fToggleFrom(pEvent.currentTarget as HTMLElement);
}

export function fGaugeTriggerEnter(pEvent: PointerEvent): void {
  if (pEvent.pointerType !== 'mouse') return;
  fOpenFrom(pEvent.currentTarget as HTMLElement);
}

export function fGaugeTriggerLeave(pEvent: PointerEvent): void {
  if (pEvent.pointerType !== 'mouse') return;
  fScheduleClose();
}

export function fGaugeCardEnter(pEvent: PointerEvent): void {
  if (pEvent.pointerType !== 'mouse') return;
  clearTimeout(vCloseTimer);
}

export function fGaugeCardLeave(pEvent: PointerEvent): void {
  if (pEvent.pointerType !== 'mouse') return;
  fScheduleClose();
}

function fScheduleClose(): void {
  clearTimeout(vCloseTimer);
  vCloseTimer = setTimeout(() => {
    cGaugePopup.open = false;
  }, cContextGaugeCloseGraceMs);
}
