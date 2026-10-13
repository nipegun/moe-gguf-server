/**
 * Scroll container binding and navigation guard for the ChatScreen.
 *
 * Binds the `AutoScrollController` to `document.documentElement`, exposes
 * the container for programmatic scrolling, and flags an `isNavigating`
 * window during route changes so the controller can reset without its
 * scroll handler seeing spurious events from layout shifts.
 */

import { afterNavigate, beforeNavigate } from '$app/navigation';
import type { AutoScrollController } from './use-auto-scroll.svelte';

export function fUseChatScreenScroll(pAutoScroll: AutoScrollController) {
  let vChatScrollContainer: HTMLElement | undefined = $state();
  let vIsNavigating = $state(false);

  function fHandleScroll(pEvent: UIEvent) {
    // Ignore scroll events caused by navigation layout changes or by our own
    // programmatic scrolls so they don't accidentally disable auto-scroll.
    if (vIsNavigating || !pEvent.isTrusted) return;
    pAutoScroll.mHandleScroll();
  }

  beforeNavigate(() => {
    vIsNavigating = true;
    pAutoScroll.mResetScrollState();
  });

  afterNavigate(() => {
    setTimeout(() => {
      vIsNavigating = false;
      pAutoScroll.mResetScrollState();
    }, 10);
  });

  $effect(() => {
    vChatScrollContainer = document.documentElement;
    pAutoScroll.mSetContainer(vChatScrollContainer);
  });

  return {
    get chatScrollContainer() {
      return vChatScrollContainer;
    },
    handleScroll: fHandleScroll
  };
}
