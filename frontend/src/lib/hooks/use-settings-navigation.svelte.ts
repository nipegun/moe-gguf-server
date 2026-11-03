import { page } from '$app/state';
import { beforeNavigate } from '$app/navigation';
import { dSettingsReferrer } from '$lib/stores/settings-referrer.svelte';
import { cRoutes } from '$lib/constants/routes';

export interface ChatSettings {
  reset: () => void;
}

export function fUseSettingsNavigation() {
  const cSubroute = $state({
    activePanel: 'chat' as 'chat' | 'settings' | 'mcp',
    chatSettingsRef: undefined as ChatSettings | undefined
  });

  const cIsSettingsRoute = $derived(!!page.route.id?.startsWith('/settings'));

  beforeNavigate(({ to: pTo, from: pFrom }) => {
    if (pTo?.route?.id?.startsWith('/settings') && !pFrom?.route?.id?.startsWith('/settings')) {
      dSettingsReferrer.url = window.location.hash || cRoutes.START;
    }
  });

  $effect(() => {
    if (cSubroute.activePanel === 'settings' && cSubroute.chatSettingsRef) {
      cSubroute.chatSettingsRef.reset();
    }
  });

  // Return to chat when navigating to a new route
  $effect(() => {
    void page.url;

    cSubroute.activePanel = 'chat';
  });

  return {
    get panel() {
      return cSubroute;
    },

    get isSettingsRoute() {
      return cIsSettingsRoute;
    }
  };
}
