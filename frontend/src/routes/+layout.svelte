<script lang="ts">
  import { fGetLanguage } from '$lib/i18n';
  import { fApiUrl } from '$lib/api-url';
  import '../app.css';
  import { base } from '$app/paths';
  import { browser } from '$app/environment';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { untrack } from 'svelte';
  import { onMount } from 'svelte';

  import { DialogLocalShellAskpass, SidebarNavigation, StatusBar } from '$lib/components/app';
  import { PwaMetaTags, PwaRefreshAlert } from '$lib/components/pwa';
  import { pwaAssetsHead } from 'virtual:pwa-assets/head';

  import { cChatStore } from '$lib/stores/chat.svelte';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { fIsRouterMode, cServerStore } from '$lib/stores/server.svelte';
  import { fConfig, cSettingsStore } from '$lib/stores/settings.svelte';
  import { ModeWatcher } from 'mode-watcher';
  import { cRoutes } from '$lib/constants/routes';
  import { RouterService } from '$lib/services/router.service';
  import { Toaster } from 'svelte-sonner';
  import { cModelsStore } from '$lib/stores/models.svelte';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { cUiSettingsStore } from '$lib/stores/ui-settings.svelte';
  import { cAuthorizationHeader, cBearerPrefix, cTooltipDelayDuration } from '$lib/constants';
  import { cFaviconPaths, cFaviconSelectors } from '$lib/constants/pwa';
  import { fUseKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
  import { fUsePwa } from '$lib/hooks/use-pwa.svelte';
  import { fConversations } from '$lib/stores/conversations.svelte';
  import { cIsMobile } from '$lib/stores/viewport.svelte';
  import { cTheme } from '$lib/stores/theme.svelte';
  import { dBuildInfoStore } from '$lib/stores/build-info.svelte';

  import { cSettingsKeys } from '$lib/constants';

  let { children: vChildren } = $props();

  $effect(() => {
    document.documentElement.lang = fGetLanguage();
    document.documentElement.dir = fGetLanguage() === 'he-IL' ? 'rtl' : 'ltr';
  });
  let vInnerHeight = $state<number | undefined>();
  let vInnerWidth = $state(browser ? window.innerWidth : 0);

  let vChatSidebar:
    | {
        activateSearchMode?: () => void;
        editActiveConversation?: () => void;
      }
    | undefined = $state();

  let vShowBuildVersion = $derived(fConfig()[cSettingsKeys.SHOW_BUILD_VERSION] as boolean);
  let vShowStatusBar = $derived(fConfig()[cSettingsKeys.SHOW_STATUS_BAR] !== false);

  // --status-bar-height (app.css) lifts the fixed and bottom-anchored elements above the bar
  $effect(() => {
    document.documentElement.classList.toggle('has-status-bar', vShowStatusBar);
  });

  // Keep the hook object intact: destructuring needRefreshByStorage reads the getter once and freezes it
  const cPwa = fUsePwa();
  const { needRefresh: cNeedRefresh, updateServiceWorker: cUpdateServiceWorker } = cPwa;

  function fUpdateFavicon() {
    const cDark = cTheme.isSystemDark;

    let vIcoLink = document.querySelector(cFaviconSelectors.ICO_48X48) as HTMLLinkElement | null;
    if (vIcoLink) {
      vIcoLink.href = cDark ? cFaviconPaths.ICO_DARK : cFaviconPaths.ICO_LIGHT;
    }

    let vSvgLink = document.querySelector(cFaviconSelectors.SVG_ANY) as HTMLLinkElement | null;
    if (vSvgLink) {
      vSvgLink.href = cDark ? cFaviconPaths.SVG_DARK : cFaviconPaths.SVG_LIGHT;
    }
  }

  function fNavigateToConversation(pDirection: -1 | 1) {
    const cAllConvs = fConversations();

    if (cAllConvs.length === 0) return;

    const cCurrentId = page.params.id;

    if (!cCurrentId) {
      goto(RouterService.mChat(cAllConvs[pDirection === 1 ? 0 : cAllConvs.length - 1].id));

      return;
    }

    const cIdx = cAllConvs.findIndex((pC) => pC.id === cCurrentId);

    if (cIdx === -1) return;

    const cTargetIdx = cIdx + pDirection;

    if (cTargetIdx >= 0 && cTargetIdx < cAllConvs.length) {
      goto(RouterService.mChat(cAllConvs[cTargetIdx].id));
    } else {
      goto(cRoutes.NEW_CHAT);
    }
  }

  // Global keyboard shortcuts
  const { handleKeydown: cHandleKeydown } = fUseKeyboardShortcuts({
    editActiveConversation: () => vChatSidebar?.editActiveConversation?.(),
    navigateToPrevConversation: () => fNavigateToConversation(-1),
    navigateToNextConversation: () => fNavigateToConversation(1)
  });

  function fCheckApiKey() {
    const cApiKey = fConfig().apiKey;

    // Without a stored key there is nothing to re-validate here; the keyless
    // 401 case is handled by validateApiKey() at navigation time, and the
    // reload below must never fire in a keyless loop.
    if (!cApiKey || cApiKey.trim() === '') {
      return;
    }

    untrack(() => {
      if (
        (page.route.id === '/(chat)' || page.route.id === '/(chat)/chat/[id]') &&
        page.status !== 401 &&
        page.status !== 403
      ) {
        const dHeaders: Record<string, string> = {
          'Content-Type': 'application/json',
          [cAuthorizationHeader]: `${cBearerPrefix}${cApiKey.trim()}`
        };

        fetch(fApiUrl('/props'), { headers: dHeaders })
          .then((pResponse) => {
            if (pResponse.status === 401 || pResponse.status === 403) {
              window.location.reload();
            }
          })
          .catch((pE) => {
            console.error('Error checking API key:', pE);
          });
      }
    });
  }

  onMount(() => {
    fUpdateFavicon();
    // the settings of the interface are kept by the server (~/.mgs/config/ui-settings.json)
    void cUiSettingsStore.mLoad();
    // snapshot of every backend running stream on first load, populates the sidebar spinners
    // so the user sees each conv that has a live inference, even ones not opened yet
    void cChatStore.mSyncRemoteRunningStreams();
  });

  // refresh that snapshot when the tab returns to the foreground, a stream may have advanced
  // or ended while it was hidden. snapshot only, no polling
  function fHandleVisibilityChange() {
    if (document.visibilityState !== 'visible') return;
    void cChatStore.mSyncRemoteRunningStreams();
  }

  $effect(() => {
    void cTheme.isSystemDark;

    fUpdateFavicon();
  });

  // Initialize server properties on app load (run once)
  $effect(() => {
    // Only fetch if we don't already have props
    if (!cServerStore.vProps) {
      untrack(() => {
        cServerStore.mFetch();
      });
    }
  });

  // Sync settings when server props and the saved settings are loaded
  $effect(() => {
    const cServerProps = cServerStore.vProps;

    if (cServerProps && cUiSettingsStore.vIsLoaded) {
      untrack(() => {
        cSettingsStore.mSyncWithServerDefaults();
      });
    }
  });

  // Mirror the server's local shell MCP server (--local-shell-mcp-server) in the MCP server list:
  // added and enabled when offered, removed when the server no longer offers it
  $effect(() => {
    const cServerProps = cServerStore.vProps;

    if (cServerProps && cUiSettingsStore.vIsLoaded) {
      untrack(() => {
        cMcpStore.mSyncShellServer(Boolean(cServerProps.shell_mcp_server));
      });
    }
  });

  // Inject custom CSS at runtime through an action on the head style node
  // textContent keeps the value as text, never parsed as HTML
  function fCustomCss(pNode: HTMLStyleElement) {
    $effect(() => {
      pNode.textContent = (fConfig().customCss as string | undefined) ?? '';
    });
  }

  // Fetch router models when in router mode (for status and modalities)
  // Wait for models to be loaded first, run only once
  let vRouterModelsFetched = false;

  $effect(() => {
    const cIsRouter = fIsRouterMode();
    const cModelsCount = cModelsStore.vModels.length;

    // Only fetch router models once when we have models loaded and in router mode
    if (cIsRouter && cModelsCount > 0 && !vRouterModelsFetched) {
      vRouterModelsFetched = true;

      untrack(() => {
        cModelsStore.mFetchRouterModels();
      });
    }
  });

  // Live model status and load progress via the /models/sse feed (router mode)
  $effect(() => {
    if (!browser) return;
    if (!fIsRouterMode()) return;

    untrack(() => {
      cModelsStore.mSubscribeStatus();
    });

    return () => {
      cModelsStore.mUnsubscribeStatus();
    };
  });

  // Background MCP server health checks on app load.
  // Health-check every configured server with a URL - including disabled ones -
  // so the /mcp-servers page can display health metadata for servers that are
  // currently turned off. Disabled servers never get promoted to active
  // connections (see runHealthCheck), so their tools/prompts/resources stay
  // out of the chat-side stores.
  // Only IDLE servers are checked; already-resolved (SUCCESS / ERROR) servers
  // keep their existing state, so adding or removing a server does not flash
  // every other card back through skeleton state.
  $effect(() => {
    if (!browser) return;

    const cMcpServers = cMcpStore.mGetServers();

    const cServersWithUrls = cMcpServers.filter((pS) => pS.url.trim());

    if (cServersWithUrls.length > 0) {
      untrack(() => {
        // Run health checks in background (don't await)
        cMcpStore.mRunHealthChecksForServers(cServersWithUrls, true).catch((pError) => {
          console.warn('[layout] MCP health checks failed:', pError);
        });
      });
    }
  });

  // Monitor API key changes and redirect to error page if removed or changed when required
  $effect(() => {
    fCheckApiKey();
  });
</script>

<svelte:head>
  {#if pwaAssetsHead.themeColor}
    <meta name="theme-color" content={pwaAssetsHead.themeColor.content} />
  {/if}

  {#if fConfig().customCss}
    <style use:customCss></style>
  {/if}

  {#each pwaAssetsHead.links as link (link.href)}
    <link {...link} />
  {/each}

  <PwaMetaTags />
</svelte:head>

<svelte:window
  onkeydown={cHandleKeydown}
  bind:innerHeight={vInnerHeight}
  bind:innerWidth={vInnerWidth}
/>
<svelte:document onvisibilitychange={fHandleVisibilityChange} />

<Tooltip.Provider delayDuration={cTooltipDelayDuration}>
  <div class="flex flex-col md:flex-row">
    <SidebarNavigation
      onSearchClick={() => {
        if (cIsMobile.current) {
          goto(cRoutes.SEARCH);
        } else if (vChatSidebar?.activateSearchMode) {
          vChatSidebar.activateSearchMode();
        }
      }}
    />

    <div class="flex-1">
      {@render vChildren?.()}
    </div>
  </div>

  {#if vShowStatusBar}
    <StatusBar />
  {/if}

  <!-- sudo password prompts of the local shell MCP server, on any page -->
  <DialogLocalShellAskpass />

  <ModeWatcher />

  <Toaster richColors />
</Tooltip.Provider>

<!-- PWA update prompt + version -->
<div
  class="fixed right-4 bottom-[calc(1rem+var(--status-bar-height))] z-9999 flex flex-col items-end gap-1"
>
  {#if vShowBuildVersion && dBuildInfoStore.value}
    <span class="text-[10px] tabular-nums text-muted-foreground">{dBuildInfoStore.value}</span>
  {/if}

  <PwaRefreshAlert
    needRefresh={$cNeedRefresh || cPwa.needRefreshByStorage}
    forceReload={cPwa.needRefreshByStorage}
    updateServiceWorker={cUpdateServiceWorker}
  />
</div>
