<script lang="ts">
  import { fTranslate } from '../../lib/i18n';

  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { browser } from '$app/environment';

  import { SearchInput, SidebarNavigationSearchResults } from '$lib/components/app';
  import { cRoutes } from '$lib/constants/routes';
  import { RouterService } from '$lib/services/router.service';
  import { cConversationsStore, fConversations } from '$lib/stores/conversations.svelte';
  import { cChatStore } from '$lib/stores/chat.svelte';
  import { cIsMobile } from '$lib/stores/viewport.svelte';

  let vSearchQuery = $state('');
  let vSearchInputRef = $state<HTMLInputElement | null>(null);

  let vCurrentChatId = $derived(page.params.id);

  let vFilteredConversations = $derived.by(() => {
    const cQuery = vSearchQuery.trim().toLowerCase();
    if (cQuery.length === 0) return [];
    return fConversations().filter((pC) => pC.name.toLowerCase().includes(cQuery));
  });

  // Search page is intended for mobile; on desktop the sidebar already exposes
  // in-place search, so bounce back to a chat.
  $effect(() => {
    if (browser && !cIsMobile.current) {
      goto(cRoutes.NEW_CHAT, { replaceState: true });
    }
  });

  async function fSelectConversation(pId: string) {
    await goto(RouterService.mChat(pId));
  }

  async function fHandleEditConversation(pId: string) {
    const cConversation = fConversations().find((pC) => pC.id === pId);
    if (!cConversation) return;

    const cNewName = window.prompt(fTranslate('text_4f10b6dc1d44'), cConversation.name);
    if (cNewName && cNewName.trim()) {
      await cConversationsStore.mUpdateConversationName(pId, cNewName.trim());
    }
  }

  async function fHandleDeleteConversation(pId: string) {
    const cConversation = fConversations().find((pC) => pC.id === pId);
    if (!cConversation) return;

    const cConfirmed = window.confirm(fTranslate('text_749f8d33a6ef', { p0: cConversation.name }));
    if (!cConfirmed) return;

    await cConversationsStore.mDeleteConversation(pId, { deleteWithForks: false });
  }

  function fHandleStopGeneration(pId: string) {
    cChatStore.mStopGenerationForChat(pId);
  }

  function fHandleBack() {
    if (history.length > 1) {
      history.back();
    } else {
      goto(cRoutes.NEW_CHAT);
    }
  }
</script>

<svelte:head>
  <title>{fTranslate('text_3a11cdd20068')}</title>
</svelte:head>

<div class="fixed top-0 z-10 left-0 right-0 p-2">
  <SearchInput
    autofocus
    bind:value={vSearchQuery}
    bind:ref={vSearchInputRef}
    onClose={fHandleBack}
    placeholder={fTranslate('text_e90dd0173105')}
  />
</div>

<div class="p-2 pt-16">
  <SidebarNavigationSearchResults
    searchQuery={vSearchQuery}
    filteredConversations={vFilteredConversations}
    currentChatId={vCurrentChatId}
    onSelect={fSelectConversation}
    onEdit={fHandleEditConversation}
    onDelete={fHandleDeleteConversation}
    onStop={fHandleStopGeneration}
  />
</div>
