<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { cConversationsStore } from '$lib/stores/conversations.svelte';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { debounce, uuid } from '$lib/utils';
  import { KeyboardKey } from '$lib/enums';
  import type { MCPPromptInfo, GetPromptResult, MCPServerSettingsEntry } from '$lib/types';
  import { SvelteMap } from 'svelte/reactivity';
  import {
    ChatFormPickerPopover,
    ChatFormPickerList,
    ChatFormPickerListItem,
    ChatFormPickerItemHeader,
    ChatFormPickerListItemSkeleton,
    ChatFormPromptPickerArgumentForm
  } from '$lib/components/app/chat';
  import Badge from '$lib/components/ui/badge/badge.svelte';

  interface Props {
    class?: string;
    isOpen?: boolean;
    searchQuery?: string;
    onClose?: () => void;
    onPromptLoadStart?: (
      pPlaceholderId: string,
      pPromptInfo: MCPPromptInfo,
      pArgs?: Record<string, string>
    ) => void;
    onPromptLoadComplete?: (pPlaceholderId: string, pResult: GetPromptResult) => void;
    onPromptLoadError?: (pPlaceholderId: string, pError: string) => void;
  }

  let {
    class: vClassName = '',
    isOpen: vIsOpen = false,
    searchQuery: vSearchQuery = '',
    onClose: vOnClose,
    onPromptLoadStart: vOnPromptLoadStart,
    onPromptLoadComplete: vOnPromptLoadComplete,
    onPromptLoadError: vOnPromptLoadError
  }: Props = $props();

  let vPrompts = $state<MCPPromptInfo[]>([]);
  let vIsLoading = $state(false);
  let vSelectedPrompt = $state<MCPPromptInfo | null>(null);
  let vPromptArgs = $state<Record<string, string>>({});
  let vSelectedIndex = $state(0);
  let vInternalSearchQuery = $state('');
  let vPromptError = $state<string | null>(null);
  let vSelectedIndexBeforeArgumentForm = $state<number | null>(null);

  let vSuggestions = $state<Record<string, string[]>>({});
  let vLoadingSuggestions = $state<Record<string, boolean>>({});
  let vActiveAutocomplete = $state<string | null>(null);
  let vAutocompleteIndex = $state(0);

  let vServerSettingsMap = $derived.by(() => {
    const cServers = cMcpStore.mGetServers();
    const cMap = new SvelteMap<string, MCPServerSettingsEntry>();

    for (const cServer of cServers) {
      cMap.set(cServer.id, cServer);
    }

    return cMap;
  });

  $effect(() => {
    if (vIsOpen) {
      fLoadPrompts();
      vSelectedIndex = 0;
    } else {
      vSelectedPrompt = null;
      vPromptArgs = {};
      vPromptError = null;
    }
  });

  $effect(() => {
    if (vFilteredPrompts.length > 0 && vSelectedIndex >= vFilteredPrompts.length) {
      vSelectedIndex = 0;
    }
  });

  async function fLoadPrompts() {
    vIsLoading = true;

    try {
      const cPerChatOverrides = cConversationsStore.mGetAllMcpServerOverrides();

      const cInitialized = await cMcpStore.mEnsureInitialized(cPerChatOverrides);

      if (!cInitialized) {
        vPrompts = [];

        return;
      }

      vPrompts = await cMcpStore.mGetAllPrompts();
    } catch (vError) {
      console.error('[ChatFormPickerMcpPrompts] Failed to load prompts:', vError);
      vPrompts = [];
    } finally {
      vIsLoading = false;
    }
  }

  function fHandlePromptClick(pPrompt: MCPPromptInfo) {
    const cArgs = pPrompt.arguments ?? [];

    if (cArgs.length > 0) {
      vSelectedIndexBeforeArgumentForm = vSelectedIndex;
      vSelectedPrompt = pPrompt;
      vPromptArgs = {};
      vPromptError = null;

      requestAnimationFrame(() => {
        const cFirstInput = document.querySelector(`#arg-${cArgs[0].name}`) as HTMLInputElement;
        if (cFirstInput) {
          cFirstInput.focus();
        }
      });
    } else {
      fExecutePrompt(pPrompt, {});
    }
  }

  async function fExecutePrompt(pPrompt: MCPPromptInfo, pArgs: Record<string, string>) {
    vPromptError = null;

    const cPlaceholderId = uuid();

    const cNonEmptyArgs = Object.fromEntries(
      Object.entries(pArgs).filter(([, pValue]) => pValue.trim() !== '')
    );
    const cArgsToPass = Object.keys(cNonEmptyArgs).length > 0 ? cNonEmptyArgs : undefined;

    vOnPromptLoadStart?.(cPlaceholderId, pPrompt, cArgsToPass);
    vOnClose?.();

    try {
      const cResult = await cMcpStore.mGetPrompt(pPrompt.serverName, pPrompt.name, pArgs);
      vOnPromptLoadComplete?.(cPlaceholderId, cResult);
    } catch (vError) {
      const cErrorMessage =
        vError instanceof Error ? vError.message : fTranslate('text_f2c329cfa93e');
      vOnPromptLoadError?.(cPlaceholderId, cErrorMessage);
    }
  }

  function fHandleArgumentSubmit(pEvent: SubmitEvent) {
    pEvent.preventDefault();

    if (vSelectedPrompt) {
      fExecutePrompt(vSelectedPrompt, vPromptArgs);
    }
  }

  const cFetchCompletions = debounce(async (pArgName: string, pValue: string) => {
    if (!vSelectedPrompt || pValue.length < 1) {
      vSuggestions[pArgName] = [];

      return;
    }

    if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
      console.log('[ChatFormPickerMcpPrompts] Fetching completions for:', {
        serverName: vSelectedPrompt.serverName,
        promptName: vSelectedPrompt.name,
        argName: pArgName,
        value: pValue
      });
    }

    vLoadingSuggestions[pArgName] = true;

    try {
      const cResult = await cMcpStore.mGetPromptCompletions(
        vSelectedPrompt.serverName,
        vSelectedPrompt.name,
        pArgName,
        pValue
      );

      if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
        console.log('[ChatFormPickerMcpPrompts] Autocomplete result:', {
          argName: pArgName,
          value: pValue,
          result: cResult,
          suggestionsCount: cResult?.values.length ?? 0
        });
      }

      if (cResult && cResult.values.length > 0) {
        // Filter out empty strings from suggestions
        const cFilteredValues = cResult.values.filter((pV) => pV.trim() !== '');

        if (cFilteredValues.length > 0) {
          vSuggestions[pArgName] = cFilteredValues;
          vActiveAutocomplete = pArgName;
          vAutocompleteIndex = 0;
        } else {
          vSuggestions[pArgName] = [];
        }
      } else {
        vSuggestions[pArgName] = [];
      }
    } catch (vError) {
      console.error('[ChatFormPickerMcpPrompts] Failed to fetch completions:', vError);
      vSuggestions[pArgName] = [];
    } finally {
      vLoadingSuggestions[pArgName] = false;
    }
  }, 200);

  function fHandleArgInput(pArgName: string, pValue: string) {
    vPromptArgs[pArgName] = pValue;
    cFetchCompletions(pArgName, pValue);
  }

  function fSelectSuggestion(pArgName: string, pValue: string) {
    vPromptArgs[pArgName] = pValue;
    vSuggestions[pArgName] = [];
    vActiveAutocomplete = null;
  }

  function fHandleArgKeydown(pEvent: KeyboardEvent, pArgName: string) {
    const cArgSuggestions = vSuggestions[pArgName] ?? [];

    // Handle Escape - return to prompt selection list
    if (pEvent.key === KeyboardKey.ESCAPE) {
      pEvent.preventDefault();
      pEvent.stopPropagation();
      fHandleCancelArgumentForm();
      return;
    }

    if (cArgSuggestions.length === 0 || vActiveAutocomplete !== pArgName) return;

    if (pEvent.key === KeyboardKey.ARROW_DOWN) {
      pEvent.preventDefault();
      vAutocompleteIndex = Math.min(vAutocompleteIndex + 1, cArgSuggestions.length - 1);
    } else if (pEvent.key === KeyboardKey.ARROW_UP) {
      pEvent.preventDefault();
      vAutocompleteIndex = Math.max(vAutocompleteIndex - 1, 0);
    } else if (pEvent.key === KeyboardKey.ENTER && cArgSuggestions[vAutocompleteIndex]) {
      pEvent.preventDefault();
      pEvent.stopPropagation();
      fSelectSuggestion(pArgName, cArgSuggestions[vAutocompleteIndex]);
    }
  }

  function fHandleArgBlur(pArgName: string) {
    // Delay to allow click on suggestion
    setTimeout(() => {
      if (vActiveAutocomplete === pArgName) {
        vSuggestions[pArgName] = [];
        vActiveAutocomplete = null;
      }
    }, 150);
  }

  function fHandleArgFocus(pArgName: string) {
    if ((vSuggestions[pArgName]?.length ?? 0) > 0) {
      vActiveAutocomplete = pArgName;
    }
  }

  function fHandleCancelArgumentForm() {
    // Restore the previously selected prompt index
    if (vSelectedIndexBeforeArgumentForm !== null) {
      vSelectedIndex = vSelectedIndexBeforeArgumentForm;
      vSelectedIndexBeforeArgumentForm = null;
    }
    vSelectedPrompt = null;
    vPromptArgs = {};
    vPromptError = null;
  }

  export function fHandleKeydown(pEvent: KeyboardEvent): boolean {
    if (!vIsOpen) return false;

    if (pEvent.key === KeyboardKey.ESCAPE) {
      pEvent.preventDefault();
      if (vSelectedPrompt) {
        // Return to prompt selection list, keeping the selected prompt active
        fHandleCancelArgumentForm();
      } else {
        vOnClose?.();
      }

      return true;
    }

    if (pEvent.key === KeyboardKey.ARROW_DOWN) {
      pEvent.preventDefault();
      if (vFilteredPrompts.length > 0) {
        vSelectedIndex = (vSelectedIndex + 1) % vFilteredPrompts.length;
      }

      return true;
    }

    if (pEvent.key === KeyboardKey.ARROW_UP) {
      pEvent.preventDefault();
      if (vFilteredPrompts.length > 0) {
        vSelectedIndex = vSelectedIndex === 0 ? vFilteredPrompts.length - 1 : vSelectedIndex - 1;
      }

      return true;
    }

    if (pEvent.key === KeyboardKey.ENTER && !vSelectedPrompt) {
      pEvent.preventDefault();
      if (vFilteredPrompts[vSelectedIndex]) {
        fHandlePromptClick(vFilteredPrompts[vSelectedIndex]);
      }

      return true;
    }

    return false;
  }

  let vFilteredPrompts = $derived.by(() => {
    const cSortedServers = cMcpStore.mGetServers();
    const dServerOrderMap = new Map(cSortedServers.map((pServer, pIndex) => [pServer.id, pIndex]));

    const cSortedPrompts = [...vPrompts].sort((pA, pB) => {
      const cOrderA = dServerOrderMap.get(pA.serverName) ?? Number.MAX_SAFE_INTEGER;
      const cOrderB = dServerOrderMap.get(pB.serverName) ?? Number.MAX_SAFE_INTEGER;
      return cOrderA - cOrderB;
    });

    const cQuery = (vSearchQuery || vInternalSearchQuery).toLowerCase();
    if (!cQuery) return cSortedPrompts;

    return cSortedPrompts.filter(
      (pPrompt) =>
        pPrompt.name.toLowerCase().includes(cQuery) ||
        pPrompt.title?.toLowerCase().includes(cQuery) ||
        pPrompt.description?.toLowerCase().includes(cQuery)
    );
  });

  let vShowSearchInput = $derived(vPrompts.length > 3);
</script>

<ChatFormPickerPopover
  bind:isOpen={vIsOpen}
  class={vClassName}
  srLabel="Open prompt picker"
  onClose={vOnClose}
  onKeydown={fHandleKeydown}
>
  {#if vSelectedPrompt}
    {@const prompt = vSelectedPrompt}
    {@const server = vServerSettingsMap.get(prompt.serverName)}
    {@const serverLabel = server ? cMcpStore.mGetServerLabel(server) : prompt.serverName}

    <div class="p-4">
      <ChatFormPickerItemHeader
        {server}
        {serverLabel}
        title={prompt.title || prompt.name}
        description={prompt.description}
      >
        {#snippet titleExtra()}
          {#if prompt.arguments?.length}
            <Badge variant="secondary">
              {prompt.arguments.length}
              {fTranslate('text_b25f03dedd69')}{prompt.arguments.length > 1 ? 's' : ''}
            </Badge>
          {/if}
        {/snippet}
      </ChatFormPickerItemHeader>

      <ChatFormPromptPickerArgumentForm
        prompt={vSelectedPrompt}
        promptArgs={vPromptArgs}
        suggestions={vSuggestions}
        loadingSuggestions={vLoadingSuggestions}
        activeAutocomplete={vActiveAutocomplete}
        autocompleteIndex={vAutocompleteIndex}
        promptError={vPromptError}
        onArgInput={fHandleArgInput}
        onArgKeydown={fHandleArgKeydown}
        onArgBlur={fHandleArgBlur}
        onArgFocus={fHandleArgFocus}
        onSelectSuggestion={fSelectSuggestion}
        onSubmit={fHandleArgumentSubmit}
        onCancel={fHandleCancelArgumentForm}
      />
    </div>
  {:else}
    <ChatFormPickerList
      items={vFilteredPrompts}
      isLoading={vIsLoading}
      selectedIndex={vSelectedIndex}
      bind:searchQuery={vInternalSearchQuery}
      showSearchInput={vShowSearchInput}
      searchPlaceholder="Search prompts..."
      emptyMessage={fTranslate('text_a3b9f9811e4d')}
      itemKey={(prompt) => prompt.serverName + ':' + prompt.name}
    >
      {#snippet item(prompt, index, isSelected)}
        {@const server = vServerSettingsMap.get(prompt.serverName)}
        {@const serverLabel = server ? cMcpStore.mGetServerLabel(server) : prompt.serverName}

        <ChatFormPickerListItem
          dataIndex={index}
          {isSelected}
          onclick={() => fHandlePromptClick(prompt)}
        >
          <ChatFormPickerItemHeader
            {server}
            {serverLabel}
            title={prompt.title || prompt.name}
            description={prompt.description}
          >
            {#snippet titleExtra()}
              {#if prompt.arguments?.length}
                <Badge variant="secondary">
                  {prompt.arguments.length}
                  {fTranslate('text_b25f03dedd69')}{prompt.arguments.length > 1 ? 's' : ''}
                </Badge>
              {/if}
            {/snippet}
          </ChatFormPickerItemHeader>
        </ChatFormPickerListItem>
      {/snippet}

      {#snippet skeleton()}
        <ChatFormPickerListItemSkeleton titleWidth="w-32" showBadge />
      {/snippet}
    </ChatFormPickerList>
  {/if}
</ChatFormPickerPopover>
