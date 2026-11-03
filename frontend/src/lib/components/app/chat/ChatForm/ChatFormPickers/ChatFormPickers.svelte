<script lang="ts">
  import ChatFormPickerMcpPrompts from './ChatFormPickerMcpPrompts/ChatFormPickerMcpPrompts.svelte';
  import ChatFormPickerMcpResources from './ChatFormPickerMcpResources.svelte';
  import type { GetPromptResult, MCPPromptInfo } from '$lib/types';

  interface Props {
    isPromptPickerOpen?: boolean;
    promptSearchQuery?: string;
    isInlineResourcePickerOpen?: boolean;
    resourceSearchQuery?: string;
    onPromptPickerClose?: () => void;
    onInlineResourcePickerClose?: () => void;
    onInlineResourceSelect?: () => void;
    onPromptLoadStart?: (
      pPlaceholderId: string,
      pPromptInfo: MCPPromptInfo,
      pArgs?: Record<string, string>
    ) => void;
    onPromptLoadComplete?: (pPlaceholderId: string, pResult: GetPromptResult) => void;
    onPromptLoadError?: (pPlaceholderId: string, pError: string) => void;
    onInlineResourceBrowse?: () => void;
  }

  let {
    isPromptPickerOpen: vIsPromptPickerOpen,
    promptSearchQuery: vPromptSearchQuery,
    isInlineResourcePickerOpen: vIsInlineResourcePickerOpen,
    resourceSearchQuery: vResourceSearchQuery,
    onPromptPickerClose: vOnPromptPickerClose,
    onInlineResourcePickerClose: vOnInlineResourcePickerClose,
    onInlineResourceSelect: vOnInlineResourceSelect,
    onPromptLoadStart: vOnPromptLoadStart,
    onPromptLoadComplete: vOnPromptLoadComplete,
    onPromptLoadError: vOnPromptLoadError,
    onInlineResourceBrowse: vOnInlineResourceBrowse
  }: Props = $props();

  let vPromptPickerRef: ChatFormPickerMcpPrompts | undefined = $state(undefined);
  let vResourcePickerRef: ChatFormPickerMcpResources | undefined = $state(undefined);

  /**
   * Delegates keyboard events to the active picker child.
   * Returns true if the event was handled.
   */
  export function fHandleKeydown(pEvent: KeyboardEvent): boolean {
    if (vIsPromptPickerOpen && vPromptPickerRef?.fHandleKeydown(pEvent)) {
      return true;
    }

    if (vIsInlineResourcePickerOpen && vResourcePickerRef?.fHandleKeydown(pEvent)) {
      return true;
    }

    return false;
  }
</script>

<ChatFormPickerMcpPrompts
  bind:this={vPromptPickerRef}
  isOpen={vIsPromptPickerOpen}
  searchQuery={vPromptSearchQuery}
  onClose={vOnPromptPickerClose}
  onPromptLoadStart={vOnPromptLoadStart}
  onPromptLoadComplete={vOnPromptLoadComplete}
  onPromptLoadError={vOnPromptLoadError}
/>

<ChatFormPickerMcpResources
  bind:this={vResourcePickerRef}
  isOpen={vIsInlineResourcePickerOpen}
  searchQuery={vResourceSearchQuery}
  onClose={vOnInlineResourcePickerClose}
  onResourceSelect={vOnInlineResourceSelect}
  onBrowse={vOnInlineResourceBrowse}
/>
