<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import {
    Trash2,
    Pencil,
    MoreHorizontal,
    Download,
    Loader2,
    Square,
    GitBranch,
    Pin,
    PinOff,
    ListChecks
  } from '@lucide/svelte';
  import { DropdownMenuActions } from '$lib/components/app';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import { cForkTreeDepthPadding } from '$lib/constants';
  import { RouterService } from '$lib/services/router.service';
  import { fGetAllLoadingChats } from '$lib/stores/chat.svelte';
  import { cConversationsStore } from '$lib/stores/conversations.svelte';
  import { TruncatedText } from '$lib/components/app';
  import { onMount } from 'svelte';

  interface Props {
    isActive?: boolean;
    depth?: number;
    conversation: DatabaseConversation;
    isSelectionMode?: boolean;
    isSelected?: boolean;
    onDelete?: (pId: string) => void;
    onEdit?: (pId: string) => void;
    onSelect?: (pId: string) => void;
    onStop?: (pId: string) => void;
    onToggleSelect?: (pId: string) => void;
    onEnterSelectionMode?: (pId: string) => void;
    onSelectionClick?: (pId: string, pOptions: { shiftKey: boolean }) => void;
    onRowMouseDown?: (pId: string, pEvent: MouseEvent) => void;
  }

  let {
    conversation: vConversation,
    onDelete: vOnDelete,
    onEdit: vOnEdit,
    onSelect: vOnSelect,
    onStop: vOnStop,
    onToggleSelect: vOnToggleSelect,
    onEnterSelectionMode: vOnEnterSelectionMode,
    onSelectionClick: vOnSelectionClick,
    onRowMouseDown: vOnRowMouseDown,
    isActive: vIsActive = false,
    isSelectionMode: vIsSelectionMode = false,
    isSelected: vIsSelected = false,
    depth: vDepth = 0
  }: Props = $props();

  let vRenderActionsDropdown = $state(false);
  let vDropdownOpen = $state(false);

  let vIsLoading = $derived(fGetAllLoadingChats().includes(vConversation.id));

  function fHandleEdit(pEvent: Event) {
    pEvent.stopPropagation();
    vOnEdit?.(vConversation.id);
  }

  function fHandleDelete(pEvent: Event) {
    pEvent.stopPropagation();
    vOnDelete?.(vConversation.id);
  }

  function fHandleStop(pEvent: Event) {
    pEvent.stopPropagation();
    vOnStop?.(vConversation.id);
  }

  function fHandleTogglePin() {
    cConversationsStore.mToggleConversationPin(vConversation.id);
  }

  function fHandleEnterSelectionMode(pEvent: Event) {
    pEvent.stopPropagation();
    vOnEnterSelectionMode?.(vConversation.id);
  }

  function fHandleGlobalEditEvent(pEvent: Event) {
    const cCustomEvent = pEvent as CustomEvent<{ conversationId: string }>;

    if (cCustomEvent.detail.conversationId === vConversation.id && vIsActive) {
      fHandleEdit(pEvent);
    }
  }

  function fHandleMouseLeave() {
    if (!vDropdownOpen) {
      vRenderActionsDropdown = false;
    }
  }

  function fHandleMouseOver() {
    if (vIsSelectionMode) return;
    vRenderActionsDropdown = true;
  }

  function fHandleSelect(pEvent: MouseEvent) {
    if (vIsSelectionMode) {
      vOnSelectionClick?.(vConversation.id, { shiftKey: pEvent.shiftKey });
    } else {
      vOnSelect?.(vConversation.id);
    }
  }

  function fHandleCheckboxClick(pEvent: MouseEvent) {
    pEvent.stopPropagation();
    if (vIsSelectionMode) {
      vOnSelectionClick?.(vConversation.id, { shiftKey: pEvent.shiftKey });
    } else {
      vOnToggleSelect?.(vConversation.id);
    }
  }

  function fHandleRowMouseDown(pEvent: MouseEvent) {
    vOnRowMouseDown?.(vConversation.id, pEvent);
  }

  function fHandleCheckboxKeydown(pEvent: KeyboardEvent) {
    if (pEvent.key !== ' ' && pEvent.key !== 'Enter') return;
    pEvent.stopPropagation();
    pEvent.preventDefault();
    if (vIsSelectionMode) {
      vOnSelectionClick?.(vConversation.id, { shiftKey: pEvent.shiftKey });
    } else {
      vOnToggleSelect?.(vConversation.id);
    }
  }

  $effect(() => {
    if (!vDropdownOpen) {
      vRenderActionsDropdown = false;
    }
  });

  onMount(() => {
    document.addEventListener('edit-active-conversation', fHandleGlobalEditEvent as EventListener);

    return () => {
      document.removeEventListener(
        'edit-active-conversation',
        fHandleGlobalEditEvent as EventListener
      );
    };
  });
</script>

<!-- svelte-ignore a11y_mouse_events_have_key_events -->
<button
  class="group flex min-h-9 w-full cursor-pointer items-center justify-between space-x-3 rounded-lg py-1.5 text-left transition-colors hover:bg-foreground/10 {vIsActive
    ? 'bg-foreground/5 text-accent-foreground'
    : ''} {vIsSelected ? 'bg-primary/10 hover:bg-primary/15' : ''} {vIsSelectionMode
    ? 'is-selection-mode'
    : ''} px-2"
  data-conversation-row={vConversation.id}
  onclick={(e) => fHandleSelect(e)}
  onmouseover={fHandleMouseOver}
  onmouseleave={fHandleMouseLeave}
  onmousedown={(e) => fHandleRowMouseDown(e)}
  onfocusin={fHandleMouseOver}
  onfocusout={(e) => {
    if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
      fHandleMouseLeave();
    }
  }}
>
  <div
    class="flex min-w-0 flex-1 items-center gap-2"
    style:padding-left="{vDepth * cForkTreeDepthPadding}px"
  >
    {#if vIsSelectionMode}
      <div
        class="shrink-0"
        onclick={(e) => fHandleCheckboxClick(e)}
        onkeydown={fHandleCheckboxKeydown}
        role="checkbox"
        aria-checked={vIsSelected}
        aria-label={vIsSelected
          ? fTranslate('text_602c0f50ebf7', { p0: vConversation.name })
          : fTranslate('text_8d29790dbe82', { p0: vConversation.name })}
        tabindex="-1"
      >
        <Checkbox
          checked={vIsSelected}
          aria-label={vIsSelected
            ? fTranslate('text_602c0f50ebf7', { p0: vConversation.name })
            : fTranslate('text_8d29790dbe82', { p0: vConversation.name })}
        />
      </div>
    {/if}

    {#if vDepth > 0}
      <Tooltip.Root>
        <Tooltip.Trigger>
          <!-- prevent another nested button element -->
          {#snippet child({ props })}
            <a
              {...props}
              href={RouterService.mChat(vConversation.forkedFromConversationId)}
              class="flex shrink-0 items-center text-muted-foreground transition-colors hover:text-foreground"
            >
              <GitBranch class="h-3.5 w-3.5" />
            </a>
          {/snippet}
        </Tooltip.Trigger>

        <Tooltip.Content>
          <p>{fTranslate('text_4d71ef1fdda5')}</p>
        </Tooltip.Content>
      </Tooltip.Root>
    {/if}

    {#if vIsLoading}
      <Tooltip.Root>
        <Tooltip.Trigger>
          <div
            class="stop-button flex {cIconClassDefault} shrink-0 cursor-pointer items-center justify-center rounded text-muted-foreground transition-colors hover:text-foreground"
            onclick={fHandleStop}
            onkeydown={(e) => e.key === 'Enter' && fHandleStop(e)}
            role="button"
            tabindex="0"
            aria-label={fTranslate('text_d4cb298c2663')}
          >
            <Loader2 class="loading-icon h-3.5 w-3.5 animate-spin" />

            <Square class="stop-icon hidden h-3 w-3 fill-current text-destructive" />
          </div>
        </Tooltip.Trigger>

        <Tooltip.Content>
          <p>{fTranslate('text_d4cb298c2663')}</p>
        </Tooltip.Content>
      </Tooltip.Root>
    {/if}

    <TruncatedText text={vConversation.name} class="text-sm font-medium" showTooltip={false} />
  </div>

  {#if !vIsSelectionMode && vRenderActionsDropdown}
    <div class="actions flex items-center">
      <DropdownMenuActions
        triggerIcon={MoreHorizontal}
        triggerTooltip="More actions"
        bind:open={vDropdownOpen}
        actions={[
          {
            icon: vConversation.pinned ? PinOff : Pin,
            label: vConversation.pinned
              ? fTranslate('text_ee3c71613054')
              : fTranslate('text_ff1cee744146'),
            onclick: (e: Event) => {
              e.stopPropagation();
              fHandleTogglePin();
            }
          },
          {
            icon: Pencil,
            label: fTranslate('text_464c4ffd019e'),
            onclick: fHandleEdit,
            shortcut: ['shift', 'cmd', 'e']
          },
          {
            icon: Download,
            label: fTranslate('text_3664895579f0'),
            onclick: (e: Event) => {
              e.stopPropagation();
              cConversationsStore.mDownloadConversation(vConversation.id);
            },
            shortcut: ['shift', 'cmd', 's']
          },
          {
            icon: ListChecks,
            label: fTranslate('text_2a78025de6aa'),
            onclick: fHandleEnterSelectionMode
          },
          {
            icon: Trash2,
            label: fTranslate('text_e2d0a54968ea'),
            onclick: fHandleDelete,
            variant: 'destructive',
            shortcut: ['shift', 'cmd', 'd'],
            separator: true
          }
        ]}
      />
    </div>
  {/if}
</button>

<style>
  button {
    :global([data-slot='dropdown-menu-trigger']:not([data-state='open'])) {
      opacity: 0;
    }

    &:is(:hover) :global([data-slot='dropdown-menu-trigger']),
    &:focus-within :global([data-slot='dropdown-menu-trigger']) {
      opacity: 1;
    }
    @media (max-width: 768px) {
      :global([data-slot='dropdown-menu-trigger']) {
        opacity: 1 !important;
      }
    }

    &.is-selection-mode :global([data-slot='dropdown-menu-trigger']) {
      display: none !important;
    }

    .stop-button {
      :global(.stop-icon) {
        display: none;
      }

      :global(.loading-icon) {
        display: block;
      }
    }

    &:is(:hover) .stop-button {
      :global(.stop-icon) {
        display: block;
      }

      :global(.loading-icon) {
        display: none;
      }
    }
  }
</style>
