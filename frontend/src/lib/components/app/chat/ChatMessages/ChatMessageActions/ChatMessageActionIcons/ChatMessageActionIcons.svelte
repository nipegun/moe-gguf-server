<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { Edit, Copy, RefreshCw, Trash2, ArrowRight, GitBranch } from '@lucide/svelte';
  import {
    ActionIcon,
    ChatMessageActionIconsBranchingControls,
    DialogConfirmation
  } from '$lib/components/app';
  import { Switch } from '$lib/components/ui/switch';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import Input from '$lib/components/ui/input/input.svelte';
  import Label from '$lib/components/ui/label/label.svelte';
  import { MessageRole } from '$lib/enums';
  import { fActiveConversation } from '$lib/stores/conversations.svelte';

  interface Props {
    role: MessageRole.USER | MessageRole.ASSISTANT;
    justify: 'start' | 'end';
    actionsPosition: 'left' | 'right';
    siblingInfo?: ChatMessageSiblingInfo | null;
    showDeleteDialog: boolean;
    deletionInfo: {
      totalCount: number;
      userMessages: number;
      assistantMessages: number;
      messageTypes: string[];
    } | null;
    onCopy: () => void;
    onEdit?: () => void;
    onRegenerate?: () => void;
    onContinue?: () => void;
    onForkConversation?: (pOptions: { name: string; includeAttachments: boolean }) => void;
    onDelete: () => void;
    onConfirmDelete: () => void;
    onNavigateToSibling?: (pSiblingId: string) => void;
    onShowDeleteDialogChange: (pShow: boolean) => void;
    showRawOutputSwitch?: boolean;
    rawOutputEnabled?: boolean;
    onRawOutputToggle?: (pEnabled: boolean) => void;
  }

  let {
    actionsPosition: vActionsPosition,
    deletionInfo: vDeletionInfo,
    justify: vJustify,
    onCopy: vOnCopy,
    onEdit: vOnEdit,
    onConfirmDelete: vOnConfirmDelete,
    onContinue: vOnContinue,
    onDelete: vOnDelete,
    onForkConversation: vOnForkConversation,
    onNavigateToSibling: vOnNavigateToSibling,
    onShowDeleteDialogChange: vOnShowDeleteDialogChange,
    onRegenerate: vOnRegenerate,
    role: vRole,
    siblingInfo: vSiblingInfo = null,
    showDeleteDialog: vShowDeleteDialog,
    showRawOutputSwitch: vShowRawOutputSwitch = false,
    rawOutputEnabled: vRawOutputEnabled = false,
    onRawOutputToggle: vOnRawOutputToggle
  }: Props = $props();

  let vShowForkDialog = $state(false);
  let vForkName = $state('');
  let vForkIncludeAttachments = $state(true);

  function fHandleConfirmDelete() {
    vOnConfirmDelete();
    vOnShowDeleteDialogChange(false);
  }

  function fHandleOpenForkDialog() {
    const cConv = fActiveConversation();

    vForkName = fTranslate('text_205e60607abc', { p0: cConv?.name ?? 'Conversation' });
    vForkIncludeAttachments = true;
    vShowForkDialog = true;
  }

  function fHandleConfirmFork() {
    vOnForkConversation?.({ name: vForkName.trim(), includeAttachments: vForkIncludeAttachments });
    vShowForkDialog = false;
  }
</script>

<div class="relative {vJustify === 'start' ? 'mt-2' : ''} flex h-6 items-center justify-between">
  <div
    class="{vActionsPosition === 'left'
      ? 'left-0'
      : 'right-0'} flex items-center gap-2 opacity-100 transition-opacity"
  >
    {#if vSiblingInfo && vSiblingInfo.totalSiblings > 1}
      <ChatMessageActionIconsBranchingControls
        siblingInfo={vSiblingInfo}
        onNavigateToSibling={vOnNavigateToSibling}
      />
    {/if}

    <div
      class="pointer-events-auto inset-0 flex items-center gap-1 opacity-100 transition-all duration-150"
    >
      <ActionIcon icon={Copy} tooltip={fTranslate('text_e21f935f11d7')} onclick={vOnCopy} />

      {#if vOnEdit}
        <ActionIcon icon={Edit} tooltip={fTranslate('text_464c4ffd019e')} onclick={vOnEdit} />
      {/if}

      {#if vRole === MessageRole.ASSISTANT && vOnRegenerate}
        <ActionIcon
          icon={RefreshCw}
          tooltip={fTranslate('text_1651031bf58d')}
          onclick={() => vOnRegenerate()}
        />
      {/if}

      {#if vRole === MessageRole.ASSISTANT && vOnContinue}
        <ActionIcon
          icon={ArrowRight}
          tooltip={fTranslate('text_31fbef162594')}
          onclick={vOnContinue}
        />
      {/if}

      {#if vOnForkConversation}
        <ActionIcon
          icon={GitBranch}
          tooltip={fTranslate('text_0e5f7f6732e0')}
          onclick={fHandleOpenForkDialog}
        />
      {/if}

      <ActionIcon icon={Trash2} tooltip={fTranslate('text_e2d0a54968ea')} onclick={vOnDelete} />
    </div>
  </div>

  {#if vShowRawOutputSwitch}
    <div class="flex items-center gap-2">
      <span class="text-xs text-muted-foreground">{fTranslate('text_31d5cb1cf038')}</span>
      <Switch
        checked={vRawOutputEnabled}
        onCheckedChange={(checked) => vOnRawOutputToggle?.(checked)}
      />
    </div>
  {/if}
</div>

<DialogConfirmation
  bind:open={vShowDeleteDialog}
  title={fTranslate('text_43e94c3cb95e')}
  description={vDeletionInfo && vDeletionInfo.totalCount > 1
    ? fTranslate('text_9c6b7e82eb81', {
        p0: vDeletionInfo.totalCount,
        p1: vDeletionInfo.userMessages,
        p2: vDeletionInfo.userMessages > 1 ? 's' : '',
        p3: vDeletionInfo.assistantMessages,
        p4: vDeletionInfo.assistantMessages > 1 ? 's' : ''
      })
    : fTranslate('text_24e7dc980667')}
  confirmText={vDeletionInfo && vDeletionInfo.totalCount > 1
    ? fTranslate('text_5b5338fe8b24', { p0: vDeletionInfo.totalCount })
    : fTranslate('text_e2d0a54968ea')}
  cancelText={fTranslate('text_19766ed6ccb2')}
  variant="destructive"
  icon={Trash2}
  onConfirm={fHandleConfirmDelete}
  onCancel={() => vOnShowDeleteDialogChange(false)}
/>

<DialogConfirmation
  bind:open={vShowForkDialog}
  title={fTranslate('text_84b04588e14c')}
  description={fTranslate('text_5af90254c54e')}
  confirmText={fTranslate('text_8e5b1a73152c')}
  cancelText={fTranslate('text_19766ed6ccb2')}
  icon={GitBranch}
  onConfirm={fHandleConfirmFork}
  onCancel={() => (vShowForkDialog = false)}
>
  <div class="flex flex-col gap-4 py-2">
    <div class="flex flex-col gap-2">
      <Label for="fork-name">{fTranslate('text_7e8cd2056da7')}</Label>

      <Input
        id="fork-name"
        class="text-foreground"
        placeholder={fTranslate('text_1be3523505c4')}
        type="text"
        bind:value={vForkName}
      />
    </div>

    <div class="flex items-center gap-2">
      <Checkbox
        id="fork-attachments"
        checked={vForkIncludeAttachments}
        onCheckedChange={(checked) => {
          vForkIncludeAttachments = checked === true;
        }}
      />

      <Label for="fork-attachments" class="cursor-pointer text-sm font-normal">
        {fTranslate('text_c159fa23ee8e')}
      </Label>
    </div>
  </div>
</DialogConfirmation>
