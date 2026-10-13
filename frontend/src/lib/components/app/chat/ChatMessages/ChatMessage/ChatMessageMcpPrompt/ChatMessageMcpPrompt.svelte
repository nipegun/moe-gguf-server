<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import {
    ChatMessageActionIcons,
    ChatMessageEditForm,
    ChatMessageMcpPromptContent
  } from '$lib/components/app';
  import { getMessageEditContext } from '$lib/contexts';
  import { MessageRole, McpPromptVariant } from '$lib/enums';
  import type { DatabaseMessageExtraMcpPrompt } from '$lib/types';

  interface Props {
    class?: string;
    message: DatabaseMessage;
    mcpPrompt: DatabaseMessageExtraMcpPrompt;
    siblingInfo?: ChatMessageSiblingInfo | null;
    showDeleteDialog: boolean;
    deletionInfo: {
      totalCount: number;
      userMessages: number;
      assistantMessages: number;
      messageTypes: string[];
    } | null;
    onCopy: () => void;
    onEdit: () => void;
    onDelete: () => void;
    onConfirmDelete: () => void;
    onNavigateToSibling?: (pSiblingId: string) => void;
    onShowDeleteDialogChange: (pShow: boolean) => void;
  }

  let {
    class: vClassName = '',
    message: vMessage,
    mcpPrompt: vMcpPrompt,
    siblingInfo: vSiblingInfo = null,
    showDeleteDialog: vShowDeleteDialog,
    deletionInfo: vDeletionInfo,
    onCopy: vOnCopy,
    onEdit: vOnEdit,
    onDelete: vOnDelete,
    onConfirmDelete: vOnConfirmDelete,
    onNavigateToSibling: vOnNavigateToSibling,
    onShowDeleteDialogChange: vOnShowDeleteDialogChange
  }: Props = $props();

  // Get edit context
  const cEditCtx = getMessageEditContext();
</script>

<div
  aria-label={fTranslate('text_0c58f271260a')}
  class="group flex flex-col items-end gap-3 md:gap-2 {vClassName}"
  role="group"
>
  {#if cEditCtx.isEditing}
    <ChatMessageEditForm />
  {:else}
    <ChatMessageMcpPromptContent
      prompt={vMcpPrompt}
      variant={McpPromptVariant.MESSAGE}
      class="w-full max-w-[80%]"
    />

    {#if vMessage.timestamp}
      <div class="max-w-[80%]">
        <ChatMessageActionIcons
          actionsPosition="right"
          deletionInfo={vDeletionInfo}
          justify="end"
          onConfirmDelete={vOnConfirmDelete}
          onCopy={vOnCopy}
          onDelete={vOnDelete}
          onEdit={vOnEdit}
          onNavigateToSibling={vOnNavigateToSibling}
          onShowDeleteDialogChange={vOnShowDeleteDialogChange}
          siblingInfo={vSiblingInfo}
          showDeleteDialog={vShowDeleteDialog}
          role={MessageRole.USER}
        />
      </div>
    {/if}
  {/if}
</div>
