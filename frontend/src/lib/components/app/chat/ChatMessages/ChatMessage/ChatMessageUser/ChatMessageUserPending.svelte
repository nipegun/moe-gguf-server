<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { ActionIcon, ChatMessageEditForm, ChatMessageUserBubble } from '$lib/components/app';
  import { ArrowUp, Edit, Trash2 } from '@lucide/svelte';
  import { fUseMessageEditContext } from '$lib/hooks/use-message-edit-context.svelte';

  interface Props {
    class?: string;
    content: string;
    extras?: DatabaseMessageExtra[];
    onSendImmediately: () => void;
    onEdit: (pNewContent: string, pExtras?: DatabaseMessageExtra[]) => void;
    onDelete: () => void;
  }

  let {
    class: vClassName = '',
    content: vContent,
    extras: vExtras = [],
    onSendImmediately: vOnSendImmediately,
    onEdit: vOnEdit,
    onDelete: vOnDelete
  }: Props = $props();

  const cEditCtx = fUseMessageEditContext({
    getContent: () => vContent,
    getExtras: () => vExtras,
    onSave: (pContent, pExtras) => vOnEdit(pContent, pExtras)
  });
</script>

<div
  aria-label={fTranslate('text_99c5e7655065')}
  class="group flex flex-col items-end gap-3 transition-opacity hover:opacity-80 md:gap-2 {vClassName} sticky bottom-[calc(8rem+var(--status-bar-height))]"
  role="group"
>
  {#if cEditCtx.isEditing}
    <ChatMessageEditForm />
  {:else}
    <ChatMessageUserBubble
      content={vContent}
      attachments={vExtras}
      textColorClass="text-muted-foreground"
      cardBgClass="dark:bg-primary/8"
      maxHeightStyle="overflow-wrap: anywhere; word-break: break-word;"
    />

    <div class="max-w-[80%]">
      <div class="relative flex h-6 items-center justify-between">
        <div class="right-0 flex items-center gap-2 opacity-100 transition-opacity">
          <div
            class="pointer-events-auto inset-0 flex items-center gap-1 opacity-0 transition-all duration-150 group-hover:opacity-100"
          >
            <ActionIcon
              icon={Edit}
              tooltip={fTranslate('text_464c4ffd019e')}
              onclick={cEditCtx.handleEdit}
            />
            <ActionIcon
              icon={Trash2}
              tooltip={fTranslate('text_e2d0a54968ea')}
              onclick={vOnDelete}
            />
            <ActionIcon
              icon={ArrowUp}
              tooltip={fTranslate('text_d330d7c6066f')}
              onclick={vOnSendImmediately}
            />
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>
