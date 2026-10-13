<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import * as Dialog from '$lib/components/ui/dialog';
  import { ConversationSelection } from '$lib/components/app';

  interface Props {
    conversations: DatabaseConversation[];
    messageCountMap?: Map<string, number>;
    mode: 'export' | 'import';
    onCancel: () => void;
    onConfirm: (pSelectedConversations: DatabaseConversation[]) => void;
    open?: boolean;
  }

  let {
    conversations: vConversations,
    messageCountMap: vMessageCountMap = new Map(),
    mode: vMode,
    onCancel: vOnCancel,
    onConfirm: vOnConfirm,
    open: vOpen = $bindable(false)
  }: Props = $props();

  let vConversationSelectionRef: ConversationSelection | undefined = $state();

  let vPreviousOpen = $state(false);

  $effect(() => {
    if (vOpen && !vPreviousOpen && vConversationSelectionRef) {
      vConversationSelectionRef.fReset();
    } else if (!vOpen && vPreviousOpen) {
      vOnCancel();
    }

    vPreviousOpen = vOpen;
  });
</script>

<Dialog.Root bind:open={vOpen}>
  <Dialog.Portal>
    <Dialog.Overlay class="z-1000000" />

    <Dialog.Content class="z-1000001 max-w-2xl">
      <Dialog.Header>
        <Dialog.Title>
          {fTranslate('text_950fdc1e46c2')}
          {vMode === 'export' ? fTranslate('text_3664895579f0') : fTranslate('text_2cff9baabf56')}
        </Dialog.Title>

        <Dialog.Description>
          {#if vMode === 'export'}
            {fTranslate('text_2ae6f6a0cc95')}
          {:else}
            {fTranslate('text_22d3a722a92b')}
          {/if}
        </Dialog.Description>
      </Dialog.Header>

      <ConversationSelection
        bind:this={vConversationSelectionRef}
        isOpen={vOpen}
        conversations={vConversations}
        messageCountMap={vMessageCountMap}
        mode={vMode}
        onCancel={vOnCancel}
        onConfirm={vOnConfirm}
      />
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>
