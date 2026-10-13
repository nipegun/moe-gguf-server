<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { Eye } from '@lucide/svelte';
  import { ActionIcon, ActionIconCopyToClipboard } from '$lib/components/app';
  import { FileTypeText } from '$lib/enums';

  interface Props {
    code: string;
    language: string;
    disabled?: boolean;
    onPreview?: (pCode: string, pLanguage: string) => void;
  }

  let {
    code: vCode,
    language: vLanguage,
    disabled: vDisabled = false,
    onPreview: vOnPreview
  }: Props = $props();

  const cShowPreview = $derived(vLanguage?.toLowerCase() === FileTypeText.HTML);
</script>

<div class="code-block-actions">
  <ActionIconCopyToClipboard
    text={vCode}
    canCopy={!vDisabled}
    ariaLabel={vDisabled ? fTranslate('text_ea6d26c39694') : fTranslate('text_49a0053f3b0d')}
  />

  {#if cShowPreview}
    <ActionIcon
      icon={Eye}
      tooltip={vDisabled ? fTranslate('text_ea6d26c39694') : fTranslate('text_7e8611fc24e8')}
      disabled={vDisabled}
      onclick={() => vOnPreview!(vCode, vLanguage)}
    />
  {/if}
</div>
