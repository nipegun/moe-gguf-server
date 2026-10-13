<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { Copy } from '@lucide/svelte';
  import { copyToClipboard } from '$lib/utils';
  import ActionIcon from './ActionIcon.svelte';

  interface Props {
    ariaLabel?: string;
    canCopy?: boolean;
    text: string;
  }

  let {
    ariaLabel: vAriaLabel = fTranslate('text_b749e205302c'),
    canCopy: vCanCopy = true,
    text: vText
  }: Props = $props();
</script>

<ActionIcon
  icon={Copy}
  tooltip={vAriaLabel}
  iconSize={cIconClassDefault}
  disabled={!vCanCopy}
  onclick={() => vCanCopy && copyToClipboard(vText)}
/>
