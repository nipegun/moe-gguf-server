<script lang="ts">
  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { Plus } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cAttachmentTooltipText } from '$lib/constants';

  interface Props {
    disabled?: boolean;
    onclick?: (pE: MouseEvent) => void;
  }

  let { disabled: vDisabled = false, onclick: vOnclick }: Props = $props();
</script>

<Tooltip.Root>
  <Tooltip.Trigger class="w-full">
    <Button
      class="file-upload-button md:h-8 md:w-8 h-9 w-9 rounded-full p-0"
      disabled={vDisabled}
      onclick={vOnclick}
      variant="secondary"
      type="button"
    >
      <span class="sr-only">{cAttachmentTooltipText}</span>

      <Plus class={cIconClassDefault} />
    </Button>
  </Tooltip.Trigger>

  <Tooltip.Content>
    <p>{cAttachmentTooltipText}</p>
  </Tooltip.Content>
</Tooltip.Root>
