<script lang="ts">
  import { fTranslate } from '../../../../../i18n';

  import { ArrowUp } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import * as Tooltip from '$lib/components/ui/tooltip';

  interface Props {
    canSend?: boolean;
    disabled?: boolean;
    showErrorState?: boolean;
    tooltipLabel?: string;
  }

  let {
    canSend: vCanSend = false,
    disabled: vDisabled = false,
    showErrorState: vShowErrorState = false,
    tooltipLabel: vTooltipLabel
  }: Props = $props();

  let vIsDisabled = $derived(!vCanSend || vDisabled);
</script>

{#snippet submitButton(props = {})}
  <Button
    type="submit"
    disabled={vIsDisabled}
    class={[
      'md:h-8 md:w-8 h-9 w-9 rounded-full p-0',
      vShowErrorState &&
        'bg-red-400/10 text-red-400 hover:bg-red-400/20 hover:text-red-400 disabled:opacity-100'
    ]}
    {...props}
  >
    <span class="sr-only">{fTranslate('text_f6f4688ff23d')}</span>
    <ArrowUp class="h-12 w-12" />
  </Button>
{/snippet}

{#if vTooltipLabel}
  <Tooltip.Root>
    <Tooltip.Trigger>
      {@render submitButton()}
    </Tooltip.Trigger>

    <Tooltip.Content>
      <p>{vTooltipLabel}</p>
    </Tooltip.Content>
  </Tooltip.Root>
{:else}
  {@render submitButton()}
{/if}
