<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import * as AlertDialog from '$lib/components/ui/alert-dialog';
  import { AlertTriangle, TimerOff } from '@lucide/svelte';
  import { ErrorDialogType } from '$lib/enums';

  interface Props {
    open: boolean;
    type: ErrorDialogType;
    message: string;
    contextInfo?: { n_prompt_tokens: number; n_ctx: number };
    onOpenChange?: (pOpen: boolean) => void;
  }

  let {
    open: vOpen = $bindable(),
    type: vType,
    message: vMessage,
    contextInfo: vContextInfo,
    onOpenChange: vOnOpenChange
  }: Props = $props();

  const cIsTimeout = $derived(vType === ErrorDialogType.TIMEOUT);
  const cTitle = $derived(cIsTimeout ? 'TCP Timeout' : fTranslate('text_1e2a5d69cec7'));
  const cDescription = $derived(
    cIsTimeout ? fTranslate('text_6cf7c7451458') : fTranslate('text_8766dcd6d3b6')
  );
  const cIconClass = $derived(cIsTimeout ? 'text-destructive' : 'text-amber-500');
  const cBadgeClass = $derived(
    cIsTimeout
      ? 'border-destructive/40 bg-destructive/10 text-destructive'
      : 'border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400'
  );

  function fHandleOpenChange(pNewOpen: boolean) {
    vOpen = pNewOpen;
    vOnOpenChange?.(pNewOpen);
  }
</script>

<AlertDialog.Root open={vOpen} onOpenChange={fHandleOpenChange}>
  <AlertDialog.Content>
    <AlertDialog.Header>
      <AlertDialog.Title class="flex items-center gap-2">
        {#if cIsTimeout}
          <TimerOff class={`h-5 w-5 ${cIconClass}`} />
        {:else}
          <AlertTriangle class={`h-5 w-5 ${cIconClass}`} />
        {/if}

        {cTitle}
      </AlertDialog.Title>

      <AlertDialog.Description>
        {cDescription}
      </AlertDialog.Description>
    </AlertDialog.Header>

    <div class={`rounded-lg border px-4 py-3 text-sm ${cBadgeClass}`}>
      <p class="font-medium">{vMessage}</p>

      {#if vContextInfo}
        <div class="mt-2 space-y-1 text-xs opacity-80">
          <p>
            <span class="font-medium">{fTranslate('text_8dc0accba0c3')}</span>

            {vContextInfo.n_prompt_tokens.toLocaleString()}
          </p>

          {#if vContextInfo.n_ctx}
            <p>
              <span class="font-medium">{fTranslate('text_4c16417781c9')}</span>

              {vContextInfo.n_ctx.toLocaleString()}
            </p>
          {/if}
        </div>
      {/if}
    </div>

    <AlertDialog.Footer>
      <AlertDialog.Action onclick={() => fHandleOpenChange(false)}
        >{fTranslate('text_7d9eb7acb13e')}</AlertDialog.Action
      >
    </AlertDialog.Footer>
  </AlertDialog.Content>
</AlertDialog.Root>
