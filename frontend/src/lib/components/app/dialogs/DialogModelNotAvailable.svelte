<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import * as AlertDialog from '$lib/components/ui/alert-dialog';
  import { AlertTriangle, ArrowRight } from '@lucide/svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';

  interface Props {
    open: boolean;
    modelName: string;
    availableModels?: string[];
    onOpenChange?: (pOpen: boolean) => void;
  }

  let {
    open: vOpen = $bindable(),
    modelName: vModelName,
    availableModels: vAvailableModels = [],
    onOpenChange: vOnOpenChange
  }: Props = $props();

  function fHandleOpenChange(pNewOpen: boolean) {
    vOpen = pNewOpen;
    vOnOpenChange?.(pNewOpen);
  }

  function fHandleSelectModel(pModel: string) {
    // Build URL with selected model, preserving other params
    const cUrl = new URL(page.url);
    cUrl.searchParams.set('model', pModel);

    fHandleOpenChange(false);
    goto(cUrl.toString());
  }
</script>

<AlertDialog.Root open={vOpen} onOpenChange={fHandleOpenChange}>
  <AlertDialog.Content class="max-w-lg">
    <AlertDialog.Header>
      <AlertDialog.Title class="flex items-center gap-2">
        <AlertTriangle class="h-5 w-5 text-amber-500" />
        {fTranslate('text_da6d30754af8')}
      </AlertDialog.Title>

      <AlertDialog.Description>{fTranslate('text_f04c4e9c86d0')}</AlertDialog.Description>
    </AlertDialog.Header>

    <div class="space-y-3">
      <div class="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm">
        <p class="font-medium text-amber-600 dark:text-amber-400">
          {fTranslate('text_65d7deb02f4b')}
          <code class="rounded bg-amber-500/20 px-1.5 py-0.5">{vModelName}</code>
        </p>
      </div>

      {#if vAvailableModels.length > 0}
        <div class="text-sm">
          <p class="mb-2 font-medium text-muted-foreground">{fTranslate('text_2444e1620433')}</p>
          <div class="max-h-48 space-y-1 overflow-y-auto rounded-md border p-1">
            {#each vAvailableModels as model (model)}
              <button
                type="button"
                class="group flex w-full items-center justify-between gap-2 rounded-sm px-3 py-2 text-left text-sm transition-colors hover:bg-accent hover:text-accent-foreground"
                onclick={() => fHandleSelectModel(model)}
              >
                <span class="min-w-0 truncate font-mono text-xs">{model}</span>
                <ArrowRight
                  class="{cIconClassDefault} shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100"
                />
              </button>
            {/each}
          </div>
        </div>
      {/if}
    </div>

    <AlertDialog.Footer>
      <AlertDialog.Action onclick={() => fHandleOpenChange(false)}
        >{fTranslate('text_19766ed6ccb2')}</AlertDialog.Action
      >
    </AlertDialog.Footer>
  </AlertDialog.Content>
</AlertDialog.Root>
