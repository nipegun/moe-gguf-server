<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { Server } from '@lucide/svelte';
  import { ServerStatus } from '$lib/components/app';
  import { fade } from 'svelte/transition';

  interface Props {
    class?: string;
    message?: string;
  }

  let { class: vClassName = '', message: vMessage = fTranslate('text_8f76bcb220f6') }: Props =
    $props();
</script>

<div class="flex h-full items-center justify-center {vClassName}">
  <div class="text-center">
    <div class="mb-4" in:fade={{ duration: 300 }}>
      <div class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
        <Server class="h-8 w-8 animate-pulse text-muted-foreground" />
      </div>

      <h2 class="mb-2 text-xl font-semibold">{fTranslate('text_1de1d0635221')}</h2>

      <p class="text-sm text-muted-foreground">
        {vMessage}
      </p>
    </div>

    <div class="mt-4">
      <ServerStatus class="justify-center" />
    </div>
  </div>
</div>
