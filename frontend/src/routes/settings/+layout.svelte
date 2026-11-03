<script lang="ts">
  import { fTranslate } from '../../lib/i18n';

  import { X } from '@lucide/svelte';
  import { goto } from '$app/navigation';
  import { browser } from '$app/environment';
  import { page } from '$app/state';
  import { ActionIcon } from '$lib/components/app';
  import { cSettingsFallbackExitRoute } from '$lib/constants';

  let { children: vChildren } = $props();

  let vPreviousRouteId = $state<string | null>(null);

  $effect(() => {
    const cCurrentId = page.route.id;
    return () => {
      vPreviousRouteId = cCurrentId;
    };
  });

  function fHandleClose() {
    const cPrevIsSettings = vPreviousRouteId?.startsWith('/settings');
    if (browser && window.history.length > 1 && !cPrevIsSettings) {
      history.back();
    } else {
      goto(cSettingsFallbackExitRoute);
    }
  }
</script>

<div class="fixed top-4.5 right-4 z-50 md:hidden">
  <ActionIcon icon={X} tooltip={fTranslate('text_7d9eb7acb13e')} onclick={fHandleClose} />
</div>

<div class="min-h-full">
  {@render vChildren?.()}
</div>
