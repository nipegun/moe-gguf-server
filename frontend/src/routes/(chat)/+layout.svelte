<script lang="ts">
  import { page } from '$app/state';
  import { ChatScreen } from '$lib/components/app';

  let { children: vChildren } = $props();

  let vShowCenteredEmpty = $derived(!page.params.id);
</script>

<ChatScreen showCenteredEmpty={vShowCenteredEmpty} />

{@render vChildren?.()}
