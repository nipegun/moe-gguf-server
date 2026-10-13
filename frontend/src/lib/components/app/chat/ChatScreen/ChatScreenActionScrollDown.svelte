<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { ArrowDown } from '@lucide/svelte';
  import ActionIcon from '$lib/components/app/actions/ActionIcon.svelte';

  let { onclick: vOnclick }: { onclick: (pE?: MouseEvent) => void } = $props();
</script>

<div class="pointer-events-auto flex justify-center relative h-0">
  <ActionIcon
    icon={ArrowDown}
    onclick={vOnclick}
    ariaLabel="Scroll to bottom"
    tooltip={fTranslate('text_6a36c1b26f1c')}
    size="lg"
    iconSize={cIconClassDefault}
    class="h-9 w-9 rounded-full bg-accent text-accent-foreground absolute bottom-4 shadow-md"
  />
</div>
