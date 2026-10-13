<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import type { Snippet } from 'svelte';
  import * as Popover from '$lib/components/ui/popover';

  interface Props {
    class?: string;
    isOpen?: boolean;
    srLabel?: string;
    onClose?: () => void;
    onKeydown?: (pEvent: KeyboardEvent) => void;
    children: Snippet;
  }

  let {
    class: vClassName = '',
    isOpen: vIsOpen = $bindable(false),
    srLabel: vSrLabel = fTranslate('text_b8db1614cbe6'),
    onClose: vOnClose,
    onKeydown: vOnKeydown,
    children: vChildren
  }: Props = $props();
</script>

<Popover.Root
  bind:open={vIsOpen}
  onOpenChange={(open) => {
    if (!open) {
      vOnClose?.();
    }
  }}
>
  <Popover.Trigger
    class="pointer-events-none absolute inset-0 opacity-0"
    tabindex={-1}
    aria-hidden="true"
  >
    <span class="sr-only">{vSrLabel}</span>
  </Popover.Trigger>

  <Popover.Content
    side="top"
    align="start"
    sideOffset={12}
    class="w-[var(--bits-popover-anchor-width)] max-w-none rounded-xl border-border/50 p-0 shadow-xl {vClassName}"
    onkeydown={vOnKeydown}
    onOpenAutoFocus={(event) => event.preventDefault()}
  >
    {@render vChildren()}
  </Popover.Content>
</Popover.Root>
